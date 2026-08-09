"""Dark count rate (DCR) maps, per quadrant and full-sensor.

DCR is accumulated dark counts divided by the time the sensor was actually
collecting. That live time is NOT the frame period: under the current
``short_window`` firmware only the user-set ``open_shutter_time_s`` of each
~9 us frame is photon-sensitive, so ``open_shutter_time_s`` is required.

What counts as "a count" depends on the acquisition program, exactly as in
'hitmap_analysis':

    * ``mode="count"`` (``S*C`` / ``ORC``) - the per-pixel photon counter is
      summed over frames.
    * ``mode="timestamp"`` (``S*T`` / ``ORT``) - the counter field does **not**
      exist; 'unpack' returns the low bits of the coarse timestamp in its place
      (see the note in 'unpack'). Summing it gives a number tens of times too
      large. Here the count is the number of frames carrying a valid timestamp,
      ``(ts > 0).sum(axis=2)``.

In timestamp mode only the *first* dark count of a frame is recorded, so the
occupancy saturates at one per ``open_shutter_time_s`` (125 kcps at an 8 µs
window) and under-reports below it. ``pileup_correct=True`` inverts that
censoring under Poisson statistics; see 'compute_dcr_32'.

See ``docs/guide/dcr.md``.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import io, plots, reduce, store, timing

# Re-exported for backwards compatibility: CLK_PERIOD now lives in
# 'dapkel.core.timing'. Kept importable as 'dcr_analysis.CLK_PERIOD'.
from dapkel.core.timing import CLK_PERIOD  # noqa: F401

__all__ = [
    # stage 1 - compute the DCR map from raw '.bin' files
    "compute_dcr_32",
    "compute_dcr_64",
    # stage 1 - compute and persist ('.npy' + '.meta.json' sidecar)
    "compute_and_save_dcr_32",
    "compute_and_save_dcr_64",
    # stage 2 - plot from an already computed map
    "plot_heatmap",
    "plot_distribution",
    # drivers - folder in, figures on disk
    "collect_and_plot_dcr_32",
    "collect_and_plot_dcr_64",
]

#: Analysis name: the stage-1 file stem and results sub-folder.
_KIND = "dcr"

# SPAD (micropixel) layout inside one 2x2 macropixel, as (drow, dcol) array
# offsets. The indices run CLOCKWISE, not row-major::
#
#     S0 (0,0)   S1 (0,1)
#     S3 (1,0)   S2 (1,1)
#
# Kept identical to hitmap_analysis and crosstalk_analysis so the (64, 64)
# assembly and the cross-talk directions agree.
_SPAD_LAYOUT = {
    "S0C": (0, 0),
    "S1C": (0, 1),
    "S2C": (1, 1),
    "S3C": (1, 0),
}

#: Readout mode -> the letter the acquisition program appends to the SPAD tag.
#: ``S0C`` is the counting program, ``S0T`` the timestamping one; the geometry
#: in '_SPAD_LAYOUT' is the same for both.
_MODE_TAG_LETTER = {"count": "C", "timestamp": "T"}


def _quadrant_tag(tag: str, mode: str) -> str:
    """Return the SPAD tag of '_SPAD_LAYOUT' spelled for a readout mode."""
    return tag[:-1] + _MODE_TAG_LETTER[mode]


def _check_mode(mode: str, tag: str = "") -> None:
    """Validate ``mode``, and catch the tag/mode mismatch that inflates DCR.

    A ``*T`` tag read with ``mode='count'`` sums coarse-timestamp bits as if
    they were photon counts, which silently multiplies the rate by ~50. That is
    a hard error rather than a warning, because the resulting map looks
    plausible.
    """
    if mode not in _MODE_TAG_LETTER:
        raise ValueError(f"mode must be 'timestamp' or 'count', got {mode!r}")

    stem = tag.upper().rstrip("0123456789")
    expected = {"T": "timestamp", "C": "count"}.get(stem[-1:] if stem else "")
    if expected is not None and expected != mode:
        raise ValueError(
            f"tag {tag!r} is a {expected}-mode acquisition but mode={mode!r} "
            f"was requested. In timestamp mode the photon-count field holds "
            f"coarse-timestamp bits, not counts, so summing it overstates the "
            f"DCR by more than an order of magnitude. Pass mode='{expected}'."
        )


def _pileup_correct(occupancy: np.ndarray, nframes_total: int) -> np.ndarray:
    """Undo the one-count-per-frame censoring of timestamp mode.

    Each frame is a Bernoulli trial: with a true dark count rate ``r`` and a
    live window ``w``, a frame fires with probability ``p = 1 - exp(-r*w)``
    because only the first count is timestamped. So the maximum-likelihood rate
    is ``r = -ln(1 - p) / w``, i.e. the occupancy count is replaced by
    ``-nframes_total * ln(1 - hits/nframes_total)`` before the division by the
    live time.

    A pixel that fires in *every* frame has no upper bound (``p = 1``); it is
    clipped to the ``p = 1 - 1/(2*nframes_total)`` half-count level, so a
    saturated pixel reads as a finite lower bound rather than ``inf``.
    """
    p = occupancy / nframes_total
    p = np.minimum(p, 1.0 - 0.5 / nframes_total)
    return -nframes_total * np.log1p(-p)


def _accumulate_dcr(
    files: list[str],
    nframes: int,
    frame_time_s: float,
    *,
    mode: str = "count",
    pileup_correct: bool = False,
    label: str = "",
) -> np.ndarray:
    """Unpack a list of '.bin' files and compute the (32, 32) DCR map.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files to unpack and accumulate.
    nframes : int
        Number of frames stored in each '.bin' file.
    frame_time_s : float
        Length of a single frame, in seconds.
    mode : str, optional
        ``'count'`` (default) sums the photon-count field; ``'timestamp'``
        counts the frames carrying a valid timestamp. See the module docstring.
    pileup_correct : bool, optional
        ``'timestamp'`` mode only: invert the one-count-per-frame censoring
        (see '_pileup_correct'). The default is False.
    label : str, optional
        Label used in the progress printout. The default is "".

    Returns
    -------
    np.ndarray
        The (32, 32) dark count rate map, in counts per second.

    Raises
    ------
    ValueError
        Raised when the total acquisition time works out to zero.
    """
    _check_mode(mode)
    timestamp = mode == "timestamp"
    count_sum = reduce.accumulate_frames(
        files,
        (lambda ts, pc: (ts > 0).sum(axis=2))
        if timestamp
        else (lambda ts, pc: pc.sum(axis=2)),
        nframes=nframes,
        need_time_series=timestamp,
        label=label,
    )
    total_frames = nframes * len(files)
    total_time = total_frames * frame_time_s
    if total_time == 0:
        raise ValueError(
            "Total acquisition time is zero — check nframes and --exp-time."
        )
    if timestamp and pileup_correct:
        count_sum = _pileup_correct(count_sum, total_frames)
    dcr = count_sum / total_time
    return dcr


def compute_dcr_32(
    folder: str,
    nframes: int,
    exp_time_s: float | None = None,
    tag: str = "",
    *,
    mode: str = "count",
    pileup_correct: bool = False,
    firmware_version: str = "short_window",
    open_shutter_time_s: float | None = 500e-9,
) -> np.ndarray:
    """Unpack the binary data and compute the (32, 32) DCR map.

    Normalised by the *live* time per frame, which depends on the firmware
    (see 'core.timing.resolve_live_time'): ``'short_window'`` counts only
    ``open_shutter_time_s`` as live, ``'full_window'`` the whole frame.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time_s : float | None, optional
        Legacy exposure time in seconds (``firmware_version='full_window'``
        only): 0 → 9 µs actual, 10e-6 → 19 µs. If None, the 9 µs
        free-running base is used. The default is None.
    tag : str, optional
        SPAD quadrant tag - 'S0C'...'S3C' in counting mode, 'S0T'...'S3T' (or
        'ORT') in timestamping mode. The default is "", which matches every
        '.bin' file in the folder.
    mode : str, optional
        ``'count'`` (default) sums the per-pixel photon counter; use
        ``'timestamp'`` for ``S*T``/``ORT`` data, where that field holds
        coarse-timestamp bits instead and the count is the number of frames
        that fired. A tag whose letter contradicts ``mode`` is an error.
    pileup_correct : bool, optional
        ``'timestamp'`` mode only. Timestamp mode records at most one count per
        frame, so the raw rate saturates at ``1 / open_shutter_time_s`` (125 kcps at an
        8 µs window) and under-reports well before that. With True the Poisson
        censoring is inverted, ``r = -ln(1 - p) / open_shutter_time_s``. The default is
        False, which reports the raw occupancy rate.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; see
        'resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_window'``). The default is ``500e-9``
        (500 ns).

    Returns
    -------
    np.ndarray
        The (32, 32) dark count rate map, in counts per second.

    Raises
    ------
    ValueError
        Raised when ``firmware_version='short_window'`` but no ``open_shutter_time_s``
        is given, or when ``mode`` is unknown or contradicts ``tag``.
    """
    _check_mode(mode, tag)
    live, _ = timing.resolve_live_time(
        firmware_version, nframes,
        open_shutter_time_s=open_shutter_time_s, exp_time_s=exp_time_s,
    )
    if live is None:
        raise ValueError(
            "firmware_version='short_window' needs open_shutter_time_s (the "
            "photon-sensitive window per frame, in seconds, e.g. 200e-9)."
        )
    # The DCR quadrant tags are matched with a leading underscore ('*_S0C*')
    # so a tag cannot match a filename that merely contains those characters.
    files = io.find_bin_files(folder, tag, require_separator=True)
    return _accumulate_dcr(
        files, nframes, live,
        mode=mode, pileup_correct=pileup_correct, label=tag,
    )


def compute_dcr_64(
    folder: str,
    nframes: int,
    exp_time_s: float | None = None,
    *,
    mode: str = "count",
    pileup_correct: bool = False,
    firmware_version: str = "short_window",
    open_shutter_time_s: float | None = 500e-9,
) -> np.ndarray:
    """Unpack the binary data and compute the (64, 64) DCR map.

    Interleaves the four quadrant maps onto the full sensor. Normalised by
    the live time per frame; see 'compute_dcr_32'.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time_s : float | None, optional
        Legacy exposure time in seconds (``firmware_version='full_window'``
        only). If None, the 9 µs free-running base is used. The default
        is None.
    mode : str, optional
        ``'count'`` (default, ``S*C`` files) or ``'timestamp'`` (``S*T``
        files); see 'compute_dcr_32'. The mode also selects which quadrant
        files are looked for.
    pileup_correct : bool, optional
        ``'timestamp'`` mode only; see 'compute_dcr_32'. The default is False.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; see
        'resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_window'``). The default is ``500e-9``
        (500 ns).

    Returns
    -------
    np.ndarray
        The (64, 64) dark count rate map, in counts per second.

    Raises
    ------
    FileNotFoundError
        Raised when no '.bin' files for any of the four SPAD tags are
        found in the folder.
    ValueError
        Raised when ``firmware_version='short_window'`` but no ``open_shutter_time_s``
        is given, or when ``mode`` is unknown.
    """
    _check_mode(mode)
    live, _ = timing.resolve_live_time(
        firmware_version, nframes,
        open_shutter_time_s=open_shutter_time_s, exp_time_s=exp_time_s,
    )
    if live is None:
        raise ValueError(
            "firmware_version='short_window' needs open_shutter_time_s (the "
            "photon-sensitive window per frame, in seconds, e.g. 200e-9)."
        )
    frame_time_s = live

    # The layout is keyed by the counting-mode tags; timestamping writes the
    # same four quadrants as 'S0T'...'S3T'.
    tag_files = {}
    for tag in _SPAD_LAYOUT:
        files = glob.glob(
            os.path.join(folder, f"*_{_quadrant_tag(tag, mode)}*.bin")
        )
        if files:
            tag_files[tag] = sorted(
                files,
                key=lambda fp: int(
                    "".join(filter(str.isdigit, os.path.basename(fp))) or "0"
                ),
            )
    if not tag_files:
        letter = _MODE_TAG_LETTER[mode]
        raise FileNotFoundError(
            f"No S0{letter}/S1{letter}/S2{letter}/S3{letter} .bin files "
            f"found in:\n  {folder}"
        )

    quadrants = {
        tag: _accumulate_dcr(
            files, nframes, frame_time_s,
            mode=mode, pileup_correct=pileup_correct,
            label=_quadrant_tag(tag, mode),
        )
        for tag, files in tag_files.items()
    }
    return reduce.assemble_64(quadrants, _SPAD_LAYOUT)


def plot_heatmap(dcr: np.ndarray, cmap: str | None = None) -> plt.Figure:
    """Plot a dark count rate map as a 2D heatmap.

    Parameters
    ----------
    dcr : np.ndarray
        Dark count rate map, e.g. from 'compute_dcr_32'/'compute_dcr_64'.
    cmap : str | None, optional
        Matplotlib colormap name. The default is None — use the active
        style's ``image.cmap``.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    rows, cols = dcr.shape
    median, maximum, _ = plots.map_stats(dcr)
    return plots.sensor_map(
        dcr,
        title=(
            f"DCR map  {rows}×{cols} SPADs\n"
            f"median {median:.0f}  max {maximum:.0f}  cps"
        ),
        cmap=cmap,
    )


def plot_distribution(dcr: np.ndarray) -> plt.Figure:
    """Plot the sorted per-pixel dark count rate distribution.

    Parameters
    ----------
    dcr : np.ndarray
        Dark count rate map, e.g. from 'compute_dcr_32'/'compute_dcr_64'.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    return plots.sorted_distribution(
        dcr,
        ylabel="DCR  (cps)",
        title="Sorted DCR distribution",
        logy=True,  # DCR spans decades
        fmt=".0f",
        unit="cps",
    )



def _warn_if_saturated(
    dcr: np.ndarray,
    mode: str,
    pileup_correct: bool,
    open_shutter_time_s: float | None,
) -> None:
    """Print how close a timestamp-mode map is to the one-per-frame ceiling.

    Silent for counting mode, which has no such ceiling, and for an already
    corrected map. Printed rather than raised: a saturated *pixel* is normal in
    a DCR map, it is only the number attached to it that stops being a rate.
    """
    if mode != "timestamp" or pileup_correct or not open_shutter_time_s:
        return
    ceiling = 1.0 / open_shutter_time_s
    worst = float(dcr.max()) / ceiling
    if worst < 0.05:
        return
    n_hot = int((dcr > 0.5 * ceiling).sum())
    print(
        f"  ! timestamp mode records at most one count per frame, so this "
        f"map saturates at {ceiling:.3g} cps. The hottest pixel is at "
        f"{worst * 100:.0f}% of that"
        + (
            f" and {n_hot} pixel{'s are' if n_hot > 1 else ' is'} past 50%"
            if n_hot
            else ""
        )
        + ", so those rates read low. Pass pileup_correct=True for the "
        "Poisson-corrected rate."
    )


def _dcr_meta(
    path: str,
    nframes: int,
    tag: str,
    exp_time_s: float | None,
    firmware_version: str,
    open_shutter_time_s: float | None,
    mode: str = "count",
    pileup_correct: bool = False,
) -> dict:
    """Build the acquisition metadata sidecar for a saved DCR map."""
    frame_live_s, live_source = timing.resolve_live_time(
        firmware_version,
        nframes,
        open_shutter_time_s=open_shutter_time_s,
        exp_time_s=exp_time_s,
    )
    return {
        "tag": tag,
        "nframes": int(nframes),
        "mode": mode,
        "pileup_correct": bool(pileup_correct) and mode == "timestamp",
        "firmware_version": firmware_version,
        "open_shutter_time_s": open_shutter_time_s,
        "exp_time_s": exp_time_s,
        "frame_live_s": frame_live_s,
        "live_time_source": live_source,
        "unit": "cps",
    }


def compute_and_save_dcr_32(
    path: str,
    nframes: int,
    exp_time_s: float | None = None,
    tag: str = "",
    *,
    mode: str = "count",
    pileup_correct: bool = False,
    firmware_version: str = "short_window",
    open_shutter_time_s: float | None = 500e-9,
) -> str:
    """Compute the (32, 32) DCR map and save it under ``processed/``.

    The stage-1 half of the DCR analysis: unpack once, write the array plus a
    '.meta.json' sidecar, and return the path. 'collect_and_plot_dcr_32' can
    then replot it with ``from_saved=True`` without re-unpacking.

    Parameters
    ----------
    path : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time_s : float | None, optional
        Value passed to Kelpie_v2.exe, in seconds. The default is None.
    tag : str, optional
        SPAD quadrant tag ('S0C'...'S3C', or 'S0T'...'S3T'/'ORT' in
        timestamping mode). The default is "".
    mode : str, optional
        ``'count'`` (default) or ``'timestamp'``; see 'compute_dcr_32'.
    pileup_correct : bool, optional
        ``'timestamp'`` mode only; see 'compute_dcr_32'. The default is False.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds. The default is
        ``500e-9`` (500 ns).

    Returns
    -------
    str
        Path to the saved '.npy'.
    """
    dcr = compute_dcr_32(
        path, nframes, exp_time_s, tag,
        mode=mode, pileup_correct=pileup_correct,
        firmware_version=firmware_version, open_shutter_time_s=open_shutter_time_s,
    )
    return store.save_map(
        dcr,
        path,
        kind=_KIND,
        tag=tag,
        meta=_dcr_meta(
            path, nframes, tag, exp_time_s, firmware_version, open_shutter_time_s,
            mode, pileup_correct,
        ),
    )


def compute_and_save_dcr_64(
    path: str,
    nframes: int,
    exp_time_s: float | None = None,
    *,
    mode: str = "count",
    pileup_correct: bool = False,
    firmware_version: str = "short_window",
    open_shutter_time_s: float | None = 500e-9,
) -> str:
    """Assemble the (64, 64) full-sensor DCR map and save it.

    Parameters
    ----------
    path : str
        Path to the folder with the four SPAD quadrants' '.bin' files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time_s : float | None, optional
        Value passed to Kelpie_v2.exe, in seconds. The default is None.
    mode : str, optional
        ``'count'`` (default) or ``'timestamp'``; see 'compute_dcr_32'.
    pileup_correct : bool, optional
        ``'timestamp'`` mode only; see 'compute_dcr_32'. The default is False.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds. The default is
        ``500e-9`` (500 ns).

    Returns
    -------
    str
        Path to the saved '.npy'.
    """
    dcr = compute_dcr_64(
        path, nframes, exp_time_s,
        mode=mode, pileup_correct=pileup_correct,
        firmware_version=firmware_version, open_shutter_time_s=open_shutter_time_s,
    )
    return store.save_map(
        dcr,
        path,
        kind=_KIND,
        tag="64",
        meta=_dcr_meta(
            path, nframes, "64", exp_time_s, firmware_version, open_shutter_time_s,
            mode, pileup_correct,
        ),
    )


def collect_and_plot_dcr_32(
    path: str,
    nframes: int,
    exp_time_s: float | None = None,
    tag: str = "",
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    cmap: str | None = None,
    *,
    mode: str = "count",
    pileup_correct: bool = False,
    firmware_version: str = "short_window",
    open_shutter_time_s: float | None = 500e-9,
    from_saved: bool = False,
) -> np.ndarray:
    """Unpack, compute, and plot the (32, 32) DCR map for a SPAD tag.

    Computes (and saves) the map, then writes both the heatmap and the
    sorted distribution to ``results/dcr``.

    Parameters
    ----------
    path : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time_s : float | None, optional
        Value passed to Kelpie_v2.exe, in seconds. See
        'compute_dcr_32' for details. The default is None.
    tag : str, optional
        SPAD quadrant tag ('S0C'...'S3C', or 'S0T'...'S3T'/'ORT' in
        timestamping mode). The default is "", which matches every '.bin'
        file in the folder.
    daughterboard_number : str | None, optional
        Camera daughterboard number, used to look up the hot/warm
        pixel mask. Mask support is not implemented yet. The default
        is None.
    motherboard_number : str | None, optional
        Camera motherboard number, used together with
        'daughterboard_number' to look up the hot/warm pixel mask.
        Mask support is not implemented yet. The default is None.
    cmap : str | None, optional
        Matplotlib colormap name for the heatmap. The default is None —
        use the active style's ``image.cmap``.
    mode : str, optional
        ``'count'`` (default) or ``'timestamp'``; see 'compute_dcr_32'.
    pileup_correct : bool, optional
        ``'timestamp'`` mode only; see 'compute_dcr_32'. The default is False.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; see
        'resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_window'``). The default is ``500e-9``
        (500 ns).
    from_saved : bool, optional
        Load the map saved by 'compute_and_save_dcr_32' instead of
        re-unpacking the '.bin' files - use this to retry a colormap. The
        default is False, which computes and then saves the array.

    Returns
    -------
    np.ndarray
        The (32, 32) dark count rate map, in counts per second.

    """
    # TODO: once per-board hot/warm pixel mask files are available,
    # use daughterboard_number/motherboard_number to load and apply
    # the mask here (see daplis.functions.utils.apply_mask).

    print(
        f"\n> > > Collecting DCR data for tag '{tag or 'all'}' "
        f"(mode='{mode}') and plotting the heatmap and distribution < < <\n"
    )

    if from_saved:
        dcr, _ = store.load_map(path, kind=_KIND, tag=tag)
    else:
        dcr = compute_dcr_32(
            path, nframes, exp_time_s, tag,
            mode=mode, pileup_correct=pileup_correct,
            firmware_version=firmware_version, open_shutter_time_s=open_shutter_time_s,
        )
        store.save_map(
            dcr,
            path,
            kind=_KIND,
            tag=tag,
            meta=_dcr_meta(
                path, nframes, tag, exp_time_s, firmware_version, open_shutter_time_s,
                mode, pileup_correct,
            ),
            quiet=True,
        )
        _warn_if_saturated(dcr, mode, pileup_correct, open_shutter_time_s)

    name = os.path.basename(os.path.normpath(path))
    tag_suffix = tag if tag else "dcr"
    results_dir = store.results_dir(path, _KIND, create=False)

    fig_heatmap = plot_heatmap(dcr, cmap=cmap)
    store.save_figure(fig_heatmap, results_dir, f"{name}_{tag_suffix}_heatmap.png")
    plt.close(fig_heatmap)

    fig_distribution = plot_distribution(dcr)
    store.save_figure(
        fig_distribution,
        results_dir,
        f"{name}_{tag_suffix}_distribution.png",
    )
    plt.close(fig_distribution)

    return dcr


def collect_and_plot_dcr_64(
    path: str,
    nframes: int,
    exp_time_s: float | None = None,
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    cmap: str | None = None,
    *,
    mode: str = "count",
    pileup_correct: bool = False,
    firmware_version: str = "short_window",
    open_shutter_time_s: float | None = 500e-9,
    from_saved: bool = False,
) -> np.ndarray:
    """Unpack, compute, and plot the (64, 64) full sensor DCR map.

    Computes (and saves) the map, then writes both the heatmap and the
    sorted distribution to ``results/dcr``.

    Parameters
    ----------
    path : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time_s : float | None, optional
        Value passed to Kelpie_v2.exe, in seconds. See
        'compute_dcr_64' for details. The default is None.
    daughterboard_number : str | None, optional
        Camera daughterboard number, used to look up the hot/warm
        pixel mask. Mask support is not implemented yet. The default
        is None.
    motherboard_number : str | None, optional
        Camera motherboard number, used together with
        'daughterboard_number' to look up the hot/warm pixel mask.
        Mask support is not implemented yet. The default is None.

    cmap : str | None, optional
        Matplotlib colormap name for the heatmap. The default is None —
        use the active style's ``image.cmap``.
    mode : str, optional
        ``'count'`` (default) or ``'timestamp'``; see 'compute_dcr_32'.
    pileup_correct : bool, optional
        ``'timestamp'`` mode only; see 'compute_dcr_32'. The default is False.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; see
        'resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_window'``). The default is ``500e-9``
        (500 ns).
    from_saved : bool, optional
        Load the map saved by 'compute_and_save_dcr_64' instead of
        re-unpacking the '.bin' files. The default is False, which computes
        and then saves the array.

    Returns
    -------
    np.ndarray
        The (64, 64) dark count rate map, in counts per second.


    """
    # TODO: once per-board hot/warm pixel mask files are available,
    # use daughterboard_number/motherboard_number to load and apply
    # the mask here (see daplis.functions.utils.apply_mask).

    print(
        f"\n> > > Collecting DCR data for the full sensor (mode='{mode}') "
        "and plotting the heatmap and distribution < < <\n"
    )

    if from_saved:
        dcr, _ = store.load_map(path, kind=_KIND, tag="64")
    else:
        dcr = compute_dcr_64(
            path, nframes, exp_time_s,
            mode=mode, pileup_correct=pileup_correct,
            firmware_version=firmware_version, open_shutter_time_s=open_shutter_time_s,
        )
        store.save_map(
            dcr,
            path,
            kind=_KIND,
            tag="64",
            meta=_dcr_meta(
                path, nframes, "64", exp_time_s, firmware_version, open_shutter_time_s,
                mode, pileup_correct,
            ),
            quiet=True,
        )
        _warn_if_saturated(dcr, mode, pileup_correct, open_shutter_time_s)

    name = os.path.basename(os.path.normpath(path))
    results_dir = store.results_dir(path, _KIND, create=False)

    fig_heatmap = plot_heatmap(dcr, cmap=cmap)
    store.save_figure(
        fig_heatmap,
        results_dir,
        f"{name}_full_heatmap.png",
    )

    fig_distribution = plot_distribution(dcr)
    store.save_figure(
        fig_distribution,
        results_dir,
        f"{name}_full_distribution.png",
    )

    return dcr
