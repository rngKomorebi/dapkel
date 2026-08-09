"""Per-pixel sensor occupancy (hitmaps), in counts and in photon rate.

Two modes, matched to what 'unpack' returns:

    * ``mode="timestamp"`` (``ORT``) - counts frames carrying a valid
      first-photon timestamp. At most one firing per macropixel per frame.
    * ``mode="count"`` (``S*C``/``ORC``) - sums the per-pixel photon counts.

Both are turned into a photon rate the same way: divide by the total *live*
(shutter-open) time of the acquisition,
``nframes * n_files * open_shutter_time_s`` - not the full ~9 us frame period.
Only the open shutter window is photon-sensitive; normalising by the whole frame
would dilute the rate by the frame/window ratio instead of reporting the flux
actually seen during the exposure. See ``docs/guide/hitmap.md``.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import io, plots, reduce, store, timing

__all__ = [
    # stage 1 - compute the hitmap from raw '.bin' files
    "compute_hitmap",
    "compute_hitmap_64",
    # stage 1 - compute and persist ('.npy' + '.meta.json' sidecar)
    "compute_and_save_hitmap",
    "compute_and_save_hitmap_64",
    # stage 2 - plot from an already computed map
    "plot_hitmap",
    # drivers - load a saved hitmap, figures on disk
    "collect_and_plot_hitmap",
]

#: Analysis name: the stage-1 file stem and the results sub-folder.
_KIND = "hitmap"

# Default open shutter time, in seconds - the usual short_window setting. Used
# only as a fallback for a sidecar saved before 'frame_live_s' was recorded in
# it; every new hitmap stores the live time 'timing.resolve_live_time' resolved
# from the firmware version and open_shutter_time_s it was given.
_DEFAULT_OPEN_SHUTTER_TIME_S = 500e-9

# SPAD quadrant layout on the full (64, 64) sensor: each S*C tag reads one
# micropixel per 2x2 macropixel, so the four (32, 32) maps interleave onto
# the full grid. The indices run CLOCKWISE, not row-major::
#
#     S0 (0,0)   S1 (0,1)
#     S3 (1,0)   S2 (1,1)
#
# Kept identical to dcr_analysis and crosstalk_analysis so the assembly and
# the cross-talk directions agree.
_SPAD_LAYOUT = {
    "S0C": (0, 0),
    "S1C": (0, 1),
    "S2C": (1, 1),
    "S3C": (1, 0),
}




def _accumulate_hitmap(
    files: list[str],
    mode: str,
    *,
    nframes: int,
    label: str = "",
) -> np.ndarray:
    """Unpack a list of '.bin' files and accumulate the (32, 32) hitmap.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files to unpack and accumulate.
    mode : str
        ``'timestamp'`` for the ORT occupancy map (per pixel, the number of
        frames carrying a valid first-photon timestamp), or ``'count'`` for
        the accumulated photon-count map (S*C / ORC).
    nframes : int
        Number of frames per file.
    label : str, optional
        Label used in the progress printout. The default is "".

    Returns
    -------
    np.ndarray
        The (32, 32) hitmap accumulated over every frame of every file.

    Raises
    ------
    ValueError
        Raised when ``mode`` is not 'timestamp' or 'count'.
    """
    if mode not in ("timestamp", "count"):
        raise ValueError(f"mode must be 'timestamp' or 'count', got {mode!r}")

    if mode == "timestamp":
        # Occupancy: frames with a valid first-photon timestamp. 'unpack'
        # leaves non-fired slots at <= 0, so validity is 'ts > 0' - a fixed
        # property of the decoding, not a tunable threshold. Do NOT use
        # photon_counts here — in ORT those bits are the low bits of the
        # coarse timestamp, not a real count (see 'unpack').
        extract = lambda ts, pc: (ts > 0).sum(axis=2)  # noqa: E731
    else:  # count
        extract = lambda ts, pc: pc.sum(axis=2)  # noqa: E731

    return reduce.accumulate_frames(
        files,
        extract,
        nframes=nframes,
        need_time_series=mode == "timestamp",
        label=label,
    )


def compute_hitmap(
    folder: str,
    nframes: int,
    mode: str = "timestamp",
    tag: str = "ORT",
    max_files: int | None = None,
) -> np.ndarray:
    """Unpack the binary data and compute the (32, 32) hitmap.

    Builds a single-readout hitmap from the '.bin' files matching ``tag``.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    mode : str, optional
        ``'timestamp'`` for the ORT occupancy hitmap (per pixel, the number
        of frames with a valid first-photon timestamp), or ``'count'`` for
        the accumulated photon-count hitmap (S*C / ORC count programs). The
        default is ``'timestamp'``.
    tag : str, optional
        Filename fragment selecting the files (e.g. ``'ORT'``, ``'ORC'``,
        or a SPAD quadrant tag ``'S0C'``). Use ``''`` to match every
        '.bin' file in the folder. The default is ``'ORT'``.
    max_files : int | None, optional
        Process at most this many files (after natural sorting). Useful
        for a quick preview over a large batch. The default is None (all
        files).

    Returns
    -------
    np.ndarray
        The (32, 32) hitmap.
    """
    files = io.find_bin_files(folder, tag)
    if max_files is not None:
        files = files[:max_files]
    return _accumulate_hitmap(
        files, mode, nframes=nframes, label=tag or mode
    )


def compute_hitmap_64(
    folder: str,
    nframes: int,
    max_files: int | None = None,
) -> np.ndarray:
    """Unpack the four S*C tags and assemble the (64, 64) count hitmap.

    Accumulates the four SPAD tags' photon counts (``mode='count'``) and
    interleaves them onto the full sensor. Only the count programs (S*C)
    tile this way; ORT/ORC are (32, 32) and use 'compute_hitmap'.

    Parameters
    ----------
    folder : str
        Path to the folder with the S*C '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    max_files : int | None, optional
        Process at most this many files per tag. The default is None.

    Returns
    -------
    np.ndarray
        The (64, 64) accumulated-count hitmap for the full sensor.

    Raises
    ------
    FileNotFoundError
        Raised when no '.bin' files for any of the four SPAD tags are
        found in the folder.
    """
    tag_files = {}
    for tag in _SPAD_LAYOUT:
        files = glob.glob(os.path.join(folder, f"*_{tag}*.bin"))
        if files:
            files = sorted(
                files,
                key=lambda fp: int(
                    "".join(filter(str.isdigit, os.path.basename(fp))) or "0"
                ),
            )
            tag_files[tag] = files[:max_files] if max_files else files
    if not tag_files:
        raise FileNotFoundError(
            "No S0C/S1C/S2C/S3C .bin files found in:\n  " + folder
        )

    quadrants = {
        tag: _accumulate_hitmap(files, "count", nframes=nframes, label=tag)
        for tag, files in tag_files.items()
    }
    return reduce.assemble_64(quadrants, _SPAD_LAYOUT)


def plot_hitmap(
    hitmap: np.ndarray,
    cmap: str | None = None,
    *,
    clabel: str = "hits",
    name: str = "Hitmap",
    show_total: bool = True,
) -> plt.Figure:
    """Render a (32, 32)/(64, 64) map as a 2D image.

    Used both for the raw counts/occupancy map and for the derived rate map
    (see 'collect_and_plot_hitmap'); the caller sets the colorbar label,
    title name and whether a total is meaningful.

    Parameters
    ----------
    hitmap : np.ndarray
        Map to plot, e.g. from 'compute_hitmap'/'compute_hitmap_64', or a
        rate map (counts divided by the active acquisition time).
    cmap : str | None, optional
        Matplotlib colormap name. The default is None — use the active
        style's ``image.cmap``.
    clabel : str, optional
        Colorbar label / units (e.g. ``'counts'``, ``'cps'``). The default
        is ``'hits'``.
    name : str, optional
        Map name used in the title (e.g. ``'Hitmap'``, ``'Photon rate'``).
        The default is ``'Hitmap'``.
    show_total : bool, optional
        Include the summed total in the title. Meaningful for counts, not
        for a rate map. The default is True.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    rows, cols = hitmap.shape
    median, maximum, total = plots.map_stats(hitmap)

    stats = f"median {median:.3g}  max {maximum:.3g}"
    if show_total:
        stats += f"  total {total:.3g}"

    return plots.sensor_map(
        hitmap,
        title=f"{name}  {rows}×{cols}\n{stats}  [{clabel}]",
        clabel=clabel,
        cmap=cmap,
    )






def _acquisition_meta(
    files: list[str],
    mode: str,
    tag: str,
    nframes: int,
    firmware_version: str,
    exp_time_s: float | None,
    open_shutter_time_s: float | None,
) -> dict:
    """Build the acquisition metadata saved alongside a hitmap.

    Records what is needed to turn counts into a photon rate later: the frame
    count, the live (shutter-open) time per frame resolved from the firmware,
    and their product - the total live time the rate divides by. Nothing here
    is read back off the '.bin' files, so a saved hitmap stays reduceable to a
    rate without them.

    Parameters
    ----------
    files : list[str]
        The '.bin' files that were accumulated.
    mode : str
        ``'timestamp'`` or ``'count'``.
    tag : str
        The tag used to select the files.
    nframes : int
        Frames per file, as the acquisition was set.
    firmware_version : str
        ``'short_window'`` or ``'full_window'``; see
        'core.timing.resolve_live_time'.
    exp_time_s : float | None
        Legacy exposure time requested of 'full_window' firmware, in seconds.
    open_shutter_time_s : float | None
        Photon-sensitive shutter window per frame, in seconds. Required for
        ``firmware_version='short_window'`` - this is what the rate divides
        by.

    Returns
    -------
    dict
        The metadata dictionary.

    Raises
    ------
    ValueError
        Raised when ``firmware_version='short_window'`` but no
        ``open_shutter_time_s`` is given.
    """
    nfr = int(nframes)
    total_frames = nfr * len(files)
    live_s, live_source = timing.resolve_live_time(
        firmware_version,
        nfr,
        open_shutter_time_s=open_shutter_time_s,
        exp_time_s=exp_time_s,
    )
    if live_s is None:
        raise ValueError(
            "firmware_version='short_window' needs open_shutter_time_s (the "
            "photon-sensitive shutter time per frame, in seconds, e.g. "
            "500e-9)."
        )
    return {
        "mode": mode,
        "tag": tag,
        "n_files": len(files),
        "nframes": nfr,
        "total_frames": total_frames,
        "firmware_version": firmware_version,
        "exp_time_s": exp_time_s,
        "open_shutter_time_s": open_shutter_time_s,
        "frame_live_s": live_s,
        "live_time_source": live_source,
        "live_time_s": total_frames * live_s,
    }


def compute_and_save_hitmap(
    path: str,
    nframes: int,
    mode: str = "timestamp",
    tag: str = "ORT",
    *,
    firmware_version: str = "short_window",
    exp_time_s: float | None = None,
    open_shutter_time_s: float | None = 500e-9,
    max_files: int | None = None,
) -> str:
    """Unpack the '.bin' files and save the (32, 32) hitmap to '.npy'.

    Unpacks once and saves ``processed/<name>_<tag>_hitmap.npy`` plus a
    ``.meta.json`` sidecar recording the frame count and the live time, so
    'collect_and_plot_hitmap' can later render the photon-rate map without
    re-unpacking.

    Parameters
    ----------
    path : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames the acquisition was set to record per '.bin' file.
        Also recorded in the sidecar: the photon rate is
        ``hits / (nframes * n_files * open_shutter_time_s)``, so this is what sets the
        denominator. Do not infer it from the file size - the exe dumps a
        fixed, larger DDR3 region, so the file is bigger than the acquisition.
    mode : str, optional
        ``'timestamp'`` (ORT occupancy) or ``'count'`` (S*C / ORC photon
        counts). See 'compute_hitmap'. The default is ``'timestamp'``.
    tag : str, optional
        Filename fragment selecting the files. The default is ``'ORT'``.
    firmware_version : str, optional
        Which firmware took the data, i.e. how much of each ~9 µs frame is
        photon-sensitive: ``'short_window'`` (the default) means only
        ``open_shutter_time_s`` is live; ``'full_window'`` means the whole
        ``exp_time_s + 9 µs`` cycle is live. See
        'core.timing.resolve_live_time'.
    exp_time_s : float | None, optional
        Legacy exposure time requested of ``firmware_version='full_window'``,
        in seconds; the firmware adds 9 µs to it. Required for
        ``firmware_version='full_window'``, ignored otherwise. The default
        is None.
    open_shutter_time_s : float | None, optional
        Photon-sensitive shutter window per frame, in seconds (50-500 ns on
        the short-window firmware). This is the rate's denominator - required
        for ``firmware_version='short_window'``. The default is ``500e-9``
        (500 ns).
    max_files : int | None, optional
        Process at most this many files. The default is None (all files).

    Returns
    -------
    str
        Path to the saved '.npy' hitmap.
    """
    print(
        f"\n> > > Computing hitmap (mode='{mode}', tag='{tag or 'all'}') "
        "and saving the array < < <\n"
    )
    hitmap = compute_hitmap(path, nframes, mode, tag, max_files)
    print(
        f"  {mode} hitmap: total {hitmap.sum():.0f} hits  "
        f"median {np.median(hitmap):.0f}  max {hitmap.max():.0f}"
    )

    files = io.find_bin_files(path, tag)
    if max_files is not None:
        files = files[:max_files]
    meta = _acquisition_meta(
        files, mode, tag, nframes, firmware_version, exp_time_s, open_shutter_time_s
    )
    out_path = store.save_map(
        hitmap, path, kind=_KIND, tag=(tag or mode), meta=meta, quiet=True
    )
    print(f"\n> > > Hitmap saved to {out_path} < < <")
    return out_path


def compute_and_save_hitmap_64(
    path: str,
    nframes: int,
    *,
    firmware_version: str = "short_window",
    exp_time_s: float | None = None,
    open_shutter_time_s: float | None = 500e-9,
    max_files: int | None = None,
) -> str:
    """Assemble and save the (64, 64) full-sensor count hitmap to '.npy'.

    Like 'compute_and_save_hitmap' but for the four-quadrant (64, 64)
    accumulated-count map (see 'compute_hitmap_64'); saved as
    ``<name>_64_hitmap.npy`` in ``processed``, with a ``.meta.json``
    sidecar (frame count per SPAD tag; each quadrant is acquired for the
    same duration) for the rate map.

    Parameters
    ----------
    path : str
        Path to the folder with the S*C '.bin' data files.
    nframes : int
        Number of frames the acquisition was set to record per '.bin' file.
    firmware_version : str, optional
        ``'short_window'`` (default, only ``open_shutter_time_s`` is live) or
        ``'full_window'`` (``exp_time_s`` + 9 µs, entirely live); see
        'compute_and_save_hitmap'.
    exp_time_s : float | None, optional
        Legacy exposure time requested of ``firmware_version='full_window'``,
        in seconds. The default is None.
    open_shutter_time_s : float | None, optional
        Photon-sensitive shutter window per frame, in seconds - the rate's
        denominator; see 'compute_and_save_hitmap'. The default is ``500e-9``
        (500 ns).
    max_files : int | None, optional
        Process at most this many files per tag. The default is None.

    Returns
    -------
    str
        Path to the saved '.npy' hitmap.
    """
    print("\n> > > Computing (64, 64) full-sensor hitmap and saving < < <\n")
    hitmap = compute_hitmap_64(path, nframes, max_files)
    print(
        f"  counts hitmap: total {hitmap.sum():.0f} hits  "
        f"median {np.median(hitmap):.0f}  max {hitmap.max():.0f}"
    )

    # Frame count from one SPAD tag (all four span the same duration).
    tag_files: list[str] = []
    for tag in _SPAD_LAYOUT:
        tag_files = io.find_bin_files(path, f"_{tag}")
        if tag_files:
            break
    if max_files is not None:
        tag_files = tag_files[:max_files]
    meta = _acquisition_meta(
        tag_files,
        "count",
        "64",
        nframes,
        firmware_version,
        exp_time_s,
        open_shutter_time_s,
    )
    out_path = store.save_map(
        hitmap, path, kind=_KIND, tag="64", meta=meta, quiet=True
    )
    print(f"\n> > > Hitmap saved to {out_path} < < <")
    return out_path


def _live_seconds(meta: dict) -> tuple[float | None, float]:
    """Return the acquisition's ``(total live seconds, live seconds/frame)``.

    ``total_frames * frame_live_s``: the shutter-open time of the run - only
    ``open_shutter_time_s`` under short-window firmware, the whole ``exp_time_s + 9 µs``
    cycle under full-window - so this is the denominator of the photon rate.
    Both factors are read from the sidecar, the ``nframes``/``open_shutter_time_s``/
    firmware the stage-1 call was given, and neither is re-derived from the
    '.bin' files: their size is a fixed DDR3 dump larger than the acquisition,
    and the ``frame_rate_cnt.txt`` beside them is written by an exe we do not
    control, so nothing here depends on it (see
    'core.timing.resolve_live_time').

    A '.npy' whose sidecar has no frame count (saved before the sidecar existed)
    gets no rate map rather than one normalised by a guess. A sidecar with a
    frame count but no resolved live time predates ``open_shutter_time_s`` being
    recorded, and falls back to the 500 ns default shutter time.
    """
    live = float(meta.get("frame_live_s") or _DEFAULT_OPEN_SHUTTER_TIME_S)
    total_frames = meta.get("total_frames")
    if not total_frames:
        return None, live
    return float(total_frames) * live, live


def collect_and_plot_hitmap(
    path: str | None = None,
    mode: str = "timestamp",
    tag: str = "ORT",
    *,
    npy_path: str | None = None,
    cmap: str | None = None,
    rate: bool = True,
) -> np.ndarray:
    """Load a saved hitmap '.npy' and save its hitmap plot(s).

    Reads the saved array - no re-unpacking - and writes two figures to
    ``results/hitmap``: the counts/occupancy map and the **photon-rate** map,
    ``hits / (nframes * n_files * open_shutter_time_s)`` in Hz. Both modes normalise the
    same way, by the live (shutter-open) time of the acquisition; the
    derivation is in ``docs/guide/hitmap.md``. The frame count and live time
    come from the '.meta.json' sidecar written at stage 1.

    Locate the '.npy' from ``path`` or directly via ``npy_path``, which
    ignores ``path``.

    Parameters
    ----------
    path : str | None, optional
        Data-folder path used to locate the saved '.npy'. May be None when
        ``npy_path`` is given. Required otherwise.
    mode : str, optional
        Used with ``tag`` to build the default '.npy' name (the suffix is
        ``tag or mode``) and to label the plots. The default is
        ``'timestamp'``.
    tag : str, optional
        Used to build the default '.npy' name. The default is ``'ORT'``.
    npy_path : str | None, optional
        Explicit path to a saved hitmap '.npy' to plot directly; when given,
        ``path`` / ``mode`` / ``tag`` are ignored (mode/units then come from
        the sidecar). The default is None.
    cmap : str | None, optional
        Matplotlib colormap name for the hitmaps. The default is None —
        use the active style's ``image.cmap``.
    rate : bool, optional
        Also save the photon-rate hitmap. The default is True.

    Returns
    -------
    np.ndarray
        The loaded hitmap (counts/occupancy).

    Raises
    ------
    FileNotFoundError
        Raised when the '.npy' hitmap cannot be found.
    ValueError
        Raised when neither ``path`` nor ``npy_path`` is given.
    """
    if npy_path is not None:
        npy_path = os.path.abspath(npy_path)
        # <root>/processed/<name>.npy -> the figures belong under <root>.
        root = os.path.dirname(os.path.dirname(npy_path))
        base = os.path.basename(npy_path).removesuffix(".npy")
    else:
        if path is None:
            raise ValueError(
                "Provide either path (to locate the saved hitmap) or npy_path."
            )
        root = path
        base = store.map_file_name(
            os.path.basename(os.path.normpath(path)), _KIND, tag or mode
        ).removesuffix(".npy")

    hitmap, meta = store.load_map(
        path, kind=_KIND, tag=(tag or mode), npy_path=npy_path
    )
    mode_eff = meta.get("mode", mode)
    counts_clabel = "photon counts" if mode_eff == "count" else "frames fired"
    print(
        f"  hitmap {hitmap.shape[0]}×{hitmap.shape[1]}: total "
        f"{hitmap.sum():.0f}  median {np.median(hitmap):.0f}  "
        f"max {hitmap.max():.0f}  [{counts_clabel}]"
    )

    # Figures go to 'results/hitmap'; the array stays in 'processed/'.
    results_dir = store.results_dir(root, _KIND)

    # 1) counts / occupancy map
    fig_counts = plot_hitmap(hitmap, cmap=cmap, clabel=counts_clabel)
    store.save_figure(fig_counts, results_dir, f"{base}.png")
    plt.close(fig_counts)

    # 2) photon-rate map: hits / live seconds, i.e.
    #    hits / (nframes * n_files * open_shutter_time_s). Only the acquisition window
    #    inside each ~9 us frame is photon-sensitive (short-window firmware),
    #    so this is the photon flux during the exposure, not diluted by the
    #    readout dead time, for both modes.
    #
    #    In 'timestamp' mode only the *first* photon of a frame is
    #    timestamped, so the map saturates at one firing per open_shutter_time_s - e.g.
    #    2 MHz at a 500 ns window - and under-reports as it approaches that.
    if rate:
        live_time_s, frame_live_s = _live_seconds(meta)
        if live_time_s:
            rate_map = hitmap / live_time_s
            print(
                f"  photon rate: {meta.get('total_frames')} frames x "
                f"{frame_live_s * 1e9:.0f} ns shutter = "
                f"{live_time_s * 1e3:.3f} ms live time  "
                f"median {np.median(rate_map):.3g}  "
                f"max {rate_map.max():.3g} Hz"
            )
            fig_rate = plot_hitmap(
                rate_map,
                cmap=cmap,
                clabel="Hz",
                name="Photon rate",
                show_total=False,
            )
            store.save_figure(fig_rate, results_dir, f"{base}_rate.png")
            # plt.close(fig_rate)  # left open on purpose, for interactive use
        else:
            print(
                "  (no rate map — the sidecar has no frame count; recompute "
                "with compute_and_save_hitmap(..., nframes=<frames>))"
            )

    return hitmap
