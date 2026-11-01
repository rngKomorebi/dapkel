"""ARCHIVED from dapkel.functions.dcr_analysis (branch modules-restructure, 2026-10-31).

Compute-only (no save) entry points and the separate 32/64 variants. Now one 'process_dcr' (the tag picks 32 or 64) and 'plot_dcr'.

Reference only: not packaged, not tested, not maintained. Copied verbatim
from the pre-restructure module (commit 6b49a4b), so it still refers to
helpers and names that no longer exist in 'src/'. To revive a piece, port it
into the live module rather than importing it from here.
"""

from __future__ import annotations
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from dapkel.core import io, plots, reduce, store, timing
from dapkel.core.timing import CLK_PERIOD  # noqa: F401


def compute_dcr_32(
    path: str,
    nframes: int,
    exp_time_s: float | None = None,
    tag: str = "",
    *,
    mode: str = "count",
    pileup_correct: bool = False,
    firmware_version: str = "short_exposure",
    open_shutter_time_s: float | None = 500e-9,
) -> np.ndarray:
    """Unpack the binary data and compute the (32, 32) DCR map.

    Normalised by the *live* time per frame, which depends on the firmware
    (see 'core.timing.resolve_live_time'): ``'short_exposure'`` counts only
    ``open_shutter_time_s`` as live, ``'long_exposure'`` the whole frame.

    Parameters
    ----------
    path : str
        Path to the path with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time_s : float | None, optional
        Legacy exposure time in seconds (``firmware_version='long_exposure'``
        only): 0 → 9 µs actual, 10e-6 → 19 µs. If None, the 9 µs
        free-running base is used. The default is None.
    tag : str, optional
        SPAD quadrant tag - 'S0C'...'S3C' in counting mode, 'S0T'...'S3T' (or
        'ORT') in timestamping mode. The default is "", which matches every
        '.bin' file in the path.
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
        ``'short_exposure'`` (default) or ``'long_exposure'``; see
        'resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_exposure'``). The default is ``500e-9``
        (500 ns).

    Returns
    -------
    np.ndarray
        The (32, 32) dark count rate map, in counts per second.

    Raises
    ------
    ValueError
        Raised when ``firmware_version='short_exposure'`` but no ``open_shutter_time_s``
        is given, or when ``mode`` is unknown or contradicts ``tag``.
    """
    _check_mode(mode, tag)
    live, _ = timing.resolve_live_time(
        firmware_version, nframes,
        open_shutter_time_s=open_shutter_time_s, exp_time_s=exp_time_s,
    )
    if live is None:
        raise ValueError(
            "firmware_version='short_exposure' needs open_shutter_time_s (the "
            "photon-sensitive window per frame, in seconds, e.g. 200e-9)."
        )
    # The DCR quadrant tags are matched with a leading underscore ('*_S0C*')
    # so a tag cannot match a filename that merely contains those characters.
    files = io.find_bin_files(path, tag, require_separator=True)
    return _accumulate_dcr(
        files, nframes, live,
        mode=mode, pileup_correct=pileup_correct, label=tag,
    )


def compute_dcr_64(
    path: str,
    nframes: int,
    exp_time_s: float | None = None,
    *,
    mode: str = "count",
    pileup_correct: bool = False,
    firmware_version: str = "short_exposure",
    open_shutter_time_s: float | None = 500e-9,
) -> np.ndarray:
    """Unpack the binary data and compute the (64, 64) DCR map.

    Interleaves the four quadrant maps onto the full sensor. Normalised by
    the live time per frame; see 'compute_dcr_32'.

    Parameters
    ----------
    path : str
        Path to the path with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time_s : float | None, optional
        Legacy exposure time in seconds (``firmware_version='long_exposure'``
        only). If None, the 9 µs free-running base is used. The default
        is None.
    mode : str, optional
        ``'count'`` (default, ``S*C`` files) or ``'timestamp'`` (``S*T``
        files); see 'compute_dcr_32'. The mode also selects which quadrant
        files are looked for.
    pileup_correct : bool, optional
        ``'timestamp'`` mode only; see 'compute_dcr_32'. The default is False.
    firmware_version : str, optional
        ``'short_exposure'`` (default) or ``'long_exposure'``; see
        'resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_exposure'``). The default is ``500e-9``
        (500 ns).

    Returns
    -------
    np.ndarray
        The (64, 64) dark count rate map, in counts per second.

    Raises
    ------
    FileNotFoundError
        Raised when no '.bin' files for any of the four SPAD tags are
        found in the path.
    ValueError
        Raised when ``firmware_version='short_exposure'`` but no ``open_shutter_time_s``
        is given, or when ``mode`` is unknown.
    """
    _check_mode(mode)
    live, _ = timing.resolve_live_time(
        firmware_version, nframes,
        open_shutter_time_s=open_shutter_time_s, exp_time_s=exp_time_s,
    )
    if live is None:
        raise ValueError(
            "firmware_version='short_exposure' needs open_shutter_time_s (the "
            "photon-sensitive window per frame, in seconds, e.g. 200e-9)."
        )
    frame_time_s = live

    # The layout is keyed by the counting-mode tags; timestamping writes the
    # same four quadrants as 'S0T'...'S3T'.
    tag_files = {}
    for tag in _SPAD_LAYOUT:
        files = glob.glob(
            os.path.join(path, f"*_{_quadrant_tag(tag, mode)}*.bin")
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
            f"found in:\n  {path}"
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
    firmware_version: str = "short_exposure",
    open_shutter_time_s: float | None = 500e-9,
    reuse: bool = False,
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
        ``'short_exposure'`` (default) or ``'long_exposure'``; see
        'resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_exposure'``). The default is ``500e-9``
        (500 ns).
    reuse : bool, optional
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

    if reuse:
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
    firmware_version: str = "short_exposure",
    open_shutter_time_s: float | None = 500e-9,
    reuse: bool = False,
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
        ``'short_exposure'`` (default) or ``'long_exposure'``; see
        'resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_exposure'``). The default is ``500e-9``
        (500 ns).
    reuse : bool, optional
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

    if reuse:
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
