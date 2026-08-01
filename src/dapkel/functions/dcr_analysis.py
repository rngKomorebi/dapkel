"""Dark count rate (DCR) maps, per quadrant and full-sensor.

DCR is accumulated photon counts divided by the time the sensor was actually
collecting. That live time is NOT the frame period: under the current
``short_window`` firmware only the user-set ``acq_window`` of each ~9 us frame is
photon-sensitive, so ``acq_window`` is required.

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





def _accumulate_dcr(
    files: list[str],
    nframes: int,
    frame_time: float,
    *,
    label: str = "",
) -> np.ndarray:
    """Unpack a list of '.bin' files and compute the (32, 32) DCR map.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files to unpack and accumulate.
    nframes : int
        Number of frames stored in each '.bin' file.
    frame_time : float
        Length of a single frame, in seconds.
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
    photon_sum = reduce.accumulate_frames(
        files,
        lambda ts, pc: pc.sum(axis=2),
        nframes=nframes,
        need_time_series=False,
        label=label,
    )
    total_time = nframes * len(files) * frame_time
    if total_time == 0:
        raise ValueError(
            "Total acquisition time is zero — check --exp-time / frame_rate_cnt.txt."
        )
    dcr = photon_sum / total_time
    return dcr


def compute_dcr_32(
    folder: str,
    nframes: int,
    exp_time: float | None = None,
    tag: str = "",
    *,
    firmware_version: str = "short_window",
    acq_window: float | None = None,
) -> np.ndarray:
    """Unpack the binary data and compute the (32, 32) DCR map.

    Normalised by the *live* time per frame, which depends on the firmware
    (see 'core.timing.resolve_live_time'): ``'short_window'`` counts only
    ``acq_window`` as live, ``'full_window'`` the whole frame.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time : float | None, optional
        Legacy exposure time in seconds (``firmware_version='full_window'``
        only): 0 → 9 µs actual, 10e-6 → 19 µs. If None, read from
        'frame_rate_cnt.txt', falling back to 9 µs. The default is None.
    tag : str, optional
        SPAD quadrant tag ('S0C', 'S1C', 'S2C', 'S3C'). The default is
        "", which matches every '.bin' file in the folder.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; see
        'resolve_live_time'.
    acq_window : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_window'``, e.g. ``200e-9``). The default
        is None.

    Returns
    -------
    np.ndarray
        The (32, 32) dark count rate map, in counts per second.

    Raises
    ------
    ValueError
        Raised when ``firmware_version='short_window'`` but no ``acq_window``
        is given.
    """
    live, _ = timing.resolve_live_time(
        folder, firmware_version, nframes,
        acq_window=acq_window, exp_time=exp_time,
    )
    if live is None:
        raise ValueError(
            "firmware_version='short_window' needs acq_window (the "
            "photon-sensitive window per frame, in seconds, e.g. 200e-9)."
        )
    # The DCR quadrant tags are matched with a leading underscore ('*_S0C*')
    # so a tag cannot match a filename that merely contains those characters.
    files = io.find_bin_files(folder, tag, require_separator=True)
    return _accumulate_dcr(files, nframes, live, label=tag)


def compute_dcr_64(
    folder: str,
    nframes: int,
    exp_time: float | None = None,
    *,
    firmware_version: str = "short_window",
    acq_window: float | None = None,
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
    exp_time : float | None, optional
        Legacy exposure time in seconds (``firmware_version='full_window'``
        only). If None, read from 'frame_rate_cnt.txt', falling back to
        9 µs. The default is None.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; see
        'resolve_live_time'.
    acq_window : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_window'``). The default is None.

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
        Raised when ``firmware_version='short_window'`` but no ``acq_window``
        is given.
    """
    live, _ = timing.resolve_live_time(
        folder, firmware_version, nframes,
        acq_window=acq_window, exp_time=exp_time,
    )
    if live is None:
        raise ValueError(
            "firmware_version='short_window' needs acq_window (the "
            "photon-sensitive window per frame, in seconds, e.g. 200e-9)."
        )
    frame_time = live

    tag_files = {}
    for tag in _SPAD_LAYOUT:
        files = glob.glob(os.path.join(folder, f"*_{tag}*.bin"))
        if files:
            tag_files[tag] = sorted(
                files,
                key=lambda fp: int(
                    "".join(filter(str.isdigit, os.path.basename(fp))) or "0"
                ),
            )
    if not tag_files:
        raise FileNotFoundError(
            "No S0C/S1C/S2C/S3C .bin files found in:\n  " + folder
        )

    quadrants = {
        tag: _accumulate_dcr(files, nframes, frame_time, label=tag)
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



def _dcr_meta(
    path: str,
    nframes: int,
    tag: str,
    exp_time: float | None,
    firmware_version: str,
    acq_window: float | None,
) -> dict:
    """Build the acquisition metadata sidecar for a saved DCR map."""
    frame_live_s, live_source = timing.resolve_live_time(
        path,
        firmware_version,
        nframes,
        acq_window=acq_window,
        exp_time=exp_time,
    )
    return {
        "tag": tag,
        "nframes": int(nframes),
        "firmware_version": firmware_version,
        "acq_window_s": acq_window,
        "exp_time_s": exp_time,
        "frame_live_s": frame_live_s,
        "live_time_source": live_source,
        "unit": "cps",
    }


def compute_and_save_dcr_32(
    path: str,
    nframes: int,
    exp_time: float | None = None,
    tag: str = "",
    *,
    firmware_version: str = "short_window",
    acq_window: float | None = None,
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
    exp_time : float | None, optional
        Value passed to Kelpie_v2.exe, in seconds. The default is None.
    tag : str, optional
        SPAD quadrant tag ('S0C', 'S1C', 'S2C', 'S3C'). The default is "".
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``.
    acq_window : float | None, optional
        Photon-sensitive window per frame, in seconds. The default is None.

    Returns
    -------
    str
        Path to the saved '.npy'.
    """
    dcr = compute_dcr_32(
        path, nframes, exp_time, tag,
        firmware_version=firmware_version, acq_window=acq_window,
    )
    return store.save_map(
        dcr,
        path,
        kind=_KIND,
        tag=tag,
        meta=_dcr_meta(
            path, nframes, tag, exp_time, firmware_version, acq_window
        ),
    )


def compute_and_save_dcr_64(
    path: str,
    nframes: int,
    exp_time: float | None = None,
    *,
    firmware_version: str = "short_window",
    acq_window: float | None = None,
) -> str:
    """Assemble the (64, 64) full-sensor DCR map and save it.

    Parameters
    ----------
    path : str
        Path to the folder with the four SPAD quadrants' '.bin' files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time : float | None, optional
        Value passed to Kelpie_v2.exe, in seconds. The default is None.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``.
    acq_window : float | None, optional
        Photon-sensitive window per frame, in seconds. The default is None.

    Returns
    -------
    str
        Path to the saved '.npy'.
    """
    dcr = compute_dcr_64(
        path, nframes, exp_time,
        firmware_version=firmware_version, acq_window=acq_window,
    )
    return store.save_map(
        dcr,
        path,
        kind=_KIND,
        tag="64",
        meta=_dcr_meta(
            path, nframes, "64", exp_time, firmware_version, acq_window
        ),
    )


def collect_and_plot_dcr_32(
    path: str,
    nframes: int,
    exp_time: float | None = None,
    tag: str = "",
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    cmap: str | None = None,
    *,
    firmware_version: str = "short_window",
    acq_window: float | None = None,
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
    exp_time : float | None, optional
        Value passed to Kelpie_v2.exe, in seconds. See
        'compute_dcr_32' for details. The default is None.
    tag : str, optional
        SPAD quadrant tag ('S0C', 'S1C', 'S2C', 'S3C'). The default is
        "", which matches every '.bin' file in the folder.
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
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; see
        'resolve_live_time'.
    acq_window : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_window'``). The default is None.
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
        f"\n> > > Collecting DCR data for tag '{tag or 'all'}' and "
        "plotting the heatmap and distribution < < <\n"
    )

    if from_saved:
        dcr, _ = store.load_map(path, kind=_KIND, tag=tag)
    else:
        dcr = compute_dcr_32(
            path, nframes, exp_time, tag,
            firmware_version=firmware_version, acq_window=acq_window,
        )
        store.save_map(
            dcr,
            path,
            kind=_KIND,
            tag=tag,
            meta=_dcr_meta(
                path, nframes, tag, exp_time, firmware_version, acq_window
            ),
            quiet=True,
        )

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
    exp_time: float | None = None,
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    cmap: str | None = None,
    *,
    firmware_version: str = "short_window",
    acq_window: float | None = None,
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
    exp_time : float | None, optional
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
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; see
        'resolve_live_time'.
    acq_window : float | None, optional
        Photon-sensitive window per frame, in seconds (required for
        ``firmware_version='short_window'``). The default is None.
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
        "\n> > > Collecting DCR data for the full sensor and plotting "
        "the heatmap and distribution < < <\n"
    )

    if from_saved:
        dcr, _ = store.load_map(path, kind=_KIND, tag="64")
    else:
        dcr = compute_dcr_64(
            path, nframes, exp_time,
            firmware_version=firmware_version, acq_window=acq_window,
        )
        store.save_map(
            dcr,
            path,
            kind=_KIND,
            tag="64",
            meta=_dcr_meta(
                path, nframes, "64", exp_time, firmware_version, acq_window
            ),
            quiet=True,
        )

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
