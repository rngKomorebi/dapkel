"""Module for computing and plotting dark count rate (DCR) maps.

This script utilizes an unpacking module used specifically for the
Kelpie v2 DDR3 binary data output.

This file can also be imported as a module and contains the following
functions:

    * compute_dcr_32 - unpack the '.bin' files for a single SPAD tag
    and compute the (32, 32) dark count rate map. Use this on its own
    when only the numbers are needed, with no plot.

    * compute_dcr_64 - unpack the '.bin' files for all four SPAD tags
    (S0C, S1C, S2C, S3C) and assemble the (64, 64) dark count rate map.
    Use this on its own when only the numbers are needed, with no
    plot.

    * plot_heatmap - plot a dark count rate map as a 2D heatmap, given
    an already computed DCR map.

    * plot_distribution - plot the sorted per-pixel dark count rate
    distribution, given an already computed DCR map.

    * collect_and_plot_dcr_32 - given the data file parameters (path,
    number of frames, frame length), unpack and compute the (32, 32)
    dark count rate map for a single SPAD tag once and save both the
    heatmap and the distribution plot.

    * collect_and_plot_dcr_64 - given the data file parameters (path,
    number of frames, frame length), unpack and compute the full
    sensor (64, 64) dark count rate map once and save both the
    heatmap and the distribution plot.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dapkel.functions.unpack import unpack

CLK_PERIOD = 5e-9  # 200 MHz
_FREE_RUNNING_US = 9.0  # base frame period when exp_time=0 is sent to the exe
# actual = user_exp_time + 9 µs  (0 → 9 µs, 10 µs → 19 µs)

_SPAD_LAYOUT = {
    "S0C": (0, 0),
    "S1C": (0, 1),
    "S2C": (1, 0),
    "S3C": (1, 1),
}


def _resolve_frame_time(
    folder: str, explicit: float | None, nframes: int
) -> tuple[float, str]:
    """Return (frame_time_seconds, human-readable source string).

    Mirrors the GUI's _run_analysis logic exactly:
      1. explicit --exp-time supplied → exp_time + 9 µs overhead
      2. frame_rate_cnt.txt present  → ticks * CLK_PERIOD / nframes
      3. fallback                    → 9 µs free-running base
    """
    if explicit is not None:
        ft = explicit + _FREE_RUNNING_US * 1e-6
        return (
            ft,
            f"exp_time={explicit * 1e6:.1f} µs + {_FREE_RUNNING_US:.0f} µs overhead",
        )

    cnt_file = os.path.join(folder, "frame_rate_cnt.txt")
    try:
        ticks = int(open(cnt_file).read().strip())
        if ticks > 0:
            ft = ticks * CLK_PERIOD / nframes
            return ft, f"frame_rate_cnt.txt → {ft * 1e6:.4f} µs"
    except (OSError, ValueError):
        pass

    ft = _FREE_RUNNING_US * 1e-6
    return ft, f"fallback free-running ({_FREE_RUNNING_US} µs)"


def _sorted_bin_files(folder: str, tag: str) -> list[str]:
    """Find and naturally sort the '.bin' files for a SPAD tag.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    tag : str
        SPAD quadrant tag ('S0C', 'S1C', 'S2C', 'S3C'), or '' to match
        every '.bin' file in the folder.

    Returns
    -------
    list[str]
        Sorted list of paths to the matching '.bin' files.

    Raises
    ------
    FileNotFoundError
        Raised when no '.bin' files match the tag in the folder.
    """
    pattern = f"*_{tag}*.bin" if tag else "*.bin"
    files = sorted(
        glob.glob(os.path.join(folder, pattern)),
        key=lambda fp: int(
            "".join(filter(str.isdigit, os.path.basename(fp))) or "0"
        ),
    )
    if not files:
        raise FileNotFoundError(
            f"No .bin files matching '{pattern}' found in:\n  {folder}"
        )
    return files


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
    photon_sum = np.zeros((32, 32), dtype=np.float64)
    for fp in tqdm(files):
        _, pc = unpack(fp, nframes, compute_time_series=False)
        photon_sum += pc.sum(axis=2)
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
) -> np.ndarray:
    """Unpack the binary data and compute the (32, 32) DCR map.

    Finds the '.bin' files for the requested SPAD tag, unpacks them,
    and computes the dark count rate map for a single quadrant. Use
    this directly when only the DCR numbers are needed, with no plot.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time : float | None, optional
        Value passed to Kelpie_v2.exe, in seconds. 0 → 9 µs actual,
        10e-6 → 19 µs actual. If None, the frame length is read from
        'frame_rate_cnt.txt', falling back to 9 µs. The default is
        None.
    tag : str, optional
        SPAD quadrant tag ('S0C', 'S1C', 'S2C', 'S3C'). The default is
        "", which matches every '.bin' file in the folder.

    Returns
    -------
    np.ndarray
        The (32, 32) dark count rate map, in counts per second.
    """
    frame_time, ft_source = _resolve_frame_time(folder, exp_time, nframes)
    files = _sorted_bin_files(folder, tag)
    return _accumulate_dcr(files, nframes, frame_time, label=tag)


def compute_dcr_64(
    folder: str,
    nframes: int,
    exp_time: float | None = None,
) -> np.ndarray:
    """Unpack the binary data and compute the (64, 64) DCR map.

    Finds the '.bin' files for each of the four SPAD tags, unpacks
    them, and assembles the dark count rate map for the full sensor.
    Use this directly when only the DCR numbers are needed, with no
    plot.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    exp_time : float | None, optional
        Value passed to Kelpie_v2.exe, in seconds. 0 → 9 µs actual,
        10e-6 → 19 µs actual. If None, the frame length is read from
        'frame_rate_cnt.txt', falling back to 9 µs. The default is
        None.

    Returns
    -------
    np.ndarray
        The (64, 64) dark count rate map, in counts per second.

    Raises
    ------
    FileNotFoundError
        Raised when no '.bin' files for any of the four SPAD tags are
        found in the folder.
    """
    frame_time, ft_source = _resolve_frame_time(folder, exp_time, nframes)

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

    dcr_map = np.zeros((64, 64), dtype=np.float64)
    for tag, (dr, dc) in _SPAD_LAYOUT.items():
        if tag not in tag_files:
            continue
        dcr_32 = _accumulate_dcr(
            tag_files[tag],
            nframes,
            frame_time,
            label=tag,
        )
        rows = np.arange(32) * 2 + dr
        cols = np.arange(32) * 2 + dc
        dcr_map[np.ix_(rows, cols)] = dcr_32

    return dcr_map


def plot_heatmap(dcr: np.ndarray, cmap: str = "PuBuGn_r") -> plt.Figure:
    """Plot a dark count rate map as a 2D heatmap.

    Parameters
    ----------
    dcr : np.ndarray
        Dark count rate map, e.g. from 'compute_dcr_32'/'compute_dcr_64'.
    cmap : str, optional
        Matplotlib colormap name. The default is "PuBuGn_r".

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    rows, cols = dcr.shape
    median = float(np.median(dcr))
    vmax = float(np.percentile(dcr, 99))  # clip hot pixels for colorscale

    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(dcr, cmap=cmap, aspect="equal", vmax=vmax)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    title = f"DCR map  {rows}×{cols} SPADs"

    title = f"\nmedian {median:.0f}  max {dcr.max():.0f}  cps"
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    return fig


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
    median = float(np.median(dcr))
    mean = float(dcr.mean())

    sorted_dcr = np.sort(dcr.ravel())
    pct = np.linspace(0, 100, len(sorted_dcr))

    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.semilogy(
        pct,
        sorted_dcr,
        ".",
        markersize=2,
        alpha=0.7,
        label="_nolegend_",
    )
    ax.axhline(
        median,
        linestyle="--",
        linewidth=2,
        label=f"Median  {median:.0f} cps",
    )
    ax.axhline(
        mean,
        linestyle="--",
        linewidth=1,
        label=f"Mean    {mean:.0f} cps",
    )

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Pixel percentile  (%)")
    ax.set_ylabel("DCR  (cps)")
    title = "Sorted DCR distribution"
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return fig


def _save_figure(
    fig: plt.Figure, results_dir: str, file_name: str
) -> str | None:
    """Save a figure into the results folder.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    results_dir : str
        Folder the figure should be saved into. Created if missing.
    file_name : str
        Name of the '.png' file to save the figure as.

    Returns
    -------
    str | None
        Path the figure was saved to.
    """
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, file_name)

    fig.savefig(out_path)
    print(f"\n> > > Plot is saved as {file_name} in {results_dir} < < <")
    return out_path


def collect_and_plot_dcr_32(
    path: str,
    nframes: int,
    exp_time: float | None = None,
    tag: str = "",
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    cmap: str = "PuBuGn_r",
) -> np.ndarray:
    """Unpack, compute, and plot the (32, 32) DCR map for a SPAD tag.

    Convenience wrapper: unpacks the data once via 'compute_dcr_32',
    then plots and saves both the heatmap and the sorted distribution
    into the 'results/dcr' folder created (if it does not already
    exist) in the same folder where the data are, without unpacking
    the data twice.

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
    cmap : str, optional
        Matplotlib colormap name for the heatmap. The default is
        "PuBuGn_r".

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

    dcr = compute_dcr_32(path, nframes, exp_time, tag)

    name = os.path.basename(os.path.normpath(path))
    tag_suffix = tag if tag else "dcr"
    results_dir = os.path.join(path, "results", "dcr")

    fig_heatmap = plot_heatmap(dcr, cmap=cmap)
    _save_figure(fig_heatmap, results_dir, f"{name}_{tag_suffix}_heatmap.png")
    plt.close(fig_heatmap)

    fig_distribution = plot_distribution(dcr)
    _save_figure(
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
    cmap: str = "PuBuGn_r",
) -> np.ndarray:
    """Unpack, compute, and plot the (64, 64) full sensor DCR map.

    Convenience wrapper: unpacks the data once via 'compute_dcr_64',
    then plots and saves both the heatmap and the sorted distribution
    into the 'results/dcr' folder created (if it does not already
    exist) in the same folder where the data are, without unpacking
    the data twice.

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

    cmap : str, optional
        Matplotlib colormap name for the heatmap. The default is
        "PuBuGn_r".

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

    dcr = compute_dcr_64(path, nframes, exp_time)

    name = os.path.basename(os.path.normpath(path))
    results_dir = os.path.join(path, "results", "dcr")

    fig_heatmap = plot_heatmap(dcr, cmap=cmap)
    _save_figure(
        fig_heatmap,
        results_dir,
        f"{name}_full_heatmap.png",
    )

    fig_distribution = plot_distribution(dcr)
    _save_figure(
        fig_distribution,
        results_dir,
        f"{name}_full_distribution.png",
    )

    return dcr
