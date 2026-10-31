"""ARCHIVED from dapkel.functions.crosstalk_analysis (branch modules-restructure, 2026-10-31).

Compute-only (no save) entry point and the standalone folder combiner. Now 'process_crosstalk' (single pairing or a parent folder of pairings) and 'plot_crosstalk'.

Reference only: not packaged, not tested, not maintained. Copied verbatim
from the pre-restructure module (commit 6b49a4b), so it still refers to
helpers and names that no longer exist in 'src/'. To revive a piece, port it
into the live module rather than importing it from here.
"""

from __future__ import annotations
import glob
import os
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
from dapkel.core import plots, reduce, store


def compute_crosstalk(
    path: str,
    nframes: int,
    tag: str = _DEFAULT_PREFIX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Unpack the coinc/OR data and compute the (32, 32) cross-talk map.

    Returns the per-macropixel ratio ``sum(coinc) / sum(OR)`` accumulated
    over every matching '.bin' file. Macropixels with ``OR = 0`` are NaN.

    Parameters
    ----------
    path : str
        Path to the path with the coinc/OR '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    tag : str, optional
        Filename prefix selecting the measurement batch. The default is
        ``'CT_'`` (the main 100-file batch), which excludes the smaller
        preliminary ``'CT01_'``-style batches.

    Returns
    -------
    ct_map : np.ndarray
        The (32, 32) cross-talk map (dimensionless ratio, NaN where OR=0).
    coinc_sum : np.ndarray
        The (32, 32) accumulated coincidence counts.
    or_sum : np.ndarray
        The (32, 32) accumulated OR counts.
    pair : str
        The detected micropixel-pair tag (e.g. ``'S0S1'``), or ``''``.
    """
    pair = _detect_pair(path, tag)

    coinc_files = _ct_files(path, "coinc", tag)
    or_files = _ct_files(path, "OR", tag)

    coinc_sum = _accumulate_counts(
        coinc_files, nframes=nframes, label=f"{pair or 'CT'} coinc"
    )
    or_sum = _accumulate_counts(
        or_files, nframes=nframes, label=f"{pair or 'CT'} OR"
    )

    # OR = union >= coinc = intersection, so the ratio is in [0, 1].
    with np.errstate(divide="ignore", invalid="ignore"):
        ct_map = np.where(or_sum > 0, coinc_sum / or_sum, np.nan)

    return ct_map, coinc_sum, or_sum, pair


def collect_and_plot_crosstalk(
    path: str,
    nframes: int,
    tag: str = _DEFAULT_PREFIX,
    cmap: str | None = None,
    *,
    reuse: bool = False,
) -> np.ndarray:
    """Unpack, compute, and plot the (32, 32) cross-talk map for a folder.

    Computes (and saves) the map, then writes both the heatmap and the
    sorted distribution to ``results/crosstalk``.

    Parameters
    ----------
    path : str
        Path to the folder with the coinc/OR '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file. Ignored when
        ``reuse=True`` - nothing is unpacked then.
    tag : str, optional
        Filename prefix selecting the measurement batch. The default is
        ``'CT_'``.
    cmap : str | None, optional
        Matplotlib colormap name for the heatmap. The default is None —
        use the active style's ``image.cmap``.

    Returns
    -------
    np.ndarray
        The (32, 32) cross-talk map (dimensionless ratio, NaN where OR=0).
    """
    print(
        "\n> > > Collecting cross-talk data and plotting the heatmap and "
        "distribution < < <\n"
    )

    if reuse:
        pair = _detect_pair(path, tag)
        ct_map, _ = store.load_map(path, kind=_KIND, tag=pair)
    else:
        ct_map, coinc_sum, or_sum, pair = compute_crosstalk(
            path, nframes, tag
        )
        store.save_map(
            ct_map,
            path,
            kind=_KIND,
            tag=pair,
            meta={
                "pair": pair,
                "tag": tag,
                "unit": "ratio",
                "total_coinc": float(np.nansum(coinc_sum)),
                "total_or": float(np.nansum(or_sum)),
                "undefined_macropixels": int((~np.isfinite(ct_map)).sum()),
            },
            quiet=True,
        )

    finite = np.isfinite(ct_map)
    print(
        f"  pair {pair or '?'}: "
        f"median CT {np.nanmedian(ct_map) * 100:.2f}%  "
        f"mean {np.nanmean(ct_map) * 100:.2f}%  "
        f"max {np.nanmax(ct_map) * 100:.2f}%  "
        f"({(~finite).sum()} macropixels with OR=0 -> NaN)"
    )

    name = os.path.basename(os.path.normpath(path))
    pair_suffix = pair.lower() if pair else "ct"
    results_dir = store.results_dir(path, _KIND, create=False)

    fig_heatmap = plot_ct_heatmap(ct_map, pair=pair, cmap=cmap)
    store.save_figure(
        fig_heatmap, results_dir, f"{name}_{pair_suffix}_ct_heatmap.png"
    )
    plt.close(fig_heatmap)

    fig_distribution = plot_ct_distribution(ct_map, pair=pair)
    store.save_figure(
        fig_distribution,
        results_dir,
        f"{name}_{pair_suffix}_ct_distribution.png",
    )
    plt.close(fig_distribution)

    return ct_map


def combine_directional_crosstalk(
    parent_path: str,
    nframes: int,
    tag: str = _DEFAULT_PREFIX,
    cmap: str | None = None,
) -> np.ndarray:
    """Combine the per-direction cross-talk of pixel S0 into a (64, 64) map.

    Scans the sub-folders of ``parent_path`` (one folder per micropixel
    pairing: S0S1 horizontal, S0S3 vertical, S0S2 diagonal), computes each
    (32, 32) directional cross-talk map via 'compute_crosstalk', and lays
    them onto a single (64, 64) micropixel grid. In each 2x2 macropixel
    block the neighbour cells carry S0's cross-talk in that direction and
    the S0 cell carries the mean of the available directions.

    Both a combined (64, 64) heatmap and a per-direction median bar chart
    are saved into ``parent_path/results/crosstalk``.

    Parameters
    ----------
    parent_path : str
        Folder holding one sub-folder per pairing (e.g. '01'/'02'/'03').
        Sub-folders named 'results' and any without a recognisable
        ``S0S<n>`` pairing are ignored.
    nframes : int
        Number of frames stored in each '.bin' file. The pairings are
        acquired with the same setting, so one value covers every
        sub-folder.
    tag : str, optional
        Filename prefix selecting the measurement batch, passed through to
        'compute_crosstalk'. The default is ``'CT_'``.
    cmap : str | None, optional
        Matplotlib colormap name for the heatmap. The default is None —
        use the active style's ``image.cmap``.

    Returns
    -------
    np.ndarray
        The (64, 64) combined cross-talk map (dimensionless ratio, NaN
        where a direction is missing or OR=0).

    Raises
    ------
    FileNotFoundError
        Raised when no sub-folder yields a recognisable directional map.
    """
    print(
        "\n> > > Combining per-direction S0 cross-talk into a 64x64 map < < <\n"
    )

    maps: dict[str, np.ndarray] = {}
    offsets: dict[str, tuple[int, int]] = {}

    for entry in sorted(os.scandir(parent_path), key=lambda e: e.name):
        if not entry.is_dir() or entry.name == "results":
            continue
        pair = _detect_pair(entry.path, tag)
        if not pair:
            continue
        direction = _direction_of_pair(pair)
        if direction is None:
            continue
        name, off = direction

        ct_map, _, _, _ = compute_crosstalk(entry.path, nframes, tag)
        maps[name] = ct_map
        offsets[name] = off
        print(
            f"  {entry.name}: {pair} ({name}) "
            f"median CT {np.nanmedian(ct_map) * 100:.2f}%"
        )

    if not maps:
        raise FileNotFoundError(
            f"No recognisable S0S<n> cross-talk sub-folders found in:\n"
            f"  {parent_path}"
        )

    missing = [
        d for d in ("horizontal", "vertical", "diagonal") if d not in maps
    ]
    if missing:
        print(f"  WARNING: missing direction(s): {', '.join(missing)}")

    # Lay each directional (32, 32) map onto its neighbour cells of the
    # (64, 64) grid; the S0 cells get the per-macropixel mean over the
    # directions that were actually measured.
    combined = np.full((64, 64), np.nan)
    for name, (drow, dcol) in offsets.items():
        combined[drow::2, dcol::2] = maps[name]

    # RuntimeWarning "Mean of empty slice" for macropixels with no measured
    # direction is expected (they stay NaN) — silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        combined[0::2, 0::2] = np.nanmean(
            np.stack([maps[n] for n in maps]), axis=0
        )

    medians: dict[str, float] = {}
    for name in ("horizontal", "vertical", "diagonal"):
        if name in maps:
            medians[name] = float(np.nanmedian(maps[name]) * 100.0)
    medians["overall (mean)"] = float(
        np.nanmedian(combined[0::2, 0::2]) * 100.0
    )

    print("\n  Median cross-talk by direction:")
    for label, val in medians.items():
        print(f"    {label:<16s} {val:6.2f} %")

    name = os.path.basename(os.path.normpath(parent_path))
    results_dir = store.results_dir(parent_path, _KIND, create=False)

    fig_heatmap = plot_combined_ct_heatmap(combined, cmap=cmap)
    store.save_figure(fig_heatmap, results_dir, f"{name}_combined_ct_heatmap.png")
    plt.close(fig_heatmap)

    fig_summary = plot_ct_direction_summary(medians)
    store.save_figure(fig_summary, results_dir, f"{name}_ct_direction_summary.png")
    plt.close(fig_summary)

    return combined
