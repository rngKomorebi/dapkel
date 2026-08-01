"""Optical cross-talk (CT) maps from the on-chip coinc/OR counters.

``CT = sum(coinc) / sum(OR)`` per macropixel - dimensionless, so the frame time
cancels and no exposure time is needed. NaN where ``OR = 0``.

NOTE: the SPAD indices run CLOCKWISE inside a 2x2 macropixel::

    S0 (0,0)   S1 (0,1)
    S3 (1,0)   S2 (1,1)

so ``S0S1`` is horizontal, ``S0S3`` VERTICAL and ``S0S2`` DIAGONAL - the reverse
of a row-major reading. See ``docs/guide/crosstalk.md``.
"""

from __future__ import annotations

import glob
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import plots, reduce, store

__all__ = [
    # stage 1 - compute the cross-talk map from raw '.bin' files
    "compute_crosstalk",
    # stage 1 - compute and persist ('.npy' + '.meta.json' sidecar)
    "compute_and_save_crosstalk",
    # stage 2 - plot from an already computed map
    "plot_ct_heatmap",
    "plot_ct_distribution",
    "plot_combined_ct_heatmap",
    "plot_ct_direction_summary",
    # drivers - folder in, figures on disk
    "collect_and_plot_crosstalk",
    "combine_directional_crosstalk",
]

#: Analysis name: the stage-1 file stem and results sub-folder.
_KIND = "crosstalk"

# Filename prefix of the main measurement batch. The '01' preliminary
# batches use a 'CT01_'/'CT02_'/... prefix and are excluded by default so
# the two acquisitions are not silently mixed.
_DEFAULT_PREFIX = "CT_"

_PAIR_RE = re.compile(r"(S\dS\d)", re.IGNORECASE)

# Micropixel layout inside one 2x2 macropixel (array indices, S0 at
# top-left). The SPAD indices run CLOCKWISE, not row-major::
#
#     S0 (0,0)   S1 (0,1)
#     S3 (1,0)   S2 (1,1)
#
# so S1 is S0's horizontal neighbour, S3 its vertical neighbour, and S2 its
# diagonal (corner) neighbour. The "other" micropixel index of an S0S<n>
# pair therefore fixes both the neighbour direction and the cell offset used
# when a (32, 32) directional map is expanded onto the (64, 64) micropixel
# grid.
_DIRECTIONS = {
    1: ("horizontal", (0, 1)),
    2: ("diagonal", (1, 1)),
    3: ("vertical", (1, 0)),
}



def _ct_files(folder: str, kind: str, prefix: str) -> list[str]:
    """Find the coinc or OR '.bin' files for a cross-talk measurement.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    kind : str
        Either ``'coinc'`` or ``'OR'``.
    prefix : str
        Filename prefix selecting the measurement batch (e.g. ``'CT_'``).

    Returns
    -------
    list[str]
        Paths to the matching '.bin' files (order is irrelevant; the
        counts are summed).

    Raises
    ------
    ValueError
        Raised when ``kind`` is not 'coinc' or 'OR'.
    FileNotFoundError
        Raised when no matching '.bin' files are found.
    """
    if kind not in ("coinc", "OR"):
        raise ValueError(f"kind must be 'coinc' or 'OR', got {kind!r}")

    files = []
    for fp in glob.glob(os.path.join(folder, "*.bin")):
        base = os.path.basename(fp)
        if not base.startswith(prefix):
            continue
        # Classify by the counter token. 'coinc' never contains an
        # upper-case 'OR', so a case-sensitive test cleanly separates the
        # two: coinc files carry 'coinc', OR files carry 'OR'.
        is_coinc = "coinc" in base
        if (
            kind == "coinc"
            and is_coinc
            or kind == "OR"
            and not is_coinc
            and "OR" in base
        ):
            files.append(fp)

    if not files:
        raise FileNotFoundError(
            f"No {prefix}*{kind}* .bin files found in:\n  {folder}"
        )
    return sorted(files)


def _detect_pair(folder: str, prefix: str) -> str:
    """Extract the micropixel-pair tag (e.g. 'S0S1') from the filenames.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    prefix : str
        Filename prefix selecting the measurement batch.

    Returns
    -------
    str
        The pair tag, upper-cased (e.g. ``'S0S1'``), or ``''`` if none of
        the filenames carry a recognisable ``S<d>S<d>`` token.
    """
    for fp in glob.glob(os.path.join(folder, prefix + "*.bin")):
        m = _PAIR_RE.search(os.path.basename(fp))
        if m:
            return m.group(1).upper()
    return ""


def _direction_of_pair(pair: str) -> tuple[str, tuple[int, int]] | None:
    """Map an ``S0S<n>`` pair tag to its neighbour direction and cell offset.

    Parameters
    ----------
    pair : str
        Pair tag such as ``'S0S1'`` (order-insensitive, case-insensitive).

    Returns
    -------
    tuple[str, tuple[int, int]] or None
        ``(name, (drow, dcol))`` where ``name`` is 'horizontal',
        'vertical', or 'diagonal' and ``(drow, dcol)`` is the neighbour's
        offset within the 2x2 macropixel block (see '_DIRECTIONS'). Returns
        ``None`` when the tag is not an S0-based pair with a known partner.
    """
    idx = [int(d) for d in re.findall(r"S(\d)", pair)]
    others = [i for i in idx if i != 0]
    if len(idx) != 2 or 0 not in idx or not others:
        return None
    return _DIRECTIONS.get(others[0])


def _accumulate_counts(files: list[str], *, label: str = "") -> np.ndarray:
    """Unpack a list of '.bin' files and sum the (32, 32) photon counts.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files to unpack and accumulate.
    label : str, optional
        Label used in the progress printout. The default is "".

    Returns
    -------
    np.ndarray
        The (32, 32) total counts summed over every frame of every file.
    """
    return reduce.accumulate_frames(
        files,
        lambda ts, pc: pc.sum(axis=2),
        need_time_series=False,
        label=label,
    )


def compute_crosstalk(
    folder: str,
    file_prefix: str = _DEFAULT_PREFIX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Unpack the coinc/OR data and compute the (32, 32) cross-talk map.

    Returns the per-macropixel ratio ``sum(coinc) / sum(OR)`` accumulated
    over every matching '.bin' file. Macropixels with ``OR = 0`` are NaN.

    Parameters
    ----------
    folder : str
        Path to the folder with the coinc/OR '.bin' data files.
    file_prefix : str, optional
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
    pair = _detect_pair(folder, file_prefix)

    coinc_files = _ct_files(folder, "coinc", file_prefix)
    or_files = _ct_files(folder, "OR", file_prefix)

    coinc_sum = _accumulate_counts(coinc_files, label=f"{pair or 'CT'} coinc")
    or_sum = _accumulate_counts(or_files, label=f"{pair or 'CT'} OR")

    # OR = union >= coinc = intersection, so the ratio is in [0, 1].
    with np.errstate(divide="ignore", invalid="ignore"):
        ct_map = np.where(or_sum > 0, coinc_sum / or_sum, np.nan)

    return ct_map, coinc_sum, or_sum, pair


def plot_ct_heatmap(
    ct_map: np.ndarray,
    pair: str = "",
    cmap: str | None = None,
) -> plt.Figure:
    """Plot a cross-talk map as a 2D heatmap (values in percent).

    Parameters
    ----------
    ct_map : np.ndarray
        Cross-talk map from 'compute_crosstalk' (dimensionless ratio).
    pair : str, optional
        Micropixel-pair tag for the title (e.g. ``'S0S1'``). The default
        is "".
    cmap : str | None, optional
        Matplotlib colormap name. The default is None — use the active
        style's ``image.cmap``.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    median, maximum, _ = plots.map_stats(ct_map, scale=100.0)
    pair_str = f"{pair}  " if pair else ""
    return plots.sensor_map(
        ct_map,
        scale=100.0,
        title=f"{pair_str}cross-talk\nmedian {median:.1f}  max {maximum:.1f}  %",
        clabel="CT (%)",
        cmap=cmap,
        xlabel="Macropixel column",
        ylabel="Macropixel row",
    )


def plot_ct_distribution(ct_map: np.ndarray, pair: str = "") -> plt.Figure:
    """Plot the sorted per-macropixel cross-talk distribution (percent).

    Parameters
    ----------
    ct_map : np.ndarray
        Cross-talk map from 'compute_crosstalk' (dimensionless ratio).
    pair : str, optional
        Micropixel-pair tag for the title/label. The default is "".

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    pair_str = f"{pair} " if pair else ""
    return plots.sorted_distribution(
        ct_map,
        scale=100.0,  # ratio -> percent; NaN (OR=0) macropixels are dropped
        ylabel="Cross-talk  (%)",
        xlabel="Macropixel percentile  (%)",
        title=f"Sorted {pair_str}cross-talk distribution",
        fmt=".1f",
        unit="%",
    )



def compute_and_save_crosstalk(
    path: str, file_prefix: str = _DEFAULT_PREFIX
) -> str:
    """Compute the (32, 32) cross-talk map and save it under ``processed/``.

    The stage-1 half of the cross-talk analysis: unpack the coinc/OR files
    once, write the array plus a '.meta.json' sidecar, and return the path.
    'collect_and_plot_crosstalk' can then replot it with ``from_saved=True``
    without re-unpacking.

    Parameters
    ----------
    path : str
        Path to the folder with the coinc/OR '.bin' data files.
    file_prefix : str, optional
        Filename prefix selecting the measurement batch. The default is
        ``'CT_'``.

    Returns
    -------
    str
        Path to the saved '.npy'.
    """
    ct_map, coinc_sum, or_sum, pair = compute_crosstalk(path, file_prefix)
    return store.save_map(
        ct_map,
        path,
        kind=_KIND,
        tag=pair,
        meta={
            "pair": pair,
            "file_prefix": file_prefix,
            "unit": "ratio",
            "total_coinc": float(np.nansum(coinc_sum)),
            "total_or": float(np.nansum(or_sum)),
            "undefined_macropixels": int((~np.isfinite(ct_map)).sum()),
        },
    )


def collect_and_plot_crosstalk(
    path: str,
    file_prefix: str = _DEFAULT_PREFIX,
    cmap: str | None = None,
    *,
    from_saved: bool = False,
) -> np.ndarray:
    """Unpack, compute, and plot the (32, 32) cross-talk map for a folder.

    Computes (and saves) the map, then writes both the heatmap and the
    sorted distribution to ``results/crosstalk``.

    Parameters
    ----------
    path : str
        Path to the folder with the coinc/OR '.bin' data files.
    file_prefix : str, optional
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

    if from_saved:
        pair = _detect_pair(path, file_prefix)
        ct_map, _ = store.load_map(path, kind=_KIND, tag=pair)
    else:
        ct_map, coinc_sum, or_sum, pair = compute_crosstalk(path, file_prefix)
        store.save_map(
            ct_map,
            path,
            kind=_KIND,
            tag=pair,
            meta={
                "pair": pair,
                "file_prefix": file_prefix,
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


def plot_combined_ct_heatmap(
    combined: np.ndarray,
    cmap: str | None = None,
) -> plt.Figure:
    """Plot the combined (64, 64) directional cross-talk map (percent).

    Each 2x2 block is one macropixel: the top-left cell holds the mean
    cross-talk of pixel S0, and the other three cells hold S0's cross-talk
    towards its horizontal (S1), vertical (S3) and diagonal (S2) neighbours
    - the SPAD indices run clockwise, see '_DIRECTIONS'.

    Parameters
    ----------
    combined : np.ndarray
        The (64, 64) map from 'combine_directional_crosstalk'
        (dimensionless ratio, NaN where a direction is missing/OR=0).
    cmap : str | None, optional
        Matplotlib colormap name. The default is None — use the active
        style's ``image.cmap``.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    median, _, _ = plots.map_stats(combined, scale=100.0)
    return plots.sensor_map(
        combined,
        scale=100.0,
        title=(
            "S0 directional cross-talk\n"
            "2x2 block: mean | horiz / vert | diag  "
            f"(median {median:.1f} %)"
        ),
        clabel="CT (%)",
        cmap=cmap,
        xlabel="Micropixel column",
        ylabel="Micropixel row",
    )


def plot_ct_direction_summary(medians: dict[str, float]) -> plt.Figure:
    """Bar-chart the median cross-talk per direction (percent).

    Parameters
    ----------
    medians : dict[str, float]
        Ordered ``label -> median CT (percent)`` mapping, e.g.
        ``{'horizontal': 1.2, 'vertical': 0.9, 'diagonal': 0.3,
        'overall (mean)': 0.8}``.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    labels = list(medians.keys())
    values = [medians[k] for k in labels]

    fig, ax = plt.subplots()

    bars = ax.bar(labels, values)
    for rect, val in zip(bars, values, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Median CT  (%)")
    ax.set_title("Cross-talk by direction")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def combine_directional_crosstalk(
    parent_path: str,
    file_prefix: str = _DEFAULT_PREFIX,
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
    file_prefix : str, optional
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
        pair = _detect_pair(entry.path, file_prefix)
        if not pair:
            continue
        direction = _direction_of_pair(pair)
        if direction is None:
            continue
        name, off = direction

        ct_map, _, _, _ = compute_crosstalk(entry.path, file_prefix)
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
