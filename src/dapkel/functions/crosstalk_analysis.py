"""Optical cross-talk (CT) maps from the on-chip coinc/OR counters.

    process_crosstalk   raw '.bin' -> processed/<name>_<tag>_crosstalk.npy
    plot_crosstalk      that array -> results/crosstalk/ figures

``CT = sum(coinc) / sum(OR)`` per macropixel - dimensionless, so the frame time
cancels and no exposure time is needed. NaN where ``OR = 0``.

A folder holds one micropixel pairing (``S0S1``...), read from the file names.
Point 'process_crosstalk' at a *parent* folder of pairings instead and it also
combines them: S0's cross-talk towards its horizontal, vertical and diagonal
neighbour laid onto one (64, 64) micropixel map.

NOTE: the SPAD indices run CLOCKWISE inside a 2x2 macropixel
('core.reduce.SPAD_LAYOUT')::

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

__all__ = ["process_crosstalk", "plot_crosstalk"]

#: Analysis name: the stage-1 file stem and results sub-folder.
_KIND = "crosstalk"

#: S0's partner index -> the direction of that neighbour, whose offset inside
#: the macropixel is 'core.reduce.SPAD_LAYOUT'.
_DIRECTIONS = {1: "horizontal", 3: "vertical", 2: "diagonal"}


def _ct_map(path: str, nframes: int, tag: str) -> tuple[np.ndarray, dict] | None:
    """Accumulate one pairing's (32, 32) CT map, or None if ``path`` has none.

    The coinc and OR files are told apart by their counter token, case
    sensitively: coinc files carry ``'coinc'``, OR files ``'OR'``, and
    ``'coinc'`` never contains an upper-case ``'OR'``.
    """
    names = sorted(glob.glob(os.path.join(path, f"{tag}*.bin")))
    coinc = [f for f in names if "coinc" in os.path.basename(f)]
    union = [
        f
        for f in names
        if "coinc" not in os.path.basename(f) and "OR" in os.path.basename(f)
    ]
    if not coinc or not union:
        return None
    found = (re.search(r"S\dS\d", os.path.basename(f), re.IGNORECASE) for f in names)
    pair = next((m.group(0).upper() for m in found if m), "")

    def _total(files: list[str], kind: str) -> np.ndarray:
        return reduce.accumulate_frames(
            files,
            lambda ts, pc: pc.sum(axis=2),
            nframes=nframes,
            need_time_series=False,
            label=f"{pair or 'CT'} {kind}",
        )

    coinc_sum, or_sum = _total(coinc, "coinc"), _total(union, "OR")
    # OR = union >= coinc = intersection, so the ratio is in [0, 1].
    with np.errstate(divide="ignore", invalid="ignore"):
        ct = np.where(or_sum > 0, coinc_sum / or_sum, np.nan)
    return ct, {
        "pair": pair,
        "total_coinc": float(coinc_sum.sum()),
        "total_or": float(or_sum.sum()),
        "undefined_macropixels": int((~np.isfinite(ct)).sum()),
    }


def process_crosstalk(path: str, nframes: int, tag: str = "CT_") -> str:
    """Unpack the coinc/OR files, compute the cross-talk map and save it.

    On a folder holding one pairing's '.bin' files this saves its (32, 32)
    macropixel map. On a *parent* folder - one sub-folder per pairing, e.g.
    S0S1 / S0S3 / S0S2 - it processes and saves each sub-folder the same way,
    then saves the combined (64, 64) map in the parent: in every 2x2 block the
    neighbour cells hold S0's cross-talk in that direction and the S0 cell the
    mean over the directions measured.

    Parameters
    ----------
    path : str
        Folder with the coinc/OR '.bin' files, or a parent of such folders.
    nframes : int
        Frames the acquisition recorded per '.bin' file (one value for every
        pairing).
    tag : str, optional
        Filename prefix selecting the measurement batch. The default is
        ``'CT_'``, the main batch, which excludes the preliminary
        ``'CT01_'``-style ones so two acquisitions are never mixed.

    Returns
    -------
    str
        Path to the saved '.npy' (the combined map for a parent folder).

    Raises
    ------
    FileNotFoundError
        Raised when neither ``path`` nor any sub-folder holds a pairing.
    """
    single = _ct_map(path, nframes, tag)
    if single is not None:
        ct, meta = single
        out_path = store.save_map(
            ct, path, kind=_KIND, tag=tag, meta={**meta, "unit": "ratio"}
        )
        print(
            f"  {meta['pair'] or '?'}: median CT {np.nanmedian(ct) * 100:.2f}%  "
            f"max {np.nanmax(ct) * 100:.2f}%  "
            f"({meta['undefined_macropixels']} macropixels with OR=0 -> NaN)"
        )
        return out_path

    print("\n> > > Combining per-direction S0 cross-talk into a 64x64 map < < <\n")
    maps: dict[str, np.ndarray] = {}
    combined = np.full((64, 64), np.nan)
    for entry in sorted(os.scandir(path), key=lambda e: e.name):
        if not entry.is_dir() or entry.name in (store.PROCESSED_DIR, store.RESULTS_DIR):
            continue
        found = _ct_map(entry.path, nframes, tag)
        if found is None:
            continue
        ct, meta = found
        idx = [int(d) for d in re.findall(r"S(\d)", meta["pair"])]
        partner = next((i for i in idx if i != 0), None)
        if len(idx) != 2 or 0 not in idx or partner not in _DIRECTIONS:
            print(f"  {entry.name}: {meta['pair'] or '?'} is not S0-based, skipped")
            continue
        store.save_map(
            ct, entry.path, kind=_KIND, tag=tag, meta={**meta, "unit": "ratio"}
        )
        direction = _DIRECTIONS[partner]
        maps[direction] = ct
        drow, dcol = reduce.SPAD_LAYOUT[partner]
        combined[drow::2, dcol::2] = ct
        print(
            f"  {entry.name}: {meta['pair']} ({direction}) median CT "
            f"{np.nanmedian(ct) * 100:.2f}%"
        )
    if not maps:
        raise FileNotFoundError(
            f"No '{tag}*' coinc/OR files in {path} or any sub-folder of it."
        )
    missing = [d for d in _DIRECTIONS.values() if d not in maps]
    if missing:
        print(f"  WARNING: missing direction(s): {', '.join(missing)}")

    # A macropixel with no measured direction stays NaN; its "Mean of empty
    # slice" warning is expected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        combined[0::2, 0::2] = np.nanmean(np.stack(list(maps.values())), axis=0)

    medians = {d: float(np.nanmedian(m) * 100.0) for d, m in maps.items()}
    medians["overall (mean)"] = float(np.nanmedian(combined[0::2, 0::2]) * 100.0)
    return store.save_map(
        combined,
        path,
        kind=_KIND,
        tag=tag,
        meta={"combined": True, "median_ct_percent": medians, "unit": "ratio"},
    )


def plot_crosstalk(
    path: str, tag: str = "CT_", *, cmap: str | None = None
) -> np.ndarray:
    """Load a saved cross-talk map and save its figures.

    A single pairing gets a heatmap and a sorted distribution; a combined
    parent-folder map gets the (64, 64) directional heatmap and a bar chart of
    the median cross-talk per direction. All values are drawn in percent.

    Parameters
    ----------
    path : str
        Folder that 'process_crosstalk' was run on.
    tag : str, optional
        The prefix it was run with. The default is ``'CT_'``.
    cmap : str | None, optional
        Matplotlib colormap name for the heatmap. The default is None - the
        active style's.

    Returns
    -------
    np.ndarray
        The cross-talk map (dimensionless ratio, NaN where undefined).

    Raises
    ------
    FileNotFoundError
        Raised when no saved map exists for ``tag``.
    """
    ct, meta = store.load_map(path, kind=_KIND, tag=tag)
    name = store.map_file_name(
        os.path.basename(os.path.normpath(path)), _KIND, tag
    ).removesuffix(".npy")
    results_dir = store.results_dir(path, _KIND)
    median, maximum, _ = plots.map_stats(ct, scale=100.0)

    if meta.get("combined"):
        fig = plots.sensor_map(
            ct,
            scale=100.0,
            title=(
                "S0 directional cross-talk\n2x2 block: mean | horiz / vert | "
                f"diag  (median {median:.1f} %)"
            ),
            clabel="CT (%)",
            cmap=cmap,
            xlabel="Micropixel column",
            ylabel="Micropixel row",
        )
        store.save_figure(fig, results_dir, f"{name}_heatmap.png")
        plt.close(fig)

        medians = meta["median_ct_percent"]
        fig, ax = plt.subplots()
        bars = ax.bar(list(medians), list(medians.values()))
        ax.bar_label(bars, fmt="%.2f")
        ax.set_ylabel("Median CT  (%)")
        ax.set_title("Cross-talk by direction")
        ax.grid(True, axis="y")
        fig.tight_layout()
        store.save_figure(fig, results_dir, f"{name}_direction_summary.png")
        plt.close(fig)
        return ct

    pair = meta.get("pair", "")
    fig = plots.sensor_map(
        ct,
        scale=100.0,
        title=f"{pair}  cross-talk\nmedian {median:.1f}  max {maximum:.1f}  %",
        clabel="CT (%)",
        cmap=cmap,
        xlabel="Macropixel column",
        ylabel="Macropixel row",
    )
    store.save_figure(fig, results_dir, f"{name}_heatmap.png")
    plt.close(fig)

    fig = plots.sorted_distribution(
        ct,
        scale=100.0,  # ratio -> percent; NaN (OR=0) macropixels are dropped
        ylabel="Cross-talk  (%)",
        xlabel="Macropixel percentile  (%)",
        title=f"Sorted {pair} cross-talk distribution",
        fmt=".1f",
        unit="%",
    )
    store.save_figure(fig, results_dir, f"{name}_distribution.png")
    plt.close(fig)
    return ct
