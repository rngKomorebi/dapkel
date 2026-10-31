"""Coincidence / timing-jitter (delta-t) analysis for the timestamping programs.

    process_coincidences   raw '.bin' -> processed/<name>_<tag>_coincidences.npy
    plot_coincidences      that grid  -> results/coincidences/ the fitted peak

In every frame where both pixels of a pair hold a timestamp, ``t_b - t_a`` is
one entry of the delta-t distribution; correlated pairs (SPDC signal/idler) pile
up at zero. Stage 1 bins those differences onto a fixed native grid *as it
finds them* and keeps only the counts - one row per pixel pair, one column per
grid cell - so the artifact is sized by the grid (~100 MB), not by the run, and
stage 2 can rebin it to any width or window in memory. Uncalibrated data lands
on a one-code grid and is lossless; a calibration smears the code lattice, so
each difference moves by at most half a cell (~3.6 ps at the default), far
below the detector jitter.

The ORT accidental background is a TRIANGLE, not flat: the free-running
oscillator has no cycle counter, so two uncorrelated phases difference into a
triangle that is zero at +/- one TDC full scale - 20 ns at 50 MHz, 100 ns at
10 MHz, 500 ns at 2 MHz. 'plot_coincidences' fits ``Gaussian + triangle`` on the raw
histogram, or measures the triangle away entirely with ``subtract_background``
(below) and fits ``Gaussian + constant`` on the residual - the model follows
from the data, it is not a choice. See ``docs/ort_triangle_background.md``.

The full scale is **measured**, never asked for: the largest code a run holds is
the range its TDC swept ('core.timing.full_scale_from_max_code'). Nor is the
open shutter time it - an 8 us shutter on a 10 MHz clock still yields ~100 ns
timestamps - so that belongs to rate normalisation and not here.

**Background subtraction.** An SPDC pair is always in one frame, so pairing
pixel A in frame ``i`` with pixel B in frame ``i + k`` (``k >= 1``) samples the
accidentals only, at the same rates. Subtract the mean of a few such lags from
lag 0 and the triangle cancels bin for bin, leaving the peak on a zero
baseline: a model-free coincidence count, and a test of whether the pedestal
was accidental at all. It does not improve CAR - the residual keeps the Poisson
noise of what it removed. See ``docs/guide/background_subtraction.md``.

UNITS: every time parameter is in seconds with an ``_s`` suffix; differences,
grids and plot axes are in picoseconds ('core.timing.to_ps'), whether or not a
calibration was applied - uncalibrated codes are converted with the nominal
'timing.CODE_S' on the way in, because a saved *difference* of two codes could
never be calibrated afterwards.

Check a fresh acquisition with 'dapkel.functions.data_quality' first.
See ``docs/guide/coincidences.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from dapkel.core import calibration, io, store, timing

# 'core.pairs' is aliased because 'plot_coincidences' takes a 'pairs' argument.
from dapkel.core import pairs as pixel_pairs
from dapkel.core.pairs import Pixel
from dapkel.functions.unpack import unpack

__all__ = ["process_coincidences", "plot_coincidences"]

#: Analysis name: the results sub-folder for the figures.
_KIND = "coincidences"

# Nominal code -> ps for uncalibrated data. The ring's own LSB, so it does not
# change with the reference clock - only the number of codes per full scale does.
_TS_CODE_PS = timing.to_ps(timing.CODE_S)

# Native grid cells per TDC code when a calibration is applied. It smears the
# integer code lattice, so there is none to land on; 1/10 of a code (~7 ps) is
# ~5x finer than the detector jitter. Raw codes are integers and need no
# subdivision.
_GRID_SUBDIV = 10

# Margin on the measured full scale when sizing the grid. The support comes from
# the largest code in the first file, an extreme value that runs slightly low,
# and the ring's per-pixel gain spreads real times a few % past the nominal code
# width. Whatever still lands outside is counted in 'n_outside', never dropped
# silently.
_SUPPORT_MARGIN = 1.2

# Files between checkpoint writes. The whole grid is rewritten each time - cheap
# against unpacking 50 files - so a run that dies leaves a valid artifact up to
# its last checkpoint, and 'resume' continues it.
_FLUSH_EVERY = 50

#: Frame offsets used to estimate the accidentals. The background's Poisson
#: noise enters the residual divided by their number; at eight it is ~12 % of
#: the residual variance, so more lags buy almost nothing.
_BACKGROUND_LAGS = (1, 2, 3, 4, 5, 6, 7, 8)

# Fitted-histogram bins grouped into one bin of the subtraction diagnostic's
# full-support panels: they show that the residual is flat and zero far from the
# peak, which needs counts per bin rather than resolution. Odd, so the coarse
# edges fall between fine bins.
_WIDE_REBIN = 25


def _counts_file(path: str, tag: str, *, create: bool = True) -> str:
    """Return ``processed/<name>_<tag>_coincidences.npy`` for a data folder.

    The tag is in the name so several programs (ORT, S0T, ...) recorded into one
    folder keep their own artifact - and, since figure names derive from it,
    their own figures.
    """
    name = os.path.basename(os.path.normpath(path))
    return os.path.join(
        store.processed_dir(path, create=create),
        store.map_file_name(name, "coincidences", tag),
    )


def _pixel_timestamps(
    time_series: np.ndarray,
    pixels: Sequence[Pixel],
    taps: np.ndarray | None,
) -> dict[Pixel, np.ndarray]:
    """Per-pixel, per-frame timestamps in ps, NaN where the pixel did not fire.

    A frame holds a timestamp when ``time_series > 0`` - empty slots decode to
    <= 0 in 'unpack'. NOTE: never ``photon_count > 0``; in timestamp mode those
    bits are part of the coarse code. With ``taps`` (the (32, 32, 8) compact
    calibration) each pixel's codes go through its own tap widths before any
    differencing, which is the only point a per-code calibration can act;
    without, the nominal code width.
    """
    out: dict[Pixel, np.ndarray] = {}
    for r, c in pixels:
        code = time_series[r, c]
        valid = code > 0
        t = np.full(code.shape, np.nan)
        t[valid] = (
            calibration.codes_to_ps(code[valid], taps[r, c])
            if taps is not None
            else code[valid] * _TS_CODE_PS
        )
        out[(r, c)] = t
    return out


def process_coincidences(
    path: str,
    nframes: int,
    pixels: Sequence[Sequence[Pixel]],
    *,
    tag: str = "ORT",
    mode: str = "all_pairs",
    bin_width_s: float | None = None,
    apply_TDC_calibration: bool = True,
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    subtract_background: bool = False,
    background_lags: int | Sequence[int] = _BACKGROUND_LAGS,
    max_files: int | None = None,
    rewrite: bool = False,
    resume: bool = True,
) -> str:
    """Unpack the timestamps, histogram every pair's delta-t and save the grid.

    Writes ``processed/<name>_<tag>_coincidences.npy``: ``(n_pairs, n_cells)``
    int64 counts on a native grid centred on zero (zero is a cell centre by
    construction) spanning the run's measured full scale, plus a sidecar with
    the grid, the pair labels and progress. With ``subtract_background`` a
    leading lag axis is added, lag 0 - the ordinary histogram - first.

    The grid is rewritten every 50 files, so a run that dies leaves a valid
    artifact up to its last checkpoint and ``resume`` continues it.

    Parameters
    ----------
    path : str
        Folder with the timestamping '.bin' files.
    nframes : int
        Frames the acquisition recorded per '.bin' file.
    pixels : Sequence[Sequence[tuple[int, int]]]
        Two groups ``[group_a, group_b]`` - e.g. the signal and idler blobs read
        off the hitmap. A bare ``(row, col)`` is a one-pixel group.
    tag : str, optional
        File tag. ``'S0T'``...``'S3T'`` read one micropixel and get that SPAD's
        calibration; ``'ORT'`` reads all four and gets the mean of theirs. The
        default is ``'ORT'``.
    mode : str, optional
        ``'all_pairs'`` (default): every A x B combination - use it when it is
        not known which pixel of a blob catches the photon. ``'1v1'``: zip the
        groups element-wise (equal lengths).
    bin_width_s : float | None, optional
        Width of one grid cell, in seconds. The default is None: one TDC code
        ('timing.CODE_S') uncalibrated, where that is exact, and a tenth of one
        with a calibration.
    apply_TDC_calibration : bool, optional
        Convert codes with the board's packaged compact calibration before
        differencing, correcting the ring's DNL at whatever clock the run used.
        The default is True, which needs ``daughterboard_number`` and
        ``motherboard_number``. False uses the nominal code width.
    daughterboard_number : str | None, optional
        Daughterboard id, e.g. ``'D0'``. The default is None.
    motherboard_number : str | None, optional
        Motherboard id, e.g. ``'M0'``. The default is None.
    subtract_background : bool, optional
        Also histogram frame-shifted pairs (``background_lags``), one grid plane
        each, so 'plot_coincidences' can measure and subtract the accidentals. Costs
        one differencing pass and one plane of grid per lag; the '.bin'
        decoding, which dominates, is paid once. The default is False.
    background_lags : int | Sequence[int], optional
        Frame offsets for the accidental estimate, all >= 1 (lag 0 is the
        signal and always included). The default is ``(1, ..., 8)``.
    max_files : int | None, optional
        Use at most this many files. The default is None (all).
    rewrite : bool, optional
        Overwrite a *complete* artifact for this data set - after a 5 s
        countdown naming it ('store.confirm_rewrite'). The default is False,
        which raises instead.
    resume : bool, optional
        Continue an incomplete artifact computed from the same inputs. The
        default is True.

    Returns
    -------
    str
        Path to the saved '.npy'.

    Raises
    ------
    FileExistsError
        Raised when an artifact exists and ``rewrite`` is False - complete, or
        computed from different inputs.
    ValueError
        Raised on a bad ``mode`` or ``background_lags``, or a calibration
        request without a board.
    """
    if subtract_background:
        extra = (
            [background_lags]
            if isinstance(background_lags, (int, np.integer))
            else list(background_lags)
        )
        lags = [0, *(int(k) for k in extra)]
        if len(lags) < 2 or min(lags[1:]) < 1 or len(set(lags)) != len(lags):
            raise ValueError(
                "background_lags must be distinct frame offsets >= 1 (lag 0 is "
                f"the signal); got {background_lags!r}"
            )
    else:
        lags = [0]

    files = io.find_bin_files(path, tag)[:max_files]
    pair_list = pixel_pairs.pair_list(pixels, mode)
    labels = [pixel_pairs.pair_label(a, b) for a, b in pair_list]
    all_pixels = list(dict.fromkeys(p for pair in pair_list for p in pair))

    taps = None
    board = None
    if apply_TDC_calibration:
        if daughterboard_number is None or motherboard_number is None:
            raise ValueError(
                "apply_TDC_calibration=True but no board was given. Pass "
                "daughterboard_number and motherboard_number, or set "
                "apply_TDC_calibration=False for the nominal code width."
            )
        board = f"{daughterboard_number}_{motherboard_number}"
        taps = calibration.board_taps(daughterboard_number, motherboard_number, tag)
    if bin_width_s is None:
        bin_width_s = timing.CODE_S / (_GRID_SUBDIV if taps is not None else 1)

    # The largest code is what the reference clock left in the data: the TDC
    # full scale, hence the widest a difference can be. One file is enough - it
    # is set by the clock, not the count - and the margin covers the rest.
    ts0, _ = unpack(files[0], nframes, compute_time_series=True)
    max_code = int(ts0[ts0 > 0].max()) if (ts0 > 0).any() else 0
    support_s = _SUPPORT_MARGIN * timing.full_scale_from_max_code(max_code)
    zero_index = int(np.ceil(support_s / bin_width_s))
    n_bins = 2 * zero_index + 1
    grid_ps = timing.to_ps(bin_width_s)

    counts_path = _counts_file(path, tag)
    # Everything that must match for a partial run to be resumed. Files are
    # fingerprinted by (name, size): progress is "the first N of them".
    digest = hashlib.sha1()
    for fp in files:
        digest.update(f"{os.path.basename(fp)}:{os.path.getsize(fp)};".encode())
    signature = {
        "n_files": len(files),
        "files_digest": digest.hexdigest(),
        "labels_digest": hashlib.sha1("|".join(labels).encode()).hexdigest(),
        "mode": mode,
        "nframes": int(nframes),
        "bin_width_s": float(bin_width_s),
        "support_s": float(support_s),
        "calibration": board,
        "lags": lags,
    }

    counts = np.zeros(
        (len(labels), n_bins) if len(lags) == 1 else (len(lags), len(labels), n_bins),
        dtype=np.int64,
    )
    start = total = outside = 0
    existing = store.read_meta(counts_path) if os.path.isfile(counts_path) else {}
    if existing.get("signature") == signature and not existing.get("complete"):
        if resume:
            saved, _ = store.load_map(npy_path=counts_path)
            # The sidecar is moved into place after the array, so a crash
            # between the two can pair a newer array with an older sidecar.
            # Resuming on that would count files twice; start over instead.
            if saved.shape == counts.shape and int(saved.sum()) == existing.get(
                "total_in_grid"
            ):
                counts = saved
                start = int(existing["files_done"])
                total = int(existing["total"])
                outside = int(existing["n_outside"])
                print(f"\n> > > Resuming at file {start} of {len(files)} < < <")
            else:
                print("\n  The saved grid disagrees with its sidecar; starting over.")
    elif os.path.isfile(counts_path):
        if not rewrite:
            raise FileExistsError(
                f"{counts_path} exists"
                + (
                    f" and covers all {len(files)} file(s)"
                    if existing.get("signature") == signature
                    else " but was computed from different inputs"
                )
                + ". Pass rewrite=True to replace it."
            )
        store.confirm_rewrite([counts_path])

    def _checkpoint(files_done: int, complete: bool) -> None:
        """Write the grid and its sidecar, each via a temporary moved into place."""
        meta = {
            "signature": signature,
            "labels": labels,
            "unit": "ps",
            "tag": tag,
            "calibration": board,
            "bin_width_s": float(bin_width_s),
            "grid_ps": float(grid_ps),
            "support_s": float(support_s),
            "max_code": max_code,
            "n_bins": n_bins,
            "zero_index": zero_index,
            "files_done": int(files_done),
            "n_files": len(files),
            "complete": complete,
            "total": int(total),
            "n_outside": int(outside),
            "total_in_grid": int(counts.sum()),
        }
        if len(lags) > 1:
            # Lag k pairs frame i with i + k, so it loses the last k frames of
            # every file; the subtraction rescales by this.
            meta["lags"] = lags
            meta["frames_used"] = [
                max(int(nframes) - k, 0) * int(files_done) for k in lags
            ]
        tmp_npy = counts_path + ".tmp.npy"
        tmp_meta = store.meta_path(counts_path) + ".tmp"
        np.save(tmp_npy, counts)
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        os.replace(tmp_npy, counts_path)
        os.replace(tmp_meta, store.meta_path(counts_path))

    print(
        f"\n> > > Histogramming delta-t from {len(files) - start} '{tag}' file(s) "
        f"onto {'x'.join(map(str, counts.shape))} cells of "
        f"{bin_width_s * 1e12:.2f} ps spanning +/-{support_s * 1e9:.1f} ns "
        f"({counts.nbytes / 1024**2:.0f} MB)"
        + (f", lags {lags}" if len(lags) > 1 else "")
        + " < < <\n"
    )
    planes = counts if len(lags) > 1 else counts[np.newaxis]
    for i, fp in enumerate(tqdm(files[start:], desc=tag), start=start + 1):
        ts, _ = unpack(fp, nframes, compute_time_series=True)
        pixel_ts = _pixel_timestamps(ts, all_pixels, taps)
        for plane, lag in zip(planes, lags, strict=True):
            n = max(int(nframes) - lag, 0)
            for row, (a, b) in enumerate(pair_list):
                # B shifted by 'lag' frames: lag 0 is the coincidence, lag > 0 a
                # pure accidental sample. NaN where either pixel did not fire.
                d = pixel_ts[b][lag : lag + n] - pixel_ts[a][:n]
                d = d[np.isfinite(d)]
                if not d.size:
                    continue
                # np.add.at, not bincount: one file adds a handful of entries
                # per pair, and bincount would pay for the whole grid each time.
                idx = np.rint(d / grid_ps).astype(np.int64) + zero_index
                keep = (idx >= 0) & (idx < n_bins)
                total += d.size
                outside += d.size - int(np.count_nonzero(keep))
                np.add.at(plane[row], idx[keep], 1)
        if i % _FLUSH_EVERY == 0:
            _checkpoint(i, False)
    _checkpoint(len(files), True)

    print(
        f"\n> > > {total} difference(s) across {len(labels)} pair(s) binned; "
        f"{outside} outside +/-{support_s * 1e9:.1f} ns. Saved to "
        f"{counts_path} < < <"
    )
    if len(lags) > 1:
        per_lag = counts.sum(axis=(1, 2))
        print(
            f"  lag 0: {per_lag[0]}; shifted lags: {per_lag[1:].mean():.0f} on "
            f"average -> a raw excess of {per_lag[0] - per_lag[1:].mean():.0f}."
        )
    if outside:
        print(
            f"  NOTE: {100 * outside / max(total, 1):.2f}% of differences fell "
            "outside the grid (counted, not binned): the run's timestamps reach "
            "further than its first file suggested."
        )
    return counts_path


def _snap_binning(
    bin_width_s: float, window_s: float, cell_s: float
) -> tuple[float, float]:
    """Round a binning onto the native grid, and say so if it moved.

    * the **width** to a whole number of cells, or bins alternate 9, 10, 9 ...
      cells and a smooth distribution grows a ``1 / n_cells`` sawtooth (it moved
      the peak bin by 8 % on real data);
    * that number to an **odd** one: bins are centred on zero, so an odd width
      puts their edges on cell boundaries, an even one through cell centres -
      forcing whole cells one way (~1 % of the counts on a real run);
    * the **window** to a whole number of bins, so zero is a bin centre.
    """
    raw = bin_width_s / cell_s
    cells = max(int(round(raw)), 1)
    if cells % 2 == 0:
        cells += 1 if raw >= cells else -1  # the nearer odd neighbour
    width = cells * cell_s
    window = max(int(round(window_s / width)), 1) * width
    moved = []
    if abs(width - bin_width_s) > 1e-21:
        moved.append(
            f"bin_width {bin_width_s * 1e12:.2f} -> {width * 1e12:.2f} ps "
            f"({cells} cells)"
        )
    if abs(window - window_s) > 1e-21:
        moved.append(f"window {window_s * 1e12:.0f} -> {window * 1e12:.0f} ps")
    if moved:
        print(f"  snapped to the {cell_s * 1e12:.2f} ps grid: {'; '.join(moved)}.")
    return width, window


def _rebin(
    counts: np.ndarray, centers: np.ndarray, bin_width_s: float, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Regroup histogram cells onto bins centred on zero, within +/- window.

    Each cell goes whole to the bin its centre falls in, so the result is exact
    when the width is an odd number of cells ('_snap_binning'). ``counts`` is
    one histogram or one row per lag, all rebinned onto the same bins - the lags
    are only subtractable if they share them.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The rebinned counts (same leading shape) and the bin centres in ps.
    """
    width_ps = timing.to_ps(bin_width_s)
    n = int(np.ceil(timing.to_ps(window_s) / width_ps - 1e-9))
    edges = (np.arange(2 * n + 2) - n - 0.5) * width_ps
    idx = np.searchsorted(edges, centers, side="right") - 1
    inside = (idx >= 0) & (idx < 2 * n + 1)
    binned = np.stack(
        [
            np.bincount(idx[inside], weights=row[inside], minlength=2 * n + 1)
            for row in np.atleast_2d(counts)
        ]
    ).astype(np.int64)
    return (
        binned if np.ndim(counts) == 2 else binned[0],
        (np.arange(2 * n + 1) - n) * width_ps,
    )


def _subtract(
    counts: np.ndarray, centers: np.ndarray, frames_used: Sequence[int]
) -> dict:
    """Lag 0 minus the mean of the shifted lags, each rescaled to lag 0's frames.

    Lag ``k`` loses the last ``k`` frames of every file, so it is scaled by
    ``frames[0] / frames[k]`` - ~0.01 % at typical frame counts, applied because
    it is free. The error is Poisson: the signal plus the variance of the mean
    of K rescaled histograms, written out rather than approximated as bkg / K.
    """
    counts = np.asarray(counts, dtype=np.float64)
    frames = np.asarray(frames_used, dtype=np.float64)
    scale = (frames[0] / frames)[1:, None]
    k = counts.shape[0] - 1
    background = (counts[1:] * scale).mean(axis=0)
    return {
        "centers": centers,
        "signal": counts[0],
        "background": background,
        "residual": counts[0] - background,
        "error": np.sqrt(counts[0] + (counts[1:] * scale**2).sum(axis=0) / k**2),
        "bin_width_s": float(np.median(np.diff(centers))) * 1e-12,
        "k": k,
    }


def _subtraction_figure(
    sub: dict, full: dict | None, *, name: str, label: str, window_s: float
) -> plt.Figure:
    """Overlay, residual and peak, each over +/- ``window_s``.

    The middle panel is the evidence, and the one people skip: if the residual
    is not flat and zero away from the peak, the pedestal was not accidental and
    no number from the bottom panel means anything. Arrays are sliced rather
    than the axes merely limited, or the y scale stays set by off-screen
    triangle and the peak is a stub.
    """

    def _clip(d: dict) -> dict:
        keep = np.abs(d["centers"]) <= window_s * 1e12
        return {
            k: v[keep] if isinstance(v, np.ndarray) and v.shape == keep.shape else v
            for k, v in d.items()
        }

    wide, zoom = _clip(full if full is not None else sub), _clip(sub)
    fig, axes = plt.subplots(3, 1)
    x = wide["centers"] / 1000.0
    per = f"{wide['bin_width_s'] * 1e12:.0f} ps"
    axes[0].plot(x, wide["signal"], lw=1.2, label="same frame (lag 0)")
    axes[0].plot(
        x,
        wide["background"],
        lw=1.2,
        label=f"shifted frames (mean of {wide['k']} lags)",
    )
    axes[0].set_ylabel(f"per {per}")
    axes[0].set_title(f"{label.upper()} delta-t, {name}")
    axes[1].fill_between(
        x, -wide["error"], wide["error"], alpha=0.35, label="+/- 1 sigma"
    )
    axes[1].plot(x, wide["residual"], lw=1.0, label="lag 0 - accidentals")
    axes[1].set_ylabel(f"excess / {per}")
    axes[1].set_title("Residual over the plotted window")
    xz = zoom["centers"] / 1000.0
    axes[2].fill_between(
        xz, -zoom["error"], zoom["error"], alpha=0.35, label="+/- 1 sigma"
    )
    axes[2].step(xz, zoom["residual"], where="mid", lw=1.0, label="subtracted")
    axes[2].set_ylabel(f"excess / {zoom['bin_width_s'] * 1e12:.0f} ps")
    axes[2].set_title("Coincidence peak on a zero baseline")
    for i, ax in enumerate(axes):
        if i:
            ax.axhline(0, color="k", lw=0.8)
        # Outside, top-aligned: at the house font size these captions cover the
        # peak wherever they sit inside a panel. 'tight_layout' measures them,
        # so the column they need is reserved without a 'rect'.
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            handlelength=1.4,
            labelspacing=0.5,
        )
        ax.grid(True, lw=0.5, alpha=0.6)
        ax.set_xlabel("Timestamp difference  (ns)")
        ax.set_xlim(-window_s * 1e9, window_s * 1e9)
    fig.tight_layout()
    return fig


def _subtract_background(
    lag_counts: np.ndarray,
    centers: np.ndarray,
    frames_used: Sequence[int],
    *,
    name: str,
    results_dir: str,
    bin_width_s: float,
    zoom_window_s: float,
    fit_window_s: float | None,
    label: str,
) -> dict:
    """Turn ``(n_lags, n_bins)`` counts over the full support into the residual.

    Returns '_subtract' over the zoom window plus ``model_free`` - the residual
    summed within ``fit_window_s`` of zero: the coincidence count with no peak
    shape assumed - ``full`` (the same over the whole support, coarsely binned,
    for the control) and ``png_path`` of the three-panel diagnostic.
    """
    peak_window_s = zoom_window_s if fit_window_s is None else fit_window_s
    keep = np.abs(centers) <= timing.to_ps(zoom_window_s)
    sub = _subtract(lag_counts[:, keep], centers[keep], frames_used)

    in_peak = np.abs(sub["centers"]) <= timing.to_ps(peak_window_s)
    n = float(sub["residual"][in_peak].sum())
    n_err = float(np.sqrt((sub["error"][in_peak] ** 2).sum()))
    n_bkg = float(sub["background"][in_peak].sum())
    sub["model_free"] = {
        "n": n,
        "n_err": n_err,
        "significance": n / n_err if n_err else np.inf,
        "n_background": n_bkg,
        # Coincidences over accidentals in the window - NOT the fitted
        # peak-height ratio; the two answer different questions.
        "car": n / n_bkg if n_bkg else np.inf,
        "window_s": float(peak_window_s),
        "n_bins": int(np.count_nonzero(in_peak)),
    }

    support_s = timing.to_s(float(np.abs(centers).max()))
    wide_counts, wide_centers = _rebin(
        lag_counts, centers, _WIDE_REBIN * bin_width_s, support_s
    )
    full = _subtract(wide_counts, wide_centers, frames_used)
    sub["full"] = full

    far = np.abs(full["centers"]) > max(5 * timing.to_ps(peak_window_s), 15000.0)
    far_sum = float(full["residual"][far].sum())
    far_err = float(np.sqrt((full["error"][far] ** 2).sum()))
    print(
        f"\n> > > lag 0 {int(full['signal'].sum())} vs accidentals "
        f"{full['background'].sum():.0f} (mean of {full['k']} shifted lag(s)) "
        f"over the whole support < < <\n"
        f"  coincidences within +/-{peak_window_s * 1e12:.0f} ps: {n:.0f} +/- "
        f"{n_err:.0f}  ({sub['model_free']['significance']:.1f} sigma), on "
        f"{n_bkg:.0f} accidentals -> CAR {sub['model_free']['car']:.3f}\n"
        f"  control, |delta| beyond the peak: {far_sum:.0f} +/- {far_err:.0f} "
        f"({far_sum / far_err if far_err else np.inf:+.1f} sigma from zero) - "
        "this must be ~0, or the pedestal is not purely accidental."
    )

    # The coarse full-support binning only earns its place when the plotted
    # range is the whole support; over a zoom it would draw a dozen wide bins.
    fig = _subtraction_figure(
        sub,
        full if zoom_window_s >= support_s else None,
        name=name,
        label=label,
        window_s=zoom_window_s,
    )
    sub["png_path"] = store.save_figure(
        fig, results_dir, f"{name}_{label}_background_subtraction.png"
    )
    return sub


def _fit_peak(
    x: np.ndarray,
    y: np.ndarray,
    *,
    half_base: float | None = None,
    fit_window: float | None = None,
) -> dict:
    """Fit a Gaussian peak on a triangle (``half_base`` given) or a constant.

    Triangle: the raw ORT histogram, fitted over all of it, with ``half_base``
    (the measured full scale, in ps) seeding a free parameter. Constant: a
    lag-subtracted residual, fitted within ``fit_window`` ps of the tallest bin;
    its level is unbounded below because that baseline is zero-mean noise.

    A fit that fails - including one whose sigma the covariance cannot tell from
    zero, which a wrong background model produces - comes back ``ok=False`` with
    a RuntimeWarning, so a plot without a curve always says why.

    Returns
    -------
    dict
        ``amp``, ``mu``, ``sigma``, ``sigma_err``, ``bkg`` (constant or triangle
        apex), ``half_base`` (triangle), ``fwhm``, ``fwhm_err``,
        ``jitter_per_detector`` (sigma / sqrt 2), its ``_err``, and ``ok``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    result = {
        "amp": np.nan,
        "mu": 0.0,
        "sigma": np.nan,
        "sigma_err": np.nan,
        "bkg": np.nan,
        "fwhm": np.nan,
        "fwhm_err": np.nan,
        "jitter_per_detector": np.nan,
        "jitter_per_detector_err": np.nan,
        "ok": False,
    }

    def _give_up(why: str) -> dict:
        warnings.warn(
            f"Gaussian peak fit did not converge ({why}); the plot shows the "
            "histogram without a fitted curve and sigma/FWHM are NaN.",
            RuntimeWarning,
            stacklevel=3,
        )
        return result

    if half_base is None:
        tail = np.r_[y[: len(y) // 3], y[-(len(y) // 3) :]]
        bkg0 = float(np.median(tail)) if tail.size else 0.0
        if fit_window is not None:
            sel = np.abs(x - x[np.argmax(y)]) <= fit_window
            x, y = x[sel], y[sel]
        if len(x) < 4:
            return _give_up(f"only {len(x)} bins in the fit window")
        w = np.clip(y - bkg0, 0.0, None)
        if w.sum() <= 0:
            return _give_up("no counts above the background level")
        mu0 = float(np.sum(w * x) / w.sum())
        sigma0 = float(np.sqrt(np.sum(w * (x - mu0) ** 2) / w.sum())) or x[1] - x[0]

        def model(x, amp, mu, sigma, bkg):
            return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) + bkg

        p0 = [float(y.max() - bkg0), mu0, sigma0, bkg0]
        lo = [0.0, float(x.min()), 0.0, -np.inf]
        hi = [np.inf, float(x.max()), np.inf, np.inf]
    else:
        if len(x) < 6:
            return _give_up(f"only {len(x)} bins")
        # Apex seed: the median just off zero (clear of the peak), scaled back
        # up to x = 0 along the triangle.
        dx = float(np.median(np.diff(x)))
        near = (np.abs(x) > 3 * dx) & (np.abs(x) < 0.15 * half_base)
        apex0 = (
            float(np.median(y[near]) / (1.0 - np.median(np.abs(x[near])) / half_base))
            if near.any()
            else float(np.median(y))
        )
        excess = np.clip(
            y - apex0 * np.clip(1.0 - np.abs(x) / half_base, 0.0, None), 0.0, None
        )
        mu0 = float(x[np.argmax(excess)])
        # Sigma from the excess near the peak only: far out the apex-seed
        # mismatch leaves a broad residual that would inflate the moment.
        w = np.where(np.abs(x - mu0) < 0.05 * half_base, excess, 0.0)
        sigma0 = (
            float(np.sqrt(np.sum(w * (x - mu0) ** 2) / w.sum())) if w.sum() else 0.0
        ) or 5 * dx

        def model(x, amp, mu, sigma, bkg, hb):
            tri = np.clip(1.0 - np.abs(x) / hb, 0.0, None)
            return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) + bkg * tri

        p0 = [float(excess.max()), mu0, sigma0, apex0, half_base]
        lo = [0.0, float(x.min()), 0.0, 0.0, 0.5 * half_base]
        hi = [np.inf, float(x.max()), 0.5 * half_base, np.inf, 2.0 * half_base]

    # curve_fit rejects a seed on or outside a bound, so clamp strictly inside.
    p0 = [
        min(
            max(v, lo_i + 1e-9 * (abs(lo_i) + 1)) if np.isfinite(lo_i) else v,
            hi_i - 1e-9 * (abs(hi_i) + 1) if np.isfinite(hi_i) else np.inf,
        )
        for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)
    ]
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        return _give_up(str(exc).strip().splitlines()[0])

    amp, mu, sigma, bkg = popt[:4]
    if amp <= 0 or not np.isfinite(sigma) or sigma <= 0:
        return _give_up(f"fit returned amp={amp:.3g}, sigma={sigma:.3g}")
    sigma_err = float(np.sqrt(pcov[2, 2])) if np.all(np.isfinite(pcov)) else np.nan
    if np.isfinite(sigma_err) and sigma_err >= sigma:
        # Converged on *something*, but the width is not determined by the data
        # - a Gaussian let loose on a residual that is still a triangle came out
        # 39 ns +/- 2.7 us on a real run. Not a width.
        return _give_up(
            f"sigma = {sigma:.3g} +/- {sigma_err:.3g} - the width is not "
            "determined by the data (is the background model right?)"
        )
    sigma = float(abs(sigma))
    result.update(
        amp=float(amp),
        mu=float(mu),
        sigma=sigma,
        sigma_err=sigma_err,
        bkg=float(bkg),
        fwhm=2.355 * sigma,
        fwhm_err=2.355 * sigma_err,
        jitter_per_detector=sigma / np.sqrt(2),
        jitter_per_detector_err=sigma_err / np.sqrt(2),
        ok=True,
    )
    if half_base is not None:
        result["half_base"] = float(popt[4])
    return result


def _pm(value: float, err: float, sep: str = " ± ") -> str:
    """``'123 ± 4 ps'``, or ``'123 ps'`` when the error is unknown."""
    if not np.isfinite(err):
        return f"{value:.0f} ps"
    return f"{value:.0f}{sep}{err:.0f} ps"


def plot_coincidences(
    path: str,
    tag: str = "ORT",
    *,
    bin_width_s: float = timing.CODE_S,
    plot_window_s: float | None = None,
    fit_window_s: float | None = 1e-9,
    pairs: Sequence[str] | None = None,
    label: str = "spdc",
    title: str | None = None,
    subtract_background: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rebin the saved grid, fit the coincidence peak and save the figure.

    Nothing is read but the one saved array, and the rebinning is in memory, so
    changing ``bin_width_s`` or ``plot_window_s`` costs milliseconds. The
    binning is snapped onto the grid - a whole, odd number of cells per bin and
    a whole number of bins per window - which is what makes the rebin exact.

    The fit model is not a parameter: on the raw histogram it is ``Gaussian +
    triangle`` with the measured full scale seeding the half-base; with
    ``subtract_background`` it is ``Gaussian + constant`` within
    ``fit_window_s`` of the peak, there being no triangle left.

    Parameters
    ----------
    path : str
        Data folder that 'process_coincidences' was run on.
    tag : str, optional
        The tag it was run with. The default is ``'ORT'``.
    bin_width_s : float, optional
        Histogram bin width, in seconds. The default is one TDC code.
    plot_window_s : float | None, optional
        Half-range around zero, in seconds. The default is None: the measured
        TDC full scale, the whole support the triangle fit wants.
    fit_window_s : float | None, optional
        Half-range around the peak, in seconds - the flat fit's range and the
        window the model-free coincidence count is summed over. Ignored by the
        triangle fit. The default is ``1e-9``.
    pairs : Sequence[str] | None, optional
        Pool only these pixel-pair labels (``"ra,ca-rb,cb"``). The default is
        None (every pair).
    label : str, optional
        Short label for the title and file name (``'spdc'``, ``'hbt'``). The
        default is ``'spdc'``.
    title : str | None, optional
        Override for the plot title. The default is None.
    subtract_background : bool, optional
        Subtract the frame-shifted accidentals before fitting and save the
        three-panel diagnostic beside the fit; read its middle panel before
        quoting anything. Needs a grid processed with
        ``subtract_background=True``. The default is False.

    Returns
    -------
    counts : np.ndarray
        The fitted histogram - the residual with ``subtract_background``.
    centers : np.ndarray
        Bin centres in picoseconds, to re-fit without re-reading anything.
    fit : dict
        ``amp``, ``mu``, ``sigma``, ``sigma_err``, ``bkg``, ``fwhm``,
        ``jitter_per_detector`` (each with ``_err``), ``car`` (peak-to-
        background, raw histogram only), ``n`` and ``ok``; with
        ``subtract_background`` also ``model_free``, ``subtraction`` and
        ``subtraction_png``.

    Raises
    ------
    FileNotFoundError
        Raised when no saved grid exists for ``tag``.
    ValueError
        Raised when ``subtract_background`` is asked of a grid without lags.
    """
    counts_path = _counts_file(path, tag, create=False)
    grid, meta = store.load_map(npy_path=counts_path)
    if not meta.get("complete", True):
        print(
            f"  NOTE: {os.path.basename(counts_path)} is incomplete - "
            f"{meta.get('files_done')} of {meta.get('n_files')} file(s)."
        )
    labels = list(meta["labels"])
    pair_axis = grid.ndim - 2
    if pairs is not None:
        missing = [p for p in pairs if p not in labels]
        if missing:
            print(f"  no such pixel-pair row(s): {missing}")
        grid = np.take(
            grid, [labels.index(p) for p in pairs if p in labels], axis=pair_axis
        )
        labels = [p for p in pairs if p in labels]
    grid = grid.sum(axis=pair_axis)  # (n_cells,), or (n_lags, n_cells)
    cell_ps = float(meta["grid_ps"])
    grid_centers = (np.arange(grid.shape[-1]) - int(meta["zero_index"])) * cell_ps

    name = os.path.basename(counts_path).removesuffix("_coincidences.npy")
    results_dir = store.results_dir(path, _KIND)
    # The grid was sized from the run's own largest code, so this is the
    # measured full scale - the triangle's half-base, with no clock stated.
    full_scale_s = timing.full_scale_from_max_code(int(meta["max_code"]))
    if plot_window_s is None:
        plot_window_s = full_scale_s
    bin_width_s, plot_window_s = _snap_binning(
        bin_width_s, plot_window_s, timing.to_s(cell_ps)
    )

    sub = None
    if subtract_background:
        if grid.ndim != 2:
            raise ValueError(
                "subtract_background=True needs lag planes. Reprocess with "
                "process_coincidences(..., subtract_background=True)."
            )
        # One binning over the whole support for the fit and the diagnostic, so
        # the panels cannot disagree about a bin edge.
        lag_counts, lag_centers = _rebin(
            grid,
            grid_centers,
            bin_width_s,
            timing.to_s(float(np.abs(grid_centers).max())),
        )
        sub = _subtract_background(
            lag_counts,
            lag_centers,
            meta["frames_used"],
            name=name,
            results_dir=results_dir,
            bin_width_s=bin_width_s,
            zoom_window_s=plot_window_s,
            fit_window_s=fit_window_s,
            label=label,
        )
        counts, centers = sub["residual"], sub["centers"]
        # Summing a residual's bins *is* the coincidence count; anything outside
        # the peak is noise around zero.
        n = int(round(sub["model_free"]["n"]))
        fit = _fit_peak(
            centers,
            counts,
            fit_window=None if fit_window_s is None else timing.to_ps(fit_window_s),
        )
        fit["car"] = np.nan  # a ratio to a zero baseline is a division by noise
    else:
        signal = grid[0] if grid.ndim == 2 else grid
        counts, centers = _rebin(signal, grid_centers, bin_width_s, plot_window_s)
        n = int(counts.sum())
        print(
            f"\n> > > {int(signal.sum())} difference(s) across {len(labels)} "
            f"pair(s); {n} inside +/-{plot_window_s * 1e9:.2f} ns, rebinned to "
            f"{counts.size} x {bin_width_s * 1e12:.0f} ps < < <"
        )
        fit = _fit_peak(centers, counts, half_base=timing.to_ps(full_scale_s))
        fit["car"] = (fit["amp"] + fit["bkg"]) / fit["bkg"] if fit["bkg"] else np.inf
    fit["n"] = n

    fig, ax = plt.subplots()
    ax.bar(centers, counts, width=timing.to_ps(bin_width_s), label=f"data ({n} pairs)")
    if fit["ok"]:
        window_ps = timing.to_ps(plot_window_s)
        xs = np.linspace(-window_ps, window_ps, 4000)
        shape = (
            np.clip(1.0 - np.abs(xs) / fit["half_base"], 0.0, None)
            if "half_base" in fit
            else 1.0
        )
        jitter = _pm(fit["jitter_per_detector"], fit["jitter_per_detector_err"])
        ax.plot(
            xs,
            fit["amp"] * np.exp(-((xs - fit["mu"]) ** 2) / (2.0 * fit["sigma"] ** 2))
            + fit["bkg"] * shape,
            "r-",
            linewidth=2,
            label=(
                f"sigma = {_pm(fit['sigma'], fit['sigma_err'])}\n"
                f"FWHM = {_pm(fit['fwhm'], fit['fwhm_err'])}\n"
                f"per-detector = {jitter}"
            ),
        )
    ax.set_xlabel("Timestamp difference  (ps)")
    ax.set_ylabel(
        "Coincidences - accidentals" if subtract_background else "Coincidences"
    )
    ax.set_title(title or f"{label.upper()} coincidence peak  ({name})")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend()
    stem = "coincidences_sub" if subtract_background else "coincidences"
    store.save_figure(fig, results_dir, f"{name}_{label}_{stem}.png")

    if fit["ok"]:
        print(
            f"  peak: mu {fit['mu']:.0f} ps  sigma "
            f"{_pm(fit['sigma'], fit['sigma_err'], ' +- ')}  FWHM "
            f"{_pm(fit['fwhm'], fit['fwhm_err'], ' +- ')}  per-detector "
            f"{_pm(fit['jitter_per_detector'], fit['jitter_per_detector_err'], ' +- ')}"
            + ("" if subtract_background else f"  peak/bkg {fit['car']:.2f}")
        )
    else:
        print(
            "  no Gaussian peak fit - check the blob selection, and that the run "
            "has proper (stopped, fine) timing."
        )
    if sub is not None:
        fit["model_free"] = sub["model_free"]
        fit["subtraction"] = sub
        fit["subtraction_png"] = sub["png_path"]
    return counts, centers, fit
