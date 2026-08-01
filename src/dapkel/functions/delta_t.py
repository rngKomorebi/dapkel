"""Coincidence / timing-jitter (delta-t) pipeline for the ORT program.

Drives 'unpack' and 'calc_diff' over a folder, pools the per-pixel-pair
timestamp differences into a '.feather' table, then histograms, fits and plots
the coincidence peak.

NOTE on fitting: the ORT accidental background is TRIANGULAR, not flat, because
the free-running oscillator has no cycle counter. Use
'fit_gaussian_on_triangle' over a wide window; 'fit_gaussian_peak' (flat
background) is valid only on a narrow window around the peak, and a flat fit
over a wide one inflates the fitted sigma.

Check a fresh acquisition with 'dapkel.functions.data_quality' first.
See ``docs/guide/coincidences.md`` and ``docs/ort_triangle_background.md``.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.optimize import curve_fit
from tqdm import tqdm

from dapkel.core import io, store
from dapkel.functions import calc_diff as cd
from dapkel.functions import tdc_calibration as tc
from dapkel.functions.calc_diff import Pixel
from dapkel.functions.unpack import unpack

__all__ = [
    # stage 1 - difference timestamps and persist to feather
    "calculate_and_save_timestamp_differences",
    "combine_delta_t_feathers",
    # fit models and fitting
    "gaussian",
    "gaussian_plus_triangle",
    "fit_gaussian_peak",
    "fit_gaussian_on_triangle",
    # drivers - load saved feathers, figures on disk
    "collect_and_plot_timestamp_differences",
]

#: Analysis name: the results sub-folder for the figures.
_KIND = "coincidences"

# Average raw-code -> picosecond conversion: ~100 ns oscillator period over
# ~1300 codes. Replace with a per-code LUT (density test) via 'time_lut'.
_TS_CODE_PS = 77.0




def _structure_pixel_timestamps(
    time_series: np.ndarray,
    pixels: Sequence[Pixel],
    valid_min: float,
    time_lut: np.ndarray | None,
) -> dict[Pixel, np.ndarray]:
    """Extract per-pixel, per-frame timestamps for a set of pixels.

    Parameters
    ----------
    time_series : np.ndarray
        The (32, 32, nframes) TDC codes from 'unpack'.
    pixels : Sequence[tuple[int, int]]
        The ``(row, col)`` pixels to extract.
    valid_min : float
        A frame holds a valid timestamp when ``time_series > valid_min``;
        empty slots decode to the ``unpack`` sentinel (<= 0). NOTE: do not
        use ``photon_count > 0`` — in timestamp mode those bits are part of
        the coarse code.
    time_lut : np.ndarray | None
        Density-test lookup table mapping an integer TDC code -> picoseconds,
        applied here before differencing so a non-linear LUT is handled
        correctly. Two shapes are accepted:

        * 1D ``(n_codes,)`` — one table shared by every pixel;
        * 3D ``(32, 32, n_codes)`` — a per-pixel table (as produced by
          'tdc_calibration.collect_and_save_luts'); pixel ``(r, c)`` is
          calibrated with ``time_lut[r, c]``.

        Codes outside the calibrated range ``0 .. n_codes-1`` have no entry
        and are dropped (NaN). If None, the raw codes are kept.

    Returns
    -------
    dict[tuple[int, int], np.ndarray]
        Mapping ``(row, col) -> 1D float array`` of per-frame timestamps
        (codes, or ps if ``time_lut`` given), NaN where the pixel had no
        photon that frame.
    """
    per_pixel_lut = time_lut is not None and time_lut.ndim == 3
    n_codes = time_lut.shape[-1] if time_lut is not None else 0

    out: dict[Pixel, np.ndarray] = {}
    for r, c in pixels:
        code = time_series[r, c].astype(np.float64)
        valid = code > valid_min
        if time_lut is not None:
            idx = code.astype(np.int64)
            # Codes past the calibrated range have no LUT entry; treat them
            # as invalid rather than indexing out of bounds.
            usable = valid & (idx >= 0) & (idx < n_codes)
            lut_pix = time_lut[r, c] if per_pixel_lut else time_lut
            vals = np.full(code.shape, np.nan)
            vals[usable] = lut_pix[idx[usable]]
        else:
            vals = np.where(valid, code, np.nan)
        out[(r, c)] = vals
    return out


def calculate_and_save_timestamp_differences(
    path: str,
    pixels: Sequence[Sequence[Pixel]],
    rewrite: bool = False,
    *,
    tag: str = "ORT",
    mode: str = "all_pairs",
    delta_window: float | None = None,
    valid_min: float = 0.0,
    apply_TDC_calibration: bool = True,
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    spad: int | str = "average",
    nframes: int | None = None,
    max_files: int | None = None,
) -> str:
    """Unpack ORT data, compute blob-vs-blob delta-t, and save to feather.

    Writes ``path/processed/<name>_delta_t.feather`` with **one column per
    pixel pair** (named ``"ra,ca-rb,cb"``), NaN-padded to a common length;
    every requested pair gets a column even if it saw no coincidences.

    With ``apply_TDC_calibration=True`` (default) the board's density-test LUT
    converts timestamps to picoseconds *before* differencing, saving
    ``delta_ps``; otherwise raw ``delta_code`` is saved. Which one it is goes
    into the feather's schema metadata as ``delta_unit``.

    Parameters
    ----------
    path : str
        Path to the folder with the ORT '.bin' data files.
    pixels : Sequence[Sequence[tuple[int, int]]]
        Two groups ``[group_a, group_b]`` — the signal and idler blobs read
        off the hitmap, each a list of ``(row, col)`` pixels.
    rewrite : bool, optional
        Overwrite an existing '.feather' for this data set. A guard against
        accidental overwrites. The default is False.
    tag : str, optional
        Filename fragment selecting the files. The default is ``'ORT'``.
    mode : str, optional
        ``'all_pairs'`` (default) or ``'1v1'``; see
        'calc_diff.calculate_differences'.
    delta_window : float | None, optional
        Keep only differences with ``abs(delta) <= delta_window`` while
        accumulating (bounds memory). In code units, or ps if ``time_lut``
        is given. The default is None (keep all).
    valid_min : float, optional
        A frame is valid when ``time_series > valid_min``. The default is
        0.0.
    apply_TDC_calibration : bool, optional
        Apply the density-test TDC calibration (per-pixel code -> ps LUT)
        before differencing, so the saved ``delta_ps`` is already calibrated.
        The default is True, in which case the packaged board LUT is loaded
        via ``daughterboard_number`` / ``motherboard_number`` (a ValueError
        is raised if either is missing). Set False to save raw ``delta_code``
        and calibrate later.
    daughterboard_number : str | None, optional
        Daughterboard id (e.g. ``'D0'``) used to pick the packaged board LUT
        via 'tdc_calibration.load_board_lut'. The default is None.
    motherboard_number : str | None, optional
        Motherboard id (e.g. ``'M0'``) used with ``daughterboard_number`` to
        pick the packaged board LUT. The default is None.
    spad : int | str, optional
        Which SPAD's LUT to use: ``'average'`` (default) for the mean of the
        board's four SPAD LUTs, an int ``0..3`` for a specific SPAD, or a tag
        fragment like ``'SPAD0_S0'``. ORT cannot tell which micropixel fired,
        so ``'average'`` is the safe default; choose a SPAD only when the
        optics fix the micropixel. A missing specific SPAD falls back to the
        average (with a warning).
    nframes : int | None, optional
        Frames per file. When None (default) it is derived from file size.
    max_files : int | None, optional
        Process at most this many files. The default is None (all files).

    Returns
    -------
    str
        Path to the saved '.feather' file.

    Raises
    ------
    FileExistsError
        Raised when the '.feather' already exists and ``rewrite`` is False.
    """
    files = io.find_bin_files(path, tag)
    if max_files is not None:
        files = files[:max_files]

    name = os.path.basename(os.path.normpath(path))
    out_path = os.path.join(
        store.processed_dir(path), f"{name}_delta_t.feather"
    )
    if os.path.isfile(out_path) and not rewrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass rewrite=True to overwrite."
        )

    # Resolve the density-test LUT to apply (per-pixel, code -> ps). None
    # leaves the raw codes in place (saved as delta_code).
    time_lut = None
    if apply_TDC_calibration:
        if daughterboard_number is not None and motherboard_number is not None:
            time_lut = tc.load_board_lut(
                daughterboard_number, motherboard_number, spad
            )
        else:
            raise ValueError(
                "apply_TDC_calibration=True but no board was given. Pass "
                "daughterboard_number and motherboard_number, or set "
                "apply_TDC_calibration=False to save raw codes."
            )

    if mode not in ("all_pairs", "1v1"):
        raise ValueError(f"mode must be 'all_pairs' or '1v1', got {mode!r}")

    unit = "ps" if time_lut is not None else "code"

    # Build the stable, ordered list of pair labels up front (matching the
    # keys 'calculate_differences' produces) so every requested pair gets a
    # column, in a deterministic order, even if it never fires.
    group_a = cd.as_pixel_list(pixels[0])
    group_b = cd.as_pixel_list(pixels[1])
    if mode == "1v1":
        if len(group_a) != len(group_b):
            raise ValueError(
                "mode='1v1' needs equal-length groups "
                f"(got {len(group_a)} and {len(group_b)})."
            )
        pair_list = list(zip(group_a, group_b, strict=True))
    else:
        pair_list = [(a, b) for a in group_a for b in group_b]
    labels = [f"{a[0]},{a[1]}-{b[0]},{b[1]}" for a, b in pair_list]
    all_pixels = list(dict.fromkeys(group_a + group_b))

    print(
        f"\n> > > Collecting delta-t (mode='{mode}', tag='{tag}') from "
        f"{len(files)} file(s) into {len(labels)} pixel-pair column(s), "
        f"saving to {os.path.basename(out_path)} < < <\n"
    )

    pair_chunks: dict[str, list[np.ndarray]] = {lbl: [] for lbl in labels}
    for fp in tqdm(files, desc=tag or "delta_t"):
        nf = nframes if nframes is not None else io.frames_in_file(fp)
        ts, _ = unpack(fp, nf, compute_time_series=True)
        pixel_ts = _structure_pixel_timestamps(
            ts, all_pixels, valid_min, time_lut
        )
        deltas = cd.calculate_differences(
            pixel_ts, pixels, delta_window=delta_window, mode=mode
        )
        for label, arr in deltas.items():
            if arr.size:
                pair_chunks[label].append(arr)

    # One column per pair, NaN-padded to the longest pair's length.
    pair_arrays = {
        lbl: (np.concatenate(v) if v else np.empty(0, dtype=np.float64))
        for lbl, v in pair_chunks.items()
    }
    total = _write_delta_feather(out_path, pair_arrays, labels, unit)

    print(
        f"\n> > > {total} differences across {len(labels)} pixel-pair "
        f"column(s) saved to {out_path} (unit '{unit}') < < <"
    )
    return out_path


def _write_delta_feather(
    out_path: str,
    pair_arrays: dict[str, np.ndarray],
    labels: Sequence[str],
    unit: str,
) -> int:
    """Write a wide, NaN-padded one-column-per-pair delta-t '.feather'.

    Each pair's 1D array becomes a column (in ``labels`` order), padded with
    NaN to the longest pair's length. The ``delta_unit`` (``'code'`` / ``'ps'``)
    is stored in the schema metadata so the reader knows the column units.

    The Arrow table is assembled column-by-column (one ``pyarrow`` array per
    pair) rather than through a single consolidated ``pandas`` block; for the
    combined feathers, where the padded width (n_pairs x longest pair) can
    reach several GiB, that avoids a single huge contiguous allocation.

    Parameters
    ----------
    out_path : str
        Path to write the '.feather' to.
    pair_arrays : dict[str, np.ndarray]
        Mapping pair label -> 1D array of differences (already NaN-free).
    labels : Sequence[str]
        Column order; every label must be a key of ``pair_arrays``.
    unit : str
        ``'code'`` or ``'ps'``, recorded in the schema metadata.

    Returns
    -------
    int
        Total number of (non-padding) differences written.
    """
    maxlen = max((a.size for a in pair_arrays.values()), default=0)
    total = int(sum(a.size for a in pair_arrays.values()))

    columns: list[pa.Array] = []
    for lbl in labels:
        arr = np.asarray(pair_arrays[lbl], dtype=np.float64)
        if arr.size < maxlen:
            col = np.full(maxlen, np.nan, dtype=np.float64)
            col[: arr.size] = arr
        else:
            col = arr
        columns.append(pa.array(col))
        del col

    # Feather V2 == the Arrow IPC file format; the plain ipc writer keeps the
    # schema metadata and avoids the deprecated feather.write_feather wrapper.
    table = (
        pa.table(dict(zip(list(labels), columns, strict=True)))
        if labels
        else pa.table({})
    )
    md = dict(table.schema.metadata or {})
    md[b"delta_unit"] = unit.encode()
    md[b"n_pairs"] = str(len(labels)).encode()
    table = table.replace_schema_metadata(md)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pa.ipc.new_file(out_path, table.schema) as writer:
        writer.write_table(table)
    return total


def _read_delta_feather(feather_path: str) -> tuple[pd.DataFrame, str | None]:
    """Read a delta-t '.feather' into a DataFrame plus its unit.

    Parameters
    ----------
    feather_path : str
        Path to a '.feather' written by
        'calculate_and_save_timestamp_differences' or
        'combine_delta_t_feathers'.

    Returns
    -------
    df : pd.DataFrame
        The wide (one-column-per-pair) or legacy single-column table.
    unit : str | None
        ``'code'`` / ``'ps'`` from the schema metadata, or None if absent.
    """
    with pa.ipc.open_file(feather_path) as reader:
        table = reader.read_all()
    meta = table.schema.metadata or {}
    unit = meta.get(b"delta_unit", b"").decode() or None
    return table.to_pandas(), unit


def combine_delta_t_feathers(
    path: str,
    *,
    out_name: str | None = None,
    rewrite: bool = False,
    feathers: Sequence[str] | None = None,
    combined_dirname: str = "combined",
) -> str:
    """Pool the delta-t feathers under a folder into one combined feather.

    Finds every ``*_delta_t.feather`` beneath ``path`` (each written by
    'calculate_and_save_timestamp_differences' for one acquisition), pools
    them per pixel-pair column — concatenating the differences of matching
    pairs and taking the union of pairs across runs — and writes a single
    combined '.feather' into ``path/<combined_dirname>``. The combined table
    keeps the one-column-per-pair, NaN-padded layout and the ``delta_unit``
    metadata, so it plots exactly like a single-run feather (pass it to
    'collect_and_plot_timestamp_differences' via ``feather_path``).

    Parameters
    ----------
    path : str
        Parent folder holding the per-run subfolders (e.g.
        ``.../SPDC_external``). Searched recursively for
        ``*_delta_t.feather``; anything already inside the ``combined``
        subfolder is skipped.
    out_name : str | None, optional
        Base name for the combined file (``<out_name>_delta_t.feather``). The
        default is None (``<folder name>_combined``).
    rewrite : bool, optional
        Overwrite an existing combined feather. The default is False.
    feathers : Sequence[str] | None, optional
        Explicit list of feather paths to combine, bypassing the recursive
        search under ``path``. The default is None (auto-discover).
    combined_dirname : str, optional
        Name of the output subfolder created under ``path``. The default is
        ``'combined'``.

    Returns
    -------
    str
        Path to the saved combined '.feather'.

    Raises
    ------
    FileNotFoundError
        Raised when no ``*_delta_t.feather`` files are found.
    FileExistsError
        Raised when the combined feather exists and ``rewrite`` is False.
    ValueError
        Raised when the inputs mix ``delta_unit`` ('code' and 'ps').
    """
    combined_dir = os.path.join(store.processed_dir(path), combined_dirname)
    combined_abs = os.path.abspath(combined_dir)

    if feathers is None:
        found = sorted(
            glob.glob(os.path.join(path, "**", "*_delta_t.feather"),
                      recursive=True)
        )
        # Never fold a previous combined output back into itself.
        feathers = [
            f for f in found
            if not os.path.abspath(f).startswith(combined_abs + os.sep)
        ]
    if not feathers:
        raise FileNotFoundError(
            f"No *_delta_t.feather files found under:\n  {path}"
        )

    name = out_name or f"{os.path.basename(os.path.normpath(path))}_combined"
    out_path = os.path.join(combined_dir, f"{name}_delta_t.feather")
    if os.path.isfile(out_path) and not rewrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass rewrite=True to overwrite."
        )

    print(
        f"\n> > > Combining {len(feathers)} delta-t feather(s) under "
        f"{path} into {os.path.basename(out_path)} < < <\n"
    )

    pair_chunks: dict[str, list[np.ndarray]] = {}
    labels: list[str] = []  # first-seen order across all feathers
    units: set[str] = set()
    for fp in feathers:
        df, unit = _read_delta_feather(fp)
        if unit is not None:
            units.add(unit)
        for col in df.columns:
            vals = df[col].to_numpy(dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if col not in pair_chunks:
                pair_chunks[col] = []
                labels.append(col)
            pair_chunks[col].append(vals)

    if len(units) > 1:
        raise ValueError(
            f"Cannot combine feathers with mixed delta_unit {sorted(units)}."
        )
    unit = units.pop() if units else "code"

    pair_arrays = {
        c: (np.concatenate(v) if v else np.empty(0, dtype=np.float64))
        for c, v in pair_chunks.items()
    }
    total = _write_delta_feather(out_path, pair_arrays, labels, unit)

    print(
        f"\n> > > Combined {len(feathers)} feather(s): {total} differences "
        f"across {len(labels)} pixel-pair column(s) -> {out_path} "
        f"(unit '{unit}') < < <"
    )
    return out_path


def gaussian(
    x: np.ndarray, amp: float, mu: float, sigma: float, bkg: float
) -> np.ndarray:
    """Gaussian peak on a flat background.

    Parameters
    ----------
    x : np.ndarray
        Input positions.
    amp : float
        Peak amplitude above the background.
    mu : float
        Peak centre.
    sigma : float
        Standard deviation of the Gaussian.
    bkg : float
        Flat background level.

    Returns
    -------
    np.ndarray
        Model values ``amp * exp(-(x - mu)**2 / (2 sigma**2)) + bkg``.
    """
    return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) + bkg


def gaussian_plus_triangle(
    x: np.ndarray,
    amp: float,
    mu: float,
    sigma: float,
    bkg: float,
    half_base: float,
) -> np.ndarray:
    """Gaussian peak on a symmetric triangular background.

    The correct accidental model for ORT: uncorrelated free-running phases
    difference into a triangle (zero at ``+/- half_base``), with correlated
    pairs adding a Gaussian at ``mu ~ 0``. Why:
    ``docs/ort_triangle_background.md``.

    Parameters
    ----------
    x : np.ndarray
        Input positions (timestamp differences, same units as ``half_base``).
    amp : float
        Gaussian amplitude above the triangle.
    mu : float
        Gaussian centre.
    sigma : float
        Gaussian standard deviation.
    bkg : float
        Triangle height at its apex (``x = 0``).
    half_base : float
        Half-width of the triangle base; the background is zero for
        ``abs(x) >= half_base`` (physically ~one oscillator period, ~100 ns).

    Returns
    -------
    np.ndarray
        Model values ``amp * exp(-(x-mu)^2 / 2 sigma^2) + bkg * tri(x)`` where
        ``tri(x) = max(0, 1 - abs(x) / half_base)``.
    """
    tri = np.clip(1.0 - np.abs(x) / half_base, 0.0, None)
    return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) + bkg * tri


def fit_gaussian_on_triangle(
    bin_centers: np.ndarray,
    counts: np.ndarray,
    half_base: float,
    fit_half_base: bool = True,
) -> dict:
    """Fit a Gaussian peak on a triangular background (Kelpie ORT model).

    Use this rather than 'fit_gaussian_peak' for ORT delta-t over a wide
    window: the background is triangular, and a flat-background fit lets the
    peak swallow the triangle and reports an inflated sigma.

    Parameters
    ----------
    bin_centers : np.ndarray
        Histogram bin centres (should span roughly the full +/- half_base).
    counts : np.ndarray
        Histogram counts.
    half_base : float
        Initial triangle half-base (~one oscillator period; ~100 ns in ps, or
        ~1300 in code units). Refined when ``fit_half_base`` is True.
    fit_half_base : bool, optional
        Let ``half_base`` float in the fit. The default is True.

    Returns
    -------
    dict
        Fit results with keys ``amp``, ``mu``, ``sigma``, ``sigma_err``,
        ``bkg`` (triangle apex), ``half_base``, and ``ok``.
    """
    x = np.asarray(bin_centers, dtype=np.float64)
    y = np.asarray(counts, dtype=np.float64)

    result = {
        "amp": np.nan,
        "mu": 0.0,
        "sigma": np.nan,
        "sigma_err": np.nan,
        "bkg": np.nan,
        "half_base": half_base,
        "ok": False,
    }
    if len(x) < 6:
        return result

    # Triangle apex seed: median just off zero (skip the peak-contaminated
    # central few bins), scaled back up to x=0 along the triangle.
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    near = (np.abs(x) > 3 * dx) & (np.abs(x) < 0.15 * half_base)
    apex0 = (
        float(
            np.median(y[near]) / (1.0 - np.median(np.abs(x[near])) / half_base)
        )
        if np.any(near)
        else float(np.median(y))
    )
    tri_seed = apex0 * np.clip(1.0 - np.abs(x) / half_base, 0.0, None)
    excess = np.clip(y - tri_seed, 0.0, None)
    amp0 = float(np.max(excess)) if excess.size else float(np.max(y))
    mu0 = float(x[np.argmax(excess)]) if excess.size else 0.0
    # Seed sigma from the excess *near the peak only*: far from zero the
    # apex-seed mismatch leaves a broad positive residual that would otherwise
    # inflate the moment to ~half_base and push p0 outside the bounds.
    core = np.abs(x - mu0) < 0.05 * half_base
    w = np.where(core, excess, 0.0)
    sigma0 = (
        float(np.sqrt(np.sum(w * (x - mu0) ** 2) / np.sum(w)))
        if w.sum() > 0
        else 5 * dx
    )
    if not sigma0 > 0:
        sigma0 = 5 * dx

    lo = [0.0, x.min(), 0.0, 0.0, 0.5 * half_base]
    hi = [
        np.inf,
        x.max(),
        0.5 * half_base,
        np.inf,
        2.0 * half_base if fit_half_base else half_base * 1.0001,
    ]
    if not fit_half_base:
        lo[4] = half_base * 0.9999

    # Clamp every seed strictly inside its bounds (curve_fit rejects p0 == or
    # outside a bound).
    p0 = [amp0, mu0, sigma0, apex0, half_base]
    p0 = [
        min(
            max(v, lo_i + 1e-9 * (abs(lo_i) + 1)),
            hi_i - 1e-9 * (abs(hi_i) + 1),
        )
        for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)
    ]

    try:
        popt, pcov = curve_fit(
            gaussian_plus_triangle, x, y, p0=p0, bounds=(lo, hi), maxfev=20000
        )
    except (RuntimeError, ValueError):
        return result

    amp, mu, sigma, bkg, hb = popt
    if amp <= 0 or not np.isfinite(sigma) or sigma <= 0:
        return result
    sigma_err = (
        float(np.sqrt(pcov[2, 2])) if np.all(np.isfinite(pcov)) else np.nan
    )
    result.update(
        amp=float(amp),
        mu=float(mu),
        sigma=float(abs(sigma)),
        sigma_err=sigma_err,
        bkg=float(bkg),
        half_base=float(hb),
        ok=True,
    )
    return result


def fit_gaussian_peak(
    bin_centers: np.ndarray,
    counts: np.ndarray,
    fit_window: float | None = None,
) -> dict:
    """Fit a Gaussian-plus-background peak with ``scipy.curve_fit``.

    The flat background is seeded from the histogram tails and the peak
    centre/width from the background-subtracted moments, then all four
    parameters (``amp``, ``mu``, ``sigma``, ``bkg``) are refined by a
    least-squares fit of 'gaussian'.

    Parameters
    ----------
    bin_centers : np.ndarray
        Histogram bin centres.
    counts : np.ndarray
        Histogram counts.
    fit_window : float | None, optional
        If given, restrict the fit to ``abs(x - x_peak) <= fit_window``
        around the tallest bin. The default is None (use every bin).

    Returns
    -------
    dict
        Fit results with keys ``amp``, ``mu``, ``sigma``, ``sigma_err``,
        ``bkg``, and ``ok`` (False when the fit did not converge to a real
        peak).
    """
    x = np.asarray(bin_centers, dtype=np.float64)
    y = np.asarray(counts, dtype=np.float64)

    n = len(x)
    tail = np.r_[y[: n // 3], y[-n // 3 :]]
    bkg = float(np.median(tail)) if tail.size else 0.0

    x_peak = float(x[np.argmax(y)])
    if fit_window is not None:
        sel = np.abs(x - x_peak) <= fit_window
        x, y = x[sel], y[sel]

    result = {
        "amp": float(np.max(y) - bkg),
        "mu": x_peak,
        "sigma": np.nan,
        "sigma_err": np.nan,
        "bkg": bkg,
        "ok": False,
    }
    if len(x) < 4:
        return result

    w = np.clip(y - bkg, 0, None)
    if w.sum() <= 0:
        return result
    mu0 = float(np.sum(w * x) / np.sum(w))
    sigma0 = float(np.sqrt(np.sum(w * (x - mu0) ** 2) / np.sum(w)))
    if not sigma0 > 0:
        sigma0 = (x[1] - x[0]) if len(x) > 1 else 1.0
    p0 = [float(np.max(y) - bkg), mu0, sigma0, bkg]

    try:
        popt, pcov = curve_fit(
            gaussian,
            x,
            y,
            p0=p0,
            bounds=(
                [0.0, x.min(), 0.0, 0.0],
                [np.inf, x.max(), np.inf, np.inf],
            ),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return result

    amp, mu, sigma, bkg_fit = popt
    if amp <= 0 or not np.isfinite(sigma) or sigma <= 0:
        return result
    sigma_err = (
        float(np.sqrt(pcov[2, 2])) if np.all(np.isfinite(pcov)) else np.nan
    )
    result.update(
        amp=float(amp),
        mu=float(mu),
        sigma=float(abs(sigma)),
        sigma_err=sigma_err,
        bkg=float(bkg_fit),
        ok=True,
    )
    return result


def collect_and_plot_timestamp_differences(
    path: str | None = None,
    *,
    time_unit_ps: float = _TS_CODE_PS,
    bin_width_ps: float = _TS_CODE_PS,
    plot_window_ps: float = 3000.0,
    fit_window_ps: float | None = 1000.0,
    background: str = "flat",
    osc_period_ps: float = 100000.0,
    feather_path: str | None = None,
    pairs: Sequence[str] | None = None,
    label: str = "spdc",
    cmap_title: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Read the saved delta-t feather, fit the peak, and plot it.

    Pools the selected pair columns of a saved feather, converts to
    picoseconds (per the ``delta_unit`` metadata), histograms, fits the
    coincidence peak and saves the figure. Old single-column feathers
    (``delta_code`` / ``delta_ps``) are still read.

    Locate the feather either from ``path`` (its ``processed/``) or directly
    via ``feather_path``, which ignores ``path`` entirely - that is how a
    combined feather is plotted.

    Parameters
    ----------
    path : str | None, optional
        Path to the raw-data folder (its ``processed`` holds the
        feather, ``results`` receives the plot). May be None when
        ``feather_path`` is given. Required otherwise.
    time_unit_ps : float, optional
        Picoseconds per raw TDC code, applied to a ``delta_code`` feather.
        The default is ~77 ps.
    bin_width_ps : float, optional
        Histogram bin width in ps. The default is ~77 ps (one code).
    plot_window_ps : float, optional
        Histogram/plot half-range in ps around zero. The default is 3000.0.
    fit_window_ps : float | None, optional
        Fit half-range in ps around the tallest bin. Only used for
        ``background='flat'``. The default is 1000.0.
    background : str, optional
        Accidental-background model. ``'flat'`` (default) fits
        ``Gaussian + constant`` over ``fit_window_ps`` — the legacy daplis
        model. ``'triangle'`` fits ``Gaussian + triangular background`` via
        'fit_gaussian_on_triangle' over the whole histogram, which is the
        physically correct model for the Kelpie ORT free-running phase and
        does not inflate sigma. For ``'triangle'`` set ``plot_window_ps`` wide
        (e.g. the full ``osc_period_ps``) so the triangle is well constrained.
    osc_period_ps : float, optional
        Oscillator period in ps; equals the triangle half-base (the
        accidental background is a triangle peaking at 0 and reaching zero at
        ``+/- osc_period_ps``). The default is 100000.0 ps (~100 ns). Only
        used when ``background='triangle'``; the fit refines it.
    feather_path : str | None, optional
        Explicit feather to read and plot directly; when given, ``path`` is
        ignored and the raw-data folder is not needed (the plot is saved
        beside the feather). The default is None (derive from ``path``).
    pairs : Sequence[str] | None, optional
        Restrict pooling to these pixel-pair column names (e.g.
        ``["16,16-20,20"]``). The default is None (pool every pair column).
        Ignored for old single-column feathers.
    label : str, optional
        Short label for the title and output file (``'spdc'`` / ``'hbt'``).
        The default is ``'spdc'``.
    cmap_title : str | None, optional
        Optional override for the plot title. The default is None.

    Returns
    -------
    deltas_ps : np.ndarray
        The timestamp differences in picoseconds.
    fit : dict
        The fit results plus ``fwhm`` (2.355 sigma), ``jitter_per_detector``
        (sigma / sqrt(2)), ``car`` (peak-to-background ratio) and ``n``.

    Raises
    ------
    FileNotFoundError
        Raised when the delta-t '.feather' cannot be found.
    """
    if feather_path is not None:
        # Feather given directly: ignore path, derive the name and output
        # folder from the feather itself so a standalone or combined feather
        # can be plotted without the original raw-data folder.
        feather_path = os.path.abspath(feather_path)
        base = os.path.basename(feather_path)
        suffix = "_delta_t.feather"
        name = base[: -len(suffix)] if base.endswith(suffix) else (
            os.path.splitext(base)[0]
        )
        # <root>/processed/<name>.feather -> figures belong under <root>.
        results_dir = store.results_dir(
            os.path.dirname(os.path.dirname(feather_path)), _KIND, create=False
        )
    else:
        if path is None:
            raise ValueError(
                "Provide either path (to locate the feather) or feather_path."
            )
        name = os.path.basename(os.path.normpath(path))
        feather_path = os.path.join(
            store.processed_dir(path, create=False),
            f"{name}_delta_t.feather",
        )
        results_dir = store.results_dir(path, _KIND, create=False)

    if not os.path.isfile(feather_path):
        raise FileNotFoundError(
            f"No delta-t feather at {feather_path}. Run "
            "calculate_and_save_timestamp_differences first."
        )

    df, unit = _read_delta_feather(feather_path)

    if "delta_ps" in df.columns:  # legacy single-column layout
        pooled = df["delta_ps"].to_numpy()
        unit = unit or "ps"
    elif "delta_code" in df.columns:  # legacy single-column layout
        pooled = df["delta_code"].to_numpy()
        unit = unit or "code"
    else:  # per-pair columns, NaN-padded
        cols = (
            list(df.columns)
            if pairs is None
            else [p for p in pairs if p in df.columns]
        )
        if pairs is not None:
            missing = [p for p in pairs if p not in df.columns]
            if missing:
                print(f"  WARNING: pair column(s) not in feather: {missing}")
        vals = df[cols].to_numpy().ravel() if cols else np.empty(0)
        pooled = vals[~np.isnan(vals)]
        unit = unit or "code"

    deltas_ps = pooled if unit == "ps" else pooled * time_unit_ps

    d = deltas_ps[np.abs(deltas_ps) <= plot_window_ps]
    edges = np.arange(
        -plot_window_ps - bin_width_ps / 2,
        plot_window_ps + bin_width_ps,
        bin_width_ps,
    )
    counts, edges = np.histogram(d, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    # Comb guard: warn if the data granularity is coarser than the bin.
    nz = np.abs(deltas_ps[deltas_ps != 0])
    if nz.size:
        gran = float(np.min(nz))
        if gran > bin_width_ps * 1.5:
            print(
                f"  WARNING: delta granularity is ~{gran:.0f} ps but "
                f"bin_width={bin_width_ps:.0f} ps -> the histogram will be a "
                "comb. Bin coarser, or ensure the fine timing is populated "
                "in the acquisition."
            )

    if background not in ("flat", "triangle"):
        raise ValueError(
            f"background must be 'flat' or 'triangle', got {background!r}"
        )
    if background == "triangle":
        fit = fit_gaussian_on_triangle(
            centers, counts, half_base=osc_period_ps
        )
    else:
        fit = fit_gaussian_peak(centers, counts, fit_window=fit_window_ps)
    sigma = fit["sigma"]
    fit["fwhm"] = 2.355 * sigma if np.isfinite(sigma) else np.nan
    fit["jitter_per_detector"] = (
        sigma / np.sqrt(2) if np.isfinite(sigma) else np.nan
    )
    # CAR = peak-to-accidental ratio at delta=0 (apex for the triangle).
    fit["car"] = (
        (fit["amp"] + fit["bkg"]) / fit["bkg"] if fit["bkg"] else np.inf
    )
    fit["n"] = int(d.size)

    fig, ax = plt.subplots()
    ax.bar(
        centers,
        counts,
        width=bin_width_ps,
        # alpha=0.6,
        label=f"data ({d.size} pairs)",
    )
    if fit["ok"]:
        xs = np.linspace(-plot_window_ps, plot_window_ps, 4000)
        if background == "triangle":
            model = gaussian_plus_triangle(
                xs, fit["amp"], fit["mu"], sigma, fit["bkg"], fit["half_base"]
            )
        else:
            model = gaussian(xs, fit["amp"], fit["mu"], sigma, fit["bkg"])
        ax.plot(
            xs,
            model,
            "r-",
            linewidth=2,
            label=(
                f"sigma = {sigma:.0f} ps\n"
                f"FWHM = {fit['fwhm']:.0f} ps\n"
                f"per-detector = {fit['jitter_per_detector']:.0f} ps"
            ),
        )
    ax.set_xlabel("Timestamp difference  (ps)")
    ax.set_ylabel("Coincidences")
    ax.set_title(cmap_title or f"{label.upper()} coincidence peak  ({name})")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend()
    # fig.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    out_png = os.path.join(results_dir, f"{name}_{label}_delta_t.png")
    fig.savefig(out_png)
    # plt.close(fig)
    print(
        f"\n> > > Plot saved as {os.path.basename(out_png)} in {results_dir}"
    )
    if fit["ok"]:
        print(
            f"  peak: mu {fit['mu']:.0f} ps  sigma {sigma:.0f} ps  "
            f"FWHM {fit['fwhm']:.0f} ps  per-detector "
            f"{fit['jitter_per_detector']:.0f} ps  peak/bkg {fit['car']:.2f}"
        )
    else:
        print(
            "  no Gaussian peak fit — check blob selection / that the run "
            "has proper (stopped, fine) timing."
        )

    return deltas_ps, fit



