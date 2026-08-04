"""Coincidence / timing-jitter (delta-t) pipeline for the ORT program.

Drives 'unpack' and 'calc_diff' over a folder, pools the per-pixel-pair
timestamp differences into a '.feather' table, then histograms, fits and plots
the coincidence peak.

Both halves stream. Stage 1 flushes size-capped part files as it goes, and the
read side histograms the feather one Arrow record batch at a time rather than
loading it - a 10 000-file run's combined feather reaches ~135 GB, which no
amount of RAM makes loadable. Since a histogram over fixed edges is additive,
the streamed counts are bin-for-bin what loading the whole table would have
given; they are saved beside the feather so re-fitting costs a small load.

NOTE on fitting: the ORT accidental background is TRIANGULAR, not flat, because
the free-running oscillator has no cycle counter. Use
'fit_gaussian_on_triangle' over a wide window; 'fit_gaussian_peak' (flat
background) is valid only on a narrow window around the peak, and a flat fit
over a wide one inflates the fitted sigma. Or do not model it at all:
``subtract_background=True`` measures it from frame-shifted pairs and subtracts
it, leaving the peak on a zero baseline
('dapkel.functions.background_subtraction', ``docs/guide/background_subtraction.md``).

Check a fresh acquisition with 'dapkel.functions.data_quality' first.
See ``docs/guide/coincidences.md`` and ``docs/ort_triangle_background.md``.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import os
import time
from collections.abc import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from scipy.optimize import curve_fit
from tqdm import tqdm

from dapkel.core import io, store

# 'core.pairs' is aliased because half the functions here take a 'pairs'
# argument (pixel-pair *labels* to pool), which would shadow the module.
from dapkel.core import pairs as pixel_pairs
from dapkel.functions import background_subtraction as bs
from dapkel.functions import calc_diff as cd
from dapkel.functions import tdc_calibration as tc
from dapkel.functions.calc_diff import Pixel
from dapkel.functions.unpack import unpack

__all__ = [
    # stage 1 - difference timestamps and persist to feather
    "calculate_and_save_timestamp_differences",
    "combine_delta_t_parts",
    "combine_delta_t_feathers",
    # stage 1b - stream the feather down to a histogram
    "compute_and_save_delta_histogram",
    # stage 1' - histogram straight off the '.bin', no feather at all
    "calculate_and_save_delta_counts",
    "load_delta_counts",
    "rebin_delta_counts",
    # fit models and fitting
    "gaussian",
    "gaussian_plus_triangle",
    "two_gaussians_on_triangle",
    "fit_gaussian_peak",
    "fit_gaussian_on_triangle",
    "fit_two_gaussians_on_triangle",
    # drivers - load saved feathers, figures on disk
    "collect_and_plot_timestamp_differences",
    "collect_and_plot_delta_counts",
]

#: Analysis name: the results sub-folder for the figures.
_KIND = "coincidences"

# Average raw-code -> picosecond conversion: ~100 ns oscillator period over
# ~1300 codes. Replace with a per-code LUT (density test) via 'time_lut'.
_TS_CODE_PS = 77.0

# Size cap for one part file, in MB. Chosen against what else is resident
# during the loop rather than in the abstract: 'unpack' already holds a
# (32, 32, nframes) float64 array, which is ~82 MB at 10 000 frames, so the
# part buffer stops being the binding constraint well below that and there is
# nothing to gain from paying the per-part write and open cost more often. A
# smaller cap (10 MB) is perfectly safe, just ~6x more part files - which is
# why this is a module constant and not a parameter: sharding is a memory
# strategy, it cannot change the numbers, and no call site needs to tune it.
_PART_MB = 64.0

# Suffix of the folder holding part files, as
# '<dataset>_delta_t_parts/<dataset>_delta_t_<run>_<i>.feather'. Note the parts
# do NOT match the '*_delta_t.feather' glob 'combine_delta_t_feathers' uses, so
# they can never be pooled twice.
_PARTS_SUFFIX = "_parts"

# Suffix of the per-run manifest written alongside the parts. Part numbering
# restarts at 0 on every run, so the file names alone cannot say which run a
# part belongs to: a re-run overwrites '<stem>_0..N' and leaves a longer dead
# run's '<stem>_N+1..M' in place, where a folder-wide glob then silently pools
# both. The manifest is the record of which parts one run actually wrote, and
# it is rewritten on every flush so a run that dies is still described by it.
_MANIFEST_SUFFIX = "_manifest.json"

# Column-name marker for a frame-shifted (lag > 0) pair in a feather:
# '6,10-21,22' is lag 0, '6,10-21,22@lag3' is the same pair with group B
# shifted three frames. Lag 0 keeping the bare label is what makes a lagged
# feather a strict *superset* - every existing reader still finds exactly the
# columns it expects, and only a reader that knows about lags asks for more.
# '@' cannot occur in a pair label (which is digits, commas and one hyphen), so
# the two can never be confused.
_LAG_COL_SEP = "@lag"

# Schema-metadata key holding the lag bookkeeping a lagged feather needs on the
# read side: the lag set and the frame count per file. Note what is *not* in
# there - the number of files. 'subtract_background' only ever uses the ratio
# frames[0] / frames[k] = nframes / (nframes - k), in which the file count
# cancels, so a part folder from a run that died is as usable as a complete one.
_LAG_META_KEY = b"lag_meta"

# File name suffix of the histogram artifact 'compute_and_save_delta_histogram'
# writes into 'processed/'. The counts are what every downstream consumer
# actually needs, and they are ~5 MB against a ~135 GB feather, so the
# expensive streaming pass is paid once per binning rather than per plot.
_HIST_SUFFIX = "_delta_hist.npy"

# File name suffix of the counts artifact 'calculate_and_save_delta_counts'
# writes into 'processed/'. This is the *alternative* stage 1: instead of
# writing every difference and histogramming it later, the differences are
# binned onto a fixed native grid inside the per-file loop and only the counts
# are kept. Nothing downstream needs more than counts, so the artifact is a
# ~100 MB array where the feather is 3-135 GB, and it is a re-encoding rather
# than a summary for raw-code data (see '_delta_grid').
_COUNTS_SUFFIX = "_delta_counts.npy"

# Sub-divisions of one TDC code used as the native grid when the differences
# are already calibrated (unit 'ps'). A per-pixel LUT smears the code grid, so
# there is no exact integer lattice to land on; 1/10 of a code (~7.7 ps) is
# ~5x finer than the detector's own jitter, which makes the displacement it
# introduces (<= 3.85 ps) irrelevant to any fitted sigma. Raw codes need no
# subdivision - they *are* integers - so this applies to 'ps' data only.
_GRID_SUBDIV = 10

# Default half-range of the counts grid, in ps. A difference cannot exceed one
# oscillator period, so a grid this wide holds the entire physical support and
# the window only ever gets narrowed afterwards, in memory. Widen it for a
# longer period; anything landing outside is counted in 'n_outside' rather
# than silently dropped.
_SUPPORT_PS = 200000.0

# Files between checkpoint writes during the counts pass. The whole grid is
# rewritten each time, which costs one ~100 MB write - cheap against unpacking
# 50 files, and it means a run that dies has a complete, valid artifact for
# every file up to its last checkpoint. This replaces the part folder: there
# is only ever one file, so two runs cannot interleave.
_FLUSH_EVERY = 50


def _structure_pixel_timestamps(
    time_series: np.ndarray,
    pixels: Sequence[Pixel],
    time_lut: np.ndarray | None,
) -> dict[Pixel, np.ndarray]:
    """Extract per-pixel, per-frame timestamps for a set of pixels.

    A frame holds a valid timestamp when ``time_series > 0``: empty slots
    decode to the ``unpack`` sentinel (<= 0), so the threshold follows from
    the decoding rather than being a tunable. NOTE: do not use
    ``photon_count > 0`` — in timestamp mode those bits are part of the
    coarse code.

    Parameters
    ----------
    time_series : np.ndarray
        The (32, 32, nframes) TDC codes from 'unpack'.
    pixels : Sequence[tuple[int, int]]
        The ``(row, col)`` pixels to extract.
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
        valid = code > 0
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


def _lag_column(label: str, lag: int) -> str:
    """Feather column name for one pixel pair at one frame lag."""
    return label if lag == 0 else f"{label}{_LAG_COL_SEP}{lag}"


def _is_lag_column(name: str) -> bool:
    """Whether a column holds frame-shifted (background) differences."""
    return _LAG_COL_SEP in name


def _lag_meta(lags: Sequence[int], nframes: int) -> dict[bytes, bytes]:
    """Build the schema metadata recording the lag set and the frame count."""
    return {
        _LAG_META_KEY: json.dumps(
            {"lags": [int(k) for k in lags], "nframes": int(nframes)}
        ).encode()
    }


def _read_lag_meta(sources: Sequence[str]) -> dict | None:
    """Read the lag bookkeeping out of the sources' schema metadata.

    Returns None when no source carries any - i.e. an ordinary feather.

    Raises
    ------
    ValueError
        Raised when the sources disagree, which would silently mix lag sets.
    """
    found: set[str] = set()
    for src in sources:
        with _open_delta_feather(src) as reader:
            blob = (reader.schema.metadata or {}).get(_LAG_META_KEY)
        if blob:
            found.add(blob.decode())
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            "These feathers were written with different lag sets and cannot "
            f"be pooled: {sorted(found)}"
        )
    return json.loads(found.pop())


def calculate_and_save_timestamp_differences(
    path: str,
    pixels: Sequence[Sequence[Pixel]],
    rewrite: bool = False,
    *,
    nframes: int,
    tag: str = "ORT",
    mode: str = "all_pairs",
    delta_window: float | None = None,
    apply_TDC_calibration: bool = True,
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    spad: int | str = "average",
    max_files: int | None = None,
    subtract_background: bool = False,
    background_lags: int | Sequence[int] = bs.BACKGROUND_LAGS,
) -> str:
    """Unpack ORT data, compute blob-vs-blob delta-t, and save to feather.

    Writes ``path/processed/<name>_delta_t.feather`` with **one column per
    pixel pair** (named ``"ra,ca-rb,cb"``), padded to a common length; every
    requested pair gets a column even if it saw no coincidences.

    The differences are **streamed to disk as they are found**, not held until
    the end: the running buffer is flushed to
    ``processed/<name>_delta_t_parts/<name>_delta_t_<run>_<i>.feather``
    whenever it would exceed '_PART_MB', and the parts are concatenated
    into the final feather one record batch at a time. Memory therefore stays
    flat in the number of '.bin' files, and a run that dies at file 9 000 of
    10 000 leaves every completed part on disk - recover them with
    'combine_delta_t_parts'.

    The parts are **kept**. They are complete feathers in their own right and
    every read-side entry point takes a part folder as readily as a single
    file ('compute_and_save_delta_histogram' streams either one a record batch
    at a time), so they are the cheaper of the two things on disk to work
    from as well as the crash insurance. Delete the folder by hand once the
    combined feather has been checked.

    Each run tags its parts with its own ``<run>`` id and records them in a
    manifest, and a run refuses to start while another run's parts are still
    in the folder. Part numbering restarts at 0 every run, so without both of
    those a re-run would overwrite a dead run's first parts, leave its later
    ones, and the combine step would pool the two - counting some differences
    twice and inflating the feather.

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
        Overwrite an existing '.feather' (and part folder) for this data set.
        A guard against accidental overwrites: the default is False, which
        raises instead. When True and something really is about to be
        destroyed, 'store.confirm_rewrite' names it and counts down five
        seconds first, so a stale ``rewrite=True`` can still be caught with
        Ctrl-C.
    nframes : int
        Number of frames stored in each '.bin' file.
    tag : str, optional
        Filename fragment selecting the files. The default is ``'ORT'``.
    mode : str, optional
        ``'all_pairs'`` (default) or ``'1v1'``; see
        'calc_diff.calculate_differences'.
    delta_window : float | None, optional
        Keep only differences with ``abs(delta) <= delta_window`` while
        accumulating (bounds memory). In code units, or ps if ``time_lut``
        is given. The default is None (keep all).
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
    max_files : int | None, optional
        Process at most this many files. The default is None (all files).
    subtract_background : bool, optional
        Also difference the *frame-shifted* pairs, so the accidental
        background can be measured rather than fitted. The default is False,
        which writes exactly the feather it always has. When True each pair
        gains one column per lag, named ``"<pair>@lag<k>"``, while lag 0 keeps
        the bare pair name - so the feather is a strict superset and every
        existing reader still sees the same columns. **It also multiplies the
        feather by roughly ``1 + len(background_lags)``**: prefer the counts
        path ('calculate_and_save_delta_counts'), where the same lags cost one
        small grid plane each. See
        'dapkel.functions.background_subtraction' and
        ``docs/guide/background_subtraction.md``.
    background_lags : int | Sequence[int], optional
        Frame offsets used for the accidental estimate when
        ``subtract_background`` is True. The default is ``(1, ..., 8)`` - which
        is 9x the feather, so consider fewer here. A bare ``int`` is taken as a
        single lag. Lag 0 is always included and must not appear here. Ignored
        when ``subtract_background`` is False.

    Returns
    -------
    str
        Path to the saved '.feather' file.

    Raises
    ------
    FileExistsError
        Raised when the '.feather' already exists and ``rewrite`` is False, or
        when an earlier run's parts are still in the part folder.
    ValueError
        Raised on a calibration request with no board, or a 0 / negative /
        repeated entry in ``background_lags``.
    """
    lags = _background_lags(background_lags) if subtract_background else [0]

    files = io.find_bin_files(path, tag)
    if max_files is not None:
        files = files[:max_files]

    name = os.path.basename(os.path.normpath(path))
    stem = f"{name}_delta_t"
    processed = store.processed_dir(path)
    out_path = os.path.join(processed, f"{stem}.feather")
    parts_dir = os.path.join(processed, f"{stem}{_PARTS_SUFFIX}")

    if os.path.isfile(out_path) and not rewrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass rewrite=True to overwrite."
        )
    if rewrite:
        # Name the part folder too: a previous run's parts are also data, and
        # stale ones left in place would be folded into this run's output.
        store.confirm_rewrite([out_path, parts_dir])
        _clear_parts(parts_dir)
    else:
        # A run that died leaves its parts behind, and part numbering restarts
        # at 0, so this run would overwrite the dead one's first N parts and
        # leave the rest - a folder holding two runs at once. Stop instead of
        # producing a feather that counts some differences twice. Nothing is
        # deleted here; the parts are the crash insurance and they are the
        # caller's to keep, combine or discard.
        stale = _all_part_files(parts_dir)
        if stale:
            raise FileExistsError(
                f"{len(stale)} part file(s) from an earlier run are still in\n"
                f"  {parts_dir}\n"
                "Starting now would interleave both runs' parts. Either\n"
                "  - recover them:  combine_delta_t_parts(parts_dir)\n"
                "  - or discard them by re-running with rewrite=True."
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

    unit = "ps" if time_lut is not None else "code"

    # Build the stable, ordered list of pair labels up front (matching the
    # keys 'calculate_differences' produces) so every requested pair gets a
    # column, in a deterministic order, even if it never fires. Shared with
    # the counts path through 'core.pairs' - the two stage-1 paths are only
    # comparable if they agree on which pairs exist and in what order.
    labels, all_pixels = pixel_pairs.pair_labels(pixels, mode)
    # Lag 0 first and under its bare name, then one block of columns per
    # shifted lag: a reader that knows nothing about lags finds precisely the
    # feather it has always found, at the front.
    columns = [_lag_column(lbl, k) for k in lags for lbl in labels]

    print(
        f"\n> > > Collecting delta-t (mode='{mode}', tag='{tag}') from "
        f"{len(files)} file(s) into {len(columns)} pixel-pair column(s)"
        + (f" at lags {lags}" if subtract_background else "")
        + f", streaming {_PART_MB:.0f} MB parts to "
        f"{os.path.basename(parts_dir)}{os.sep} < < <\n"
    )

    writer = _PartWriter(
        parts_dir,
        stem,
        columns,
        unit,
        _PART_MB,
        extra_meta=_lag_meta(lags, nframes) if subtract_background else None,
    )
    for fp in tqdm(files, desc=tag or "delta_t"):
        ts, _ = unpack(fp, nframes, compute_time_series=True)
        pixel_ts = _structure_pixel_timestamps(ts, all_pixels, time_lut)
        if subtract_background:
            deltas = {}
            for k in lags:
                lagged = bs.compute_lagged_differences(
                    pixel_ts, pixels, delta_window, mode, frame_lag=k
                )
                deltas.update(
                    {_lag_column(lbl, k): arr for lbl, arr in lagged.items()}
                )
        else:
            # Left as it was: lag 0 is tested to be identical to this, but the
            # plain path has no reason to route through the lag machinery.
            deltas = cd.calculate_differences(
                pixel_ts, pixels, delta_window=delta_window, mode=mode
            )
        writer.add(deltas)
    parts = writer.close()

    if parts:
        print(
            f"\n> > > Combining {len(parts)} part file(s) into "
            f"{os.path.basename(out_path)} < < <"
        )
        total, _, _ = _combine_delta_feathers(parts, out_path)
    else:
        # Not one coincidence anywhere. Still write the empty table so the
        # requested pair columns and the unit are on record.
        total = _write_delta_feather(
            out_path,
            {lbl: np.empty(0, dtype=np.float64) for lbl in columns},
            columns,
            unit,
            _lag_meta(lags, nframes) if subtract_background else None,
        )

    print(
        f"\n> > > {total} differences across {len(columns)} pixel-pair "
        f"column(s) saved to {out_path} (unit '{unit}') < < <"
    )
    return out_path


class _PartWriter:
    """Buffer per-pair deltas and flush them to size-capped part feathers.

    Stage 1 used to keep every difference from every '.bin' file in RAM and
    write a single feather at the end. Over ~10 000 files that fails twice:
    the accumulated lists exhaust memory, and the final write asks pyarrow
    for one multi-GB contiguous allocation on top of what is already held.

    Buffering to a fixed cap instead makes the loop's memory ceiling one part
    plus one unpacked file, independent of how many files are processed, and
    turns the single all-or-nothing write into many small ones - so a run
    that dies partway through has still written everything up to its last
    flush.
    """

    def __init__(
        self,
        parts_dir: str,
        stem: str,
        labels: Sequence[str],
        unit: str,
        part_size_mb: float,
        run_id: str | None = None,
        extra_meta: dict[bytes, bytes] | None = None,
    ) -> None:
        self.parts_dir = parts_dir
        self.stem = stem
        self.labels = list(labels)
        self.unit = unit
        self.extra_meta = extra_meta
        self.part_bytes = max(int(part_size_mb * 1024 * 1024), 1)
        self.run_id = (
            run_id if run_id is not None else _new_run_id(parts_dir)
        )
        self.parts: list[str] = []
        self.total = 0
        self._buf: dict[str, list[np.ndarray]] = {
            lbl: [] for lbl in self.labels
        }
        self._lens: dict[str, int] = dict.fromkeys(self.labels, 0)

    @property
    def _buffered_bytes(self) -> int:
        """Bytes the buffer would occupy once written.

        Parts are written wide and padded, so what lands on disk - and what
        has to be built in memory to get there - is ``n_pairs`` x the longest
        column, not the sum of the columns. Budgeting on the padded size is
        what keeps one busy pair from inflating the part.
        """
        return len(self.labels) * max(self._lens.values(), default=0) * 8

    def add(self, deltas: dict[str, np.ndarray]) -> None:
        """Buffer one file's differences, flushing if the cap is reached."""
        for label, arr in deltas.items():
            if arr.size and label in self._buf:
                self._buf[label].append(arr)
                self._lens[label] += int(arr.size)
                self.total += int(arr.size)
        if self._buffered_bytes >= self.part_bytes:
            self.flush()

    @property
    def manifest_path(self) -> str:
        """Path of this run's manifest inside the part folder."""
        return os.path.join(
            self.parts_dir, f"{self.stem}_{self.run_id}{_MANIFEST_SUFFIX}"
        )

    def _write_manifest(self, complete: bool) -> None:
        """Record this run's parts, so a folder-wide glob cannot mix runs.

        Rewritten after every flush rather than once at the end: a run that
        dies still leaves a manifest naming every part it completed, which is
        what 'combine_delta_t_parts' needs to recover it without sweeping up
        a different run's leftovers.
        """
        os.makedirs(self.parts_dir, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "stem": self.stem,
            "unit": self.unit,
            "n_labels": len(self.labels),
            "parts": [os.path.basename(p) for p in self.parts],
            "total": self.total,
            "complete": complete,
        }
        tmp = f"{self.manifest_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.manifest_path)

    def flush(self) -> str | None:
        """Write the buffer as the next part; a no-op when it is empty."""
        if not any(self._lens.values()):
            return None
        part = os.path.join(
            self.parts_dir,
            f"{self.stem}_{self.run_id}_{len(self.parts)}.feather",
        )
        pair_arrays = {
            lbl: (
                np.concatenate(chunks)
                if chunks
                else np.empty(0, dtype=np.float64)
            )
            for lbl, chunks in self._buf.items()
        }
        _write_delta_feather(
            part, pair_arrays, self.labels, self.unit, self.extra_meta
        )
        self.parts.append(part)
        self._buf = {lbl: [] for lbl in self.labels}
        self._lens = dict.fromkeys(self.labels, 0)
        self._write_manifest(complete=False)
        return part

    def close(self) -> list[str]:
        """Flush whatever is left and return every part written."""
        self.flush()
        if self.parts:
            self._write_manifest(complete=True)
        return self.parts


def _new_run_id(parts_dir: str) -> str:
    """Return a run tag that no other run in ``parts_dir`` already uses.

    Wall-clock to the second plus the pid reads well in a folder listing and
    sorts chronologically, but it is not unique on its own: two runs started
    from the same script within the same second share both, and the second
    would then overwrite the first's parts - the very failure the run tag
    exists to prevent. So the tag is checked against the folder and given a
    ``-2``, ``-3``, ... suffix until nothing on disk answers to it.
    """
    base = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid():d}"
    taken = set(_part_runs(parts_dir))
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def _part_index(part_path: str) -> int:
    """Sort key for '<stem>_<run>_<i>.feather': the trailing int, else -1."""
    stem = os.path.basename(part_path).removesuffix(".feather")
    _, _, tail = stem.rpartition("_")
    return int(tail) if tail.isdigit() else -1


def _all_part_files(parts_dir: str) -> list[str]:
    """Every part '.feather' in the folder, whichever run wrote it."""
    if not os.path.isdir(parts_dir):
        return []
    found = glob.glob(os.path.join(parts_dir, "*_delta_t_*.feather"))
    return sorted(found, key=_part_index)


def _part_runs(parts_dir: str) -> dict[str, list[str]]:
    """Group a part folder's files by the run that wrote them.

    Manifests are authoritative: each names the parts of exactly one run.
    Part files no manifest claims are grouped under ``'legacy'`` - that is
    what a folder written before manifests existed looks like, and it is also
    where a folder holding two pre-manifest runs' parts ends up, which is
    precisely the case that used to be pooled silently.

    Returns
    -------
    dict[str, list[str]]
        Run id -> that run's parts, in write order. Empty when the folder
        holds no parts.
    """
    if not os.path.isdir(parts_dir):
        return {}

    runs: dict[str, list[str]] = {}
    claimed: set[str] = set()
    manifests = sorted(
        glob.glob(os.path.join(parts_dir, f"*{_MANIFEST_SUFFIX}"))
    )
    for mpath in manifests:
        try:
            with open(mpath, encoding="utf-8") as fh:
                payload = json.load(fh)
            run_id = str(payload["run_id"])
            names = [str(n) for n in payload["parts"]]
        except (OSError, ValueError, KeyError, TypeError):
            continue  # an unreadable manifest must not hide its parts
        present = [
            os.path.join(parts_dir, n)
            for n in names
            if os.path.isfile(os.path.join(parts_dir, n))
        ]
        claimed.update(present)
        if present:
            runs[run_id] = sorted(present, key=_part_index)

    orphans = [p for p in _all_part_files(parts_dir) if p not in claimed]
    if orphans:
        _warn_if_untagged_parts_are_mixed(orphans)
        runs["legacy"] = orphans
    return runs


def _warn_if_untagged_parts_are_mixed(parts: Sequence[str]) -> None:
    """Warn when untagged parts look like more than one run.

    Parts written before run tags existed carry no record of who wrote them,
    so they cannot be separated - but they can be *detected*: one run writes
    its parts in index order, so a part whose index is higher than its
    neighbour's while its mtime is older belongs to an earlier, longer run
    that a re-run overwrote the front of. Pooling those double-counts part of
    the data set, which is worth saying out loud even though it cannot be
    fixed after the fact.
    """
    stamped = []
    for p in parts:
        try:
            stamped.append((_part_index(p), os.stat(p).st_mtime))
        except OSError:
            return
    backwards = sum(
        1
        for (_, t_prev), (_, t_next) in zip(stamped, stamped[1:], strict=False)
        if t_next < t_prev
    )
    if backwards:
        print(
            f"  WARNING: {backwards} of {len(parts)} untagged part file(s) are "
            "older than the part before them - this folder looks like two "
            "runs, and pooling it would count some differences twice. Check "
            "the mtimes before trusting a feather built from it."
        )


def _find_parts(parts_dir: str, run: str | None = None) -> list[str]:
    """Return one run's part '.feather' files, in write order.

    Raises
    ------
    ValueError
        Raised when the folder holds more than one run and ``run`` does not
        say which to take. Pooling them is what produced feathers with
        several times the differences the data actually contains, so it is
        refused rather than guessed at.
    """
    runs = _part_runs(parts_dir)
    if not runs:
        return []
    if run is not None:
        if run not in runs:
            raise ValueError(
                f"No parts for run {run!r} in {parts_dir}. "
                f"Available: {sorted(runs)}"
            )
        return runs[run]
    if len(runs) == 1:
        return next(iter(runs.values()))

    summary = "\n".join(
        f"    run={rid!r}: {len(parts)} part(s)"
        for rid, parts in sorted(runs.items())
    )
    raise ValueError(
        f"{parts_dir} holds parts from {len(runs)} different runs:\n"
        f"{summary}\n"
        "  Pooling them would count some differences more than once. Pass "
        "run='<id>' to pick one, or move the others aside."
    )


def _clear_parts(parts_dir: str) -> None:
    """Remove every part file, leaving anything else in the folder alone.

    Only ever touches ``*_delta_t_*.feather`` and the manifests beside them,
    and only from a ``rewrite=True`` that the countdown has already announced.
    A successful run keeps its parts, so nothing else deletes them.
    """
    targets = _all_part_files(parts_dir)
    manifests = glob.glob(os.path.join(parts_dir, f"*{_MANIFEST_SUFFIX}"))
    for part in targets:
        try:
            os.remove(part)
        except OSError:
            pass
    for mpath in manifests:
        try:
            os.remove(mpath)
        except OSError:
            pass
    if os.path.isdir(parts_dir) and not os.listdir(parts_dir):
        try:
            os.rmdir(parts_dir)
        except OSError:
            pass


def _write_delta_feather(
    out_path: str,
    pair_arrays: dict[str, np.ndarray],
    labels: Sequence[str],
    unit: str,
    extra_meta: dict[bytes, bytes] | None = None,
) -> int:
    """Write a wide, null-padded one-column-per-pair delta-t '.feather'.

    Each pair's 1D array becomes a column (in ``labels`` order), padded to the
    longest pair's length. The ``delta_unit`` (``'code'`` / ``'ps'``) is stored
    in the schema metadata so the reader knows the column units.

    The Arrow table is assembled column-by-column (one ``pyarrow`` array per
    pair) rather than through a single consolidated ``pandas`` block; for the
    combined feathers, where the padded width (n_pairs x longest pair) can
    reach several GiB, that avoids a single huge contiguous allocation.

    The padding is Arrow *nulls* rather than NaN values. Both read back as NaN
    through ``to_pandas``, so this is invisible downstream, but it keeps
    padding distinguishable from data: 'combine_delta_feathers' can then count
    the real differences in a part from its null count instead of scanning it.
    Feathers written before this change pad with NaN and still read correctly.

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
    extra_meta : dict[bytes, bytes] | None, optional
        Further schema metadata to record, e.g. '_lag_meta'. The default is
        None.

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
        col = pa.array(arr)
        if arr.size < maxlen:
            # Concatenating a null run beats materialising a padded copy:
            # the data buffer stays the one numpy already owns.
            col = pa.concat_arrays(
                [col, pa.nulls(maxlen - arr.size, pa.float64())]
            )
        columns.append(col)
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
    md.update(extra_meta or {})
    table = table.replace_schema_metadata(md)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pa.ipc.new_file(out_path, table.schema) as writer:
        writer.write_table(table)
    return total


@contextlib.contextmanager
def _open_delta_feather(src: str) -> Iterator[pa.ipc.RecordBatchFileReader]:
    """Open a delta-t feather **memory-mapped**, for batch-at-a-time reading.

    ``pa.ipc.open_file(path)`` reads a whole record batch into memory the
    moment one is asked for. That is invisible on a feather assembled from
    64 MB parts, but everything written in a single batch - which is every
    feather the pre-parts code produced, and anything '_write_delta_feather'
    writes directly - then loads *entirely*. Measured: +794 MB for a 0.79 GB
    single-batch feather, so a 9.4 GB one asks for 9.4 GB in one allocation
    and the "streaming" read is streaming in name only.

    Memory mapping makes 'get_batch' cost nothing (+0 MB, same file) and lets
    the OS page in only the columns actually touched.

    Anything derived from a batch must be *copied* before this context exits -
    a zero-copy 'to_numpy' is a view into the mapping, which is unmapped on
    the way out. Boolean masking, arithmetic and 'write_batch' all copy, which
    is why every caller here is safe.
    """
    with pa.memory_map(src, "rb") as source:
        with pa.ipc.open_file(source) as reader:
            yield reader


def _read_delta_feather(feather_path: str) -> tuple[pd.DataFrame, str | None]:
    """Read a delta-t '.feather' into a DataFrame plus its unit.

    This materialises the whole table (and ``to_pandas`` copies it again), so
    it is for small feathers only - a single part, or a short run. To read a
    full-size combined feather, stream it with '_stream_delta_histogram'
    instead; nothing downstream of stage 1 needs the raw differences.

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


def _real_values(col: pa.Array) -> int:
    """Count a column's actual differences, ignoring padding.

    Parts written by this module pad with nulls; feathers written before that
    change pad with NaN. Both are excluded, so the total reported after a
    combine is the number of differences either way.
    """
    n = len(col) - col.null_count
    if n:
        n -= int(pc.sum(pc.is_nan(col)).as_py() or 0)
    return int(n)


def _combine_delta_feathers(
    sources: Sequence[str], out_path: str
) -> tuple[int, list[str], str]:
    """Concatenate delta-t feathers into one, a record batch at a time.

    The obvious implementation - read every source into a DataFrame, pool the
    columns, write once - needs the whole pooled data set resident twice over,
    which is exactly what fails on a large run. Arrow IPC files are a sequence
    of record batches sharing a schema, so instead each source's batches are
    re-emitted into a single output file: peak memory is one batch (one part),
    whatever the total.

    Columns are the union of the sources' pair labels in first-seen order; a
    source missing a pair contributes nulls for it, which is the same padding
    a pair that never fired gets anyway.

    Parameters
    ----------
    sources : Sequence[str]
        Delta-t feathers to concatenate, in output order.
    out_path : str
        Path to write the combined '.feather' to.

    Returns
    -------
    total : int
        Number of real (non-padding) differences written.
    labels : list[str]
        The combined column order.
    unit : str
        The shared ``delta_unit``, or ``'code'`` if none was recorded.

    Raises
    ------
    ValueError
        Raised when the sources mix ``delta_unit`` ('code' and 'ps').
    """
    labels: list[str] = []
    seen: set[str] = set()
    units: set[str] = set()
    carried: dict[bytes, set[bytes]] = {}
    for src in sources:
        with _open_delta_feather(src) as reader:
            schema = reader.schema
        src_unit = (schema.metadata or {}).get(b"delta_unit", b"").decode()
        if src_unit:
            units.add(src_unit)
        # Anything else the sources recorded (the lag bookkeeping, say) has to
        # survive the combine, or the parts would describe themselves and the
        # combined feather would not.
        for key, value in (schema.metadata or {}).items():
            if key not in (b"delta_unit", b"n_pairs"):
                carried.setdefault(key, set()).add(value)
        for label in schema.names:
            if label not in seen:
                seen.add(label)
                labels.append(label)

    if len(units) > 1:
        raise ValueError(
            f"Cannot combine feathers with mixed delta_unit {sorted(units)}."
        )
    unit = units.pop() if units else "code"

    metadata = {
        b"delta_unit": unit.encode(),
        b"n_pairs": str(len(labels)).encode(),
    }
    for key, values in carried.items():
        if len(values) == 1:
            metadata[key] = values.pop()
        else:
            print(
                f"  WARNING: sources disagree on schema metadata "
                f"{key.decode()}; dropping it from the combined feather."
            )

    out_schema = pa.schema(
        [pa.field(label, pa.float64()) for label in labels], metadata=metadata
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    total = 0
    with pa.ipc.new_file(out_path, out_schema) as writer:
        for src in sources:
            with _open_delta_feather(src) as reader:
                present = set(reader.schema.names)
                for i in range(reader.num_record_batches):
                    batch = reader.get_batch(i)
                    if not batch.num_rows:
                        continue
                    columns: list[pa.Array] = []
                    for label in labels:
                        if label in present:
                            col = batch.column(
                                batch.schema.get_field_index(label)
                            )
                            total += _real_values(col)
                        else:
                            col = pa.nulls(batch.num_rows, pa.float64())
                        columns.append(col)
                    writer.write_batch(
                        pa.record_batch(columns, schema=out_schema)
                    )
    return total, labels, unit


def combine_delta_t_parts(
    parts_dir: str,
    out_path: str | None = None,
    *,
    rewrite: bool = False,
    run: str | None = None,
) -> str:
    """Assemble one run's delta-t part files into its combined feather.

    'calculate_and_save_timestamp_differences' does this itself at the end of
    a successful run. Call it by hand to recover a run that died partway
    through: every part flushed before the crash is complete and usable, so
    the differences up to the last flush are not lost.

    Parameters
    ----------
    parts_dir : str
        The ``processed/<dataset>_delta_t_parts`` folder to assemble.
    out_path : str | None, optional
        Where to write the combined '.feather'. The default is None, meaning
        ``processed/<dataset>_delta_t.feather`` - the same path the run would
        have written, beside the part folder.
    rewrite : bool, optional
        Overwrite an existing combined feather. The default is False, which
        raises instead. When True, 'store.confirm_rewrite' counts down first.
    run : str | None, optional
        Which run's parts to assemble, when the folder holds more than one.
        The default is None, which requires the folder to describe exactly
        one run and raises otherwise rather than pooling them.

    Returns
    -------
    str
        Path to the saved combined '.feather'.

    Raises
    ------
    FileNotFoundError
        Raised when the folder holds no part files.
    FileExistsError
        Raised when the output exists and ``rewrite`` is False.
    ValueError
        Raised when the folder holds several runs and ``run`` is None.
    """
    parts = _find_parts(parts_dir, run)
    if not parts:
        raise FileNotFoundError(
            f"No *_delta_t_<i>.feather part files in:\n  {parts_dir}"
        )

    if out_path is None:
        stem = os.path.basename(os.path.normpath(parts_dir)).removesuffix(
            _PARTS_SUFFIX
        )
        out_path = os.path.join(
            os.path.dirname(os.path.normpath(parts_dir)), f"{stem}.feather"
        )

    if os.path.isfile(out_path):
        if not rewrite:
            raise FileExistsError(
                f"{out_path} already exists. Pass rewrite=True to overwrite."
            )
        store.confirm_rewrite(out_path)

    print(
        f"\n> > > Assembling {len(parts)} part file(s) from "
        f"{os.path.basename(os.path.normpath(parts_dir))} into "
        f"{os.path.basename(out_path)} < < <"
    )
    total, labels, unit = _combine_delta_feathers(parts, out_path)
    print(
        f"\n> > > {total} differences across {len(labels)} pixel-pair "
        f"column(s) -> {out_path} (unit '{unit}') < < <"
    )
    return out_path


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
    keeps the one-column-per-pair, padded layout and the ``delta_unit``
    metadata, so it plots exactly like a single-run feather (pass it to
    'collect_and_plot_timestamp_differences' via ``feather_path``).

    The pooling streams a record batch at a time (see
    '_combine_delta_feathers'), so combining a hundred runs costs one batch of
    memory rather than the whole pooled data set. Per-run part folders are not
    picked up: their files end in ``_<i>.feather``, so the ``*_delta_t.feather``
    search skips them and nothing is counted twice.

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
        Overwrite an existing combined feather. The default is False, which
        raises instead. When True, 'store.confirm_rewrite' counts down first.
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
    if os.path.isfile(out_path):
        if not rewrite:
            raise FileExistsError(
                f"{out_path} already exists. Pass rewrite=True to overwrite."
            )
        store.confirm_rewrite(out_path)

    print(
        f"\n> > > Combining {len(feathers)} delta-t feather(s) under "
        f"{path} into {os.path.basename(out_path)} < < <\n"
    )

    total, labels, unit = _combine_delta_feathers(feathers, out_path)

    print(
        f"\n> > > Combined {len(feathers)} feather(s): {total} differences "
        f"across {len(labels)} pixel-pair column(s) -> {out_path} "
        f"(unit '{unit}') < < <"
    )
    return out_path


def _resolve_delta_sources(target: str) -> tuple[list[str], str, str]:
    """Resolve a delta-t target to the feather(s) behind it.

    Accepts either a single '.feather' or a part folder - the parts are
    complete feathers in their own right, so a run whose combined file is
    missing (or too big to be worth writing) plots straight from them.

    Parameters
    ----------
    target : str
        Path to a delta-t '.feather', or to a ``*_delta_t_parts`` folder.

    Returns
    -------
    sources : list[str]
        The feather files to stream, in order.
    name : str
        Dataset name, for figure titles and artifact file names.
    root : str
        The data folder the artifacts belong to - ``processed/`` and
        ``results/`` sit under it.

    Raises
    ------
    FileNotFoundError
        Raised when a folder was given but holds no part files.
    """
    target = os.path.abspath(target)
    if os.path.isdir(target):
        sources = _find_parts(target)
        if not sources:
            raise FileNotFoundError(
                f"No *_delta_t_<i>.feather part files in:\n  {target}"
            )
        base = os.path.basename(os.path.normpath(target))
        name = base.removesuffix(_PARTS_SUFFIX).removesuffix("_delta_t")
    else:
        sources = [target]
        base = os.path.basename(target)
        name = base.removesuffix(".feather").removesuffix("_delta_t")
    # <root>/processed/<artifact> -> the artifacts belong to <root>.
    return sources, name, os.path.dirname(os.path.dirname(target))


def _locate_delta_target(
    path: str | None, feather_path: str | None
) -> tuple[list[str], str, str]:
    """Find the delta-t feather(s) for a data folder, or take them directly.

    ``feather_path`` wins and ignores ``path`` entirely - that is how a
    combined or standalone feather is read without its raw-data folder.
    Otherwise the run's own ``processed/<name>_delta_t.feather`` is used,
    falling back to its part folder when the combine step never happened.

    Parameters
    ----------
    path : str | None
        Path to the raw-data folder.
    feather_path : str | None
        Explicit '.feather' (or part folder) to read.

    Returns
    -------
    tuple[list[str], str, str]
        Sources, dataset name and data folder, as '_resolve_delta_sources'.

    Raises
    ------
    ValueError
        Raised when neither ``path`` nor ``feather_path`` is given.
    FileNotFoundError
        Raised when neither a combined feather nor any part file exists.
    """
    if feather_path is not None:
        return _resolve_delta_sources(feather_path)
    if path is None:
        raise ValueError(
            "Provide either path (to locate the feather) or feather_path."
        )

    name = os.path.basename(os.path.normpath(path))
    combined = os.path.join(
        store.processed_dir(path, create=False), f"{name}_delta_t.feather"
    )
    if os.path.isfile(combined):
        return [combined], name, os.path.abspath(path)

    parts_dir = combined.removesuffix(".feather") + _PARTS_SUFFIX
    parts = _find_parts(parts_dir)
    if parts:
        print(
            f"  no combined feather - streaming the {len(parts)} part "
            f"file(s) in {os.path.basename(parts_dir)}{os.sep} instead."
        )
        return parts, name, os.path.abspath(path)

    raise FileNotFoundError(
        f"No delta-t feather at {combined}. Run "
        "calculate_and_save_timestamp_differences first."
    )


def _delta_hist_edges(
    bin_width_ps: float, plot_window_ps: float
) -> np.ndarray:
    """Histogram bin edges: bins centred on 0, spanning +/- the window."""
    return np.arange(
        -plot_window_ps - bin_width_ps / 2,
        plot_window_ps + bin_width_ps,
        bin_width_ps,
    )


def _hist_signature(
    sources: Sequence[str],
    *,
    time_unit_ps: float,
    bin_width_ps: float,
    plot_window_ps: float,
    pairs: Sequence[str] | None,
    lags: Sequence[int] | None = None,
) -> dict:
    """Everything that changes the histogram, for reuse invalidation.

    The inputs are fingerprinted by (name, size, mtime) rather than content:
    hashing 135 GB to decide whether to re-read 135 GB would be self-defeating,
    and stage-1 feathers are written once and not edited in place.
    """
    digest = hashlib.sha1()
    for src in sorted(sources):
        st = os.stat(src)
        digest.update(
            f"{os.path.basename(src)}:{st.st_size}:{st.st_mtime_ns};".encode()
        )
    signature = {
        "time_unit_ps": float(time_unit_ps),
        "bin_width_ps": float(bin_width_ps),
        "plot_window_ps": float(plot_window_ps),
        "pairs": None if pairs is None else sorted(pairs),
        "n_sources": len(sources),
        "sources_digest": digest.hexdigest(),
    }
    # Added only when lags are in play, so a plain histogram keeps the
    # signature it had before they existed and stays reusable.
    if lags is not None:
        signature["lags"] = [int(k) for k in lags]
    return signature


def _pooled_columns(
    sources: Sequence[str], pairs: Sequence[str] | None
) -> tuple[list[str], str, list[str]]:
    """Decide which columns to pool, and in what unit, from the schemas alone.

    Opening an Arrow IPC file reads its footer, not its data, so this pass
    over a multi-GB feather is effectively free.

    Returns
    -------
    columns : list[str]
        Column names to pool, in first-seen order.
    unit : str
        The shared ``delta_unit`` ('code' / 'ps').
    missing : list[str]
        Requested ``pairs`` that no source has a column for.

    Raises
    ------
    ValueError
        Raised when the sources mix ``delta_unit``.
    """
    names: list[str] = []
    units: set[str] = set()
    for src in sources:
        with _open_delta_feather(src) as reader:
            schema = reader.schema
        src_unit = (schema.metadata or {}).get(b"delta_unit", b"").decode()
        if src_unit:
            units.add(src_unit)
        for label in schema.names:
            if label not in names:
                names.append(label)

    if len(units) > 1:
        raise ValueError(
            f"Cannot pool feathers with mixed delta_unit {sorted(units)}."
        )
    unit = units.pop() if units else ""

    # Legacy single-column layouts name the unit in the column itself.
    if "delta_ps" in names:
        return ["delta_ps"], unit or "ps", []
    if "delta_code" in names:
        return ["delta_code"], unit or "code", []

    # Frame-shifted columns are background, not signal. Pooling them into the
    # coincidence histogram because they happen to be in the file would inflate
    # it ~9x, so they are only ever reached through an explicit lag request
    # ('_pooled_lag_columns').
    names = [n for n in names if not _is_lag_column(n)]

    if pairs is None:
        return names, unit or "code", []
    missing = [p for p in pairs if p not in names]
    return [p for p in pairs if p in names], unit or "code", missing


def _pooled_lag_columns(
    sources: Sequence[str], pairs: Sequence[str] | None, lags: Sequence[int]
) -> tuple[list[list[str]], str, list[str]]:
    """Group the columns to pool by frame lag, in ``lags`` order.

    Returns
    -------
    groups : list[list[str]]
        One list of column names per lag, lag 0 first.
    unit : str
        The shared ``delta_unit``.
    missing : list[str]
        Requested pair/lag columns no source has.

    Raises
    ------
    ValueError
        Raised when the sources mix ``delta_unit``, or hold no lag columns at
        all.
    """
    names: list[str] = []
    units: set[str] = set()
    for src in sources:
        with _open_delta_feather(src) as reader:
            schema = reader.schema
        src_unit = (schema.metadata or {}).get(b"delta_unit", b"").decode()
        if src_unit:
            units.add(src_unit)
        for label in schema.names:
            if label not in names:
                names.append(label)

    if len(units) > 1:
        raise ValueError(
            f"Cannot pool feathers with mixed delta_unit {sorted(units)}."
        )
    unit = units.pop() if units else "code"

    wanted = pairs if pairs is not None else [
        n for n in names if not _is_lag_column(n)
    ]
    groups: list[list[str]] = []
    missing: list[str] = []
    for k in lags:
        group = []
        for label in wanted:
            column = _lag_column(label, k)
            if column in names:
                group.append(column)
            else:
                missing.append(column)
        groups.append(group)
    return groups, unit, missing


def _stream_delta_histogram(
    sources: Sequence[str],
    *,
    edges: np.ndarray,
    time_unit_ps: float,
    plot_window_ps: float,
    pairs: Sequence[str] | None = None,
    lags: Sequence[int] | None = None,
    quiet: bool = False,
) -> tuple[np.ndarray, dict]:
    """Histogram delta-t feathers one Arrow record batch at a time.

    A full run's combined feather reaches hundreds of GB, so the old read path
    - ``read_all().to_pandas()``, then ``to_numpy().ravel()`` - could not open
    it at all: it needs the whole table resident several times over. Arrow IPC
    files are a sequence of record batches, though, and a histogram over fixed
    edges is *additive*, so the counts can be summed batch by batch with one
    batch resident at a time. The result is identical to histogramming the
    pooled array in one go, bin for bin - this is the exact baseline, with no
    rounding, sampling or re-binning anywhere.

    Padding is dropped as it is met: nulls (current writers) and NaN (feathers
    written before null padding) both, exactly as the pooled path did.

    Parameters
    ----------
    sources : Sequence[str]
        Delta-t feathers to pool. Their columns need not match.
    edges : np.ndarray
        Histogram bin edges, from '_delta_hist_edges'.
    time_unit_ps : float
        Picoseconds per raw TDC code, applied when the unit is ``'code'``.
    plot_window_ps : float
        Only differences with ``abs(delta_ps) <= plot_window_ps`` are counted.
    pairs : Sequence[str] | None, optional
        Restrict pooling to these pixel-pair columns. The default is None
        (pool every pair column). Frame-shifted columns are never included
        unless ``lags`` asks for them.
    lags : Sequence[int] | None, optional
        Frame lags to histogram *separately*, lag 0 first, giving one row of
        counts each. The default is None - one pooled histogram, as before.
        All lags are accumulated in the same single pass over the feather,
        because that pass is the expensive part.
    quiet : bool, optional
        Suppress the progress bar. The default is False.

    Returns
    -------
    counts : np.ndarray
        Counts per bin, ``len(edges) - 1`` long - or ``(n_lags, n_bins)`` when
        ``lags`` is given.
    stats : dict
        ``unit``, ``n`` (differences inside the window), ``total`` (real
        differences seen), ``slots`` (table cells read, padding included),
        ``padding_fraction``, ``granularity_ps`` (smallest non-zero
        ``abs(delta_ps)``, for the comb guard), ``n_columns`` and ``missing``.
        With ``lags``, ``n`` and ``total`` count the lag-0 row only - they
        describe the measurement, not the bookkeeping around it.
    """
    if lags is None:
        groups, unit, missing = _pooled_columns(sources, pairs)
        groups = [groups]
    else:
        groups, unit, missing = _pooled_lag_columns(sources, pairs, lags)
    if missing:
        print(f"  WARNING: pair column(s) not in feather: {missing}")

    # Batch counts come from the footer, so the bar knows its length up front.
    plan: list[tuple[str, int]] = []
    for src in sources:
        with _open_delta_feather(src) as reader:
            plan.append((src, reader.num_record_batches))

    counts = np.zeros((len(groups), len(edges) - 1), dtype=np.int64)
    n_windowed = 0
    total = 0
    slots = 0
    granularity_ps = np.inf

    bar = tqdm(
        total=sum(n for _, n in plan),
        desc="delta-t histogram",
        unit="batch",
        disable=quiet,
    )
    for src, n_batches in plan:
        with _open_delta_feather(src) as reader:
            present = set(reader.schema.names)
            indices = [
                [
                    reader.schema.get_field_index(c)
                    for c in group
                    if c in present
                ]
                for group in groups
            ]
            for i in range(n_batches):
                batch = reader.get_batch(i)
                for gi, group_indices in enumerate(indices):
                    for j in group_indices:
                        # Nulls come back as NaN, so one mask drops both kinds
                        # of padding. The resident set is this batch (~one
                        # part) plus this one column's copy - never the table.
                        values = batch.column(j).to_numpy(zero_copy_only=False)
                        if gi == 0:
                            slots += values.size
                        values = values[~np.isnan(values)]
                        if not values.size:
                            continue
                        d_ps = values if unit == "ps" else values * time_unit_ps
                        if gi == 0:
                            total += d_ps.size

                            nonzero = np.abs(d_ps[d_ps != 0])
                            if nonzero.size:
                                granularity_ps = min(
                                    granularity_ps, float(nonzero.min())
                                )

                        inside = d_ps[np.abs(d_ps) <= plot_window_ps]
                        if gi == 0:
                            n_windowed += inside.size
                        if inside.size:
                            counts[gi] += np.histogram(inside, bins=edges)[0]
                del batch
                bar.update(1)
    bar.close()

    stats = {
        "unit": unit,
        "n": int(n_windowed),
        "total": int(total),
        "slots": int(slots),
        "padding_fraction": (1.0 - total / slots) if slots else 0.0,
        "granularity_ps": (
            float(granularity_ps) if np.isfinite(granularity_ps) else None
        ),
        "n_columns": len(groups[0]) if groups else 0,
        "missing": list(missing),
    }
    if lags is not None:
        stats["lags"] = [int(k) for k in lags]
        return counts, stats
    return counts[0], stats


def compute_and_save_delta_histogram(
    path: str | None = None,
    *,
    feather_path: str | None = None,
    time_unit_ps: float = _TS_CODE_PS,
    bin_width_ps: float = _TS_CODE_PS,
    plot_window_ps: float = 3000.0,
    pairs: Sequence[str] | None = None,
    subtract_background: bool = False,
    reuse: bool = True,
    quiet: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stream a delta-t feather down to a saved histogram.

    This is the expensive pass - it reads every record batch of every source -
    and it is the only thing standing between a hundreds-of-GB feather and a
    plot. The counts are saved to ``processed/<name>_delta_hist.npy`` with a
    '.meta.json' sidecar, so replotting, re-fitting or changing the background
    model costs a ~5 MB load rather than another full read.

    The saved histogram is *lossless for everything downstream*: both fit
    models take ``(bin_centers, counts)``, and so does the plot. Binning at
    ``bin_width_ps=_TS_CODE_PS`` (one TDC code) is the native resolution of
    the data, so nothing is thrown away at the default settings.

    The sidecar records the binning and a fingerprint of the sources; a saved
    histogram is only reused when all of it matches, so changing
    ``bin_width_ps``, ``plot_window_ps``, ``time_unit_ps`` or ``pairs`` - or
    rewriting the feather - re-streams automatically.

    Parameters
    ----------
    path : str | None, optional
        Path to the raw-data folder. Its ``processed/`` holds the feather and
        receives the histogram. May be None when ``feather_path`` is given.
    feather_path : str | None, optional
        Explicit delta-t '.feather' to read, or a ``*_delta_t_parts`` folder
        to read the parts directly (which skips needing the combined file at
        all). When given, ``path`` is ignored. The default is None.
    time_unit_ps : float, optional
        Picoseconds per raw TDC code, applied to a ``'code'`` feather. The
        default is ~77 ps.
    bin_width_ps : float, optional
        Histogram bin width in ps. The default is ~77 ps (one code).
    plot_window_ps : float, optional
        Histogram half-range in ps around zero. The default is 3000.0.
    pairs : Sequence[str] | None, optional
        Restrict pooling to these pixel-pair column names. The default is
        None (pool every pair column). Frame-shifted columns are never pooled
        into the signal histogram.
    subtract_background : bool, optional
        Histogram every frame lag separately instead of pooling one signal
        histogram, giving an ``(n_lags, n_bins)`` array with lag 0 - the
        ordinary histogram, bin-for-bin - as row 0. The default is False.
        Requires a feather written with
        ``calculate_and_save_timestamp_differences(..., subtract_background=True)``.
    reuse : bool, optional
        Load a matching saved histogram instead of re-streaming. The default
        is True. Pass False to force the full pass.
    quiet : bool, optional
        Suppress the progress bar and the summary. The default is False.

    Returns
    -------
    counts : np.ndarray
        Counts per bin, or ``(n_lags, n_bins)`` with ``subtract_background``.
    edges : np.ndarray
        The bin edges the counts belong to.
    info : dict
        The stats from '_stream_delta_histogram' plus ``name``, ``root``,
        ``sources``, ``hist_path`` and ``reused``. With
        ``subtract_background`` also ``lags`` and ``frames_used``.

    Raises
    ------
    ValueError
        Raised when ``subtract_background`` is asked for on a feather with no
        frame-shifted columns.
    """
    sources, name, root = _locate_delta_target(path, feather_path)
    lag_meta = _read_lag_meta(sources) if subtract_background else None
    if subtract_background and lag_meta is None:
        raise ValueError(
            "subtract_background=True needs a feather written with "
            "frame-shifted columns. Recompute stage 1 with "
            "'calculate_and_save_timestamp_differences(..., "
            "subtract_background=True)', or use the counts path."
        )
    lags = lag_meta["lags"] if lag_meta else None

    edges = _delta_hist_edges(bin_width_ps, plot_window_ps)
    signature = _hist_signature(
        sources,
        time_unit_ps=time_unit_ps,
        bin_width_ps=bin_width_ps,
        plot_window_ps=plot_window_ps,
        pairs=pairs,
        lags=lags,
    )
    hist_path = os.path.join(
        store.processed_dir(root, create=False), f"{name}{_HIST_SUFFIX}"
    )

    def _lag_info(info: dict) -> dict:
        """Attach the lag bookkeeping the subtraction needs."""
        if lag_meta is not None:
            info["lags"] = lags
            info["frames_used"] = [
                int(v) for v in bs.frames_per_lag(lag_meta["nframes"], lags)
            ]
        return info

    if reuse and os.path.isfile(hist_path):
        counts, meta = store.load_map(npy_path=hist_path)
        if (
            meta.get("signature") == signature
            and counts.shape[-1] == len(edges) - 1
        ):
            if not quiet:
                print(
                    f"\n> > > Reusing the saved delta-t histogram "
                    f"({meta.get('n', 0)} differences in the window) from "
                    f"{os.path.basename(hist_path)} - pass reuse=False to "
                    "re-stream < < <"
                )
            info = dict(meta.get("stats", {}))
            info.update(
                name=name,
                root=root,
                sources=sources,
                hist_path=hist_path,
                reused=True,
            )
            return counts, edges, _lag_info(info)

    if not quiet:
        print(
            f"\n> > > Streaming {len(sources)} delta-t feather(s) into a "
            f"{len(edges) - 1}-bin histogram ({bin_width_ps:.0f} ps bins, "
            f"+/-{plot_window_ps:.0f} ps)"
            + (f" at lags {lags}" if lags else "")
            + " < < <\n"
        )

    counts, stats = _stream_delta_histogram(
        sources,
        edges=edges,
        time_unit_ps=time_unit_ps,
        plot_window_ps=plot_window_ps,
        pairs=pairs,
        lags=lags,
        quiet=quiet,
    )

    hist_path = store.save_map(
        counts,
        root,
        kind=_KIND,
        file_name=f"{name}{_HIST_SUFFIX}",
        meta={
            "signature": signature,
            "stats": stats,
            "sources": [os.path.basename(s) for s in sources],
        },
        quiet=True,
    )

    if not quiet:
        print(
            f"\n> > > {stats['total']} difference(s) across "
            f"{stats['n_columns']} pair column(s); {stats['n']} inside "
            f"+/-{plot_window_ps:.0f} ps. Padding was "
            f"{100 * stats['padding_fraction']:.1f}% of the cells read. "
            f"Histogram saved to {hist_path} < < <"
        )
    return (
        counts,
        edges,
        _lag_info(
            {
                **stats,
                "name": name,
                "root": root,
                "sources": sources,
                "hist_path": hist_path,
                "reused": False,
            }
        ),
    )


def _resolve_time_lut(
    apply_TDC_calibration: bool,
    daughterboard_number: str | None,
    motherboard_number: str | None,
    spad: int | str,
) -> np.ndarray | None:
    """Load the density-test LUT to apply before differencing, or None.

    Mirrors what 'calculate_and_save_timestamp_differences' does inline.

    Returns
    -------
    np.ndarray | None
        The code -> ps lookup table, or None to keep raw codes.

    Raises
    ------
    ValueError
        Raised when calibration is asked for without naming a board.
    """
    if not apply_TDC_calibration:
        return None
    if daughterboard_number is None or motherboard_number is None:
        raise ValueError(
            "apply_TDC_calibration=True but no board was given. Pass "
            "daughterboard_number and motherboard_number, or set "
            "apply_TDC_calibration=False to save raw codes."
        )
    return tc.load_board_lut(daughterboard_number, motherboard_number, spad)


def _delta_grid(
    grid_ps: float, support_ps: float
) -> tuple[int, int, np.ndarray]:
    """Define the native counts grid: bins centred on zero, spanning support.

    Bin ``i`` covers ``[(i - zero) - 0.5, (i - zero) + 0.5] * grid_ps``, so
    **zero sits at a bin centre by construction**. That is the one thing the
    feather path cannot promise: its edges come from
    ``arange(-W - w/2, ...)``, which only centres a bin on zero when the
    window is an exact multiple of the width.

    For raw codes (``grid_ps`` = one code) the grid is not an approximation at
    all: ``delta_code`` is an integer, so counts-per-code carries exactly the
    information the feather did, at ~8 bytes per *bin* instead of per *event*.

    Parameters
    ----------
    grid_ps : float
        Width of one grid cell, in ps.
    support_ps : float
        Half-range to cover, in ps. Rounded up to a whole number of cells.

    Returns
    -------
    n_bins : int
        Number of cells, always odd.
    zero_index : int
        Index of the cell holding zero.
    edges : np.ndarray
        The ``n_bins + 1`` cell edges, in ps.
    """
    if grid_ps <= 0:
        raise ValueError(f"grid_ps must be positive, got {grid_ps}")
    if support_ps <= 0:
        raise ValueError(f"support_ps must be positive, got {support_ps}")
    zero_index = int(np.ceil(support_ps / grid_ps))
    n_bins = 2 * zero_index + 1
    edges = (np.arange(n_bins + 1) - zero_index - 0.5) * grid_ps
    return n_bins, zero_index, edges


def _counts_signature(
    files: Sequence[str],
    *,
    labels: Sequence[str],
    mode: str,
    nframes: int,
    delta_window: float | None,
    unit: str,
    grid_ps: float,
    support_ps: float,
    lags: Sequence[int] = (0,),
) -> dict:
    """Everything that must match for a partial counts run to be resumed.

    The file list is fingerprinted by (name, size) - a checkpoint is only
    resumable against the same files in the same order, since progress is
    recorded as "the first N of them".
    """
    digest = hashlib.sha1()
    for fp in files:
        digest.update(f"{os.path.basename(fp)}:{os.path.getsize(fp)};".encode())
    signature = {
        "n_files": len(files),
        "files_digest": digest.hexdigest(),
        "labels_digest": hashlib.sha1(
            "|".join(labels).encode()
        ).hexdigest(),
        "n_labels": len(labels),
        "mode": mode,
        "nframes": int(nframes),
        "delta_window": None if delta_window is None else float(delta_window),
        "unit": unit,
        "grid_ps": float(grid_ps),
        "support_ps": float(support_ps),
    }
    # Added only when there are lags to record, so a plain (no-subtraction) run
    # produces the same signature it did before lags existed - and every
    # artifact already on disk still resumes instead of looking foreign.
    if list(lags) != [0]:
        signature["lags"] = [int(k) for k in lags]
    return signature


def _accumulate_delta_counts(
    counts: np.ndarray,
    deltas: dict[str, np.ndarray],
    rows: dict[str, int],
    *,
    grid_ps: float,
    zero_index: int,
) -> tuple[int, int]:
    """Bin one file's differences into the resident counts grid, in place.

    'np.add.at' rather than 'np.bincount': bincount pays for the whole grid on
    every call, and a single file contributes a handful of differences per
    pair, so the fixed cost would dwarf the data by orders of magnitude.

    Returns
    -------
    total : int
        Differences seen.
    outside : int
        Differences beyond the grid's support. Counted, never dropped
        silently - if this is not ~0 the support is too narrow and the
        triangular background is being clipped.
    """
    n_bins = counts.shape[1]
    total = 0
    outside = 0
    for label, arr in deltas.items():
        row = rows.get(label)
        if row is None or not arr.size:
            continue
        arr = arr[np.isfinite(arr)]
        if not arr.size:
            continue
        total += int(arr.size)
        idx = np.rint(arr / grid_ps).astype(np.int64) + zero_index
        keep = (idx >= 0) & (idx < n_bins)
        outside += int(arr.size - np.count_nonzero(keep))
        np.add.at(counts[row], idx[keep], 1)
    return total, outside


def _background_lags(background_lags: int | Sequence[int]) -> list[int]:
    """Validate the requested frame offsets and prepend lag 0 (the signal).

    A bare int is a single lag: ``background_lags=1`` is the obvious thing to
    write for one shifted frame, and 'calc_diff' already accepts a bare
    ``(row, col)`` the same way.

    Raises
    ------
    ValueError
        Raised on an empty set, a 0 / negative entry (lag 0 is the signal, not
        a background), or a duplicate.
    """
    if isinstance(background_lags, (int, np.integer)):
        background_lags = (int(background_lags),)
    lags = [int(k) for k in background_lags]
    if not lags:
        raise ValueError(
            "subtract_background=True needs at least one background lag; got "
            "an empty background_lags."
        )
    if any(k <= 0 for k in lags):
        raise ValueError(
            f"background_lags must all be >= 1 (lag 0 is the signal), got {lags}"
        )
    if len(set(lags)) != len(lags):
        raise ValueError(f"background_lags has duplicates: {lags}")
    return [0, *lags]


def _write_counts_checkpoint(
    counts_path: str, counts: np.ndarray, meta: dict
) -> None:
    """Write the counts grid and its sidecar as atomically as two files allow.

    Both are written to temporaries and moved into place, array first. A crash
    in the microseconds between the two moves leaves a newer array beside an
    older sidecar, which would make a resume re-run files already counted - so
    the sidecar carries ``total_in_grid`` and the resume path refuses to
    continue unless the array sums to it.
    """
    tmp_npy = counts_path + ".tmp.npy"
    tmp_meta = store.meta_path(counts_path) + ".tmp"
    np.save(tmp_npy, counts)
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp_npy, counts_path)
    os.replace(tmp_meta, store.meta_path(counts_path))


def calculate_and_save_delta_counts(
    path: str,
    pixels: Sequence[Sequence[Pixel]],
    rewrite: bool = False,
    *,
    nframes: int,
    tag: str = "ORT",
    mode: str = "all_pairs",
    delta_window: float | None = None,
    apply_TDC_calibration: bool = True,
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    spad: int | str = "average",
    max_files: int | None = None,
    grid_ps: float | None = None,
    support_ps: float = _SUPPORT_PS,
    subtract_background: bool = False,
    background_lags: int | Sequence[int] = bs.BACKGROUND_LAGS,
    flush_every: int = _FLUSH_EVERY,
    resume: bool = True,
) -> str:
    """Unpack ORT data and histogram delta-t directly - no feather.

    The alternative to 'calculate_and_save_timestamp_differences'. Same loop
    (unpack -> per-pixel timestamps -> 'calc_diff.calculate_differences'), but
    the differences are binned onto a fixed native grid as they are found and
    only the counts are kept. Writes
    ``processed/<name>_delta_counts.npy``: one row per pixel pair, one column
    per grid cell - and, with ``subtract_background``, one *plane* per frame
    lag on top of that.

    The size is set by the *grid*, not by the data - a 10 000-file run and a
    100-file run produce the same ~100 MB artifact, against 3-135 GB of
    feather. That is what makes every pair plottable at once, and it makes
    re-binning free: 'rebin_delta_counts' turns the grid into any coarser
    histogram in memory, so ``bin_width_ps`` and ``plot_window_ps`` stop being
    compute-time decisions.

    **What it gives up.** No per-event data survives, so there is no
    re-calibrating with a different LUT, no re-cutting ``delta_window``, and
    no analysis that needs individual differences (time ordering, splitting by
    acquisition, higher-order correlations). Keep the feather until you have
    decided you never want those.

    **How exact it is.** For raw codes (``apply_TDC_calibration=False``) the
    default grid is one code and the result is *lossless* - integers binned
    onto their own lattice. For calibrated data the LUT smears the lattice, so
    each difference is displaced by up to ``grid_ps / 2`` (~3.85 ps at the
    default), which is far below the detector jitter but is not bit-exact
    against the feather.

    Crash insurance is one file rather than a part folder: the grid is
    rewritten every ``flush_every`` files, so a run that dies leaves a valid
    artifact covering everything up to its last checkpoint, and ``resume``
    picks it up. Two runs cannot interleave, because there is only ever one
    file to write.

    Parameters
    ----------
    path : str
        Path to the folder with the ORT '.bin' data files.
    pixels : Sequence[Sequence[tuple[int, int]]]
        Two groups ``[group_a, group_b]`` - the signal and idler blobs.
    rewrite : bool, optional
        Overwrite a *complete* counts artifact for this data set. The default
        is False, which raises instead. An incomplete one is resumed rather
        than overwritten (see ``resume``).
    nframes : int
        Number of frames stored in each '.bin' file. Part of the resume
        signature - it sets how much of each file is read, so a checkpoint
        only resumes against the same value.
    tag : str, optional
        Filename fragment selecting the files. The default is ``'ORT'``.
    mode : str, optional
        ``'all_pairs'`` (default) or ``'1v1'``.
    delta_window : float | None, optional
        Keep only differences with ``abs(delta) <= delta_window``. In code
        units, or ps if calibrating. The default is None (keep all). Note this
        is frozen into the artifact - unlike the feather, it cannot be
        narrowed afterwards, so prefer leaving it None and cutting at plot
        time.
    apply_TDC_calibration : bool, optional
        Apply the per-pixel code -> ps LUT before differencing. The default is
        True, which requires ``daughterboard_number`` and
        ``motherboard_number``.
    daughterboard_number : str | None, optional
        Daughterboard id (e.g. ``'D0'``). The default is None.
    motherboard_number : str | None, optional
        Motherboard id (e.g. ``'M0'``). The default is None.
    spad : int | str, optional
        Which SPAD's LUT to use. The default is ``'average'``.
    max_files : int | None, optional
        Process at most this many files. The default is None (all files).
    grid_ps : float | None, optional
        Width of one grid cell, in ps. The default is None, meaning one TDC
        code (~77 ps) for raw-code data - where that is exact - and one tenth
        of a code (~7.7 ps) for calibrated data.
    support_ps : float, optional
        Half-range of the grid, in ps. The default is 200 000 (200 ns), one
        oscillator period. Differences beyond it are counted in
        ``n_outside``.
    subtract_background : bool, optional
        Also histogram the differences between *frame-shifted* pixel pairs, so
        the accidental background can be measured rather than fitted. The
        default is False, which writes exactly the 2D artifact it always has.
        When True the artifact gains a leading lag axis with lag 0 - the
        ordinary within-frame histogram, bin-for-bin identical to the 2D one -
        as plane 0; see 'dapkel.functions.background_subtraction' and
        ``docs/guide/background_subtraction.md``. Costs one plane of grid per lag and one
        differencing pass per lag; the '.bin' decoding, which dominates the
        loop, is paid once either way.
    background_lags : int | Sequence[int], optional
        Frame offsets used for the accidental estimate when
        ``subtract_background`` is True. The default is ``(1, ..., 8)``. A bare
        ``int`` is taken as a single lag. Lag 0 is always included and must not
        appear here. Ignored when ``subtract_background`` is False.
    flush_every : int, optional
        Files between checkpoint writes. The default is 50.
    resume : bool, optional
        Continue an incomplete artifact for the same inputs instead of
        starting over. The default is True.

    Returns
    -------
    str
        Path to the saved '.npy'.

    Raises
    ------
    FileExistsError
        Raised when a complete artifact exists and ``rewrite`` is False.
    ValueError
        Raised on a bad ``mode``, unequal ``'1v1'`` groups, a calibration
        request with no board, a non-positive grid, or a 0 / negative / repeated
        entry in ``background_lags``.
    """
    lags = _background_lags(background_lags) if subtract_background else [0]

    files = io.find_bin_files(path, tag)
    if max_files is not None:
        files = files[:max_files]

    labels, all_pixels = pixel_pairs.pair_labels(pixels, mode)
    time_lut = _resolve_time_lut(
        apply_TDC_calibration, daughterboard_number, motherboard_number, spad
    )
    unit = "ps" if time_lut is not None else "code"

    if grid_ps is None:
        grid_ps = (
            _TS_CODE_PS / _GRID_SUBDIV if unit == "ps" else _TS_CODE_PS
        )
    # In code units the grid and the support are counts of codes, not ps.
    grid = grid_ps if unit == "ps" else grid_ps / _TS_CODE_PS
    support = support_ps if unit == "ps" else support_ps / _TS_CODE_PS
    n_bins, zero_index, _ = _delta_grid(grid, support)

    name = os.path.basename(os.path.normpath(path))
    counts_path = os.path.join(
        store.processed_dir(path), f"{name}{_COUNTS_SUFFIX}"
    )
    signature = _counts_signature(
        files,
        labels=labels,
        mode=mode,
        nframes=nframes,
        delta_window=delta_window,
        unit=unit,
        grid_ps=grid_ps,
        support_ps=support_ps,
        lags=lags,
    )

    # 2D without the lags, exactly as before, so an artifact written by an
    # older run still resumes and every downstream reader keeps working.
    counts = np.zeros(
        (len(labels), n_bins) if len(lags) == 1 else (len(lags), len(labels), n_bins),
        dtype=np.int64,
    )
    rows = {label: i for i, label in enumerate(labels)}
    start = 0
    total = 0
    outside = 0

    existing = store.read_meta(counts_path) if os.path.isfile(counts_path) else {}
    if existing.get("signature") == signature:
        if existing.get("complete"):
            if not rewrite:
                raise FileExistsError(
                    f"{counts_path} already covers all {len(files)} file(s). "
                    "Pass rewrite=True to recompute."
                )
            store.confirm_rewrite([counts_path])
        elif resume:
            saved, _ = store.load_map(npy_path=counts_path)
            # The sidecar is moved into place after the array, so a crash
            # between the two moves can pair a newer array with an older
            # sidecar. Resuming on that would re-count files, and a silently
            # inflated coincidence total is worse than starting again.
            if (
                saved.shape == counts.shape
                and int(saved.sum()) == existing.get("total_in_grid")
            ):
                counts = saved
                start = int(existing.get("files_done", 0))
                total = int(existing.get("total", 0))
                outside = int(existing.get("n_outside", 0))
                print(
                    f"\n> > > Resuming {os.path.basename(counts_path)} at "
                    f"file {start} of {len(files)} < < <"
                )
            else:
                print(
                    f"\n  {os.path.basename(counts_path)} is inconsistent "
                    "with its sidecar (interrupted mid-write); starting over."
                )
    elif os.path.isfile(counts_path) and not rewrite:
        raise FileExistsError(
            f"{counts_path} exists but was computed from different inputs "
            "(files, pairs, grid or calibration). Pass rewrite=True to "
            "replace it."
        )
    elif os.path.isfile(counts_path):
        store.confirm_rewrite([counts_path])

    def _meta(files_done: int, complete: bool) -> dict:
        meta = {
            "signature": signature,
            "labels": labels,
            "unit": unit,
            "grid_ps": float(grid_ps),
            # The cell width in the *stored* unit - codes when uncalibrated.
            # Keeping both means the ps-per-code assumption is not baked in:
            # a code grid can be re-scaled at load time, exactly as the
            # feather path's 'time_unit_ps' does.
            "grid_native": float(grid),
            "support_ps": float(support_ps),
            "n_bins": int(n_bins),
            "zero_index": int(zero_index),
            "tag": tag,
            "files_done": int(files_done),
            "n_files": len(files),
            "complete": bool(complete),
            "total": int(total),
            "n_outside": int(outside),
            "total_in_grid": int(counts.sum()),
        }
        if subtract_background:
            # 'frames_used' is what 'background_subtraction' needs to rescale
            # the shifted lags. It is deterministic - lag k loses the last k
            # frames of every file - so it is derived rather than tracked, and
            # stays right on a resume.
            meta["lags"] = lags
            meta["frames_used"] = [
                int(v) for v in bs.frames_per_lag(nframes, lags, files_done)
            ]
        return meta

    print(
        f"\n> > > Histogramming delta-t straight from {len(files) - start} "
        f"'.bin' file(s) (mode='{mode}', tag='{tag}') onto a "
        f"{'x'.join(str(d) for d in counts.shape)} grid of {grid_ps:.2f} ps "
        f"cells spanning +/-{support_ps:.0f} ps - "
        f"{counts.nbytes / 1024**2:.0f} MB, no feather"
        + (f"; lags {lags} < < <\n" if subtract_background else " < < <\n")
    )

    for i, fp in enumerate(
        tqdm(files[start:], desc=tag or "delta_counts"), start=start + 1
    ):
        ts, _ = unpack(fp, nframes, compute_time_series=True)
        pixel_ts = _structure_pixel_timestamps(ts, all_pixels, time_lut)
        if subtract_background:
            for li, lag in enumerate(lags):
                deltas = bs.compute_lagged_differences(
                    pixel_ts, pixels, delta_window, mode, frame_lag=lag
                )
                seen, out = _accumulate_delta_counts(
                    counts[li], deltas, rows, grid_ps=grid,
                    zero_index=zero_index,
                )
                total += seen
                outside += out
        else:
            # Left exactly as it was: 'compute_lagged_differences' at lag 0 is
            # tested to be identical to this, but the plain path has no reason
            # to route through it.
            deltas = cd.calculate_differences(
                pixel_ts, pixels, delta_window=delta_window, mode=mode
            )
            seen, out = _accumulate_delta_counts(
                counts, deltas, rows, grid_ps=grid, zero_index=zero_index
            )
            total += seen
            outside += out
        if i % flush_every == 0:
            _write_counts_checkpoint(counts_path, counts, _meta(i, False))

    _write_counts_checkpoint(counts_path, counts, _meta(len(files), True))

    print(
        f"\n> > > {total} difference(s) across {len(labels)} pixel-pair "
        f"row(s)"
        + (f" and {len(lags)} lag plane(s)" if subtract_background else "")
        + f" binned onto {n_bins} cells; {outside} outside "
        f"+/-{support_ps:.0f} ps. Saved to {counts_path} "
        f"(unit '{unit}') < < <"
    )
    if subtract_background:
        per_lag = counts.sum(axis=(1, 2))
        print(
            f"  lag 0: {per_lag[0]}; shifted lags: {per_lag[1:].mean():.0f} on "
            f"average -> a raw excess of {per_lag[0] - per_lag[1:].mean():.0f} "
            "over the whole grid."
        )
    if outside:
        print(
            f"  NOTE: {100 * outside / max(total, 1):.2f}% of differences "
            "fell outside the grid and are counted but not binned. Raise "
            "support_ps to keep them."
        )
    return counts_path


def load_delta_counts(
    path: str | None = None,
    *,
    counts_path: str | None = None,
    pairs: Sequence[str] | None = None,
    pool: bool = True,
    time_unit_ps: float = _TS_CODE_PS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a saved counts grid and its cell centres.

    Parameters
    ----------
    path : str | None, optional
        Data folder whose ``processed/`` holds the artifact. May be None when
        ``counts_path`` is given.
    counts_path : str | None, optional
        Explicit '.npy' to load, ignoring ``path``. The default is None.
    pairs : Sequence[str] | None, optional
        Restrict to these pixel-pair labels. The default is None (all).
    pool : bool, optional
        Sum the selected pairs into one histogram. The default is True; pass
        False to keep the pair axis.
    time_unit_ps : float, optional
        Picoseconds per raw TDC code, used to place the cell centres of an
        *uncalibrated* (``'code'``) grid. The default is ~77 ps. Ignored for a
        calibrated grid, whose cells are already in ps.

    Returns
    -------
    counts : np.ndarray
        Pooled counts, or per-pair counts when ``pool=False``. An artifact
        written with ``subtract_background`` keeps its leading lag axis, so
        the shape is ``(n_bins)`` / ``(n_pairs, n_bins)`` without lags and
        ``(n_lags, n_bins)`` / ``(n_lags, n_pairs, n_bins)`` with them - lag 0
        first.
    centers : np.ndarray
        Grid cell centres, in **picoseconds** whatever the stored unit.
    info : dict
        The sidecar, plus ``name``, ``root``, ``counts_path``, the selected
        ``labels``, the effective ``cell_ps`` and any requested pairs that
        were ``missing``. A lagged artifact's sidecar also carries ``lags``
        and ``frames_used``, which is what
        'background_subtraction.subtract_background' needs.

    Raises
    ------
    ValueError
        Raised when neither ``path`` nor ``counts_path`` is given.
    FileNotFoundError
        Raised when the artifact does not exist.
    """
    if counts_path is None:
        if path is None:
            raise ValueError("Provide either path or counts_path.")
        name = os.path.basename(os.path.normpath(path))
        counts_path = os.path.join(
            store.processed_dir(path, create=False), f"{name}{_COUNTS_SUFFIX}"
        )
    counts_path = os.path.abspath(counts_path)
    counts, meta = store.load_map(npy_path=counts_path)

    name = os.path.basename(counts_path).removesuffix(_COUNTS_SUFFIX)
    root = os.path.dirname(os.path.dirname(counts_path))

    # The pair axis is the last but one, whether or not a lag axis precedes it.
    pair_axis = counts.ndim - 2

    labels = list(meta.get("labels", []))
    missing: list[str] = []
    if pairs is not None:
        wanted = [p for p in pairs if p in labels]
        missing = [p for p in pairs if p not in labels]
        counts = np.take(
            counts, [labels.index(p) for p in wanted], axis=pair_axis
        )
        labels = wanted

    zero_index = int(meta.get("zero_index", (counts.shape[-1] - 1) // 2))
    grid_native = float(meta.get("grid_native", meta.get("grid_ps", 1.0)))
    # A code grid's cells are codes; scale them to ps here rather than at save
    # time, so a better ps-per-code can be applied without recomputing.
    cell_ps = (
        grid_native
        if meta.get("unit", "ps") == "ps"
        else grid_native * time_unit_ps
    )
    centers = (np.arange(counts.shape[-1]) - zero_index) * cell_ps

    if pool:
        counts = counts.sum(axis=pair_axis)

    info = {
        **meta,
        "name": name,
        "root": root,
        "counts_path": counts_path,
        "labels": labels,
        "missing": missing,
        "cell_ps": cell_ps,
    }
    if not meta.get("complete", True):
        print(
            f"  NOTE: {os.path.basename(counts_path)} is incomplete - "
            f"{meta.get('files_done')} of {meta.get('n_files')} file(s)."
        )
    return counts, centers, info


def rebin_delta_counts(
    counts: np.ndarray,
    centers: np.ndarray,
    *,
    bin_width_ps: float,
    plot_window_ps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a native counts grid onto a coarser histogram, in memory.

    Each grid cell is assigned whole to the target bin its *centre* falls in.
    That is exact whenever ``bin_width_ps`` is a whole multiple of the grid
    cell and the two lattices line up; otherwise a cell straddling a target
    edge goes entirely one way, which is the rebinning, not the data. Bin on a
    multiple of the grid when comparing against the feather path.

    Parameters
    ----------
    counts : np.ndarray
        1D pooled counts on the native grid, from 'load_delta_counts'.
    centers : np.ndarray
        The grid cell centres in ps, from 'load_delta_counts'.
    bin_width_ps : float
        Target bin width, in ps.
    plot_window_ps : float
        Target half-range, in ps.

    Returns
    -------
    counts : np.ndarray
        Counts per target bin.
    edges : np.ndarray
        The target bin edges - the same '_delta_hist_edges' the feather path
        uses, so the two histograms are directly comparable.
    """
    edges = _delta_hist_edges(bin_width_ps, plot_window_ps)
    counts = np.asarray(counts).ravel()
    idx = np.searchsorted(edges, centers, side="right") - 1
    inside = (idx >= 0) & (idx < len(edges) - 1)
    return (
        np.bincount(
            idx[inside], weights=counts[inside], minlength=len(edges) - 1
        ).astype(np.int64),
        edges,
    )


def _check_hist_lengths(x: np.ndarray, y: np.ndarray) -> None:
    """Reject a centres/counts pair whose lengths disagree.

    Every histogram builder here returns bin *edges* -
    'compute_and_save_delta_histogram' and 'rebin_delta_counts' both do, and so
    does '_delta_hist_edges' - while the fitters take bin *centres*. Passing
    edges straight through is the natural mistake, and unguarded it surfaces
    much later as an ``IndexError`` from a boolean mask inside the seeding code,
    which says nothing about the actual cause. ``len(x) == len(y) + 1`` is a
    positive fingerprint of it, so it gets named.

    Raises
    ------
    ValueError
        Raised when the two lengths differ.
    """
    if x.shape[0] == y.shape[0]:
        return
    hint = (
        "  - looks like bin EDGES were passed where centres are expected; "
        "convert with 0.5 * (edges[:-1] + edges[1:])."
        if x.shape[0] == y.shape[0] + 1
        else ""
    )
    raise ValueError(
        f"bin_centers has {x.shape[0]} entries but counts has {y.shape[0]}."
        + (f"\n{hint}" if hint else "")
    )


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


def two_gaussians_on_triangle(
    x: np.ndarray,
    amp_n: float,
    mu: float,
    sigma_n: float,
    amp_b: float,
    dmu: float,
    sigma_ratio: float,
    bkg: float,
    half_base: float,
) -> np.ndarray:
    """Narrow + broad Gaussian on a triangular background.

    A single Gaussian fitted to the ORT coincidence peak reports the
    area-weighted mixture of two physically distinct timing populations, which
    is why it lands near 500-700 ps while the designer's focused-laser number
    is tens of ps:

    * **narrow** - photons absorbed *inside* the SPAD depletion region, timed
      by the avalanche build-up alone (tens of ps, at or below this TDC's
      ~77 ps LSB, so it is resolution-limited here);
    * **broad** - photons absorbed in the field-free epi/substrate *below* the
      junction, which reach the multiplication region by diffusion. The true
      shape of that component is one-sided exponential, not Gaussian; a
      Gaussian is the cheap two-parameter stand-in that separates the *scales*
      without committing to the tail shape. If the residuals stay structured
      in the flanks, that is the signal to move to an exponentially modified
      Gaussian instead.

    The broad width is parameterised as a **ratio** rather than an absolute
    sigma so the two components cannot swap roles mid-fit: ``sigma_ratio >= 1``
    makes ``sigma_n`` the narrow one by construction. ``sigma_ratio = 1``
    collapses the model to a single Gaussian, so a data set that does not need
    two components degenerates gracefully instead of fitting noise.

    Parameters
    ----------
    x : np.ndarray
        Input positions (timestamp differences, same units as ``half_base``).
    amp_n : float
        Amplitude of the narrow component above the triangle.
    mu : float
        Centre of the narrow component.
    sigma_n : float
        Standard deviation of the narrow component.
    amp_b : float
        Amplitude of the broad component above the triangle.
    dmu : float
        Centre of the broad component *relative to* ``mu``. Free because a
        diffusion tail is one-sided: with unequal signal/idler wavelengths the
        two arms' tails differ and the broad component sits off-centre.
    sigma_ratio : float
        ``sigma_broad / sigma_n``, constrained to ``>= 1`` by the fitter.
    bkg : float
        Triangle height at its apex (``x = 0``).
    half_base : float
        Half-width of the triangle base; the background is zero for
        ``abs(x) >= half_base`` (~one oscillator period, ~100 ns).

    Returns
    -------
    np.ndarray
        Model values: narrow Gaussian + broad Gaussian + ``bkg * tri(x)``,
        with ``tri(x) = max(0, 1 - abs(x) / half_base)``.
    """
    sigma_b = sigma_n * sigma_ratio
    tri = np.clip(1.0 - np.abs(x) / half_base, 0.0, None)
    narrow = amp_n * np.exp(-((x - mu) ** 2) / (2.0 * sigma_n**2))
    broad = amp_b * np.exp(-((x - mu - dmu) ** 2) / (2.0 * sigma_b**2))
    return narrow + broad + bkg * tri


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

    Raises
    ------
    ValueError
        Raised when ``bin_centers`` and ``counts`` differ in length.
    """
    x = np.asarray(bin_centers, dtype=np.float64)
    y = np.asarray(counts, dtype=np.float64)
    _check_hist_lengths(x, y)

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


def fit_two_gaussians_on_triangle(
    bin_centers: np.ndarray,
    counts: np.ndarray,
    half_base: float,
    fit_half_base: bool = True,
    poisson_weights: bool = True,
) -> dict:
    """Fit a narrow + broad Gaussian on a triangular background.

    Separates the two timing populations a single Gaussian averages together
    (see 'two_gaussians_on_triangle'), so the fast avalanche core can be
    compared against a focused-laser jitter number and the slow diffusion
    component quantified instead of inflating one sigma.

    Seeded from 'fit_gaussian_on_triangle': its sigma is the area-weighted
    mixture of both components, which makes it a sound seed for the broad one
    (it carries most of the area) and the narrow component is then seeded from
    what is left over in the central few bins. Fitting eight free parameters
    from cold seeds does not converge reliably; from these it does.

    Parameters
    ----------
    bin_centers : np.ndarray
        Histogram bin centres. Should span well past the peak so the triangle
        is constrained - a window of a few ns cannot separate the components.
    counts : np.ndarray
        Histogram counts.
    half_base : float
        Initial triangle half-base (~one oscillator period; ~100 ns in ps, or
        ~1300 in code units). Refined when ``fit_half_base`` is True.
    fit_half_base : bool, optional
        Let ``half_base`` float in the fit. The default is True.
    poisson_weights : bool, optional
        Weight each bin by ``1 / sqrt(max(counts, 1))``. The default is True,
        and it matters more here than for the single-Gaussian fits: the narrow
        core is a small fraction of the total counts, so unweighted least
        squares is dominated by the triangle and can leave the core badly
        fitted while still reporting convergence. Set False to reproduce the
        unweighted behaviour of the other fitters.

    Returns
    -------
    dict
        ``amp_n``, ``mu``, ``sigma_n``, ``sigma_n_err``, ``amp_b``, ``dmu``,
        ``sigma_b``, ``sigma_b_err``, ``sigma_ratio``, ``bkg``, ``half_base``,
        ``frac_narrow`` (narrow share of the total peak area), ``fwhm_n``,
        ``fwhm_b`` and ``ok``. ``sigma_ratio`` at its lower bound of 1.0 means
        the data did not support two components.

    Raises
    ------
    ValueError
        Raised when ``bin_centers`` and ``counts`` differ in length.
    """
    x = np.asarray(bin_centers, dtype=np.float64)
    y = np.asarray(counts, dtype=np.float64)
    _check_hist_lengths(x, y)

    result = {
        "amp_n": np.nan,
        "mu": 0.0,
        "sigma_n": np.nan,
        "sigma_n_err": np.nan,
        "amp_b": np.nan,
        "dmu": 0.0,
        "sigma_b": np.nan,
        "sigma_b_err": np.nan,
        "sigma_ratio": np.nan,
        "bkg": np.nan,
        "half_base": half_base,
        "frac_narrow": np.nan,
        "fwhm_n": np.nan,
        "fwhm_b": np.nan,
        "ok": False,
    }
    if len(x) < 10:
        return result

    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0

    seed = fit_gaussian_on_triangle(
        x, y, half_base, fit_half_base=fit_half_base
    )
    if seed["ok"]:
        mu0 = seed["mu"]
        sigma_b0 = seed["sigma"]
        apex0 = seed["bkg"]
        hb0 = seed["half_base"]
        amp_b0 = seed["amp"]
    else:
        mu0 = float(x[np.argmax(y)])
        sigma_b0 = 10.0 * dx
        apex0 = float(np.median(y))
        hb0 = half_base
        amp_b0 = max(float(np.max(y)) - apex0, 1.0)

    # Narrow seed: whatever the one-Gaussian fit failed to account for in the
    # central few bins. That under-shoot at x=0 is the whole reason for this
    # model, so it is exactly the right thing to seed the second component on.
    resid = y - gaussian_plus_triangle(x, amp_b0, mu0, sigma_b0, apex0, hb0)
    core = np.abs(x - mu0) <= 3.0 * dx
    amp_n0 = float(np.max(resid[core])) if np.any(core) else 0.0
    amp_n0 = max(amp_n0, 0.05 * amp_b0)
    # The core is expected at or below the TDC LSB, so seed it a couple of bins
    # wide rather than anywhere near the mixture sigma.
    sigma_n0 = max(1.5 * dx, 1e-3 * sigma_b0)
    ratio0 = float(np.clip(sigma_b0 / sigma_n0, 1.05, 400.0))

    lo = [
        0.0,
        x.min(),
        dx / 10.0,
        0.0,
        -0.05 * half_base,
        1.0,
        0.0,
        0.5 * half_base,
    ]
    hi = [
        np.inf,
        x.max(),
        0.5 * half_base,
        np.inf,
        0.05 * half_base,
        500.0,
        np.inf,
        2.0 * half_base if fit_half_base else half_base * 1.0001,
    ]
    if not fit_half_base:
        lo[7] = half_base * 0.9999

    p0 = [amp_n0, mu0, sigma_n0, amp_b0, 0.0, ratio0, apex0, hb0]
    p0 = [
        min(
            max(v, lo_i + 1e-9 * (abs(lo_i) + 1)),
            hi_i - 1e-9 * (abs(hi_i) + 1),
        )
        for v, lo_i, hi_i in zip(p0, lo, hi, strict=True)
    ]

    kwargs = {}
    if poisson_weights:
        kwargs["sigma"] = np.sqrt(np.maximum(y, 1.0))
        kwargs["absolute_sigma"] = False

    try:
        popt, pcov = curve_fit(
            two_gaussians_on_triangle,
            x,
            y,
            p0=p0,
            bounds=(lo, hi),
            maxfev=40000,
            **kwargs,
        )
    except (RuntimeError, ValueError):
        return result

    amp_n, mu, sigma_n, amp_b, dmu, ratio, bkg, hb = popt
    if not np.isfinite(sigma_n) or sigma_n <= 0 or (amp_n <= 0 and amp_b <= 0):
        return result

    sigma_n = abs(sigma_n)
    sigma_b = sigma_n * ratio

    finite_cov = np.all(np.isfinite(pcov))
    sigma_n_err = float(np.sqrt(pcov[2, 2])) if finite_cov else np.nan
    if finite_cov:
        # sigma_b = sigma_n * ratio, so both parameters and their covariance
        # enter; taking sqrt(pcov[5,5]) alone would understate it.
        var_b = (
            ratio**2 * pcov[2, 2]
            + sigma_n**2 * pcov[5, 5]
            + 2.0 * ratio * sigma_n * pcov[2, 5]
        )
        sigma_b_err = float(np.sqrt(var_b)) if var_b > 0 else np.nan
    else:
        sigma_b_err = np.nan

    area_n = max(amp_n, 0.0) * sigma_n
    area_b = max(amp_b, 0.0) * sigma_b
    frac_narrow = (
        float(area_n / (area_n + area_b)) if (area_n + area_b) > 0 else np.nan
    )

    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    result.update(
        amp_n=float(amp_n),
        mu=float(mu),
        sigma_n=float(sigma_n),
        sigma_n_err=sigma_n_err,
        amp_b=float(amp_b),
        dmu=float(dmu),
        sigma_b=float(sigma_b),
        sigma_b_err=sigma_b_err,
        sigma_ratio=float(ratio),
        bkg=float(bkg),
        half_base=float(hb),
        frac_narrow=frac_narrow,
        fwhm_n=float(fwhm * sigma_n),
        fwhm_b=float(fwhm * sigma_b),
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
    reuse_histogram: bool = True,
    subtract_background: bool = False,
    peak_window_ps: float = 1000.0,
    wide_rebin: int = 25,
    support_ps: float = 100000.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Histogram the saved delta-t feather, fit the peak, and plot it.

    Pools the selected pair columns of a saved feather, converts to
    picoseconds (per the ``delta_unit`` metadata), histograms, fits the
    coincidence peak and saves the figure. Old single-column feathers
    (``delta_code`` / ``delta_ps``) are still read.

    The feather is **streamed** through 'compute_and_save_delta_histogram',
    one Arrow record batch at a time, and never held in memory - a full run's
    combined feather is far too large to load. The counts are identical to
    histogramming the whole pooled array at once; only the memory profile
    differs. The histogram is cached in ``processed/``, so re-fitting or
    changing the background model does not re-read the feather.

    Locate the feather either from ``path`` (its ``processed/``) or directly
    via ``feather_path``, which ignores ``path`` entirely - that is how a
    combined feather is plotted. Either may point at a ``*_delta_t_parts``
    folder to plot straight from the part files.

    With ``subtract_background`` the histogram that gets fitted is the *lag-0
    minus shifted-frames* residual: the accidental pedestal is measured from
    frame-shifted pairs and subtracted, rather than modelled. All the lags are
    histogrammed in the same single streaming pass. Needs a feather written
    with ``calculate_and_save_timestamp_differences(...,
    subtract_background=True)``.

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
        Peak-and-background model. ``'flat'`` (default) fits
        ``Gaussian + constant`` over ``fit_window_ps`` — the legacy daplis
        model. ``'triangle'`` fits ``Gaussian + triangular background`` via
        'fit_gaussian_on_triangle' over the whole histogram, which is the
        physically correct model for the Kelpie ORT free-running phase and
        does not inflate sigma. ``'two_gaussians'`` adds a second, broader
        Gaussian on the same triangle via
        'fit_two_gaussians_on_triangle' — after TDC calibration the peak shows
        a narrow core on wide tails that one Gaussian averages into a single
        inflated sigma, and this separates the two. The plot then also draws
        both components and the bare triangle. For ``'triangle'`` and
        ``'two_gaussians'`` set ``plot_window_ps`` wide (e.g. the full
        ``osc_period_ps``) so the triangle is well constrained.
    osc_period_ps : float, optional
        Oscillator period in ps; equals the triangle half-base (the
        accidental background is a triangle peaking at 0 and reaching zero at
        ``+/- osc_period_ps``). The default is 100000.0 ps (~100 ns). Only
        used when ``background`` is ``'triangle'`` or ``'two_gaussians'``; the
        fit refines it.
    feather_path : str | None, optional
        Explicit feather to read and plot directly, or a ``*_delta_t_parts``
        folder to read the parts; when given, ``path`` is ignored and the
        raw-data folder is not needed (the plot is saved beside the feather).
        The default is None (derive from ``path``).
    pairs : Sequence[str] | None, optional
        Restrict pooling to these pixel-pair column names (e.g.
        ``["16,16-20,20"]``). The default is None (pool every pair column).
        Ignored for old single-column feathers.
    label : str, optional
        Short label for the title and output file (``'spdc'`` / ``'hbt'``).
        The default is ``'spdc'``.
    cmap_title : str | None, optional
        Optional override for the plot title. The default is None.
    reuse_histogram : bool, optional
        Reuse a saved histogram whose binning and sources match, instead of
        re-streaming the feather. The default is True. Pass False to force
        the full pass.
    subtract_background : bool, optional
        Subtract the frame-shifted accidentals before fitting, and write the
        three-panel diagnostic beside the fit. The default is False. Requires a
        feather written with ``subtract_background=True``, and only makes sense
        with ``background='flat'`` - there is no triangle left to fit. Note the
        streamed histogram then spans ``support_ps`` rather than
        ``plot_window_ps``, since the residual out at +/-100 ns is the evidence
        that the pedestal was accidental.
    peak_window_ps : float, optional
        Half-range the model-free coincidence count is summed over, in ps.
        ``subtract_background`` only. The default is 1000.0.
    wide_rebin : int, optional
        Histogram bins per bin of the diagnostic's two full-support panels.
        ``subtract_background`` only. The default is 25.
    support_ps : float, optional
        Half-range the lag histograms are streamed over, in ps.
        ``subtract_background`` only. The default is 100000.0 (one oscillator
        period, the whole physical support of a difference).

    Returns
    -------
    counts : np.ndarray
        Counts per bin - the histogram that was fitted and plotted, or the
        residual when ``subtract_background`` is set.
    centers : np.ndarray
        Bin centres in picoseconds, to re-fit without re-reading the feather.
    fit : dict
        The fit results plus ``fwhm`` (2.355 sigma), ``jitter_per_detector``
        (sigma / sqrt(2)), ``car`` (peak-to-background ratio) and ``n``. With
        ``subtract_background`` also ``model_free``, ``subtraction`` and
        ``subtraction_png``.

    Raises
    ------
    FileNotFoundError
        Raised when the delta-t '.feather' cannot be found.
    ValueError
        Raised when ``subtract_background`` is asked for on a feather with no
        frame-shifted columns, or alongside a non-flat background model.
    """
    if subtract_background:
        _check_subtraction_model(background)

    # The subtraction needs the whole support in one histogram: the zoom is a
    # slice of it and the diagnostic's wide panels are a regrouping, so the
    # expensive streaming pass is still paid exactly once.
    hist_window_ps = (
        _whole_bins(support_ps, bin_width_ps)
        if subtract_background
        else plot_window_ps
    )
    counts, edges, info = compute_and_save_delta_histogram(
        path,
        feather_path=feather_path,
        time_unit_ps=time_unit_ps,
        bin_width_ps=bin_width_ps,
        plot_window_ps=hist_window_ps,
        pairs=pairs,
        subtract_background=subtract_background,
        reuse=reuse_histogram,
    )
    name = info["name"]
    results_dir = store.results_dir(info["root"], _KIND, create=False)
    centers = (edges[:-1] + edges[1:]) / 2
    n_in_window = int(info["n"])

    # Comb guard: warn if the data granularity is coarser than the bin.
    gran = info.get("granularity_ps")
    if gran is not None and gran > bin_width_ps * 1.5:
        print(
            f"  WARNING: delta granularity is ~{gran:.0f} ps but "
            f"bin_width={bin_width_ps:.0f} ps -> the histogram will be a "
            "comb. Bin coarser, or ensure the fine timing is populated "
            "in the acquisition."
        )

    sub = None
    if subtract_background:
        sub = _subtract_lag_background(
            counts,
            centers,
            info["frames_used"],
            name=name,
            root=info["root"],
            background=background,
            bin_width_ps=bin_width_ps,
            zoom_window_ps=plot_window_ps,
            peak_window_ps=peak_window_ps,
            wide_rebin=wide_rebin,
            label=label,
        )
        counts = sub["residual"]
        centers = sub["centers"]
        # The model-free count, not the bin total: the residual's bins are
        # differences of two histograms, so summing them *is* the coincidence
        # number and anything outside the peak is noise around zero.
        n_in_window = int(round(sub["model_free"]["n"]))

    counts, centers, fit = _fit_and_plot_delta(
        counts,
        centers,
        name=name,
        results_dir=results_dir,
        bin_width_ps=bin_width_ps,
        plot_window_ps=plot_window_ps,
        fit_window_ps=fit_window_ps,
        background=background,
        osc_period_ps=osc_period_ps,
        label=label,
        cmap_title=cmap_title,
        n_in_window=n_in_window,
        stem="delta_t_sub" if subtract_background else "delta_t",
        subtracted=subtract_background,
    )
    if sub is not None:
        fit["model_free"] = sub["model_free"]
        fit["subtraction"] = sub
        fit["subtraction_png"] = sub["png_path"]
    return counts, centers, fit


def _fit_and_plot_delta(
    counts: np.ndarray,
    centers: np.ndarray,
    *,
    name: str,
    results_dir: str,
    bin_width_ps: float,
    plot_window_ps: float,
    fit_window_ps: float | None,
    background: str,
    osc_period_ps: float,
    label: str,
    cmap_title: str | None,
    n_in_window: int,
    stem: str,
    subtracted: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit the coincidence peak on a histogram, plot it, and save the figure.

    Shared by both stage-2 drivers *on purpose*: the feather path and the
    counts path are only comparable if the fitting and the drawing are
    literally the same code, so that any difference between their results is
    the histogram and nothing else.

    Parameters
    ----------
    counts, centers : np.ndarray
        The histogram to fit, and its bin centres in ps.
    name : str
        Dataset name, for the title and file name.
    results_dir : str
        Folder the figure is written to; created if missing.
    bin_width_ps : float
        Bar width for the plot, in ps.
    plot_window_ps : float
        Half-range the model curve is drawn over, in ps.
    fit_window_ps : float | None
        Fit half-range around the tallest bin; ``'flat'`` background only.
    background : str
        ``'flat'``, ``'triangle'`` or ``'two_gaussians'``; see
        'collect_and_plot_timestamp_differences'.
    osc_period_ps : float
        Triangle half-base seed, in ps.
    label : str
        Short label for the title and file name.
    cmap_title : str | None
        Optional title override.
    n_in_window : int
        Differences the histogram was built from, for the legend and
        ``fit['n']``.
    stem : str
        File-name fragment identifying which path drew this
        (``'delta_t'`` / ``'delta_counts'``), so both figures coexist.
    subtracted : bool, optional
        Whether ``counts`` is a background-subtracted residual. The default is
        False. When True the fitted baseline is ~0 by construction, so ``car``
        (a peak-to-baseline *ratio*) is meaningless and is neither computed nor
        printed - use ``integrate_residual``'s window CAR instead.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        ``(counts, centers, fit)`` - what both drivers return.

    Raises
    ------
    ValueError
        Raised on an unknown ``background``.
    """
    if background not in ("flat", "triangle", "two_gaussians"):
        raise ValueError(
            "background must be 'flat', 'triangle' or 'two_gaussians', got "
            f"{background!r}"
        )
    if background == "two_gaussians":
        fit = fit_two_gaussians_on_triangle(
            centers, counts, half_base=osc_period_ps
        )
        # Present the narrow component as *the* sigma: it is the one that
        # carries the timing resolution, and it keeps the reported keys, the
        # console line and the returned contract identical across models.
        # 'amp' becomes the combined peak height above the triangle at mu, so
        # CAR stays the peak-to-accidental ratio it is for the other two.
        if fit["ok"]:
            fit["sigma"] = fit["sigma_n"]
            fit["sigma_err"] = fit["sigma_n_err"]
            fit["amp"] = fit["amp_n"] + fit["amp_b"] * np.exp(
                -(fit["dmu"] ** 2) / (2.0 * fit["sigma_b"] ** 2)
            )
        else:
            fit["sigma"] = np.nan
            fit["sigma_err"] = np.nan
            fit["amp"] = np.nan
    elif background == "triangle":
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
    # CAR = peak-to-accidental ratio at delta=0 (apex for the triangle). On a
    # subtracted residual the baseline is zero by construction, so the ratio is
    # a division by noise - it came out as 1e36 on real data. Left as NaN
    # rather than computed and disbelieved.
    if subtracted:
        fit["car"] = np.nan
    else:
        fit["car"] = (
            (fit["amp"] + fit["bkg"]) / fit["bkg"] if fit["bkg"] else np.inf
        )
    fit["n"] = n_in_window

    fig, ax = plt.subplots()
    ax.bar(
        centers,
        counts,
        width=bin_width_ps,
        # alpha=0.6,
        label=f"data ({n_in_window} pairs)",
    )
    if fit["ok"]:
        xs = np.linspace(-plot_window_ps, plot_window_ps, 4000)
        if background == "two_gaussians":
            model = two_gaussians_on_triangle(
                xs,
                fit["amp_n"],
                fit["mu"],
                fit["sigma_n"],
                fit["amp_b"],
                fit["dmu"],
                fit["sigma_ratio"],
                fit["bkg"],
                fit["half_base"],
            )
            # Same 'name = value unit' notation, same quantities and same
            # order as the one-Gaussian legend, just doubled. The _n / _b
            # suffixes are the keys of the returned fit dict, so a number read
            # off the figure can be traced straight back to it.
            fit_label = (
                f"sigma_n = {fit['sigma_n']:.0f} ps\n"
                f"sigma_b = {fit['sigma_b']:.0f} ps\n"
                f"FWHM_n = {fit['fwhm_n']:.0f} ps\n"
                f"FWHM_b = {fit['fwhm_b']:.0f} ps\n"
                f"per-detector_n = {fit['jitter_per_detector']:.0f} ps\n"
                f"narrow area = {100 * fit['frac_narrow']:.1f} %"
            )
        elif background == "triangle":
            model = gaussian_plus_triangle(
                xs, fit["amp"], fit["mu"], sigma, fit["bkg"], fit["half_base"]
            )
            fit_label = (
                f"sigma = {sigma:.0f} ps\n"
                f"FWHM = {fit['fwhm']:.0f} ps\n"
                f"per-detector = {fit['jitter_per_detector']:.0f} ps"
            )
        else:
            model = gaussian(xs, fit["amp"], fit["mu"], sigma, fit["bkg"])
            fit_label = (
                f"sigma = {sigma:.0f} ps\n"
                f"FWHM = {fit['fwhm']:.0f} ps\n"
                f"per-detector = {fit['jitter_per_detector']:.0f} ps"
            )
        ax.plot(xs, model, "r-", linewidth=2, label=fit_label)
        if background == "two_gaussians":
            # The components are the point of this model - a total curve alone
            # cannot show whether the narrow one is real or is just soaking up
            # a cusp the two Gaussians together cannot make.
            tri = fit["bkg"] * np.clip(
                1.0 - np.abs(xs) / fit["half_base"], 0.0, None
            )
            ax.plot(
                xs,
                tri
                + fit["amp_n"]
                * np.exp(
                    -((xs - fit["mu"]) ** 2) / (2.0 * fit["sigma_n"] ** 2)
                ),
                "-",
                color="darkorange",
                linewidth=1.0,
                label="narrow + triangle",
            )
            ax.plot(
                xs,
                tri
                + fit["amp_b"]
                * np.exp(
                    -((xs - fit["mu"] - fit["dmu"]) ** 2)
                    / (2.0 * fit["sigma_b"] ** 2)
                ),
                "--",
                color="darkorange",
                linewidth=1.0,
                label="broad + triangle",
            )
            ax.plot(
                xs, tri, ":", color="k", linewidth=1.0, label="ORT triangle"
            )
    ax.set_xlabel("Timestamp difference  (ps)")
    ax.set_ylabel(
        "Coincidences - accidentals" if subtracted else "Coincidences"
    )
    ax.set_title(cmap_title or f"{label.upper()} coincidence peak  ({name})")
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend()
    # fig.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    out_png = os.path.join(results_dir, f"{name}_{label}_{stem}.png")
    fig.savefig(out_png)
    # plt.close(fig)
    print(
        f"\n> > > Plot saved as {os.path.basename(out_png)} in {results_dir}"
    )
    if fit["ok"]:
        print(
            f"  peak: mu {fit['mu']:.0f} ps  sigma {sigma:.0f} ps  "
            f"FWHM {fit['fwhm']:.0f} ps  per-detector "
            f"{fit['jitter_per_detector']:.0f} ps"
            + ("" if subtracted else f"  peak/bkg {fit['car']:.2f}")
        )
        if background == "two_gaussians":
            print(
                f"  narrow: sigma {fit['sigma_n']:.0f} +- "
                f"{fit['sigma_n_err']:.0f} ps  FWHM {fit['fwhm_n']:.0f} ps  "
                f"area {100 * fit['frac_narrow']:.1f} %\n"
                f"  broad:  sigma {fit['sigma_b']:.0f} +- "
                f"{fit['sigma_b_err']:.0f} ps  FWHM {fit['fwhm_b']:.0f} ps  "
                f"offset {fit['dmu']:+.0f} ps\n"
                f"  ratio broad/narrow {fit['sigma_ratio']:.1f}"
                "   (1.0 => the data did not support two components)"
            )
    else:
        print(
            "  no Gaussian peak fit — check blob selection / that the run "
            "has proper (stopped, fine) timing."
        )

    return counts, centers, fit


def _snap_binning(
    bin_width_ps: float, plot_window_ps: float, cell_ps: float
) -> tuple[float, float]:
    """Round a binning onto the counts grid, and say so if it moved.

    Three roundings, each fixing a distinct artifact:

    * the **width** to a whole number of cells, so every bin holds the same
      number of them. Otherwise the count alternates (9, 10, 9, 10 ...) and a
      smooth distribution acquires a sawtooth of ``1 / n_cells``;
    * that number to an **odd** one, which is what makes the rebin exact.
      Bins are centred on zero, so their edges sit at half-integer multiples
      of the width; an odd width in cells puts those edges on cell
      *boundaries*, an even one puts them on cell *centres* - bisecting a
      cell at every boundary and forcing its whole population one way. On a
      real run that cost ~1% of the counts, against exactly zero for an odd
      width;
    * the **window** to a whole number of bins, so zero lands on a bin centre
      rather than up to half a bin off it.

    Returns
    -------
    tuple[float, float]
        The snapped ``(bin_width_ps, plot_window_ps)``.
    """
    raw = bin_width_ps / cell_ps
    cells = max(int(round(raw)), 1)
    if cells % 2 == 0:
        lower, upper = cells - 1, cells + 1
        cells = (
            upper
            if lower < 1 or abs(upper - raw) <= abs(raw - lower)
            else lower
        )
    width = cells * cell_ps
    bins = max(int(round(plot_window_ps / width)), 1)
    window = bins * width

    moved = []
    if abs(width - bin_width_ps) > 1e-9:
        moved.append(
            f"bin_width {bin_width_ps:.2f} -> {width:.2f} ps "
            f"({cells} cells, odd so bin edges fall between cells)"
        )
    if abs(window - plot_window_ps) > 1e-9:
        moved.append(
            f"window {plot_window_ps:.0f} -> {window:.0f} ps ({bins} bins)"
        )
    if moved:
        print(
            f"  snapped to the {cell_ps:.2f} ps grid: {'; '.join(moved)}. "
            "Every bin now holds the same number of cells and zero is a bin "
            "centre; pass snap_to_grid=False to take the numbers literally."
        )
    return width, window


def collect_and_plot_delta_counts(
    path: str | None = None,
    *,
    counts_path: str | None = None,
    time_unit_ps: float = _TS_CODE_PS,
    bin_width_ps: float = _TS_CODE_PS,
    plot_window_ps: float = 3000.0,
    fit_window_ps: float | None = 1000.0,
    background: str = "flat",
    osc_period_ps: float = 100000.0,
    pairs: Sequence[str] | None = None,
    label: str = "spdc",
    cmap_title: str | None = None,
    snap_to_grid: bool = True,
    subtract_background: bool = False,
    peak_window_ps: float = 1000.0,
    wide_rebin: int = 25,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rebin a saved counts grid, fit the peak, and plot it.

    The counterpart of 'collect_and_plot_timestamp_differences' for the
    feather-free path, and deliberately the same function underneath: only the
    histogram differs, so the two can be compared directly.

    With ``subtract_background`` the histogram that gets fitted is the *lag-0
    minus shifted-frames* residual instead of the raw one - same fit, same
    figure, on a distribution whose accidental pedestal has been measured away
    rather than modelled. Needs an artifact written with
    ``calculate_and_save_delta_counts(..., subtract_background=True)``.

    Nothing is read from disk except one ~100 MB array, and the rebinning is
    in memory, so changing ``bin_width_ps`` or ``plot_window_ps`` costs
    milliseconds rather than a full pass over the data - and every pixel pair
    is available at once instead of whichever columns were streamed.

    Parameters
    ----------
    path : str | None, optional
        Data folder whose ``processed/`` holds the counts artifact and whose
        ``results/`` receives the plot. May be None when ``counts_path`` is
        given.
    counts_path : str | None, optional
        Explicit counts '.npy', ignoring ``path``. The default is None.
    time_unit_ps : float, optional
        Picoseconds per raw TDC code, for an uncalibrated grid. The default
        is ~77 ps.
    bin_width_ps : float, optional
        Histogram bin width in ps. The default is ~77 ps (one code). Use a
        whole multiple of the artifact's grid cell for an exact rebin.
    plot_window_ps : float, optional
        Histogram/plot half-range in ps around zero. The default is 3000.0.
        Unlike the feather path, widening this is free - it is a slice of what
        is already loaded.
    fit_window_ps : float | None, optional
        Fit half-range in ps around the tallest bin. ``'flat'`` background
        only. The default is 1000.0.
    background : str, optional
        ``'flat'`` (default), ``'triangle'`` or ``'two_gaussians'``; see
        'collect_and_plot_timestamp_differences'.
    osc_period_ps : float, optional
        Triangle half-base seed in ps. The default is 100000.0.
    pairs : Sequence[str] | None, optional
        Restrict pooling to these pixel-pair labels. The default is None
        (pool every pair).
    label : str, optional
        Short label for the title and output file. The default is ``'spdc'``.
    cmap_title : str | None, optional
        Optional override for the plot title. The default is None.
    snap_to_grid : bool, optional
        Round ``bin_width_ps`` to a whole number of grid cells and
        ``plot_window_ps`` to a whole number of bins. The default is True,
        and you almost always want it - see the note below. Pass False to
        take the numbers literally.
    subtract_background : bool, optional
        Subtract the shifted-frame accidentals before fitting. The default is
        False. Requires a lagged artifact, and only makes sense with
        ``background='flat'`` - there is no triangle left to fit. Writes a
        second, three-panel figure whose middle panel is the evidence that the
        pedestal really was accidental; read it before quoting anything.
    peak_window_ps : float, optional
        Half-range the model-free coincidence count is summed over, in ps.
        ``subtract_background`` only. The default is 1000.0.
    wide_rebin : int, optional
        Grid cells per bin in the two full-support panels of the diagnostic
        figure. ``subtract_background`` only. The default is 25, which smooths
        the triangle without hiding structure in the residual.

    Returns
    -------
    counts : np.ndarray
        Counts per bin - the rebinned histogram that was fitted and plotted,
        or the residual when ``subtract_background`` is set.
    centers : np.ndarray
        Bin centres in picoseconds.
    fit : dict
        As 'collect_and_plot_timestamp_differences'. With
        ``subtract_background`` it also carries ``model_free`` (the
        'background_subtraction.integrate_residual' summary), ``subtraction``
        (the residual, background and errors) and ``subtraction_png``.

    Raises
    ------
    FileNotFoundError
        Raised when the counts artifact cannot be found.
    ValueError
        Raised when ``subtract_background`` is asked for on an artifact with no
        lag planes, or alongside a non-flat background model.

    Notes
    -----
    Re-binning already-binned data onto an incommensurate width is not a
    rounding detail, it is a **ripple**. A 71 ps bin over a 7.7 ps grid holds
    9 or 10 cells depending on where it lands, so a perfectly smooth
    distribution comes out with an ~11% bin-to-bin sawtooth - on real data
    that moved the peak bin by 8%.

    With ``snap_to_grid`` the rebin is not merely close to the feather path's
    histogram, it is **identical to it**, because the target bins then hold a
    whole odd number of cells and their edges fall between cells rather than
    through them. Verified bin-for-bin on a 3.22 GB run: zero difference at 9,
    11 and 13 cells per bin, ~1% at 8, 10 and 12. See '_snap_binning'.

    The one place the two paths still disagree is the outermost bin at each
    end, where the feather path cuts the data at ``abs(delta) <= window`` but
    draws edges half a bin beyond it, so those bins can only ever be part
    filled. Here they are filled properly.
    """
    grid_counts, grid_centers, info = load_delta_counts(
        path,
        counts_path=counts_path,
        pairs=pairs,
        pool=True,
        time_unit_ps=time_unit_ps,
    )
    if info["missing"]:
        print(f"  no such pixel-pair row(s): {info['missing']}")

    cell_ps = float(info["cell_ps"])
    if snap_to_grid:
        bin_width_ps, plot_window_ps = _snap_binning(
            bin_width_ps, plot_window_ps, cell_ps
        )
    else:
        ratio = bin_width_ps / cell_ps
        if abs(ratio - round(ratio)) > 1e-9:
            print(
                f"  WARNING: bin_width={bin_width_ps:.2f} ps is "
                f"{ratio:.2f} grid cells, so bins hold {int(ratio)} or "
                f"{int(ratio) + 1} of them and the histogram carries a "
                f"~{100 / max(int(ratio), 1):.0f}% sawtooth. Leave "
                "snap_to_grid=True unless you know you want this."
            )

    sub = None
    if subtract_background:
        _check_subtraction_model(background)
        if grid_counts.ndim != 2 or not info.get("lags"):
            raise ValueError(
                "subtract_background=True needs a counts artifact written with "
                "lag planes. Recompute stage 1 with "
                "'calculate_and_save_delta_counts(..., "
                "subtract_background=True)'."
            )
        # Rebin the native grid once, over its whole support, and let the
        # helper slice the zoom out of it: one binning for the fit and the
        # diagnostic means the panels cannot disagree about a bin edge.
        lag_counts, lag_centers = _rebin_lag_planes(
            grid_counts,
            grid_centers,
            bin_width_ps=bin_width_ps,
            plot_window_ps=_whole_bins(
                float(np.abs(grid_centers).max()), bin_width_ps
            ),
        )
        sub = _subtract_lag_background(
            lag_counts,
            lag_centers,
            info["frames_used"],
            name=info["name"],
            root=info["root"],
            background=background,
            bin_width_ps=bin_width_ps,
            zoom_window_ps=plot_window_ps,
            peak_window_ps=peak_window_ps,
            wide_rebin=wide_rebin,
            label=label,
        )
        counts = sub["residual"]
        centers = sub["centers"]
        # The model-free count, not the bin total: the residual's bins are
        # differences of two histograms, so summing them *is* the coincidence
        # number and anything outside the peak is noise around zero.
        n_in_window = int(round(sub["model_free"]["n"]))
    else:
        counts, edges = rebin_delta_counts(
            grid_counts,
            grid_centers,
            bin_width_ps=bin_width_ps,
            plot_window_ps=plot_window_ps,
        )
        centers = (edges[:-1] + edges[1:]) / 2
        n_in_window = int(counts.sum())

        print(
            f"\n> > > {int(grid_counts.sum())} difference(s) on the "
            f"{grid_centers.size}-cell grid across {len(info['labels'])} pair "
            f"row(s); {n_in_window} inside +/-{plot_window_ps:.0f} ps, rebinned "
            f"to {counts.size} x {bin_width_ps:.0f} ps < < <"
        )

    counts, centers, fit = _fit_and_plot_delta(
        counts,
        centers,
        name=info["name"],
        results_dir=store.results_dir(info["root"], _KIND, create=False),
        bin_width_ps=bin_width_ps,
        plot_window_ps=plot_window_ps,
        fit_window_ps=fit_window_ps,
        background=background,
        osc_period_ps=osc_period_ps,
        label=label,
        cmap_title=cmap_title,
        n_in_window=n_in_window,
        stem="delta_counts_sub" if subtract_background else "delta_counts",
        subtracted=subtract_background,
    )
    if sub is not None:
        fit["model_free"] = sub["model_free"]
        fit["subtraction"] = sub
        fit["subtraction_png"] = sub["png_path"]
    return counts, centers, fit


def _subtract_lag_background(
    lag_counts: np.ndarray,
    centers: np.ndarray,
    frames_used: Sequence[int],
    *,
    name: str,
    root: str,
    background: str,
    bin_width_ps: float,
    zoom_window_ps: float,
    peak_window_ps: float,
    wide_rebin: int,
    label: str,
) -> dict:
    """Turn a lag-resolved histogram into the accidental-free residual.

    Shared by both stage-2 drivers *on purpose*, exactly as '_fit_and_plot_delta'
    is: the feather path and the counts path can only be compared if the
    subtraction, the reported numbers and the figure are literally the same
    code. Each caller supplies an ``(n_lags, n_bins)`` histogram already binned
    at ``bin_width_ps`` and spanning the whole support; the zoom is a *slice* of
    it and the diagnostic's wide panels are a regrouping, so no caller re-reads
    anything.

    Parameters
    ----------
    lag_counts : np.ndarray
        ``(n_lags, n_bins)`` counts over the full support, lag 0 first.
    centers : np.ndarray
        Bin centres in ps. Must be symmetric about a bin centred on zero, so
        the zoom slice lands on the same bins a direct rebin would.
    frames_used : Sequence[int]
        Frames each lag paired up, for the rescaling.
    name, root : str
        Dataset name and data-folder root, for the title and the figure path.
    background : str
        The caller's fit model; only ``'flat'`` is meaningful here.
    bin_width_ps : float
        Width of one bin of ``lag_counts``, in ps.
    zoom_window_ps : float
        Half-range kept for the fitted/returned residual.
    peak_window_ps : float
        Half-range the model-free count is summed over.
    wide_rebin : int
        Bins of ``lag_counts`` per bin of the two full-support panels.
    label : str
        Short measurement label.

    Returns
    -------
    dict
        'background_subtraction.subtract_background' over the zoom window, plus
        ``model_free``, ``full`` (the same over the whole support) and
        ``png_path``.

    Raises
    ------
    ValueError
        Raised when the histogram has no lag rows, or the fit model is not
        flat.
    """
    if lag_counts.ndim != 2 or lag_counts.shape[0] < 2:
        raise ValueError(
            "subtract_background=True needs a histogram with lag rows; got "
            f"shape {lag_counts.shape}. Recompute stage 1 with "
            "subtract_background=True."
        )
    _check_subtraction_model(background)

    keep = np.abs(centers) <= zoom_window_ps
    sub = bs.subtract_background(
        lag_counts[:, keep], centers[keep], frames_used
    )
    sub["model_free"] = bs.integrate_residual(sub, window_ps=peak_window_ps)

    # The whole support, coarsely binned: this is the evidence panel. A
    # residual that is not flat and zero out at +/-100 ns means the pedestal
    # was not purely accidental, and then no number above is worth quoting.
    step = max(int(wide_rebin), 1)
    if step % 2 == 0:
        # Bins are centred on zero, so an odd group keeps the coarse edges on
        # fine-bin boundaries; an even one bisects a bin at every edge.
        step += 1
    wide_counts, wide_centers = _rebin_lag_planes(
        lag_counts,
        centers,
        bin_width_ps=step * bin_width_ps,
        plot_window_ps=float(np.abs(centers).max()),
    )
    full = bs.subtract_background(wide_counts, wide_centers, frames_used)
    sub["full"] = full

    peak = sub["model_free"]
    print(
        f"\n> > > lag 0 {int(full['signal'].sum())} vs accidentals "
        f"{full['background'].sum():.0f} (mean of {full['k']} shifted lag(s)) "
        f"over the whole support < < <"
    )
    print(
        f"  coincidences within +/-{peak_window_ps:.0f} ps: {peak['n']:.0f} "
        f"+/- {peak['n_err']:.0f}  ({peak['significance']:.1f} sigma), on "
        f"{peak['n_background']:.0f} accidentals -> CAR {peak['car']:.3f}"
    )
    far = np.abs(full["centers"]) > max(5 * peak_window_ps, 15000.0)
    far_sum = float(full["residual"][far].sum())
    far_err = float(np.sqrt((full["error"][far] ** 2).sum()))
    print(
        f"  control, |delta| beyond the peak: {far_sum:.0f} +/- {far_err:.0f} "
        f"({far_sum / far_err if far_err else np.inf:+.1f} sigma from zero) - "
        "this must be ~0, or the pedestal is not purely accidental."
    )

    fig = bs.plot_background_subtraction(
        sub, full=full, name=name, label=label
    )
    sub["png_path"] = store.save_figure(
        fig,
        store.results_dir(root, _KIND),
        f"{name}_{label}_background_subtraction.png",
    )
    return sub


def _check_subtraction_model(background: str) -> None:
    """Reject a background model that cannot apply to a residual.

    Checked by the drivers *before* they read anything: on the feather path the
    histogram pass can take minutes, and finding out afterwards that the model
    was never going to work is a poor trade.

    Raises
    ------
    ValueError
        Raised for anything but ``'flat'``.
    """
    if background != "flat":
        raise ValueError(
            f"subtract_background=True with background={background!r} does not "
            "mean anything: the subtraction removes the triangular pedestal, "
            "so there is no triangle left to fit. Use background='flat'."
        )


def _whole_bins(window_ps: float, bin_width_ps: float) -> float:
    """Round a half-range up to a whole number of bins.

    '_delta_hist_edges' places bin *centres* at multiples of the width starting
    from ``-window``, so zero is only a bin centre when the window is a whole
    number of bins - and unless it is, a histogram built over the support and
    one built over a narrower window sit on different lattices, which would
    make the zoom slice and a direct rebin disagree.
    """
    return float(np.ceil(window_ps / bin_width_ps) * bin_width_ps)


def _rebin_lag_planes(
    counts: np.ndarray,
    centers: np.ndarray,
    *,
    bin_width_ps: float,
    plot_window_ps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebin every lag plane of a pooled grid onto one common histogram.

    Same 'rebin_delta_counts' per plane rather than anything cleverer: the
    lags are only subtractable if their bins are the *same* bins, and reusing
    the one function is how that stays true.
    """
    planes = []
    edges = None
    for plane in counts:
        binned, edges = rebin_delta_counts(
            plane,
            centers,
            bin_width_ps=bin_width_ps,
            plot_window_ps=plot_window_ps,
        )
        planes.append(binned)
    return np.vstack(planes), (edges[:-1] + edges[1:]) / 2



