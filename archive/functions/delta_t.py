"""ARCHIVED from dapkel.functions.delta_t (branch modules-restructure, 2026-10-31).

The feather stage-1 path (every difference written to a '.feather', then streamed into a histogram), the two-Gaussian peak model, and the dense-table branch of the calibration choice. Superseded by the binned-counts path: 'process_delta_t' / 'plot_delta_t'.

Reference only: not packaged, not tested, not maintained. Copied verbatim
from the pre-restructure module (commit 6b49a4b), so it still refers to
helpers and names that no longer exist in 'src/'. To revive a piece, port it
into the live module rather than importing it from here.
"""

from __future__ import annotations
import contextlib
import glob
import hashlib
import json
import os
import re
import time
import warnings
from collections.abc import Iterator, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from scipy.optimize import curve_fit
from tqdm import tqdm
from dapkel.core import io, store, timing
from dapkel.core import pairs as pixel_pairs
from dapkel.functions import background_subtraction as bs
from dapkel.functions import calc_diff as cd
from dapkel.functions import tdc_calibration as tc
from dapkel.functions.calc_diff import Pixel
from dapkel.functions.unpack import unpack


#: How a packaged TDC calibration is stored, and therefore how it is loaded.
#: ``'table'`` is the dense per-clock density-test LUT - one width per code, one
#: file per reference clock. ``'compact'`` is the same measurement reduced to the
#: ring's 8 tap widths, which belong to the oscillator rather than to the clock,
#: so a single table expands onto whatever code range a run occupies. See
#: ``docs/guide/tdc_calibration.md``.
_TDC_MODELS = ("table", "compact")


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
    apply_TDC_calibration: bool = True,
    daughterboard_number: str | None = None,
    motherboard_number: str | None = None,
    lut_clock: str | None = None,
    tdc_model: str = "table",
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

    Differences are always saved in **picoseconds**. With
    ``apply_TDC_calibration=True`` (default) the board's density-test LUT
    converts each timestamp *before* differencing; without it the nominal
    'timing.CODE_S' does. There is no raw-code option because there would be
    nothing to do with one: a LUT maps a *code* to a time, and a saved
    difference has lost both codes, so an uncalibrated feather could never be
    calibrated afterwards anyway.

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
        Filename fragment selecting the files. The default is ``'ORT'``. It
        also settles which SPAD's TDC calibration applies, so that is not a
        parameter: ``'S0T'``...``'S3T'`` read one micropixel and get that
        SPAD's own LUT, while ``'ORT'`` reads all four and gets the average of
        theirs. See '_spad_from_tag'.
    mode : str, optional
        ``'all_pairs'`` (default) or ``'1v1'``; see
        'calc_diff.calculate_differences'.
    apply_TDC_calibration : bool, optional
        Apply the density-test TDC calibration (per-pixel code -> ps LUT)
        before differencing, so the DNL of the ring is corrected. The default
        is True, in which case the packaged board LUT is loaded via
        ``daughterboard_number`` / ``motherboard_number`` (a ValueError is
        raised if either is missing) and checked against the code range this
        run actually occupies. Set False to fall back to the nominal code
        width.
    daughterboard_number : str | None, optional
        Daughterboard id (e.g. ``'D0'``) used to pick the packaged board LUT
        via 'tdc_calibration.load_board_lut'. The default is None.
    motherboard_number : str | None, optional
        Motherboard id (e.g. ``'M0'``) used with ``daughterboard_number`` to
        pick the packaged board LUT. The default is None.
    lut_clock : str | None, optional
        Which packaged calibration to load, named by the clock tag in its file
        name - ``'2MHz'``, ``'10MHz'``, ``'50MHz'`` as shipped. The default is
        None, which lets the run's own code range pick a table that covers it,
        and that is enough while one calibration per clock is installed. It
        cannot choose between two calibrations of the *same* clock, which cover
        identically; name one here when a second is added (the qualified tag is
        whatever follows the frequency in the file name). Ignored when
        ``tdc_model='compact'``.
    tdc_model : str, optional
        Which form of the packaged calibration to apply. ``'table'`` (default)
        uses the dense per-clock density-test LUT, one width per code.
        ``'compact'`` uses the ring's 8 tap widths - the same measurement,
        averaged over every lap the run swept - expanded onto this run's code
        range. The two describe the same hardware; ``'compact'`` carries far less
        counting noise per code and needs no per-clock table, so it is the one to
        prefer once you have compared them.
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

    # Resolve the density-test LUT to apply (per-pixel, code -> ps), validated
    # against the code range this run occupies rather than against a stated
    # clock. None falls back to the nominal code width; either way the
    # differences come out in ps.
    time_lut = _resolve_time_lut(
        apply_TDC_calibration,
        daughterboard_number,
        motherboard_number,
        _spad_from_tag(tag),
        # Reading one file to learn the run's code range is only worth it when
        # a LUT has to be matched against it.
        min_codes=(
            _run_max_code(files, nframes) + 1 if apply_TDC_calibration else None
        ),
        lut_clock=lut_clock,
        tdc_model=tdc_model,
    )

    unit = "ps"
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
                    pixel_ts, pixels, mode, frame_lag=k
                )
                deltas.update(
                    {_lag_column(lbl, k): arr for lbl, arr in lagged.items()}
                )
        else:
            # Left as it was: lag 0 is tested to be identical to this, but the
            # plain path has no reason to route through the lag machinery.
            deltas = cd.calculate_differences(pixel_ts, pixels, mode=mode)
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


def _hist_signature(
    sources: Sequence[str],
    *,
    bin_width_s: float,
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
        "bin_width_s": float(bin_width_s),
        "pairs": None if pairs is None else sorted(pairs),
        "n_sources": len(sources),
        "sources_digest": digest.hexdigest(),
    }
    # Added only when lags are in play, so a plain histogram keeps the
    # signature it had before they existed and stays reusable.
    if lags is not None:
        signature["lags"] = [int(k) for k in lags]
    return signature


def _require_ps(units: set[str], sources: Sequence[str]) -> str:
    """Reject feathers that are not in picoseconds.

    Stage 1 has always recorded its unit in the schema metadata, and now only
    ever writes ``'ps'``. Feathers written by an older version may hold raw
    codes, which read as picoseconds would be wrong by a factor of ~14 without
    looking wrong, so they are refused rather than guessed at. They cannot be
    converted here either - see 'calculate_and_save_timestamp_differences'.

    Parameters
    ----------
    units : set[str]
        The ``delta_unit`` values found across the sources.
    sources : Sequence[str]
        The feathers being pooled, for the error message.

    Returns
    -------
    str
        ``'ps'``.

    Raises
    ------
    ValueError
        Raised when any source is not in picoseconds.
    """
    if units - {"ps"}:
        raise ValueError(
            f"These delta-t feathers are in {sorted(units)}, not picoseconds:\n"
            + "\n".join(f"  {os.path.basename(src)}" for src in sources)
            + "\nA saved difference of two raw codes cannot be converted after "
            "the fact - the calibration is per code and both operands are gone. "
            "Recompute stage 1 with "
            "'calculate_and_save_timestamp_differences'."
        )
    return "ps"


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

    unit = _require_ps(units, sources)

    # Legacy single-column layouts name the unit in the column itself.
    if "delta_ps" in names:
        return ["delta_ps"], unit, []

    # Frame-shifted columns are background, not signal. Pooling them into the
    # coincidence histogram because they happen to be in the file would inflate
    # it ~9x, so they are only ever reached through an explicit lag request
    # ('_pooled_lag_columns').
    names = [n for n in names if not _is_lag_column(n)]

    if pairs is None:
        return names, unit, []
    missing = [p for p in pairs if p not in names]
    return [p for p in pairs if p in names], unit, missing


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

    unit = _require_ps(units, sources)

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
        Histogram bin edges in ps, from '_delta_hist_edges' - which spans the
        whole physical support, so nothing has to be windowed away here.
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
        ``unit``, ``n`` (differences binned), ``total`` (real differences seen),
        ``slots`` (table cells read, padding included), ``padding_fraction``,
        ``granularity_s`` (smallest non-zero ``abs(delta)`` in seconds, for the
        comb guard), ``support_s`` (largest ``abs(delta)``, i.e. the measured
        TDC full scale - the triangle's half-base), ``n_outside``,
        ``n_columns`` and ``missing``.
        With ``lags``, ``n`` and ``total`` count the lag-0 row only - they
        describe the measurement, not the bookkeeping around it.
    """
    window_ps = float(max(abs(edges[0]), abs(edges[-1])))

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
    outside = 0
    granularity_ps = np.inf
    max_abs_ps = 0.0

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
                        d_ps = values
                        if gi == 0:
                            total += d_ps.size

                            magnitude = np.abs(d_ps)
                            nonzero = magnitude[magnitude != 0]
                            if nonzero.size:
                                granularity_ps = min(
                                    granularity_ps, float(nonzero.min())
                                )
                            # The largest difference present *is* the TDC full
                            # scale, hence the accidental triangle's half-base.
                            # Accumulated here because this pass already reads
                            # every value once, so measuring it is free.
                            max_abs_ps = max(max_abs_ps, float(magnitude.max()))

                        inside = d_ps[np.abs(d_ps) <= window_ps]
                        if gi == 0:
                            n_windowed += inside.size
                            outside += d_ps.size - inside.size
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
        "granularity_s": (
            timing.to_s(granularity_ps) if np.isfinite(granularity_ps) else None
        ),
        "support_s": timing.to_s(max_abs_ps) if max_abs_ps else None,
        "n_outside": int(outside),
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
    bin_width_s: float = timing.CODE_S,
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

    The histogram always spans the **whole physical support** of a difference
    ('timing.MAX_DELTA_S', ~586 ns), so there is no window to choose: at one
    code per bin that is ~16 k bins, i.e. ~131 kB. Every plot window is then a
    slice of one cached histogram rather than a reason to re-stream, and the
    largest difference actually present - the TDC full scale, hence the
    accidental triangle's half-base - is measured on the way past and returned
    as ``support_s``.

    The saved histogram is *lossless for everything downstream*: both fit
    models take ``(bin_centers, counts)``, and so does the plot. Binning at
    ``bin_width_s=timing.CODE_S`` (one TDC code) is the native resolution of
    the data, so nothing is thrown away at the default settings.

    The sidecar records the binning and a fingerprint of the sources; a saved
    histogram is only reused when all of it matches, so changing
    ``bin_width_s`` or ``pairs`` - or rewriting the feather - re-streams
    automatically.

    Parameters
    ----------
    path : str | None, optional
        Path to the raw-data folder. Its ``processed/`` holds the feather and
        receives the histogram. May be None when ``feather_path`` is given.
    feather_path : str | None, optional
        Explicit delta-t '.feather' to read, or a ``*_delta_t_parts`` folder
        to read the parts directly (which skips needing the combined file at
        all). When given, ``path`` is ignored. The default is None.
    bin_width_s : float, optional
        Histogram bin width in seconds. The default is one TDC code.
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

    edges = _delta_hist_edges(bin_width_s)
    signature = _hist_signature(
        sources,
        bin_width_s=bin_width_s,
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
                # 'n' lives under 'stats', not at the top level - reading it
                # from the wrong place made every cache hit report "0".
                n_saved = meta.get("stats", {}).get("n", 0)
                print(
                    f"\n> > > Reusing the saved delta-t histogram "
                    f"({n_saved} differences) from "
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
            f"{len(edges) - 1}-bin histogram "
            f"({bin_width_s * 1e12:.0f} ps bins over the whole "
            f"+/-{timing.MAX_DELTA_S * 1e9:.0f} ns support)"
            + (f" at lags {lags}" if lags else "")
            + " < < <\n"
        )

    counts, stats = _stream_delta_histogram(
        sources,
        edges=edges,
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
        support_s = stats.get("support_s")
        print(
            f"\n> > > {stats['total']} difference(s) across "
            f"{stats['n_columns']} pair column(s), reaching "
            + (
                f"+/-{support_s * 1e9:.2f} ns"
                if support_s
                else "no measurable support"
            )
            + f". Padding was {100 * stats['padding_fraction']:.1f}% of the "
            f"cells read. Histogram saved to {hist_path} < < <"
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


def _measured_support_s(
    counts: np.ndarray, centers: np.ndarray, info: dict
) -> float:
    """Return the TDC full scale the histogram itself shows, in seconds.

    The accidental background fills the whole measurement range, so the
    outermost populated bin *is* the full scale - which is the triangle's
    half-base and the widest window worth plotting. Preferred source is the
    streaming pass's own ``support_s`` (the largest ``abs(delta)`` it saw, exact
    to the value rather than to a bin); the outermost non-empty bin is the
    fallback for histograms cached before that was recorded.

    Parameters
    ----------
    counts : np.ndarray
        The histogram, ``(n_bins,)`` or ``(n_lags, n_bins)`` with lag 0 first.
    centers : np.ndarray
        Bin centres in ps.
    info : dict
        Stats from 'compute_and_save_delta_histogram'.

    Returns
    -------
    float
        The measured full scale in seconds, or 'timing.MAX_DELTA_S' when the
        histogram is empty and there is nothing to measure.
    """
    recorded = info.get("support_s")
    if recorded:
        return float(recorded)

    row = counts[0] if counts.ndim == 2 else counts
    filled = np.nonzero(row)[0]
    if not filled.size:
        return timing.MAX_DELTA_S
    return timing.to_s(
        float(max(abs(centers[filled[0]]), abs(centers[filled[-1]])))
    )


def collect_and_plot_timestamp_differences(
    path: str | None = None,
    *,
    bin_width_s: float = timing.CODE_S,
    plot_window_s: float | None = None,
    fit_window_s: float | None = 1e-9,
    feather_path: str | None = None,
    pairs: Sequence[str] | None = None,
    label: str = "spdc",
    title: str | None = None,
    reuse: bool = True,
    subtract_background: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Histogram the saved delta-t feather, fit the peak, and plot it.

    Pools the selected pair columns of a saved feather, histograms them, fits
    the coincidence peak and saves the figure.

    The feather is **streamed** through 'compute_and_save_delta_histogram',
    one Arrow record batch at a time, and never held in memory - a full run's
    combined feather is far too large to load. The counts are identical to
    histogramming the whole pooled array at once; only the memory profile
    differs. That histogram covers the entire physical support of a difference
    and is cached in ``processed/``, so any plot window, re-fit or change of
    background model is a slice of it rather than another pass.

    **Nothing about the TDC has to be stated.** The largest difference in the
    data is the range the TDC swept, so the triangle's half-base and the default
    plot window are measured, not configured. See 'core.timing'.

    **Nor is the fit model a choice.** The accidental pedestal is either still
    in the histogram - and on ORT it is a *triangle*, which is what the
    free-running phase produces - or it has been subtracted, and the residual
    sits on a flat zero baseline. So the model follows ``subtract_background``:
    ``Gaussian + triangle`` over the whole histogram, or ``Gaussian + constant``
    over ``fit_window_s``. To fit anything else - the narrow core and broad
    tails of 'fit_two_gaussians_on_triangle', say - re-fit the ``counts`` and
    ``centers`` this returns; nothing is re-read from disk.

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
    bin_width_s : float, optional
        Histogram bin width in seconds. The default is one TDC code
        ('core.timing.CODE_S', ~71.6 ps), the native resolution of the data.
    plot_window_s : float | None, optional
        Histogram/plot half-range in seconds around zero. The default is None,
        i.e. the measured TDC full scale - the whole support the data occupies,
        which is what the triangle models want. Narrow it for a flat fit.
    fit_window_s : float | None, optional
        Half-range around the peak, in seconds. The default is ``1e-9`` (1 ns).
        On a residual this is the fit range, measured from the tallest bin, and
        also the range the model-free coincidence count is summed over. Ignored
        by the triangle fit, which uses the whole histogram.
    feather_path : str | None, optional
        Explicit feather to read and plot directly, or a ``*_delta_t_parts``
        folder to read the parts; when given, ``path`` is ignored and the
        raw-data folder is not needed (the plot is saved beside the feather).
        The default is None (derive from ``path``).
    pairs : Sequence[str] | None, optional
        Restrict pooling to these pixel-pair column names (e.g.
        ``["16,16-20,20"]``). The default is None (pool every pair column).
    label : str, optional
        Short label for the title and output file (``'spdc'`` / ``'hbt'``).
        The default is ``'spdc'``.
    title : str | None, optional
        Optional override for the plot title. The default is None.
    reuse : bool, optional
        Reuse a saved histogram whose binning and sources match, instead of
        re-streaming the feather. The default is True. Pass False to force
        the full pass.
    subtract_background : bool, optional
        Subtract the frame-shifted accidentals before fitting, and write the
        three-panel diagnostic beside the fit. The default is False. Requires a
        feather written with ``subtract_background=True``. It also settles the
        fit model - flat rather than triangular, see above.

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
        Raised when the feather is not in picoseconds (written by a version
        that stored raw codes), or when ``subtract_background`` is asked for on
        a feather with no frame-shifted columns.
    """
    background = _background_model(subtract_background)

    counts, edges, info = compute_and_save_delta_histogram(
        path,
        feather_path=feather_path,
        bin_width_s=bin_width_s,
        pairs=pairs,
        subtract_background=subtract_background,
        reuse=reuse,
    )
    name = info["name"]
    results_dir = store.results_dir(info["root"], _KIND, create=False)
    centers = (edges[:-1] + edges[1:]) / 2

    # The data's own full scale: the triangle's half-base, and the default
    # window - so a triangle fit is correctly scoped without being told anything.
    full_scale_s = _measured_support_s(counts, centers, info)
    if plot_window_s is None:
        plot_window_s = full_scale_s

    # Comb guard: warn if the data granularity is coarser than the bin.
    gran_s = info.get("granularity_s")
    if gran_s is not None and gran_s > bin_width_s * 1.5:
        print(
            f"  WARNING: delta granularity is ~{gran_s * 1e12:.0f} ps but "
            f"bin_width={bin_width_s * 1e12:.0f} ps -> the histogram will be a "
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
            bin_width_s=bin_width_s,
            zoom_window_s=plot_window_s,
            fit_window_s=fit_window_s,
            label=label,
        )
        counts = sub["residual"]
        centers = sub["centers"]
        # The model-free count, not the bin total: the residual's bins are
        # differences of two histograms, so summing them *is* the coincidence
        # number and anything outside the peak is noise around zero.
        n_in_window = int(round(sub["model_free"]["n"]))
    else:
        # The cached histogram spans the whole support; the requested window is
        # a slice of it, which is why changing the window costs nothing.
        keep = np.abs(centers) <= timing.to_ps(plot_window_s)
        counts, centers = counts[keep], centers[keep]
        n_in_window = int(counts.sum())

    counts, centers, fit = _fit_and_plot_delta(
        counts,
        centers,
        name=name,
        results_dir=results_dir,
        bin_width_s=bin_width_s,
        plot_window_s=plot_window_s,
        fit_window_s=fit_window_s,
        background=background,
        full_scale_s=full_scale_s,
        label=label,
        title=title,
        n_in_window=n_in_window,
        stem="delta_t_sub" if subtract_background else "delta_t",
        subtracted=subtract_background,
    )
    if sub is not None:
        fit["model_free"] = sub["model_free"]
        fit["subtraction"] = sub
        fit["subtraction_png"] = sub["png_path"]
    return counts, centers, fit
