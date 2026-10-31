"""Stage 1 must stream to disk, not accumulate the whole run in RAM.

Collecting delta-t over ~10 000 '.bin' files used to hold every difference in
memory and write one feather at the end, which failed twice over: the buffers
exhausted RAM, and the final write asked pyarrow for a single multi-GB
allocation on top. The writer now flushes size-capped part files as it goes
and concatenates them a record batch at a time.

These tests pin the two properties that makes this worth doing - the parts
really do roll over, and the combined result is bit-for-bit what the old
accumulate-everything path would have produced - plus the recovery route for
a run that dies mid-way.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pyarrow as pa
import pytest

from dapkel.core import store
from dapkel.functions import delta_t as dt

LABELS = ["0,0-1,1", "0,0-2,2"]


def _writer(tmp_path, labels=LABELS, part_mb=1 / 1024):
    """Build a part writer capped at 1 KB, so a few values roll it over."""
    return dt._PartWriter(
        str(tmp_path / "parts"), "run_delta_t", labels, "ps", part_mb
    )


def _columns(path):
    """Read a delta-t feather back as label -> real values (padding dropped)."""
    df, _ = dt._read_delta_feather(path)
    return {
        col: df[col].to_numpy(dtype=np.float64)[
            ~np.isnan(df[col].to_numpy(dtype=np.float64))
        ]
        for col in df.columns
    }


def test_writer_rolls_over_past_the_size_cap(tmp_path):
    writer = _writer(tmp_path)
    # 1 KB cap / 2 columns / 8 bytes = 64 rows before a flush.
    for _ in range(10):
        writer.add({lbl: np.arange(50.0) for lbl in LABELS})
    parts = writer.close()

    assert len(parts) > 1, "the cap was never reached - nothing was streamed"
    assert [dt._part_index(p) for p in parts] == list(range(len(parts)))
    assert writer.total == 10 * 50 * len(LABELS)


def test_combined_feather_matches_the_unstreamed_result(tmp_path):
    rng = np.random.default_rng(0)
    batches = [{lbl: rng.normal(size=37) for lbl in LABELS} for _ in range(20)]

    writer = _writer(tmp_path)
    for batch in batches:
        writer.add(batch)
    parts = writer.close()

    out = str(tmp_path / "combined.feather")
    total, labels, unit = dt._combine_delta_feathers(parts, out)

    assert labels == LABELS
    assert unit == "ps"
    assert total == 20 * 37 * len(LABELS)

    got = _columns(out)
    for lbl in LABELS:
        expected = np.concatenate([b[lbl] for b in batches])
        np.testing.assert_array_equal(got[lbl], expected)


def test_combining_streams_one_batch_at_a_time(tmp_path):
    """The output must keep the parts as separate batches, not one slab.

    A single record batch would mean the combine materialised everything at
    once, which is the failure this whole path exists to avoid.
    """
    writer = _writer(tmp_path)
    for _ in range(10):
        writer.add({lbl: np.arange(50.0) for lbl in LABELS})
    parts = writer.close()

    out = str(tmp_path / "combined.feather")
    dt._combine_delta_feathers(parts, out)

    with pa.ipc.open_file(out) as reader:
        assert reader.num_record_batches == len(parts)


def test_pairs_missing_from_one_source_become_padding(tmp_path):
    """Runs that saw different pairs pool into the union of their columns."""
    a = str(tmp_path / "a_delta_t.feather")
    b = str(tmp_path / "b_delta_t.feather")
    dt._write_delta_feather(a, {"0,0-1,1": np.arange(5.0)}, ["0,0-1,1"], "ps")
    dt._write_delta_feather(b, {"9,9-8,8": np.arange(3.0)}, ["9,9-8,8"], "ps")

    out = str(tmp_path / "combined.feather")
    total, labels, unit = dt._combine_delta_feathers([a, b], out)

    assert labels == ["0,0-1,1", "9,9-8,8"]
    assert total == 8
    got = _columns(out)
    np.testing.assert_array_equal(got["0,0-1,1"], np.arange(5.0))
    np.testing.assert_array_equal(got["9,9-8,8"], np.arange(3.0))


def test_mixed_units_are_refused(tmp_path):
    a = str(tmp_path / "a_delta_t.feather")
    b = str(tmp_path / "b_delta_t.feather")
    dt._write_delta_feather(a, {"0,0-1,1": np.arange(5.0)}, ["0,0-1,1"], "ps")
    dt._write_delta_feather(b, {"0,0-1,1": np.arange(5.0)}, ["0,0-1,1"], "code")

    with pytest.raises(ValueError, match="mixed delta_unit"):
        dt._combine_delta_feathers([a, b], str(tmp_path / "out.feather"))


def test_a_raw_code_feather_is_refused_rather_than_read_as_ps(tmp_path):
    """Stage 1 now always writes ps, so a 'code' feather predates this.

    Reading it as picoseconds would be wrong by ~14x without looking wrong, and
    it cannot be fixed up either: the calibration is per code and a saved
    difference has lost both operands.
    """
    path = str(tmp_path / "old_delta_t.feather")
    dt._write_delta_feather(
        path, {"0,0-1,1": np.arange(5.0)}, ["0,0-1,1"], "code"
    )

    with pytest.raises(ValueError, match="not picoseconds"):
        dt._pooled_columns([path], None)


def test_nan_padded_legacy_feathers_still_count_correctly(tmp_path):
    """Feathers written before null padding must not count their padding."""
    legacy = str(tmp_path / "legacy_delta_t.feather")
    long_col = np.arange(10.0)
    short_col = np.full(10, np.nan)
    short_col[:4] = np.arange(4.0)
    table = pa.table({"0,0-1,1": long_col, "0,0-2,2": short_col})
    table = table.replace_schema_metadata({b"delta_unit": b"ps"})
    with pa.ipc.new_file(legacy, table.schema) as writer:
        writer.write_table(table)

    total, _, _ = dt._combine_delta_feathers(
        [legacy], str(tmp_path / "out.feather")
    )
    assert total == 14  # 10 real + 4 real, not 20


def test_parts_are_recoverable_after_a_crash(tmp_path):
    """A run that dies leaves usable parts; combine_delta_t_parts reads them."""
    processed = tmp_path / "processed"
    parts_dir = processed / "run_delta_t_parts"
    writer = dt._PartWriter(str(parts_dir), "run_delta_t", LABELS, "ps", 1 / 1024)
    for _ in range(10):
        writer.add({lbl: np.arange(50.0) for lbl in LABELS})
    # Simulate the crash: flush what is buffered, never combine.
    parts = writer.close()

    out = dt.combine_delta_t_parts(str(parts_dir))

    assert out == str(processed / "run_delta_t.feather")
    got = _columns(out)
    expected = np.concatenate([np.arange(50.0)] * 10)
    for lbl in LABELS:
        np.testing.assert_array_equal(got[lbl], expected)
    # The parts are the insurance - assembling them must not consume them.
    assert dt._find_parts(str(parts_dir)) == parts


def test_combine_parts_refuses_to_overwrite_without_rewrite(tmp_path):
    parts_dir = tmp_path / "processed" / "run_delta_t_parts"
    writer = dt._PartWriter(str(parts_dir), "run_delta_t", LABELS, "ps", 1.0)
    writer.add({lbl: np.arange(5.0) for lbl in LABELS})
    writer.close()

    out = dt.combine_delta_t_parts(str(parts_dir))
    with pytest.raises(FileExistsError):
        dt.combine_delta_t_parts(str(parts_dir))
    # ... and goes ahead when told to, after the countdown.
    assert dt.combine_delta_t_parts(str(parts_dir), rewrite=True) == out


def test_parts_are_sorted_numerically_not_lexically(tmp_path):
    """part_10 must follow part_9, or the combined order silently scrambles."""
    parts_dir = tmp_path / "parts"
    writer = dt._PartWriter(str(parts_dir), "run_delta_t", ["0,0-1,1"], "ps", 1 / 1024)
    for i in range(12):
        writer.add({"0,0-1,1": np.full(200, float(i))})
    writer.close()

    found = dt._find_parts(str(parts_dir))
    assert len(found) >= 11
    assert [dt._part_index(p) for p in found] == sorted(
        dt._part_index(p) for p in found
    )


def test_clear_parts_leaves_foreign_files_alone(tmp_path):
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    (parts_dir / "run_delta_t_0.feather").write_bytes(b"")
    keep = parts_dir / "notes.txt"
    keep.write_text("do not delete me")

    dt._clear_parts(str(parts_dir))

    assert not (parts_dir / "run_delta_t_0.feather").exists()
    assert keep.exists()


def test_confirm_rewrite_is_silent_on_a_fresh_run(tmp_path, capsys):
    assert store.confirm_rewrite(str(tmp_path / "nope.feather")) is False
    assert capsys.readouterr().out == ""


def test_confirm_rewrite_warns_and_names_the_targets(tmp_path, capsys):
    doomed = tmp_path / "run_delta_t.feather"
    doomed.write_bytes(b"x" * 2048)
    parts = tmp_path / "run_delta_t_parts"
    parts.mkdir()
    (parts / "run_delta_t_0.feather").write_bytes(b"y" * 4096)

    assert store.confirm_rewrite([str(doomed), str(parts)], delay=0) is True

    out = capsys.readouterr().out
    assert "OVERWRITTEN" in out
    assert "run_delta_t.feather" in out
    assert "run_delta_t_parts" in out
    assert "2 file(s)" not in out  # the folder holds exactly one


def test_confirm_rewrite_actually_waits(tmp_path):
    """The pause is the whole point - it is what makes Ctrl-C possible."""
    import time

    doomed = tmp_path / "run.feather"
    doomed.write_bytes(b"x")

    start = time.monotonic()
    store.confirm_rewrite(str(doomed), delay=1.0)
    assert time.monotonic() - start >= 0.9


# --- end to end, through the real entry point ------------------------------

PAIRS = [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]
NFRAMES = 20


def _fake_run(tmp_path, n_files=6):
    """Build a data folder of ORT '.bin' files full of plausible codes.

    'unpack' reads fixed-width frames and decodes every 14-bit field, so
    random bytes decode to random-but-valid coarse/fine codes - which is all
    this needs: the point is the file plumbing, not the physics.
    """
    folder = tmp_path / "run"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(n_files):
        raw = rng.integers(
            0, 256, size=NFRAMES * 4 * 64 * 8, dtype=np.uint8
        )
        raw.tofile(folder / f"data_ORT_{i}.bin")
    return folder


def test_end_to_end_streams_parts_and_combines_them(tmp_path, monkeypatch):
    folder = _fake_run(tmp_path)

    # 1 KB parts, so the buffer rolls over every few files. The cap is a
    # module constant rather than a parameter, so the test moves the constant.
    monkeypatch.setattr(dt, "_PART_MB", 1 / 1024)
    out = dt.calculate_and_save_timestamp_differences(
        str(folder),
        PAIRS,
        apply_TDC_calibration=False,
        nframes=NFRAMES,
    )

    parts_dir = tmp_path / "run" / "processed" / "run_delta_t_parts"
    parts = dt._find_parts(str(parts_dir))
    assert len(parts) > 1, "everything went into one part - nothing streamed"

    got = _columns(out)
    assert set(got) == {"0,0-1,0", "0,0-1,1", "0,1-1,0", "0,1-1,1"}
    # A frame contributes one difference per pair whenever both pixels
    # decoded a valid code, which random bytes occasionally miss.
    for values in got.values():
        assert 0.9 * 6 * NFRAMES <= values.size <= 6 * NFRAMES


def test_end_to_end_part_size_does_not_change_the_result(tmp_path, monkeypatch):
    """Sharding is a memory strategy; it must not touch the numbers."""
    results = {}
    for part_mb in (1 / 1024, 1024.0):
        folder = _fake_run(tmp_path / f"cap{part_mb}")
        monkeypatch.setattr(dt, "_PART_MB", part_mb)
        out = dt.calculate_and_save_timestamp_differences(
            str(folder),
            PAIRS,
            apply_TDC_calibration=False,
            nframes=NFRAMES,
        )
        results[part_mb] = _columns(out)

    small, large = results[1 / 1024], results[1024.0]
    assert set(small) == set(large)
    for label in small:
        np.testing.assert_array_equal(small[label], large[label])


def test_end_to_end_rewrite_guard(tmp_path, monkeypatch, capsys):
    folder = _fake_run(tmp_path, n_files=2)
    kwargs = {
        "apply_TDC_calibration": False,
        "nframes": NFRAMES,
    }
    monkeypatch.setattr(dt, "_PART_MB", 1 / 1024)
    dt.calculate_and_save_timestamp_differences(str(folder), PAIRS, **kwargs)

    with pytest.raises(FileExistsError):
        dt.calculate_and_save_timestamp_differences(
            str(folder), PAIRS, **kwargs
        )

    monkeypatch.setattr(store, "REWRITE_DELAY_S", 0.0)
    capsys.readouterr()
    dt.calculate_and_save_timestamp_differences(
        str(folder), PAIRS, rewrite=True, **kwargs
    )
    out = capsys.readouterr().out
    assert "OVERWRITTEN" in out
    assert "run_delta_t_parts" in out


def test_end_to_end_rewrite_does_not_pool_the_previous_run(tmp_path, monkeypatch):
    """Stale parts must be cleared, or a rewrite doubles the data."""
    folder = _fake_run(tmp_path, n_files=3)
    kwargs = {
        "apply_TDC_calibration": False,
        "nframes": NFRAMES,
    }
    monkeypatch.setattr(dt, "_PART_MB", 1 / 1024)
    first = _columns(
        dt.calculate_and_save_timestamp_differences(str(folder), PAIRS, **kwargs)
    )

    monkeypatch.setattr(store, "REWRITE_DELAY_S", 0.0)
    second = _columns(
        dt.calculate_and_save_timestamp_differences(
            str(folder), PAIRS, rewrite=True, **kwargs
        )
    )

    for label, values in first.items():
        np.testing.assert_array_equal(second[label], values)


def test_end_to_end_keeps_the_parts_after_combining(tmp_path, monkeypatch):
    """A successful run keeps its parts.

    They are the crash insurance, and every read-side entry point streams a
    part folder as readily as the combined file.
    """
    folder = _fake_run(tmp_path, n_files=3)
    monkeypatch.setattr(dt, "_PART_MB", 1 / 1024)
    out = dt.calculate_and_save_timestamp_differences(
        str(folder),
        PAIRS,
        apply_TDC_calibration=False,
        nframes=NFRAMES,
    )

    parts_dir = tmp_path / "run" / "processed" / "run_delta_t_parts"
    parts = dt._find_parts(str(parts_dir))
    assert len(parts) > 1

    # The kept parts and the combined feather must say the same thing.
    combined = _columns(out)
    for values in combined.values():
        assert 0.9 * 3 * NFRAMES <= values.size <= 3 * NFRAMES
    from_parts: dict[str, list] = {label: [] for label in combined}
    for part in parts:
        for label, values in _columns(part).items():
            from_parts[label].extend(values.tolist())
    for label, values in combined.items():
        np.testing.assert_allclose(from_parts[label], values)


# --- stage 1b: streaming the feather down to a histogram -------------------
#
# The combined feather of a 10 000-file run reaches hundreds of GB, so the
# read side has to stream too. These tests pin the property that makes the
# streamed histogram usable as the *baseline* every later optimisation is
# measured against: it is bin-for-bin what loading the whole table and
# histogramming it in one go would have produced.

BIN_PS = 77.0
WINDOW_PS = 3000.0
# The library takes seconds; these tests were written in ps, so convert once
# here rather than restating every number.
BIN_S = BIN_PS * 1e-12
WINDOW_S = WINDOW_PS * 1e-12
# Half-base of the accidental triangle the fixtures fake, in ps: one 10 MHz
# reference-clock period, which is what the read side has to *measure* now
# rather than be told.
FULL_SCALE_PS = 100_000.0


def _reference_histogram(feather, *, edges):
    """Histogram the old way: load everything, pool it, bin it once."""
    df, _ = dt._read_delta_feather(feather)
    vals = df.to_numpy().ravel()
    pooled = vals[~np.isnan(vals)]
    inside = pooled[np.abs(pooled) <= max(abs(edges[0]), abs(edges[-1]))]
    return np.histogram(inside, bins=edges)[0], inside.size


def _spread_deltas(rng, n):
    """Differences in ps: a peak at 0 on a wide accidental background."""
    peak = rng.normal(0.0, 300.0, size=n // 4)
    accidental = rng.uniform(-FULL_SCALE_PS, FULL_SCALE_PS, size=n - n // 4)
    return np.concatenate([peak, accidental])


def _feather_of_parts(tmp_path, n_batches=12, unit="ps"):
    """Write a multi-batch delta-t feather and return (combined, parts_dir)."""
    rng = np.random.default_rng(7)
    parts_dir = tmp_path / "parts"
    writer = dt._PartWriter(
        str(parts_dir), "run_delta_t", LABELS, unit, 1 / 1024
    )
    for _ in range(n_batches):
        writer.add({lbl: _spread_deltas(rng, 90) for lbl in LABELS})
    parts = writer.close()
    combined = str(tmp_path / "run_delta_t.feather")
    dt._combine_delta_feathers(parts, combined)
    return combined, str(parts_dir)


def test_streamed_histogram_is_identical_to_loading_the_whole_table(tmp_path):
    """The baseline property: streaming must change memory, not numbers."""
    combined, _ = _feather_of_parts(tmp_path)
    edges = dt._delta_hist_edges(BIN_S, WINDOW_S)

    counts, stats = dt._stream_delta_histogram(
        [combined], edges=edges, quiet=True
    )
    expected, expected_n = _reference_histogram(combined, edges=edges)

    assert counts.sum() > 0, "the fixture produced no counts in the window"
    np.testing.assert_array_equal(counts, expected)
    assert stats["n"] == expected_n
    assert stats["unit"] == "ps"


def test_streaming_reads_more_than_one_batch(tmp_path):
    """Guards the test above: a single-batch file would prove nothing."""
    combined, _ = _feather_of_parts(tmp_path)
    with pa.ipc.open_file(combined) as reader:
        assert reader.num_record_batches > 1


def test_parts_and_combined_give_the_same_histogram(tmp_path):
    """Plotting from the parts must not need the giant combined file."""
    combined, parts_dir = _feather_of_parts(tmp_path)
    edges = dt._delta_hist_edges(BIN_S, WINDOW_S)
    kwargs = {"edges": edges, "quiet": True}

    from_combined, stats_c = dt._stream_delta_histogram([combined], **kwargs)
    from_parts, stats_p = dt._stream_delta_histogram(
        dt._find_parts(parts_dir), **kwargs
    )

    np.testing.assert_array_equal(from_parts, from_combined)
    assert stats_p["n"] == stats_c["n"]


def test_the_streaming_pass_measures_the_support_it_saw(tmp_path):
    """The largest difference present is the TDC full scale.

    That is the triangle's half-base and the default plot window, so measuring
    it here is what replaces being told a clock period.
    """
    path = str(tmp_path / "run_delta_t.feather")
    dt._write_delta_feather(
        path,
        {"0,0-1,1": np.array([0.0, 100.0, -100.0, FULL_SCALE_PS])},
        ["0,0-1,1"],
        "ps",
    )

    counts, stats = dt._stream_delta_histogram(
        [path], edges=dt._delta_hist_edges(BIN_S), quiet=True
    )

    assert stats["unit"] == "ps"
    assert stats["granularity_s"] == 100.0e-12
    assert stats["support_s"] == pytest.approx(FULL_SCALE_PS * 1e-12)
    # The default edges span the whole physical support, so nothing is lost.
    assert counts.sum() == 4
    assert stats["n_outside"] == 0


def test_padding_is_dropped_and_measured(tmp_path):
    """Padding must not be counted - and its share is worth reporting."""
    path = str(tmp_path / "run_delta_t.feather")
    dt._write_delta_feather(
        path,
        {"0,0-1,1": np.zeros(100), "0,0-2,2": np.zeros(4)},
        ["0,0-1,1", "0,0-2,2"],
        "ps",
    )
    counts, stats = dt._stream_delta_histogram(
        [path], edges=dt._delta_hist_edges(BIN_S, WINDOW_S), quiet=True
    )

    assert counts.sum() == 104  # 104 real values, not 200 padded cells
    assert stats["total"] == 104
    assert stats["slots"] == 200
    assert stats["padding_fraction"] == pytest.approx(0.48)


def test_out_of_window_differences_are_excluded_but_still_counted(tmp_path):
    """Narrow edges still see everything - they just do not bin it.

    The drivers histogram over the whole support, but the helper accepts any
    edges, and a difference outside them has to be counted rather than lost.
    """
    path = str(tmp_path / "run_delta_t.feather")
    inside = np.zeros(6)
    outside = np.full(4, FULL_SCALE_PS)  # 100 ns >> the 3 ns edges
    dt._write_delta_feather(
        path,
        {"0,0-1,1": np.concatenate([inside, outside])},
        ["0,0-1,1"],
        "ps",
    )
    counts, stats = dt._stream_delta_histogram(
        [path], edges=dt._delta_hist_edges(BIN_S, WINDOW_S), quiet=True
    )

    assert counts.sum() == 6
    assert stats["n"] == 6
    assert stats["n_outside"] == 4
    assert stats["total"] == 10  # the granularity guard sees them all
    # ... and the support is measured from all of them, not just the binned ones.
    assert stats["support_s"] == pytest.approx(FULL_SCALE_PS * 1e-12)


def test_missing_pairs_are_reported_not_fatal(tmp_path, capsys):
    combined, _ = _feather_of_parts(tmp_path)
    counts, stats = dt._stream_delta_histogram(
        [combined],
        edges=dt._delta_hist_edges(BIN_S, WINDOW_S),
        pairs=[LABELS[0], "31,31-30,30"],
        quiet=True,
    )

    assert "31,31-30,30" in capsys.readouterr().out
    assert stats["missing"] == ["31,31-30,30"]
    assert stats["n_columns"] == 1
    assert counts.sum() > 0


def _hist_run(tmp_path):
    """Build a data folder with a combined delta-t feather in 'processed/'."""
    root = tmp_path / "run"
    processed = root / "processed"
    processed.mkdir(parents=True)
    combined, _ = _feather_of_parts(tmp_path)
    target = processed / "run_delta_t.feather"
    os.replace(combined, target)
    return root


def test_histogram_is_saved_and_then_reused(tmp_path, monkeypatch):
    """The 135 GB pass happens once; replotting loads ~KB instead."""
    root = _hist_run(tmp_path)

    counts, edges, info = dt.compute_and_save_delta_histogram(
        str(root), quiet=True
    )
    assert info["reused"] is False
    assert os.path.isfile(info["hist_path"])

    def _boom(*args, **kwargs):
        raise AssertionError("the feather was re-streamed instead of reused")

    monkeypatch.setattr(dt, "_stream_delta_histogram", _boom)
    again, again_edges, again_info = dt.compute_and_save_delta_histogram(
        str(root), quiet=True
    )

    assert again_info["reused"] is True
    np.testing.assert_array_equal(again, counts)
    np.testing.assert_array_equal(again_edges, edges)
    assert again_info["n"] == info["n"]


def test_a_different_binning_re_streams(tmp_path):
    """Reuse is keyed on the binning, or the cache would answer wrongly."""
    root = _hist_run(tmp_path)

    dt.compute_and_save_delta_histogram(str(root), quiet=True)
    counts, edges, info = dt.compute_and_save_delta_histogram(
        str(root), bin_width_s=BIN_S * 4, quiet=True
    )

    assert info["reused"] is False
    assert counts.size == len(edges) - 1
    assert counts.size < len(dt._delta_hist_edges(BIN_S)) - 1


def test_a_rewritten_feather_re_streams(tmp_path):
    """Stale counts against fresh data would be the worst kind of wrong."""
    root = _hist_run(tmp_path)
    first, _, _ = dt.compute_and_save_delta_histogram(str(root), quiet=True)

    feather = str(root / "processed" / "run_delta_t.feather")
    dt._write_delta_feather(
        feather, {"0,0-1,1": np.zeros(500)}, ["0,0-1,1"], "ps"
    )
    second, _, info = dt.compute_and_save_delta_histogram(
        str(root), quiet=True
    )

    assert info["reused"] is False
    assert second.sum() == 500
    assert not np.array_equal(second, first)


def test_missing_combined_feather_falls_back_to_the_parts(tmp_path):
    """A run that died before the combine step is still plottable."""
    root = tmp_path / "run"
    processed = root / "processed"
    processed.mkdir(parents=True)
    _feather_of_parts(tmp_path)
    os.rename(
        str(tmp_path / "parts"), str(processed / "run_delta_t_parts")
    )

    counts, _, info = dt.compute_and_save_delta_histogram(str(root), quiet=True)

    assert len(info["sources"]) > 1
    assert counts.sum() > 0


def test_no_feather_and_no_parts_is_a_clear_error(tmp_path):
    root = tmp_path / "run"
    (root / "processed").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="run_delta_t.feather"):
        dt.compute_and_save_delta_histogram(str(root), quiet=True)


def test_plot_streams_and_returns_the_histogram(tmp_path):
    """End to end: the driver never materialises the differences."""
    import matplotlib

    matplotlib.use("Agg")
    root = _hist_run(tmp_path)

    counts, centers, fit = dt.collect_and_plot_timestamp_differences(str(root))

    assert counts.shape == centers.shape
    assert fit["n"] == int(counts.sum())
    assert os.path.isfile(
        root / "results" / "coincidences" / "run_spdc_delta_t.png"
    )
    # The fitted numbers must come from the same counts the caller gets back -
    # here through the triangle, which is the model a raw histogram gets, on
    # the same measured support (a cached read, not another pass).
    _, _, info = dt.compute_and_save_delta_histogram(str(root), quiet=True)
    refit = dt.fit_gaussian_on_triangle(
        centers, counts, half_base=info["support_s"] * 1e12
    )
    assert refit["sigma"] == fit["sigma"]


# --- one run's parts must never be pooled with another's ------------------
#
# Part numbering restarts at 0 on every run. A 10 000-file run that died at
# part 300, restarted and writing 120 parts, used to overwrite parts 0-119
# and leave 120-299 from the dead run in place - and the folder-wide glob
# behind 'combine_delta_t_parts' then pooled all 300, counting the dead run's
# tail as extra coincidences. These pin the two halves of the fix.


def _run_parts(parts_dir, value, n_parts, run_id=None):
    """Write 'n_parts' parts of a constant value, as one run would."""
    writer = dt._PartWriter(
        str(parts_dir), "run_delta_t", LABELS, "ps", 1.0, run_id=run_id
    )
    for _ in range(n_parts):
        writer.add({lbl: np.full(2, float(value)) for lbl in LABELS})
        writer.flush()
    return writer.close()


def test_parts_are_tagged_and_manifested_per_run(tmp_path):
    parts_dir = tmp_path / "parts"
    dead = _run_parts(parts_dir, 111, 4, run_id="runA")
    live = _run_parts(parts_dir, 999, 2, run_id="runB")

    # Two runs, no file name collision: the dead run's parts are all intact.
    assert not set(dead) & set(live)
    assert all(os.path.isfile(p) for p in dead + live)

    runs = dt._part_runs(str(parts_dir))
    assert set(runs) == {"runA", "runB"}
    assert runs["runA"] == dead
    assert runs["runB"] == live


def test_combining_refuses_to_pool_two_runs(tmp_path):
    parts_dir = tmp_path / "parts"
    _run_parts(parts_dir, 111, 4, run_id="runA")
    _run_parts(parts_dir, 999, 2, run_id="runB")

    with pytest.raises(ValueError, match="2 different runs"):
        dt.combine_delta_t_parts(str(parts_dir))

    # Naming a run takes that run's parts, and nothing of the other's.
    out = dt.combine_delta_t_parts(
        str(parts_dir), str(tmp_path / "b.feather"), run="runB"
    )
    values = _columns(out)["0,0-1,1"]
    assert set(values) == {999.0}
    assert len(values) == 2 * 2


def test_legacy_untagged_parts_still_read_as_one_run(tmp_path):
    """A folder written before manifests existed keeps working."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    for i in range(3):
        dt._write_delta_feather(
            str(parts_dir / f"run_delta_t_{i}.feather"),
            {lbl: np.full(2, 5.0) for lbl in LABELS},
            LABELS,
            "ps",
        )
    assert list(dt._part_runs(str(parts_dir))) == ["legacy"]
    assert len(dt._find_parts(str(parts_dir))) == 3


def test_stage_one_refuses_to_start_on_top_of_stale_parts(tmp_path):
    root = tmp_path / "run"
    processed = root / "processed"
    processed.mkdir(parents=True)
    # The guard must fire before a single '.bin' is opened, so the file only
    # has to exist for 'find_bin_files' - it is never read.
    (root / "data_ORT1.bin").write_bytes(b"")
    _run_parts(processed / "run_delta_t_parts", 111, 2, run_id="dead")

    with pytest.raises(FileExistsError, match="earlier run"):
        dt.calculate_and_save_timestamp_differences(
            str(root),
            [[(0, 0)], [(1, 1)]],
            nframes=NFRAMES,
            apply_TDC_calibration=False,
        )
    # The refusal must not have destroyed the parts it refused over.
    assert len(dt._all_part_files(str(processed / "run_delta_t_parts"))) == 2


def test_two_runs_in_the_same_second_still_get_separate_ids(tmp_path):
    """Wall-clock alone is not unique: same script, same second, same pid."""
    parts_dir = tmp_path / "parts"
    dead = _run_parts(parts_dir, 111, 4)
    live = _run_parts(parts_dir, 999, 2)

    assert not set(dead) & set(live), "the re-run overwrote the dead run"
    assert len(dt._part_runs(str(parts_dir))) == 2
    assert all(os.path.isfile(p) for p in dead + live)


def test_untagged_parts_from_two_runs_are_flagged(tmp_path, capsys):
    """Legacy folders cannot be separated, but they can be called out."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    for i in range(4):
        dt._write_delta_feather(
            str(parts_dir / f"run_delta_t_{i}.feather"),
            {lbl: np.full(2, 5.0) for lbl in LABELS},
            LABELS,
            "ps",
        )
    # A re-run rewrites the low indices, so 0-1 end up NEWER than 2-3.
    for i, age in ((0, 500.0), (1, 400.0), (2, 900.0), (3, 800.0)):
        p = parts_dir / f"run_delta_t_{i}.feather"
        os.utime(p, (os.stat(p).st_atime, os.stat(p).st_mtime - age))

    dt._part_runs(str(parts_dir))
    assert "looks like two runs" in capsys.readouterr().out


# --------------------------------------------------------------------------
# reading a feather must not depend on how many batches it was written in
# --------------------------------------------------------------------------


def test_sources_are_memory_mapped_not_read_into_memory(tmp_path):
    """'pa.ipc.open_file(path)' materialises a whole record batch on demand.

    Invisible on a feather assembled from 64 MB parts, fatal on one written in
    a single batch - which is every feather the pre-parts code produced. A
    9.4 GB single-batch file then asks for 9.4 GB in one allocation, and the
    "streaming" read streams nothing.
    """
    path = str(tmp_path / "one_batch.feather")
    dt._write_delta_feather(
        path, {lbl: np.arange(100.0) for lbl in LABELS}, LABELS, "ps"
    )

    with dt._open_delta_feather(path) as reader:
        source = reader
        assert reader.num_record_batches == 1
        # Copied out, so it must survive the mapping being torn down.
        kept = reader.get_batch(0).column(0).to_numpy(zero_copy_only=False) + 0

    del source
    np.testing.assert_array_equal(kept, np.arange(100.0))


def test_one_batch_and_many_batches_histogram_identically(tmp_path):
    """The batch layout is a storage detail, never a result."""
    rng = np.random.default_rng(11)
    chunks = [{lbl: rng.normal(scale=300, size=250) for lbl in LABELS}
              for _ in range(8)]

    single = str(tmp_path / "single.feather")
    dt._write_delta_feather(
        single,
        {lbl: np.concatenate([c[lbl] for c in chunks]) for lbl in LABELS},
        LABELS,
        "ps",
    )

    writer = _writer(tmp_path)
    for chunk in chunks:
        writer.add(chunk)
    parts = writer.close()
    many = str(tmp_path / "many.feather")
    dt._combine_delta_feathers(parts, many)

    assert len(parts) > 1, "the fixture must actually roll over"
    with dt._open_delta_feather(many) as reader:
        assert reader.num_record_batches > 1

    edges = dt._delta_hist_edges(77.0e-12, 3080.0e-12)
    kwargs = dict(edges=edges, pairs=None, quiet=True)
    one_counts, one_stats = dt._stream_delta_histogram([single], **kwargs)
    many_counts, many_stats = dt._stream_delta_histogram([many], **kwargs)

    np.testing.assert_array_equal(one_counts, many_counts)
    assert one_stats["total"] == many_stats["total"] > 0


# --- subtract_background: the frame-shifted columns -----------------------
#
# The feather grows one column per pair per lag, named '<pair>@lag<k>', with
# lag 0 keeping the bare pair name. That naming is the whole compatibility
# story: every existing reader still finds exactly the columns it always did,
# and a reader that pools "every pair column" must not quietly hoover up 9x
# the data because the background happens to be in the same file.


def _lagged_feather(tmp_path, lags=(1, 2, 3), nframes=4):
    """Write a small lagged feather by hand, one known value per column."""
    path = str(tmp_path / "run_delta_t.feather")
    columns = {}
    for k in (0, *lags):
        for i, label in enumerate(LABELS):
            # lag 0 holds a peak at 0; the shifted lags hold only 'background'
            # away from it, so a subtraction has something to remove.
            values = (
                np.array([0.0, 0.0, 200.0]) if k == 0 else np.array([200.0])
            )
            columns[dt._lag_column(label, k)] = values + i
    dt._write_delta_feather(
        path,
        columns,
        list(columns),
        "ps",
        dt._lag_meta((0, *lags), nframes),
    )
    return path


def test_lag_columns_are_named_apart_from_the_pair_columns(tmp_path):
    path = _lagged_feather(tmp_path)
    with pa.ipc.open_file(path) as reader:
        names = reader.schema.names

    assert names[: len(LABELS)] == LABELS, "lag 0 must keep the bare names"
    assert "0,0-1,1@lag2" in names
    assert not any("@" in lbl for lbl in LABELS), "a pair label cannot hold @"


def test_pooling_every_pair_column_ignores_the_lag_columns(tmp_path):
    """Otherwise a plain plot of a lagged feather would inflate ~9x."""
    path = _lagged_feather(tmp_path)
    columns, _, _ = dt._pooled_columns([path], None)

    assert columns == LABELS


def test_the_lag_meta_travels_through_a_combine(tmp_path):
    """A combined feather must describe itself as its parts did."""
    parts_dir = str(tmp_path / "parts")
    writer = dt._PartWriter(
        parts_dir,
        "run_delta_t",
        [dt._lag_column(LABELS[0], k) for k in (0, 1)],
        "ps",
        1 / 1024,
        extra_meta=dt._lag_meta((0, 1), 10),
    )
    for _ in range(4):
        writer.add(
            {dt._lag_column(LABELS[0], k): np.arange(40.0) for k in (0, 1)}
        )
    parts = writer.close()
    assert len(parts) > 1, "the cap was never reached"

    combined = str(tmp_path / "combined_delta_t.feather")
    dt._combine_delta_feathers(parts, combined)

    assert dt._read_lag_meta([combined]) == {"lags": [0, 1], "nframes": 10}


def test_mixed_lag_sets_are_refused(tmp_path):
    a = str(tmp_path / "a_delta_t.feather")
    b = str(tmp_path / "b_delta_t.feather")
    for path, lags in ((a, (0, 1)), (b, (0, 2))):
        dt._write_delta_feather(
            path,
            {LABELS[0]: np.array([0.0])},
            [LABELS[0]],
            "ps",
            dt._lag_meta(lags, 10),
        )
    with pytest.raises(ValueError, match="different lag sets"):
        dt._read_lag_meta([a, b])


def test_every_lag_is_histogrammed_in_one_pass(tmp_path):
    """One row per lag, lag 0 first, and row 0 is the plain histogram."""
    path = _lagged_feather(tmp_path)
    edges = dt._delta_hist_edges(BIN_S, WINDOW_S)
    kwargs = {"edges": edges, "quiet": True}

    plain, plain_stats = dt._stream_delta_histogram([path], **kwargs)
    lagged, lag_stats = dt._stream_delta_histogram(
        [path], lags=[0, 1, 2, 3], **kwargs
    )

    assert lagged.shape == (4, plain.size)
    np.testing.assert_array_equal(lagged[0], plain)
    # The stats describe the measurement, not the background bookkeeping.
    assert lag_stats["n"] == plain_stats["n"]
    assert lag_stats["total"] == plain_stats["total"]
    # Each shifted lag holds one value per pair, at +200 ps.
    assert lagged[1:].sum() == 3 * len(LABELS)


def test_subtracting_needs_a_feather_with_lag_columns(tmp_path):
    root = _hist_run(tmp_path)
    with pytest.raises(ValueError, match="frame-shifted columns"):
        dt.compute_and_save_delta_histogram(
            str(root), subtract_background=True, quiet=True
        )


def test_the_fit_model_follows_the_histogram(tmp_path):
    """Triangle on raw data, flat on a residual - and no way to say otherwise.

    The two are one decision, not two: the accidental pedestal is either still
    in the histogram, where on ORT it is a triangle, or it has been measured
    away and the residual is flat about zero. The old fixed 'flat' default was
    the combination wrong on raw data, fitting a constant to a slope.
    """
    import matplotlib

    matplotlib.use("Agg")
    root = _hist_run(tmp_path)

    _, _, raw = dt.collect_and_plot_timestamp_differences(str(root))

    # 'half_base' is the triangle's own parameter; a flat fit has no such thing.
    assert "half_base" in raw
    assert "background" not in inspect.signature(
        dt.collect_and_plot_timestamp_differences
    ).parameters


def test_the_subtracted_driver_returns_the_residual(tmp_path):
    """End to end on the feather path: residual out, both figures on disk."""
    import matplotlib

    matplotlib.use("Agg")
    root = tmp_path / "run"
    (root / "processed").mkdir(parents=True)
    path = _lagged_feather(root / "processed")

    # Three bins in the fit window against four Gaussian parameters: this
    # fixture is about the subtraction, and the peak fit is expected to bow out.
    with pytest.warns(RuntimeWarning, match="did not converge"):
        counts, centers, fit = dt.collect_and_plot_timestamp_differences(
            feather_path=path,
            bin_width_s=100.0e-12,
            plot_window_s=500.0e-12,
            fit_window_s=100.0e-12,
            subtract_background=True,
            label="test",
        )

    assert counts.shape == centers.shape
    np.testing.assert_allclose(counts, fit["subtraction"]["residual"])
    # Two pairs, two entries each at delta = 0 and 1 ps -> both in the 0 bin,
    # against one background entry per lag well away from it.
    np.testing.assert_allclose(fit["model_free"]["n"], 4.0)
    assert np.isnan(fit["car"]), "a residual has no peak-to-baseline ratio"

    results = root / "results" / "coincidences"
    assert (results / "run_test_delta_t_sub.png").is_file()
    assert (results / "run_test_background_subtraction.png").is_file()


# --------------------------------------------------------------------------
# a flat-background fit has to survive a background-subtracted residual
# --------------------------------------------------------------------------


def _residual_peak(baseline, sigma_ps=500.0, seed=0):
    """A lag-subtracted-style histogram: a peak on zero-mean noise."""
    x = np.arange(-10000, 10000, 20.0)
    rng = np.random.default_rng(seed)
    y = (
        1300 * np.exp(-(x**2) / (2 * sigma_ps**2))
        + baseline
        + rng.normal(0, 40, x.size)
    )
    return x, y


@pytest.mark.parametrize("baseline", [-3.0, 0.0, 150.0])
def test_the_flat_fit_survives_a_zero_mean_baseline(baseline):
    """Subtraction leaves a residual whose baseline sits at or below zero.

    The flat level used to be bounded at zero from below while being seeded
    from the tail median, so on exactly the histograms 'subtract_background'
    produces the seed landed on or outside its own bound, curve_fit raised
    'x0 is infeasible', and the caller drew a plot with no fit on it.
    """
    x, y = _residual_peak(baseline)

    fit = dt.fit_gaussian_peak(x, y, fit_window=1000.0)

    assert fit["ok"]
    assert fit["sigma"] == pytest.approx(500.0, rel=0.05)


def test_a_fit_that_does_not_converge_says_so():
    """Silence is what made the failure hard to see - it must warn."""
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit = dt.fit_gaussian_peak(np.array([0.0, 1.0]), np.array([1.0, 1.0]))

    assert not fit["ok"]
    assert np.isnan(fit["sigma"])


def test_a_width_the_data_does_not_determine_is_not_a_fit(monkeypatch):
    """Converging is not succeeding.

    Fitting a Gaussian on a *flat* background over a stretch of data that has
    no peak in it - a lag-subtracted residual that is still a triangle, say,
    which is what a non-stationary run leaves - lets amplitude, width and
    baseline trade against each other freely. curve_fit returns, and the
    covariance is what says the answer is worth nothing: on a real Ne-lamp run
    it came out as sigma 39 ns +/- 2.7 us, drawn as a curve and printed as a
    number. The optimiser's basin is not reproducible across platforms, so the
    covariance is handed over directly here; the guard is what is under test.
    """
    x, y = _residual_peak(0.0)
    popt = np.array([1300.0, 0.0, 39257.0, 0.0])
    pcov = np.diag([1e6, 1e6, 2665891.0**2, 1e6])
    monkeypatch.setattr(dt, "curve_fit", lambda *a, **k: (popt, pcov))

    with pytest.warns(RuntimeWarning, match="not determined by the data"):
        fit = dt.fit_gaussian_peak(x, y, fit_window=1000.0)

    assert not fit["ok"]
    assert np.isnan(fit["sigma"])


# --------------------------------------------------------------------------
# every reported width carries its uncertainty
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "err", "kwargs", "expected"), [
    (76.5, 34.2, {}, "76 \u00b1 34 ps"),
    (76.5, 34.2, {"sep": " +- "}, "76 +- 34 ps"),
    (76.5, np.nan, {}, "76 ps"),
    (76.5, np.inf, {}, "76 ps"),
    (1234.0, 56.0, {"unit": "ns"}, "1234 \u00b1 56 ns"),
])
def test_a_width_is_printed_with_its_error(value, err, kwargs, expected):
    """A fit with no covariance still has a best-fit width worth showing.

    Printing '+- nan' beside it would read as a bug rather than as 'unknown',
    so the error is dropped instead.
    """
    assert dt._pm(value, err, **kwargs) == expected


def test_the_derived_widths_carry_derived_errors(tmp_path):
    """FWHM and per-detector are sigma times a constant - so are their errors.

    They are worked out where the values are, not in the legend, so a number
    read off a figure and the same number out of the returned dict cannot
    disagree.
    """
    import matplotlib

    matplotlib.use("Agg")
    root = _hist_run(tmp_path)

    _, _, fit = dt.collect_and_plot_timestamp_differences(str(root))

    assert fit["fwhm_err"] == pytest.approx(2.355 * fit["sigma_err"])
    assert fit["jitter_per_detector_err"] == pytest.approx(
        fit["sigma_err"] / np.sqrt(2)
    )
