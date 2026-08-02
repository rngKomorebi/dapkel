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


def test_end_to_end_streams_parts_and_combines_them(tmp_path):
    folder = _fake_run(tmp_path)

    out = dt.calculate_and_save_timestamp_differences(
        str(folder),
        PAIRS,
        apply_TDC_calibration=False,
        nframes=NFRAMES,
        part_size_mb=1 / 1024,  # 1 KB, so it rolls over every few files
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


def test_end_to_end_part_size_does_not_change_the_result(tmp_path):
    """Sharding is a memory strategy; it must not touch the numbers."""
    results = {}
    for part_mb in (1 / 1024, 1024.0):
        folder = _fake_run(tmp_path / f"cap{part_mb}")
        out = dt.calculate_and_save_timestamp_differences(
            str(folder),
            PAIRS,
            apply_TDC_calibration=False,
            nframes=NFRAMES,
            part_size_mb=part_mb,
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
        "part_size_mb": 1 / 1024,
    }
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
        "part_size_mb": 1 / 1024,
    }
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


def test_end_to_end_keep_parts_false_removes_them(tmp_path):
    folder = _fake_run(tmp_path, n_files=3)
    out = dt.calculate_and_save_timestamp_differences(
        str(folder),
        PAIRS,
        apply_TDC_calibration=False,
        nframes=NFRAMES,
        part_size_mb=1 / 1024,
        keep_parts=False,
    )

    parts_dir = tmp_path / "run" / "processed" / "run_delta_t_parts"
    assert not parts_dir.exists()
    for values in _columns(out).values():
        assert 0.9 * 3 * NFRAMES <= values.size <= 3 * NFRAMES


# --- stage 1b: streaming the feather down to a histogram -------------------
#
# The combined feather of a 10 000-file run reaches hundreds of GB, so the
# read side has to stream too. These tests pin the property that makes the
# streamed histogram usable as the *baseline* every later optimisation is
# measured against: it is bin-for-bin what loading the whole table and
# histogramming it in one go would have produced.

TIME_UNIT_PS = 77.0
BIN_PS = 77.0
WINDOW_PS = 3000.0


def _reference_histogram(
    feather, *, edges, time_unit_ps=TIME_UNIT_PS, window_ps=WINDOW_PS
):
    """Histogram the old way: load everything, pool it, bin it once."""
    df, unit = dt._read_delta_feather(feather)
    vals = df.to_numpy().ravel()
    pooled = vals[~np.isnan(vals)]
    deltas_ps = pooled if unit == "ps" else pooled * time_unit_ps
    inside = deltas_ps[np.abs(deltas_ps) <= window_ps]
    return np.histogram(inside, bins=edges)[0], inside.size


def _spread_deltas(rng, n):
    """Codes with a coincidence peak at 0 on a wide accidental background."""
    peak = rng.normal(0.0, 4.0, size=n // 4)
    accidental = rng.uniform(-1300.0, 1300.0, size=n - n // 4)
    return np.concatenate([peak, accidental])


def _feather_of_parts(tmp_path, n_batches=12, unit="code"):
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
    edges = dt._delta_hist_edges(BIN_PS, WINDOW_PS)

    counts, stats = dt._stream_delta_histogram(
        [combined],
        edges=edges,
        time_unit_ps=TIME_UNIT_PS,
        plot_window_ps=WINDOW_PS,
        quiet=True,
    )
    expected, expected_n = _reference_histogram(combined, edges=edges)

    assert counts.sum() > 0, "the fixture produced no counts in the window"
    np.testing.assert_array_equal(counts, expected)
    assert stats["n"] == expected_n
    assert stats["unit"] == "code"


def test_streaming_reads_more_than_one_batch(tmp_path):
    """Guards the test above: a single-batch file would prove nothing."""
    combined, _ = _feather_of_parts(tmp_path)
    with pa.ipc.open_file(combined) as reader:
        assert reader.num_record_batches > 1


def test_parts_and_combined_give_the_same_histogram(tmp_path):
    """Plotting from the parts must not need the giant combined file."""
    combined, parts_dir = _feather_of_parts(tmp_path)
    edges = dt._delta_hist_edges(BIN_PS, WINDOW_PS)
    kwargs = {
        "edges": edges,
        "time_unit_ps": TIME_UNIT_PS,
        "plot_window_ps": WINDOW_PS,
        "quiet": True,
    }

    from_combined, stats_c = dt._stream_delta_histogram([combined], **kwargs)
    from_parts, stats_p = dt._stream_delta_histogram(
        dt._find_parts(parts_dir), **kwargs
    )

    np.testing.assert_array_equal(from_parts, from_combined)
    assert stats_p["n"] == stats_c["n"]


def test_ps_feathers_are_not_rescaled(tmp_path):
    """A 'ps' feather is already calibrated; time_unit_ps must not apply."""
    path = str(tmp_path / "run_delta_t.feather")
    dt._write_delta_feather(
        path, {"0,0-1,1": np.array([0.0, 100.0, -100.0])}, ["0,0-1,1"], "ps"
    )
    edges = dt._delta_hist_edges(BIN_PS, WINDOW_PS)

    counts, stats = dt._stream_delta_histogram(
        [path],
        edges=edges,
        time_unit_ps=TIME_UNIT_PS,
        plot_window_ps=WINDOW_PS,
        quiet=True,
    )
    np.testing.assert_array_equal(
        counts, _reference_histogram(path, edges=edges)[0]
    )
    assert stats["unit"] == "ps"
    assert stats["granularity_ps"] == 100.0  # not 100 * 77


def test_padding_is_dropped_and_measured(tmp_path):
    """Padding must not be counted - and its share is worth reporting."""
    path = str(tmp_path / "run_delta_t.feather")
    dt._write_delta_feather(
        path,
        {"0,0-1,1": np.zeros(100), "0,0-2,2": np.zeros(4)},
        ["0,0-1,1", "0,0-2,2"],
        "code",
    )
    counts, stats = dt._stream_delta_histogram(
        [path],
        edges=dt._delta_hist_edges(BIN_PS, WINDOW_PS),
        time_unit_ps=TIME_UNIT_PS,
        plot_window_ps=WINDOW_PS,
        quiet=True,
    )

    assert counts.sum() == 104  # 104 real values, not 200 padded cells
    assert stats["total"] == 104
    assert stats["slots"] == 200
    assert stats["padding_fraction"] == pytest.approx(0.48)


def test_out_of_window_differences_are_excluded_but_still_counted(tmp_path):
    path = str(tmp_path / "run_delta_t.feather")
    inside = np.zeros(6)
    outside = np.full(4, 1000.0)  # 1000 codes * 77 ps >> 3000 ps
    dt._write_delta_feather(
        path,
        {"0,0-1,1": np.concatenate([inside, outside])},
        ["0,0-1,1"],
        "code",
    )
    counts, stats = dt._stream_delta_histogram(
        [path],
        edges=dt._delta_hist_edges(BIN_PS, WINDOW_PS),
        time_unit_ps=TIME_UNIT_PS,
        plot_window_ps=WINDOW_PS,
        quiet=True,
    )

    assert counts.sum() == 6
    assert stats["n"] == 6
    assert stats["total"] == 10  # the granularity guard sees them all


def test_missing_pairs_are_reported_not_fatal(tmp_path, capsys):
    combined, _ = _feather_of_parts(tmp_path)
    counts, stats = dt._stream_delta_histogram(
        [combined],
        edges=dt._delta_hist_edges(BIN_PS, WINDOW_PS),
        time_unit_ps=TIME_UNIT_PS,
        plot_window_ps=WINDOW_PS,
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
        str(root), bin_width_ps=BIN_PS * 4, quiet=True
    )

    assert info["reused"] is False
    assert counts.size == len(edges) - 1
    assert counts.size < len(dt._delta_hist_edges(BIN_PS, WINDOW_PS)) - 1


def test_a_rewritten_feather_re_streams(tmp_path):
    """Stale counts against fresh data would be the worst kind of wrong."""
    root = _hist_run(tmp_path)
    first, _, _ = dt.compute_and_save_delta_histogram(str(root), quiet=True)

    feather = str(root / "processed" / "run_delta_t.feather")
    dt._write_delta_feather(
        feather, {"0,0-1,1": np.zeros(500)}, ["0,0-1,1"], "code"
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

    counts, centers, fit = dt.collect_and_plot_timestamp_differences(
        str(root), background="flat"
    )

    assert counts.shape == centers.shape
    assert fit["n"] == int(counts.sum())
    assert os.path.isfile(
        root / "results" / "coincidences" / "run_spdc_delta_t.png"
    )
    # The fitted numbers must come from the same counts the caller gets back.
    refit = dt.fit_gaussian_peak(centers, counts, fit_window=1000.0)
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
