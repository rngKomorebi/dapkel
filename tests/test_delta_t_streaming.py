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
