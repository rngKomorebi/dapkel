"""Guards on the counting/timestamping split in 'dcr_analysis'.

Under the ``S*T``/``ORT`` timestamping programs there is no photon counter -
'unpack' returns the low bits of the coarse timestamp in that slot. Summing it
as if it were a count inflates the DCR by ~40-50x, and the resulting map still
looks like a plausible DCR map, so nothing downstream catches it. These tests
pin the three things that keep it caught: the extraction is chosen by ``mode``,
a tag whose letter contradicts ``mode`` is rejected, and the pile-up inversion
is the exact inverse of the one-hit-per-frame censoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from dapkel.functions import dcr_analysis

# --------------------------------------------------------------------------
# mode <-> tag agreement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ["S0T", "S3T", "ORT", "s0t"])
def test_timestamp_tag_rejects_count_mode(tag):
    """The mistake this whole module exists to prevent."""
    with pytest.raises(ValueError, match="timestamp-mode"):
        dcr_analysis._check_mode("count", tag)


@pytest.mark.parametrize("tag", ["S0C", "S3C", "ORC"])
def test_count_tag_rejects_timestamp_mode(tag):
    with pytest.raises(ValueError, match="count-mode"):
        dcr_analysis._check_mode("timestamp", tag)


@pytest.mark.parametrize(
    "mode, tag",
    [
        ("count", "S0C"),
        ("count", "ORC"),
        ("count", ""),  # '' matches every file: nothing to contradict
        ("timestamp", "S0T"),
        ("timestamp", "ORT"),
        ("timestamp", ""),
        ("count", "SPAD0"),  # tag with no mode letter at all
    ],
)
def test_consistent_combinations_pass(mode, tag):
    dcr_analysis._check_mode(mode, tag)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="'timestamp' or 'count'"):
        dcr_analysis._check_mode("counts")


def test_quadrant_tag_respells_the_layout_keys():
    """'_SPAD_LAYOUT' is keyed by the counting tags; 'T' data is the same grid."""
    assert dcr_analysis._quadrant_tag("S2C", "timestamp") == "S2T"
    assert dcr_analysis._quadrant_tag("S2C", "count") == "S2C"


# --------------------------------------------------------------------------
# which field each mode reads
# --------------------------------------------------------------------------


def _fake_frames(nframes=8):
    """One pixel fires in half the frames; the counter field is timestamp junk.

    Shaped like 'unpack' output: ``(32, 32, nframes)`` timestamps and counts.
    """
    ts = np.zeros((32, 32, nframes))
    pc = np.zeros((32, 32, nframes))
    fired = np.arange(0, nframes, 2)
    ts[5, 7, fired] = 900.0  # a valid, late timestamp
    pc[5, 7, fired] = 450.0  # what those same bits look like as a "count"
    return ts, pc


def test_modes_read_different_fields(monkeypatch):
    nframes, frame_time = 8, 1e-6
    ts, pc = _fake_frames(nframes)
    monkeypatch.setattr(
        "dapkel.core.reduce.unpack",
        lambda fp, n, compute_time_series=True: (ts, pc),
    )

    live = nframes * frame_time
    timestamped = dcr_analysis._accumulate_dcr(
        ["fake.bin"], nframes, frame_time, mode="timestamp"
    )
    counted = dcr_analysis._accumulate_dcr(
        ["fake.bin"], nframes, frame_time, mode="count"
    )

    # timestamp mode counts frames that fired: 4 of them
    assert timestamped[5, 7] == pytest.approx(4 / live)
    # count mode sums the raw field: 4 x 450, i.e. 450x too high here
    assert counted[5, 7] == pytest.approx(4 * 450 / live)
    assert counted[5, 7] > 100 * timestamped[5, 7]
    # every other pixel is dark in both
    assert timestamped.sum() == pytest.approx(timestamped[5, 7])


def test_timestamp_mode_cannot_exceed_one_count_per_frame(monkeypatch):
    """The physical ceiling that the count-mode bug blows straight past."""
    nframes, frame_time = 8, 1e-6
    ts, pc = _fake_frames(nframes)
    monkeypatch.setattr(
        "dapkel.core.reduce.unpack",
        lambda fp, n, compute_time_series=True: (ts, pc),
    )
    dcr = dcr_analysis._accumulate_dcr(
        ["a.bin", "b.bin"], nframes, frame_time, mode="timestamp"
    )
    assert dcr.max() <= 1.0 / frame_time


# --------------------------------------------------------------------------
# pile-up inversion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [1e3, 5e3, 5e4, 1.2e5])
def test_pileup_correction_recovers_the_true_rate(rate):
    """Feed in the occupancy a Poisson source of known rate would produce."""
    nframes, window = 100_000, 8e-6
    occupancy = np.array([[(1 - np.exp(-rate * window)) * nframes]])

    raw = occupancy / (nframes * window)
    corrected = dcr_analysis._pileup_correct(occupancy, nframes) / (
        nframes * window
    )

    assert corrected[0, 0] == pytest.approx(rate, rel=1e-9)
    assert raw[0, 0] <= corrected[0, 0]  # censoring can only under-report


def test_pileup_correction_is_a_no_op_at_zero():
    assert dcr_analysis._pileup_correct(np.zeros((2, 2)), 1000).tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_fully_saturated_pixel_stays_finite():
    """A pixel with p = 1 has no MLE; it must clip, not go to 'inf'."""
    nframes = 8192
    out = dcr_analysis._pileup_correct(np.array([[float(nframes)]]), nframes)
    assert np.isfinite(out).all()
    assert out[0, 0] > nframes  # a lower bound above the raw occupancy


def test_pileup_correct_is_ignored_in_count_mode(monkeypatch):
    nframes, frame_time = 8, 1e-6
    ts, pc = _fake_frames(nframes)
    monkeypatch.setattr(
        "dapkel.core.reduce.unpack",
        lambda fp, n, compute_time_series=True: (ts, pc),
    )
    plain = dcr_analysis._accumulate_dcr(
        ["fake.bin"], nframes, frame_time, mode="count"
    )
    flagged = dcr_analysis._accumulate_dcr(
        ["fake.bin"], nframes, frame_time, mode="count", pileup_correct=True
    )
    assert np.array_equal(plain, flagged)


# --------------------------------------------------------------------------
# the sidecar records which mode produced the map
# --------------------------------------------------------------------------


def test_meta_records_mode_and_correction():
    meta = dcr_analysis._dcr_meta(
        "path", 8192, "S0T", None, "short_exposure", 8e-6, "timestamp", True
    )
    assert meta["mode"] == "timestamp"
    assert meta["pileup_correct"] is True


def test_meta_never_claims_a_correction_in_count_mode():
    meta = dcr_analysis._dcr_meta(
        "path", 8192, "S0C", None, "short_exposure", 8e-6, "count", True
    )
    assert meta["pileup_correct"] is False
