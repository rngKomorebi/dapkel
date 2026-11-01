"""Guards on the counting/timestamping split, and the DCR built on it.

Under the ``S*T``/``ORT`` timestamping programs there is no photon counter -
'unpack' returns the low bits of the coarse timestamp in that slot. Summing it
as if it were a count inflates the DCR by ~40-50x, and the resulting map still
looks plausible, so nothing downstream catches it. The count mode is therefore
not a parameter at all: it is read off the tag ('core.reduce.readout_mode').
These tests pin that, and the pile-up inversion being the exact inverse of the one-hit-per-frame censoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from dapkel.core import reduce, store
from dapkel.functions import dcr_analysis

# --------------------------------------------------------------------------
# the tag decides what a count is
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ["S0T", "S3T", "ORT", "s0t", "ORT_2"])
def test_timestamping_tags_count_frames(tag):
    assert reduce.readout_mode(tag) == "timestamp"


@pytest.mark.parametrize("tag", ["S0C", "S3C", "ORC"])
def test_counting_tags_sum_the_counter(tag):
    assert reduce.readout_mode(tag) == "count"


@pytest.mark.parametrize("tag", ["", "SPAD0_S0", "run"])
def test_a_tag_without_a_program_letter_is_refused(tag):
    """Guessing here is exactly the ~50x mistake, so it is an error."""
    with pytest.raises(ValueError, match="neither a timestamping"):
        reduce.readout_mode(tag)


# --------------------------------------------------------------------------
# which field each mode reads - through the real 'process_dcr'
# --------------------------------------------------------------------------

NFRAMES = 8
WINDOW_S = 1e-6


def _fake_unpack(monkeypatch, fires=range(0, NFRAMES, 2)):
    """One pixel fires in the given frames; its counter field is timestamp junk.

    Patched where 'core.reduce' looks it up, so the analyses run unmodified.
    """
    ts = np.zeros((32, 32, NFRAMES))
    pc = np.zeros((32, 32, NFRAMES))
    fired = list(fires)
    ts[5, 7, fired] = 900.0  # a valid, late timestamp
    pc[5, 7, fired] = 450.0  # what those same bits look like as a "count"
    monkeypatch.setattr(
        "dapkel.core.reduce.unpack", lambda fp, n, compute_time_series=True: (ts, pc)
    )


def _dcr(tmp_path, tag, n_files=1, **kwargs):
    for i in range(n_files):
        (tmp_path / f"run_{tag}_{i}.bin").write_bytes(b"")
    dcr_analysis.process_dcr(
        str(tmp_path), NFRAMES, tag, open_shutter_time_s=WINDOW_S, **kwargs
    )
    return store.load_map(str(tmp_path), kind="dcr", tag=tag)


def test_modes_read_different_fields(tmp_path, monkeypatch):
    _fake_unpack(monkeypatch)
    live = NFRAMES * WINDOW_S

    timestamped, meta_t = _dcr(tmp_path, "S0T")
    counted, meta_c = _dcr(tmp_path, "S0C")

    assert timestamped[5, 7] == pytest.approx(4 / live)  # 4 photons
    assert counted[5, 7] == pytest.approx(4 * 450 / live)  # 4 x the raw field
    assert np.nansum(timestamped) == pytest.approx(timestamped[5, 7])
    assert (meta_t["mode"], meta_c["mode"]) == ("timestamp", "count")


def test_timestamp_mode_cannot_exceed_one_count_per_frame(tmp_path, monkeypatch):
    _fake_unpack(monkeypatch, fires=range(NFRAMES))
    dcr, _ = _dcr(tmp_path, "S0T", n_files=2)
    assert dcr.max() <= 1.0 / WINDOW_S


@pytest.mark.parametrize("k", [1, 4, 7])
def test_pileup_correction_is_the_poisson_inverse(tmp_path, monkeypatch, k):
    """Turn the occupancy p = k/N into the MLE rate -ln(1 - p) / window."""
    _fake_unpack(monkeypatch, fires=range(k))

    raw, _ = _dcr(tmp_path, "S0T")
    corrected, meta = _dcr(tmp_path, "S0T", pileup_correct=True)

    assert corrected[5, 7] == pytest.approx(-np.log1p(-k / NFRAMES) / WINDOW_S)
    assert raw[5, 7] <= corrected[5, 7]  # censoring can only under-report
    assert corrected[0, 0] == 0.0
    assert meta["pileup_correct"] is True


def test_a_fully_saturated_pixel_stays_finite(tmp_path, monkeypatch):
    """An occupancy of 1 has no MLE; it is clipped to the half-count level."""
    _fake_unpack(monkeypatch, fires=range(NFRAMES))
    dcr, _ = _dcr(tmp_path, "S0T", pileup_correct=True)

    assert np.isfinite(dcr).all()
    assert dcr[5, 7] > 1.0 / WINDOW_S  # a lower bound above the raw occupancy


def test_pileup_correct_is_ignored_and_not_claimed_in_count_mode(tmp_path, monkeypatch):
    _fake_unpack(monkeypatch)
    plain, _ = _dcr(tmp_path, "S0C")
    flagged, meta = _dcr(tmp_path, "S0C", pileup_correct=True)

    np.testing.assert_array_equal(plain, flagged)
    assert meta["pileup_correct"] is False


def test_short_exposure_without_a_shutter_time_is_refused(tmp_path):
    (tmp_path / "run_S0C_0.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="open_shutter_time_s"):
        dcr_analysis.process_dcr(str(tmp_path), NFRAMES, "S0C", open_shutter_time_s=None)
