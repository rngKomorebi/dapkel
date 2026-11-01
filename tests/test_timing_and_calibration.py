"""The TDC full scale is measured, and the calibration is 8 taps at any clock.

The largest code a run produced *is* the range its TDC swept, so no analysis is
told a clock. The one place a clock period is irreducible is building the
calibration: the density test fixes relative widths only. And the ring has only
8 distinct widths - code ``k`` is tap ``k % 8`` - so the calibration is those 8
per pixel, reduced from the density test, valid at every clock and applied by
'core.calibration.codes_to_ps' with no code range to fall off.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from dapkel.core import calibration, io, store, timing
from dapkel.functions import tdc_calibration as tc

# Largest code seen at each clock on the 2026.08.07 rate-vs-clock set, which is
# what 'timing.CODE_S' was measured against.
MEASURED = {
    50e6: (274, 20e-9),
    10e6: (1378, 100e-9),
    5e6: (2760, 200e-9),
    2e6: (6913, 500e-9),
}
TAPS = [1.00, 1.02, 0.97, 1.01, 1.06, 0.96, 0.97, 1.01]


@pytest.mark.parametrize(("clock", "case"), sorted(MEASURED.items()))
def test_the_full_scale_comes_back_out_of_the_largest_code(clock, case):
    max_code, full_scale_s = case
    assert timing.full_scale_from_max_code(max_code) == pytest.approx(
        full_scale_s, rel=0.05
    )


def test_a_run_with_no_valid_timestamp_cannot_be_measured():
    with pytest.raises(ValueError, match="no valid timestamp"):
        timing.full_scale_from_max_code(0)


def test_a_wrapped_coarse_counter_is_called_out():
    """Below ~1.7 MHz the counter wraps and the timestamps mean nothing."""
    with pytest.warns(UserWarning, match="wrapped"):
        timing.full_scale_from_max_code(timing.MAX_TS_CODE + 1)


def test_the_live_time_follows_the_firmware():
    assert timing.resolve_live_time(
        "short_exposure", open_shutter_time_s=1e-7
    )[0] == pytest.approx(1e-7)
    assert timing.resolve_live_time("long_exposure", exp_time_s=1e-6)[0] == (
        pytest.approx(10e-6)
    )
    with pytest.raises(ValueError, match="open_shutter_time_s"):
        timing.resolve_live_time("short_exposure")
    for gone in ("short_window", "full_window"):
        with pytest.raises(ValueError, match="firmware_version"):
            timing.resolve_live_time(gone, open_shutter_time_s=1e-7)


# --------------------------------------------------------------------------
# the density test, reduced to 8 taps
# --------------------------------------------------------------------------


def _ring_histogram(taps, n_codes, events_per_lap=5000, stop=None):
    """Build the code histogram of a ring whose 8 taps have the given widths.

    Under uniform illumination a code's counts are proportional to its width.
    ``stop`` truncates one pixel's own ruler - a faster-running oscillator.
    """
    widths = np.asarray(taps, dtype=np.float64)
    counts = np.rint(
        np.broadcast_to(
            widths[np.arange(n_codes) % calibration.TAPS_PER_LAP], (32, 32, n_codes)
        )
        * events_per_lap
    ).astype(np.int64)
    if stop is not None:
        counts[5, 7, stop:] = 0
    return counts


def test_the_tap_widths_come_back_out():
    taps = tc._histogram_to_taps(_ring_histogram(TAPS, 800), 100e-9)

    np.testing.assert_allclose(
        taps[0, 0] / taps[0, 0].mean(), np.array(TAPS) / np.mean(TAPS), rtol=1e-9
    )
    # the eight sum to the ring period, and the laps tile one clock period
    assert taps[0, 0].sum() * 800 / calibration.TAPS_PER_LAP == pytest.approx(100e3)


def test_codes_land_on_the_dense_density_test_table_exactly():
    """Same measurement, one number per tap instead of one per code.

    The dense table is the bin-centre CDF of the code histogram; a ring that
    really repeats its 8 widths every lap must reproduce it bin for bin.
    """
    counts = _ring_histogram(TAPS, 800)
    h = counts[16, 16].astype(float)
    dense = 100e3 * (np.cumsum(h) - h / 2) / h.sum()

    taps = tc._histogram_to_taps(counts, 100e-9)

    np.testing.assert_allclose(
        calibration.codes_to_ps(np.arange(800), taps[16, 16]), dense, atol=1e-9
    )


def test_a_faster_pixel_keeps_its_shorter_lap():
    """Per-pixel gain is the dominant correction - it must survive the reduction."""
    lap = tc._histogram_to_taps(_ring_histogram(TAPS, 800, stop=780), 100e-9).sum(-1)
    assert lap[5, 7] / lap[0, 0] == pytest.approx(800 / 780, rel=1e-6)


def test_one_table_serves_every_clock():
    """The taps belong to the ring; the clock only sets how many codes."""
    slow = tc._histogram_to_taps(_ring_histogram(TAPS, 4000), 500e-9)
    fast = tc._histogram_to_taps(_ring_histogram(TAPS, 800), 100e-9)
    np.testing.assert_allclose(slow, fast, rtol=1e-9)


def test_codes_convert_at_any_range_with_nothing_to_fall_off():
    """No table length, so no code is ever past the end of the calibration."""
    taps = np.array(TAPS) * 9.0
    codes = np.arange(timing.MAX_TS_CODE + 1)

    t = calibration.codes_to_ps(codes, taps)

    assert (np.diff(t) > 0).all()
    assert t[-1] == pytest.approx(
        (codes[-1] // 8) * taps.sum() + taps[: codes[-1] % 8].sum() + taps[codes[-1] % 8] / 2
    )


# --------------------------------------------------------------------------
# the packaged tables, picked by the file tag
# --------------------------------------------------------------------------


def _packaged(spad):
    from importlib.resources import files

    with files("dapkel.params").joinpath(
        "calibration_data", f"D0_M0_TDC_COMPACT_SPAD{spad}_S{spad}.npy"
    ).open("rb") as fh:
        return np.load(fh)


@pytest.mark.parametrize(("tag", "spad"), [("S0T", 0), ("S3T", 3), ("s2t", 2)])
def test_a_single_spad_tag_gets_that_spad_s_table(tag, spad):
    np.testing.assert_array_equal(calibration.board_taps("D0", "M0", tag), _packaged(spad))


def test_the_or_program_gets_the_mean_of_the_four():
    """ORT cannot say which micropixel fired, so the mean is the honest table."""
    want = np.mean([_packaged(s) for s in range(4)], axis=0)
    np.testing.assert_allclose(calibration.board_taps("D0", "M0", "ORT"), want)


def test_an_unknown_board_is_refused():
    with pytest.raises(FileNotFoundError, match="D9_M9"):
        calibration.board_taps("D9", "M9", "ORT")


# --------------------------------------------------------------------------
# end to end, off '.bin' files
# --------------------------------------------------------------------------


def test_process_and_plot_the_calibration(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(4)
    for i in range(2):
        (tmp_path / f"density_test_S0T{i}.bin").write_bytes(
            rng.bytes(io.BYTES_PER_FRAME * 20)
        )

    saved = tc.process_tdc_calibration(str(tmp_path), 20, 100e-9)
    tables = tc.plot_tdc_calibration(str(tmp_path))

    assert [os.path.basename(p) for p in saved] == ["TDC_COMPACT_S0T.npy"]
    assert tables["S0T"].shape == (32, 32, calibration.TAPS_PER_LAP)
    assert store.read_meta(saved[0])["measured_at_clock_period_s"] == 100e-9
    assert os.path.isfile(
        os.path.join(store.results_dir(str(tmp_path), "tdc_calibration"),
                     "TDC_COMPACT_S0T_qa.png")
    )
    plt.close("all")


@pytest.mark.parametrize("tag", ["S0C", "ORC"])
def test_a_counting_tag_has_no_codes_to_calibrate(tmp_path, tag):
    with pytest.raises(ValueError, match="counting program"):
        tc.process_tdc_calibration(str(tmp_path), 20, 100e-9, tags=(tag,))
