"""ARCHIVED tests (branch modules-restructure, 2026-10-31): the dense per-clock LUT, its selection by clock tag and coverage, and
its agreement with the compact form.

They exercise code that now lives in archive/functions, against the
pre-restructure API (commit 6b49a4b). Not collected by pytest
(testpaths = tests), not maintained.
"""

from __future__ import annotations
import numpy as np
import pytest
from dapkel.core import io, timing
from dapkel.functions import delta_t as dt
from dapkel.functions import tdc_calibration as tc


# Largest code seen at each clock on the 2026.08.07 rate-vs-clock set, which is
# what 'timing.CODE_S' was measured against. The full scale a run swept has to
# come back out of these to within the ring's few-percent spread.
MEASURED = {
    50e6: (274, 20e-9),
    10e6: (1378, 100e-9),
    5e6: (2760, 200e-9),
    2e6: (6913, 500e-9),
}


def test_the_physical_support_is_the_widest_code_the_format_can_express():
    """What the feather path histograms over, so nothing can fall off it."""
    assert timing.MAX_DELTA_S == timing.MAX_TS_CODE * timing.CODE_S
    # Every clock in use produces a narrower support than this bound.
    for max_code, _ in MEASURED.values():
        assert timing.full_scale_from_max_code(max_code) < timing.MAX_DELTA_S


def test_one_firmware_vocabulary_across_both_resolvers():
    """Two spellings for one setting is how a caller passes the wrong pair.

    'resolve_cycle_time' used to accept ``'short_window'`` as an alias while
    'resolve_live_time' accepted only that spelling, so the two functions
    disagreed about what a valid value was.
    """
    assert timing.FIRMWARE_VERSIONS == ("short_exposure", "long_exposure")

    for version in timing.FIRMWARE_VERSIONS:
        timing.resolve_cycle_time(version, exp_time_s=1e-6)
        timing.resolve_live_time(version, 100, open_shutter_time_s=1e-7,
                                 exp_time_s=1e-6)

    for gone in ("short_window", "full_window"):
        with pytest.raises(ValueError, match="firmware_version"):
            timing.resolve_cycle_time(gone)
        with pytest.raises(ValueError, match="firmware_version"):
            timing.resolve_live_time(gone, 100, open_shutter_time_s=1e-7)


# --------------------------------------------------------------------------
# a LUT is checked against the run, not against a stated clock
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("clock", "full_scale_s"),
                         [("50MHz", 20e-9), ("10MHz", 100e-9), ("2MHz", 500e-9)])
def test_the_packaged_lut_is_self_describing(clock, full_scale_s):
    """Its largest entry is the full scale it was built for."""
    lut = tc.load_board_lut("D0", "M0", clock=clock)

    assert tc.lut_full_scale_s(lut) == pytest.approx(full_scale_s, rel=1e-6)
    assert lut.shape[:2] == (32, 32)


@pytest.mark.parametrize("clock_hz", [50e6, 10e6, 2e6])
def test_a_lut_that_covers_the_run_is_accepted(clock_hz):
    """Every packaged clock must load cleanly for a run taken at it."""
    max_code, _ = MEASURED[clock_hz]
    tag = tc._clock_tag(1.0 / clock_hz)

    lut = tc.load_board_lut("D0", "M0", min_codes=max_code + 1, clock=tag)

    assert lut.shape[-1] >= max_code


def test_a_lut_from_another_clock_is_refused():
    """A 10 MHz LUT on a 2 MHz run drops 79 % of the singles, silently."""
    max_code = MEASURED[2e6][0]

    with pytest.raises(ValueError, match="specific to its reference clock"):
        tc.load_board_lut("D0", "M0", min_codes=max_code + 1, clock="10MHz")


def test_the_check_is_skipped_when_the_run_is_unknown():
    """``min_codes`` is derived, so 'not yet known' has to be a valid state."""
    assert tc.load_board_lut("D0", "M0", clock="10MHz", min_codes=None).size > 0


# --------------------------------------------------------------------------
# picking between calibrations: coverage first, then the caller
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "tag"), [
    ("D0_M0_TDC_LUT_SPAD0_S0.npy", ""),
    ("D0_M0_TDC_LUT_SPAD0_S0_10MHz.npy", "10MHz"),
    ("D0_M0_TDC_LUT_SPAD0_S0_2MHz_2026-08-13.npy", "2MHz_2026-08-13"),
    ("D0_M0_500ns_TDC_LUT_SPAD0_S0.npy", "500ns"),
    ("TDC_LUT_S0T_2MHz.npy", "2MHz"),
])
def test_the_clock_tag_is_read_off_either_layout(name, tag):
    """Both naming layouts, and an untagged legacy table, resolve as written."""
    assert tc._lut_clock_tag(name) == tag


@pytest.mark.parametrize("clock_hz", [50e6, 10e6, 2e6])
def test_the_run_picks_its_own_clock_without_being_told(clock_hz):
    """A 2 MHz run must not silently get the 10 MHz table, or the reverse.

    Covering the run is necessary but not sufficient - a 2 MHz table covers a
    10 MHz run five times over - so the tightest covering scale is the one that
    fits. One calibration per clock is installed, so nothing has to be named.
    """
    max_code, full_scale_s = MEASURED[clock_hz]

    lut = tc.load_board_lut("D0", "M0", min_codes=max_code + 1)

    assert tc.lut_full_scale_s(lut) == pytest.approx(full_scale_s, rel=1e-6)


def _write_luts(folder, tags, n_codes, full_scale_ps):
    """Four SPAD LUTs under one clock tag, as linear ramps."""
    for tag in tags:
        for spad in range(4):
            ramp = np.linspace(0.0, full_scale_ps, n_codes, dtype=np.float32)
            np.save(
                folder / f"D0_M0_TDC_LUT_SPAD{spad}_S{spad}_{tag}.npy",
                np.broadcast_to(ramp, (32, 32, n_codes)).copy(),
            )


def test_two_calibrations_of_one_clock_have_to_be_told_apart(tmp_path):
    """Coverage cannot choose between them - they cover identically."""
    _write_luts(tmp_path, ("10MHz", "10MHz_2026-08-13"), 1391, 100e3)

    with pytest.raises(ValueError, match="clock="):
        tc.load_lut(str(tmp_path), min_codes=1379)

    named = tc.load_lut(str(tmp_path), min_codes=1379, clock="10MHz_2026-08-13")
    assert tc.lut_full_scale_s(named) == pytest.approx(100e-9, rel=1e-6)


def test_an_exact_tag_wins_over_a_partial_one(tmp_path):
    """'10MHz' keeps meaning the table named that, dated ones notwithstanding."""
    _write_luts(tmp_path, ("10MHz",), 1391, 100e3)
    _write_luts(tmp_path, ("10MHz_2026-08-13",), 1300, 100e3)

    assert tc.load_lut(str(tmp_path), clock="10MHz").shape[-1] == 1391
    assert tc.load_lut(str(tmp_path), clock="10MHz_2026-08-13").shape[-1] == 1300


def test_a_partial_tag_resolves_when_only_one_answers_to_it(tmp_path):
    _write_luts(tmp_path, ("2MHz_2026-08-13",), 6917, 500e3)

    assert tc.lut_full_scale_s(
        tc.load_lut(str(tmp_path), clock="2MHz")
    ) == pytest.approx(500e-9, rel=1e-6)


def test_an_unknown_clock_names_what_is_installed():
    with pytest.raises(ValueError, match="No LUT tagged"):
        tc.load_board_lut("D0", "M0", clock="5MHz")


def test_the_four_spads_average_despite_ragged_code_counts():
    """Whether a SPAD ever hit its last code is chance, not calibration."""
    lut = tc.load_board_lut("D0", "M0", "average", clock="2MHz")
    one = tc.load_board_lut("D0", "M0", 0, clock="2MHz")

    assert lut.shape[:2] == (32, 32)
    assert abs(lut.shape[-1] - one.shape[-1]) <= 2
    assert tc.lut_full_scale_s(lut) == pytest.approx(500e-9, rel=1e-6)
    assert (np.diff(lut, axis=-1) >= 0).all()


def test_averaging_never_blends_two_calibrations():
    """The mean is over one calibration's four SPADs, never across clocks."""
    with pytest.raises(ValueError, match="Cannot average across"):
        tc.load_board_lut("D0", "M0", "average")


def test_a_lut_maps_the_populated_range_onto_the_clock_period():
    """The density test fixes relative widths; the clock sets the scale.

    This is the one time parameter the package cannot derive, which is why
    ``clock_period_s`` survives here and nowhere else.
    """
    counts = np.ones((32, 32, 1000), dtype=np.int64)

    lut = tc.histogram_to_lut(counts, clock_period_s=500e-9)

    assert tc.lut_full_scale_s(lut) == pytest.approx(500e-9, rel=1e-3)
    # A flat density test is a uniform TDC, so the table is a linear ramp.
    np.testing.assert_allclose(np.diff(lut[16, 16]), 500e3 / 1000, rtol=1e-9)


def _own_range(dense):
    """Last code each pixel actually produced: where its dense LUT stops rising."""
    rising = np.diff(dense, axis=-1) > 0
    return np.where(
        rising.any(-1),
        rising.shape[-1] - 1 - np.argmax(rising[:, :, ::-1], axis=-1),
        0,
    )


def test_the_packaged_tables_agree_within_the_dense_one_s_counting_noise():
    """Both are shipped for the same board, so they must describe one ring.

    They cannot agree bin for bin, and that is the point: the density test
    estimates every code from an empirical CDF, whose noise is
    ``0.5/sqrt(N) x full_scale`` ~ 569 ps per pixel here, while the compact form
    averages each tap over every lap the run swept. Agreement at that scale is
    the most the dense table can offer.
    """
    dense = tc.load_board_lut("D0", "M0", 0, clock="2MHz").astype(np.float64)
    compact = tc.compact_to_lut(
        tc.load_board_compact("D0", "M0", 0), dense.shape[-1]
    )

    code = np.arange(dense.shape[-1])
    inside = code[None, None, :] <= _own_range(dense)[..., None]
    err = np.abs(compact - dense)[inside]

    assert np.median(err) < 600.0                 # ps, the CDF noise scale
    assert np.percentile(err, 99) < 2_000.0


def test_the_compact_table_does_not_clamp_past_a_pixel_s_own_range():
    """The dense table saturates out there; the compact one keeps counting.

    A pixel's LUT is a cumulative count, so above the last code that pixel ever
    produced it is pinned at the full scale - a flat tail covering ~4 % of the
    table, where a code that did arrive would be read as the end of the range
    instead of its own time. The tap model has no such edge: laps keep
    accumulating.
    """
    dense = tc.load_board_lut("D0", "M0", 0, clock="2MHz").astype(np.float64)
    compact = tc.compact_to_lut(
        tc.load_board_compact("D0", "M0", 0), dense.shape[-1]
    )
    last = _own_range(dense)

    # every pixel's dense table really is flat above its own range ...
    beyond = last < dense.shape[-1] - 1
    assert beyond.any()
    r, c = np.argwhere(beyond)[0]
    assert np.ptp(dense[r, c, last[r, c] + 1:]) == pytest.approx(0.0, abs=1.0)
    # ... while the compact one goes on rising, by about one lap per 8 codes
    tail = np.diff(compact[r, c, last[r, c] + 1:])
    assert (tail > 0).all()


@pytest.mark.parametrize("model", ["table", "compact"])
def test_both_calibration_models_load(model):
    lut = dt._resolve_time_lut(True, "D0", "M0", 0, min_codes=1379,
                               tdc_model=model)

    assert lut.shape[:2] == (32, 32)
    assert lut.shape[-1] >= 1379


def test_an_unknown_calibration_model_is_refused():
    with pytest.raises(ValueError, match="tdc_model"):
        dt._resolve_time_lut(True, "D0", "M0", 0, min_codes=1379,
                             tdc_model="dense")


def test_the_compact_model_needs_the_runs_code_range():
    with pytest.raises(ValueError, match="min_codes"):
        dt._resolve_time_lut(True, "D0", "M0", 0, tdc_model="compact")
