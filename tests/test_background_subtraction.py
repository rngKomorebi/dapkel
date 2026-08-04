"""Unit tests for frame-shift (lag) background subtraction.

Three things are worth pinning down and none of them needs real data:

    1. ``compute_lagged_differences`` at lag 0 must be *exactly* what
       'calc_diff.calculate_differences' already produces - otherwise the
       lag-0 plane of the counts grid is not the same measurement everything
       else in the package reports, and 'delta_t' cannot promise that
       ``subtract_background=True`` leaves its own answer untouched.

    2. On a synthetic data set with a *known* number of injected pairs, the
       subtraction must recover that number. This is the whole claim of the
       module, and it is checkable to the count.

    3. The frame-count rescaling must actually happen: a lag that saw fewer
       frames has to be scaled up, not averaged in raw.
"""

from __future__ import annotations

import numpy as np
import pytest

from dapkel.functions import background_subtraction as bs
from dapkel.functions import calc_diff as cd

A = (3, 4)
B = (20, 21)
PIXELS = [[A], [B]]
LABEL = f"{A[0]},{A[1]}-{B[0]},{B[1]}"


def _random_frames(
    nframes: int, p_a: float, p_b: float, n_pairs: int, seed: int
) -> dict[tuple[int, int], np.ndarray]:
    """Two uncorrelated pixels plus ``n_pairs`` injected same-frame pairs.

    Codes are uniform on 1..1346 (the oscillator range), so the accidental
    differences form the same triangle the real data does. The injected pairs
    are simultaneous, i.e. delta == 0 exactly.
    """
    rng = np.random.default_rng(seed)
    ta = np.full(nframes, np.nan)
    tb = np.full(nframes, np.nan)

    fires_a = rng.random(nframes) < p_a
    fires_b = rng.random(nframes) < p_b
    ta[fires_a] = rng.integers(1, 1347, fires_a.sum())
    tb[fires_b] = rng.integers(1, 1347, fires_b.sum())

    # Overwrite both pixels in n_pairs distinct frames with one shared code.
    where = rng.choice(nframes, size=n_pairs, replace=False)
    shared = rng.integers(1, 1347, n_pairs).astype(float)
    ta[where] = shared
    tb[where] = shared

    return {A: ta, B: tb}


def _lag_grid(ts, lags, support=1400):
    """Histogram one synthetic data set at every lag, in raw codes."""
    n_bins = 2 * support + 1
    counts = np.zeros((len(lags), n_bins), dtype=np.int64)
    for li, lag in enumerate(lags):
        d = bs.compute_lagged_differences(ts, PIXELS, frame_lag=lag)[LABEL]
        counts[li] = np.bincount(
            d.astype(np.int64) + support, minlength=n_bins
        )
    centers = (np.arange(n_bins) - support) * 1.0
    return counts, centers


def test_lag_zero_matches_calc_diff() -> None:
    """Lag 0 must reproduce the existing within-frame difference exactly."""
    ts = _random_frames(5_000, 0.3, 0.25, n_pairs=200, seed=1)

    mine = bs.compute_lagged_differences(ts, PIXELS, frame_lag=0)
    theirs = cd.calculate_differences(ts, PIXELS)

    assert mine.keys() == theirs.keys()
    np.testing.assert_array_equal(mine[LABEL], theirs[LABEL])


def test_lag_zero_matches_calc_diff_under_a_delta_window() -> None:
    """The window cut must behave identically too, or stage 1 diverges."""
    ts = _random_frames(5_000, 0.3, 0.25, n_pairs=200, seed=11)

    mine = bs.compute_lagged_differences(ts, PIXELS, 300.0, frame_lag=0)
    theirs = cd.calculate_differences(ts, PIXELS, 300.0)

    np.testing.assert_array_equal(mine[LABEL], theirs[LABEL])
    assert np.all(np.abs(mine[LABEL]) <= 300.0)


def test_lag_shifts_group_b_by_whole_frames() -> None:
    """A lag of k pairs frame i of A with frame i+k of B."""
    ta = np.array([10.0, np.nan, 30.0, 40.0])
    tb = np.array([np.nan, 100.0, 200.0, 300.0])
    ts = {A: ta, B: tb}

    # lag 1: (i=0,j=1) -> 100-10, (i=2,j=3) -> 300-30. i=1 has no A.
    got = bs.compute_lagged_differences(ts, PIXELS, frame_lag=1)[LABEL]
    np.testing.assert_array_equal(got, np.array([90.0, 270.0]))

    # A lag longer than the run yields nothing rather than raising.
    empty = bs.compute_lagged_differences(ts, PIXELS, frame_lag=9)[LABEL]
    assert empty.size == 0


def test_negative_lag_is_rejected() -> None:
    ts = _random_frames(100, 0.3, 0.3, n_pairs=1, seed=2)
    with pytest.raises(ValueError, match="frame_lag"):
        bs.compute_lagged_differences(ts, PIXELS, frame_lag=-1)


def test_frames_per_lag_counts_the_pairable_frames() -> None:
    """Lag k loses the last k frames of every file, and never goes negative."""
    np.testing.assert_array_equal(
        bs.frames_per_lag(100, [0, 1, 2], n_files=3), [300, 297, 294]
    )
    np.testing.assert_array_equal(bs.frames_per_lag(5, [0, 5, 9]), [5, 0, 0])


def test_subtraction_recovers_the_injected_pairs() -> None:
    """The residual must integrate to the number of pairs actually injected.

    The injected pairs sit at delta == 0, so a +/-1-code window holds all of
    them; everything else is accidental and must cancel.
    """
    nframes, n_pairs = 200_000, 400
    ts = _random_frames(nframes, 0.30, 0.25, n_pairs=n_pairs, seed=7)

    lags = [0, 1, 2, 3, 4]
    counts, centers = _lag_grid(ts, lags)
    out = bs.subtract_background(
        counts, centers, bs.frames_per_lag(nframes, lags)
    )

    peak = bs.integrate_residual(out, window_ps=1.0)
    assert abs(peak["n"] - n_pairs) < 4 * peak["n_err"], (
        f"recovered {peak['n']:.0f} +/- {peak['n_err']:.0f} injected pairs, "
        f"expected {n_pairs}"
    )

    # ...and the rest of the support must be consistent with zero.
    far = np.abs(out["centers"]) > 50
    far_sum = out["residual"][far].sum()
    far_err = np.sqrt((out["error"][far] ** 2).sum())
    assert abs(far_sum) < 4 * far_err, (
        f"residual away from the peak is {far_sum:.0f} +/- {far_err:.0f}, "
        "which should be consistent with zero"
    )


def test_normalises_unequal_frame_counts() -> None:
    """A lag that saw half the frames must be scaled back up, not averaged in."""
    counts = np.array(
        [
            [0, 0, 100, 0, 0],  # lag 0
            [0, 0, 50, 0, 0],  # lag 1, from half as many frames
        ],
        dtype=np.int64,
    )
    centers = np.arange(5, dtype=float) - 2
    out = bs.subtract_background(counts, centers, [1000, 500])

    np.testing.assert_allclose(out["background"][2], 100.0)
    np.testing.assert_allclose(out["residual"][2], 0.0)
    np.testing.assert_allclose(out["scale"], [1.0, 2.0])


def test_background_is_the_mean_not_the_sum() -> None:
    """K lags estimate one background, and its noise falls as 1/sqrt(K)."""
    counts = np.zeros((5, 3), dtype=np.int64)
    counts[0, 1] = 500
    counts[1:, 1] = 400
    centers = np.array([-1.0, 0.0, 1.0])
    out = bs.subtract_background(counts, centers, [1000] * 5)

    assert out["k"] == 4
    np.testing.assert_allclose(out["background"][1], 400.0)
    np.testing.assert_allclose(out["residual"][1], 100.0)
    # sqrt(signal + bkg/K) = sqrt(500 + 100)
    np.testing.assert_allclose(out["error"][1], np.sqrt(500 + 400 / 4))


def test_rejects_bad_shapes() -> None:
    counts = np.zeros((2, 5), dtype=np.int64)
    centers = np.arange(5, dtype=float) - 2

    with pytest.raises(ValueError, match="n_lags, n_bins"):
        bs.subtract_background(np.zeros((2, 3, 5)), centers, [1, 1])
    with pytest.raises(ValueError, match="background lag"):
        bs.subtract_background(np.zeros((1, 5)), centers, [1])
    with pytest.raises(ValueError, match="centers"):
        bs.subtract_background(counts, centers[:3], [1, 1])
    with pytest.raises(ValueError, match="frames_used"):
        bs.subtract_background(counts, centers, [1, 1, 1])
    with pytest.raises(ValueError, match="positive"):
        bs.subtract_background(counts, centers, [1, 0])


def test_integrate_residual_windows_and_adds_errors_in_quadrature() -> None:
    counts = np.zeros((3, 7), dtype=np.int64)
    centers = (np.arange(7) - 3) * 100.0  # -300..300 ps
    counts[0] = [10, 10, 40, 60, 40, 10, 10]
    counts[1:] = [10, 10, 10, 10, 10, 10, 10]
    out = bs.subtract_background(counts, centers, [1000] * 3)

    got = bs.integrate_residual(out, window_ps=100.0)
    assert got["n_bins"] == 3
    # 30 + 50 + 30 above a background of 10 per bin.
    np.testing.assert_allclose(got["n"], 110.0)
    np.testing.assert_allclose(got["n_background"], 30.0)
    np.testing.assert_allclose(got["car"], 110.0 / 30.0)
    np.testing.assert_allclose(
        got["n_err"], np.sqrt((out["error"][2:5] ** 2).sum())
    )
    # A wider window sees the tails too.
    assert bs.integrate_residual(out, window_ps=300.0)["n_bins"] == 7


def test_plot_returns_a_figure_without_writing_anything() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = _random_frames(20_000, 0.3, 0.25, n_pairs=100, seed=3)
    counts, centers = _lag_grid(ts, [0, 1, 2])
    out = bs.subtract_background(counts, centers, bs.frames_per_lag(20_000, [0, 1, 2]))

    fig = bs.plot_background_subtraction(out, name="synthetic", label="test")
    assert len(fig.axes) == 3
    plt.close(fig)
