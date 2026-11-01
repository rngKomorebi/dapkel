"""ARCHIVED tests (branch modules-restructure, 2026-10-31): lag-0 differencing against calc_diff, and the input validation of the
former public background_subtraction functions.

They exercise code that now lives in archive/functions, against the
pre-restructure API (commit 6b49a4b). Not collected by pytest
(testpaths = tests), not maintained.
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


def test_lag_zero_matches_calc_diff() -> None:
    """Lag 0 must reproduce the existing within-frame difference exactly."""
    ts = _random_frames(5_000, 0.3, 0.25, n_pairs=200, seed=1)

    mine = bs.compute_lagged_differences(ts, PIXELS, frame_lag=0)
    theirs = cd.calculate_differences(ts, PIXELS)

    assert mine.keys() == theirs.keys()
    np.testing.assert_array_equal(mine[LABEL], theirs[LABEL])


def test_lag_zero_matches_calc_diff_on_a_busy_run() -> None:
    """Agreement has to hold at scale, not just on a handful of frames.

    Stage 1 picks between these two functions on one boolean, so any divergence
    would show up as 'subtract_background' quietly changing the measurement it
    is supposed to leave alone.
    """
    ts = _random_frames(5_000, 0.3, 0.25, n_pairs=200, seed=11)

    mine = bs.compute_lagged_differences(ts, PIXELS, frame_lag=0)
    theirs = cd.calculate_differences(ts, PIXELS)

    np.testing.assert_array_equal(mine[LABEL], theirs[LABEL])
    assert mine[LABEL].size > 0, "the fixture produced no coincidences"


def test_negative_lag_is_rejected() -> None:
    ts = _random_frames(100, 0.3, 0.3, n_pairs=1, seed=2)
    with pytest.raises(ValueError, match="frame_lag"):
        bs.compute_lagged_differences(ts, PIXELS, frame_lag=-1)


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
