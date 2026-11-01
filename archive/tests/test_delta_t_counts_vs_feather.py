"""ARCHIVED tests (branch modules-restructure, 2026-10-31): the counts path against the feather path, bin for bin.

They exercise code that now lives in archive/functions, against the
pre-restructure API (commit 6b49a4b). Not collected by pytest
(testpaths = tests), not maintained.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pytest
from dapkel.core import io, store
from dapkel.functions import delta_t as dt


# Two pixels per group, so 'all_pairs' gives four pair rows.
PIXELS = [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]


LABELS = ["0,0-1,0", "0,0-1,1", "0,1-1,0", "0,1-1,1"]


NFRAMES = 50


# The library takes seconds everywhere; these tests were written in ps, so a
# single conversion keeps the numbers below readable.
CODE_S = dt._TS_CODE_PS * 1e-12


@pytest.fixture(autouse=True)
def _no_rewrite_countdown(monkeypatch):
    """Skip 'confirm_rewrite's five-second pause; it is tested elsewhere."""
    monkeypatch.setattr(store, "REWRITE_DELAY_S", 0.0)


@pytest.fixture
def bin_folder(tmp_path):
    """Build a data folder of six seeded-random ORT '.bin' files."""
    rng = np.random.default_rng(20260802)
    for i in range(6):
        (tmp_path / f"ORT_{i}.bin").write_bytes(
            rng.bytes(io.BYTES_PER_FRAME * NFRAMES)
        )
    return str(tmp_path)


def _counts_run(path, **kwargs):
    """Run the counts path with the arguments these tests share."""
    return dt.calculate_and_save_delta_counts(
        path,
        PIXELS,
        tag="ORT",
        apply_TDC_calibration=False,
        nframes=NFRAMES,
        **kwargs,
    )


def _feather_run(path):
    """Run the feather path over the same files, for comparison."""
    return dt.calculate_and_save_timestamp_differences(
        path,
        PIXELS,
        tag="ORT",
        apply_TDC_calibration=False,
        nframes=NFRAMES,
    )


def _counts_path(path):
    return os.path.join(
        store.processed_dir(path, create=False),
        f"{os.path.basename(path)}_ort{dt._COUNTS_SUFFIX}",
    )


# --------------------------------------------------------------------------
# the two stage-1 paths must agree
# --------------------------------------------------------------------------


def _both_paths(bin_folder, bin_width, window):
    """Histogram the same '.bin' files down both stage-1 paths.

    The feather path now histograms the whole physical support in one pass, so
    the comparison window is a *slice* of it. Both paths place bin centres at
    multiples of the width about zero ('_delta_hist_edges' / '_delta_grid'), so
    the slice lands on exactly the bins a direct rebin produces.
    """
    _feather_run(bin_folder)
    _counts_run(bin_folder)

    full, full_edges, info = dt.compute_and_save_delta_histogram(
        bin_folder, bin_width_s=bin_width * 1e-12, quiet=True
    )
    full_centres = (full_edges[:-1] + full_edges[1:]) / 2

    grid_counts, centres, _ = dt.load_delta_counts(bin_folder)
    got, edges = dt.rebin_delta_counts(
        grid_counts,
        centres,
        bin_width_s=bin_width * 1e-12,
        plot_window_s=window * 1e-12,
    )

    keep = np.abs(full_centres) <= max(abs(edges[0]), abs(edges[-1]))
    want = full[keep]
    np.testing.assert_allclose(full_centres[keep], (edges[:-1] + edges[1:]) / 2)
    return got, want, edges, info


def test_counts_path_reproduces_the_feather_histogram_exactly(bin_folder):
    """The whole point: same '.bin' files, same histogram, bin for bin.

    Uncalibrated, so the differences are integers and the counts grid is
    lossless; binned on a whole multiple of one code, so the rebin is a plain
    regrouping. Any difference at all would be a defect, not a rounding -
    including in the outermost bins, which the feather path used to leave
    half-filled before both paths were put on one lattice.
    """
    # Wide: the fabricated codes spread over the whole grid, so a narrow
    # window around zero would compare two empty histograms. An *odd* number of
    # cells per bin, because that is what puts the target edges between cells -
    # the drivers snap to it, and comparing on an even one would only measure
    # which side of a tie a float landed ('_snap_binning').
    bin_width = 3 * dt._TS_CODE_PS
    window = 600 * bin_width

    got, want, _, info = _both_paths(bin_folder, bin_width, window)

    np.testing.assert_array_equal(got, want)
    assert want.sum() > 0, "the fixture produced no data to compare"
    assert int(info["n"]) > 0


def test_both_paths_put_zero_at_a_bin_centre_and_fill_the_edge_bins(
    bin_folder,
):
    """The lattice is shared, so the outermost bins agree too.

    The feather path used to draw edges half a bin past its window
    (``arange(-W - w/2, ...)``) while cutting the data at ``abs(delta) <= W``,
    which left the first and last bins permanently half-filled and made zero a
    bin centre only when the window happened to be a whole number of bins. Both
    paths now build edges the same way, which is what the full-support histogram
    needs: every plot window is a slice of one lattice.
    """
    bin_width = 3 * dt._TS_CODE_PS
    window = 600 * bin_width

    got, want, edges, _ = _both_paths(bin_folder, bin_width, window)
    centres = (edges[:-1] + edges[1:]) / 2

    assert got[0] == want[0]
    assert got[-1] == want[-1]
    assert got.sum() == want.sum()
    # Zero is a bin centre, and the edges sit half a bin outside the window.
    assert centres[centres.size // 2] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(centres, -centres[::-1], atol=1e-9)


def test_the_two_paths_see_the_same_number_of_differences(bin_folder):
    """Totals must match over the whole support, not just inside a window."""
    _feather_run(bin_folder)
    counts_path = _counts_run(bin_folder)

    _, _, info = dt.compute_and_save_delta_histogram(bin_folder, quiet=True)
    meta = store.read_meta(counts_path)

    assert meta["total"] == info["total"] > 0
    assert meta["n_outside"] == 0
    assert meta["total_in_grid"] == meta["total"]


def test_the_measured_support_matches_the_feather_paths(bin_folder):
    """Both stage-1 paths must agree about how far the differences reach."""
    _feather_run(bin_folder)
    counts_path = _counts_run(bin_folder)

    _, _, info = dt.compute_and_save_delta_histogram(bin_folder, quiet=True)
    grid_support_s = store.read_meta(counts_path)["support_s"]

    # The feather path measures the largest difference exactly; the grid is that
    # bound plus a margin, so it must cover it and not wildly exceed it.
    assert info["support_s"] <= grid_support_s
    assert grid_support_s < 3 * info["support_s"]


def test_the_wrong_model_cannot_be_asked_for(bin_folder):
    """A triangle on a residual used to be refused; now it is unsayable.

    The model is not a parameter of either driver - it follows the histogram -
    so the combination that had to be caught at run time no longer exists in
    the signature.
    """
    import inspect

    for driver in (
        dt.collect_and_plot_delta_counts,
        dt.collect_and_plot_timestamp_differences,
    ):
        assert "background" not in inspect.signature(driver).parameters


def test_both_stage_one_paths_give_the_same_subtracted_residual(bin_folder):
    """The feather and the counts path must agree bin-for-bin after subtraction.

    They agree on the raw histogram already
    ('test_counts_path_reproduces_the_feather_histogram_exactly'), so this pins
    the *subtraction* on top of it: the same lags, the same frame-count
    rescaling, the same bins, and - the fiddly part - the zoom window landing on
    the same lattice whether it came from slicing a support-wide histogram or
    from rebinning a native grid.
    """
    import matplotlib

    matplotlib.use("Agg")

    # An odd number of cells per bin, so the counts-path rebin is exact.
    width = 3 * CODE_S
    kwargs = {
        "subtract_background": True,
        "background_lags": (1, 2, 3),
    }
    feather = dt.calculate_and_save_timestamp_differences(
        bin_folder,
        PIXELS,
        tag="ORT",
        apply_TDC_calibration=False,
        nframes=NFRAMES,
        **kwargs,
    )
    counts_npy = _counts_run(bin_folder, **kwargs)

    _, _, from_feather = dt.collect_and_plot_timestamp_differences(
        feather_path=feather,
        bin_width_s=width,
        plot_window_s=20 * width,
        fit_window_s=2 * width,
        subtract_background=True,
        label="feather",
    )
    _, _, from_counts = dt.collect_and_plot_delta_counts(
        counts_path=counts_npy,
        bin_width_s=width,
        plot_window_s=20 * width,
        fit_window_s=2 * width,
        subtract_background=True,
        label="counts",
    )

    np.testing.assert_allclose(
        from_feather["subtraction"]["centers"],
        from_counts["subtraction"]["centers"],
    )
    np.testing.assert_array_equal(
        from_feather["subtraction"]["residual"],
        from_counts["subtraction"]["residual"],
    )
    assert (
        from_feather["model_free"]["n"] == from_counts["model_free"]["n"]
    ), "the model-free coincidence count must not depend on the stage-1 path"
