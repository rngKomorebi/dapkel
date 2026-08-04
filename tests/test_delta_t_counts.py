"""Stage 1' - histogramming delta-t straight off the '.bin', with no feather.

The feather path writes every difference and histograms it later. That is
correct, but the artifact scales with the data (3-135 GB for a long run) and
the binning is still chosen at plot time, so every change of mind costs
another full pass. The counts path bins onto a fixed native grid inside the
acquisition loop and keeps only the counts: the artifact scales with the
*grid* instead, and re-binning becomes an in-memory reduction.

That is only worth having if it gives the same answer, so the central test
here runs both stage-1 paths over the same '.bin' files and demands a
bin-for-bin identical histogram. The rest pin the properties that make the
trade safe: nothing is dropped silently, the artifact does not grow with the
run, and a run that dies is resumable without double-counting.

The '.bin' files are seeded random bytes. 'unpack' is a pure decode, so they
carry garbage physics but drive a perfectly deterministic pipeline - which is
what these tests are about.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from dapkel.core import io, store
from dapkel.functions import delta_t as dt

# Two pixels per group, so 'all_pairs' gives four pair rows.
PIXELS = [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]
LABELS = ["0,0-1,0", "0,0-1,1", "0,1-1,0", "0,1-1,1"]

NFRAMES = 50

# Random codes reach ~8 200, so differences reach ~630 000 ps. A support this
# wide holds all of them, which is what the totals tests need.
WIDE_PS = 1e6


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
        f"{os.path.basename(path)}{dt._COUNTS_SUFFIX}",
    )


# --------------------------------------------------------------------------
# the grid itself
# --------------------------------------------------------------------------


def test_grid_puts_zero_at_a_cell_centre():
    """Zero must be a cell centre - the property the feather edges lack.

    The feather path's edges come from ``arange(-W - w/2, ...)``, which only
    centres a bin on zero when the window is a whole multiple of the width.
    """
    n_bins, zero, edges = dt._delta_grid(7.7, 200e3)

    assert n_bins % 2 == 1, "an odd cell count is what centres one on zero"
    assert len(edges) == n_bins + 1
    centres = (edges[:-1] + edges[1:]) / 2
    assert centres[zero] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(centres, -centres[::-1], atol=1e-9)


def test_grid_covers_at_least_the_requested_support():
    """The support is rounded up to a whole cell, never truncated."""
    _, _, edges = dt._delta_grid(30.0, 100.0)
    assert edges[0] <= -100.0
    assert edges[-1] >= 100.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_grid_is_rejected(bad):
    with pytest.raises(ValueError):
        dt._delta_grid(bad, 100.0)
    with pytest.raises(ValueError):
        dt._delta_grid(1.0, bad)


# --------------------------------------------------------------------------
# accumulation
# --------------------------------------------------------------------------


def test_accumulate_is_lossless_for_integer_codes():
    """One cell per code is a re-encoding, not a summary.

    ``delta_code`` is an integer, so binning it onto its own lattice throws
    nothing away - the counts are exactly 'np.bincount' of the codes.
    """
    n_bins, zero, _ = dt._delta_grid(1.0, 20.0)
    counts = np.zeros((1, n_bins), dtype=np.int64)
    codes = np.array([-3.0, -3.0, 0.0, 1.0, 1.0, 1.0, 7.0])

    total, outside = dt._accumulate_delta_counts(
        counts, {"a": codes}, {"a": 0}, grid_ps=1.0, zero_index=zero
    )

    assert (total, outside) == (7, 0)
    for code, want in ((-3, 2), (0, 1), (1, 3), (7, 1)):
        assert counts[0, zero + code] == want
    assert counts.sum() == 7


def test_out_of_range_differences_are_counted_not_dropped():
    """A too-narrow support must be visible, not silent."""
    n_bins, zero, _ = dt._delta_grid(1.0, 5.0)
    counts = np.zeros((1, n_bins), dtype=np.int64)

    total, outside = dt._accumulate_delta_counts(
        counts,
        {"a": np.array([0.0, 4.0, 50.0, -50.0])},
        {"a": 0},
        grid_ps=1.0,
        zero_index=zero,
    )

    assert total == 4
    assert outside == 2
    assert counts.sum() == 2, "the two outside values must not be clamped in"


def test_unknown_pair_labels_are_ignored():
    n_bins, zero, _ = dt._delta_grid(1.0, 5.0)
    counts = np.zeros((1, n_bins), dtype=np.int64)

    total, outside = dt._accumulate_delta_counts(
        counts,
        {"a": np.array([1.0]), "not-a-row": np.array([1.0, 2.0])},
        {"a": 0},
        grid_ps=1.0,
        zero_index=zero,
    )

    assert (total, outside, counts.sum()) == (1, 0, 1)


# --------------------------------------------------------------------------
# rebinning
# --------------------------------------------------------------------------


def test_rebinning_conserves_every_count_inside_the_window():
    grid = np.arange(-100, 101) * 10.0
    counts = np.arange(grid.size, dtype=np.int64)

    binned, edges = dt.rebin_delta_counts(
        counts, grid, bin_width_ps=50.0, plot_window_ps=500.0
    )

    inside = (grid >= edges[0]) & (grid < edges[-1])
    assert binned.sum() == counts[inside].sum()
    assert binned.size == len(edges) - 1


def test_rebinning_onto_a_multiple_of_the_grid_is_a_plain_regrouping():
    """A 50 ps bin over a 10 ps grid must be the sum of each run of five."""
    grid = np.arange(-50, 51) * 10.0
    counts = np.ones(grid.size, dtype=np.int64)

    binned, _ = dt.rebin_delta_counts(
        counts, grid, bin_width_ps=50.0, plot_window_ps=250.0
    )

    assert set(np.unique(binned[1:-1])) == {5}


# --------------------------------------------------------------------------
# the two stage-1 paths must agree
# --------------------------------------------------------------------------


def _both_paths(bin_folder, bin_width, window):
    """Histogram the same '.bin' files down both stage-1 paths."""
    _feather_run(bin_folder)
    _counts_run(bin_folder)

    want, edges, info = dt.compute_and_save_delta_histogram(
        bin_folder, bin_width_ps=bin_width, plot_window_ps=window, quiet=True
    )
    grid_counts, centres, _ = dt.load_delta_counts(bin_folder)
    got, got_edges = dt.rebin_delta_counts(
        grid_counts, centres, bin_width_ps=bin_width, plot_window_ps=window
    )
    np.testing.assert_allclose(got_edges, edges)
    return got, want, edges, info


def test_counts_path_reproduces_the_feather_histogram_exactly(bin_folder):
    """The whole point: same '.bin' files, same histogram, bin for bin.

    Uncalibrated, so the differences are integers and the counts grid is
    lossless; binned on a whole multiple of one code, so the rebin is a plain
    regrouping. A difference in the interior would be a defect, not a
    rounding. The outermost bin on each side is excluded here and pinned
    separately below - the two paths genuinely disagree there, and the counts
    path is the one that is right.
    """
    # Wide: the fabricated codes spread over the whole grid, so a narrow
    # window around zero would compare two empty histograms.
    bin_width = 4 * dt._TS_CODE_PS
    window = 600 * bin_width

    got, want, _, info = _both_paths(bin_folder, bin_width, window)

    np.testing.assert_array_equal(got[1:-1], want[1:-1])
    assert want[1:-1].sum() > 0, "the fixture produced no data to compare"
    assert int(info["n"]) > 0


def test_the_feather_paths_outermost_bins_are_undercounted(bin_folder):
    """A known, pre-existing quirk of the feather path - pinned, not fixed.

    Its edges run half a bin past the window (``arange(-W - w/2, ...)``) but
    it cuts the data at ``abs(delta) <= W``, so the first and last bins can
    only ever be half-filled. The counts path assigns whole grid cells to
    whichever bin their centre falls in, so it fills them properly.

    Only the two extreme bins can differ, and only in that direction.
    """
    bin_width = 4 * dt._TS_CODE_PS
    window = 600 * bin_width

    got, want, edges, _ = _both_paths(bin_folder, bin_width, window)

    assert got[0] >= want[0]
    assert got[-1] >= want[-1]
    # The excess is exactly the range the feather path's cut threw away.
    assert edges[0] < -window
    assert edges[-1] > window
    assert got.sum() - want.sum() == (got[0] - want[0]) + (got[-1] - want[-1])


def test_the_two_paths_see_the_same_number_of_differences(bin_folder):
    """Totals must match over the whole support, not just inside a window."""
    _feather_run(bin_folder)
    counts_path = _counts_run(bin_folder, support_ps=WIDE_PS)

    _, _, info = dt.compute_and_save_delta_histogram(bin_folder, quiet=True)
    meta = store.read_meta(counts_path)

    assert meta["total"] == info["total"] > 0
    assert meta["n_outside"] == 0
    assert meta["total_in_grid"] == meta["total"]


def test_a_narrow_support_reports_what_it_lost(bin_folder):
    """The default support is narrower than these fabricated codes reach."""
    meta = store.read_meta(_counts_run(bin_folder, support_ps=200e3))

    assert meta["n_outside"] > 0
    assert meta["total_in_grid"] == meta["total"] - meta["n_outside"]


def _continuous_grid(deltas, cell):
    """Bin continuous (LUT'd) differences onto a native grid."""
    n_bins, zero, _ = dt._delta_grid(cell, 200e3)
    counts = np.zeros((1, n_bins), dtype=np.int64)
    dt._accumulate_delta_counts(
        counts, {"a": deltas}, {"a": 0}, grid_ps=cell, zero_index=zero
    )
    return counts[0], (np.arange(n_bins) - zero) * cell


@pytest.mark.parametrize("cells", [9, 11, 13])
def test_an_odd_number_of_cells_per_bin_rebins_exactly(cells):
    """Odd cells per bin puts the bin edges *between* cells, so nothing splits.

    Bins are centred on zero, so their edges lie at half-integer multiples of
    the width. An odd width in cells makes those half-integers land on cell
    boundaries; an even one lands them on cell centres, cutting a cell in two
    at every boundary. This holds for continuous data, where each difference
    is displaced onto its cell centre - the displacement cannot cross a bin
    edge, because the edges are exactly where the cells already end.
    """
    rng = np.random.default_rng(3)
    deltas = rng.normal(scale=400.0, size=50_000)
    cell = dt._TS_CODE_PS / dt._GRID_SUBDIV

    counts, centres = _continuous_grid(deltas, cell)
    width = cells * cell
    window = 60 * width
    got, edges = dt.rebin_delta_counts(
        counts, centres, bin_width_ps=width, plot_window_ps=window
    )

    np.testing.assert_array_equal(got, np.histogram(deltas, bins=edges)[0])


def test_an_even_number_of_cells_per_bin_does_not(caplog):
    """The counterexample - pinned so the odd-cell rule is not lore."""
    rng = np.random.default_rng(3)
    deltas = rng.normal(scale=400.0, size=50_000)
    cell = dt._TS_CODE_PS / dt._GRID_SUBDIV

    counts, centres = _continuous_grid(deltas, cell)
    width = 10 * cell
    window = 60 * width
    got, edges = dt.rebin_delta_counts(
        counts, centres, bin_width_ps=width, plot_window_ps=window
    )
    want = np.histogram(deltas, bins=edges)[0]

    assert got.sum() == want.sum(), "counts are still conserved"
    assert np.abs(got - want).sum() > 0, "an even width must split cells"


def test_snapping_picks_an_odd_cell_count_and_a_whole_window(capsys):
    cell = dt._TS_CODE_PS / dt._GRID_SUBDIV  # 7.7 ps

    width, window = dt._snap_binning(71.0, 10e3, cell)

    assert width / cell == pytest.approx(9.0)
    assert (window / width) == pytest.approx(round(window / width))
    assert "9 cells" in capsys.readouterr().out


def test_snapping_moves_an_even_request_off_the_even_count():
    cell = 7.7
    width, _ = dt._snap_binning(10 * cell, 1000.0, cell)
    assert round(width / cell) % 2 == 1


def test_snapping_never_returns_a_zero_width():
    width, window = dt._snap_binning(0.1, 1.0, 7.7)
    assert width == pytest.approx(7.7)
    assert window > 0


# --------------------------------------------------------------------------
# the artifact, and surviving a dead run
# --------------------------------------------------------------------------


def test_artifact_size_is_set_by_the_grid_not_the_run_length(bin_folder):
    """Six files must produce exactly the same bytes as two.

    This is the property the feather cannot have, and the reason a 10 000-file
    run is plottable at all.
    """
    two_bytes = np.load(_counts_run(bin_folder, max_files=2)).nbytes
    two_total = store.read_meta(_counts_path(bin_folder))["total"]

    six_path = _counts_run(bin_folder, max_files=6, rewrite=True)

    assert np.load(six_path).nbytes == two_bytes
    assert store.read_meta(six_path)["total"] > two_total


def test_a_complete_artifact_is_not_silently_overwritten(bin_folder):
    _counts_run(bin_folder)
    with pytest.raises(FileExistsError):
        _counts_run(bin_folder)


def test_different_inputs_will_not_overwrite_without_rewrite(bin_folder):
    """A different pixel set is a different artifact, not a resumable one."""
    _counts_run(bin_folder)
    with pytest.raises(FileExistsError):
        dt.calculate_and_save_delta_counts(
            bin_folder,
            [[(2, 2)], [(3, 3)]],
            tag="ORT",
            apply_TDC_calibration=False,
            nframes=NFRAMES,
        )


def _die_after(monkeypatch, n_files):
    """Make 'unpack' raise once it has decoded ``n_files`` files."""
    real = dt.unpack
    calls = {"n": 0}

    def dying(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > n_files:
            raise KeyboardInterrupt("simulated dead run")
        return real(*args, **kwargs)

    monkeypatch.setattr(dt, "unpack", dying)
    return real


def test_an_interrupted_run_resumes_without_double_counting(
    bin_folder, monkeypatch
):
    """A run that dies mid-way must finish to exactly the same answer.

    This is what replaces the feather path's part folder: there is one file,
    so two runs cannot interleave, and progress is "the first N files".
    """
    reference = np.load(_counts_run(bin_folder, flush_every=1))
    reference_total = store.read_meta(_counts_path(bin_folder))["total"]

    real = _die_after(monkeypatch, 3)
    with pytest.raises(KeyboardInterrupt):
        _counts_run(bin_folder, flush_every=1, rewrite=True)

    partial = store.read_meta(_counts_path(bin_folder))
    assert partial["complete"] is False
    assert partial["files_done"] == 3

    monkeypatch.setattr(dt, "unpack", real)
    resumed_path = _counts_run(bin_folder, flush_every=1)
    resumed_meta = store.read_meta(resumed_path)

    np.testing.assert_array_equal(np.load(resumed_path), reference)
    assert resumed_meta["total"] == reference_total
    assert resumed_meta["complete"] is True


def test_a_checkpoint_inconsistent_with_its_sidecar_is_not_resumed(
    bin_folder, monkeypatch
):
    """The array is moved into place before the sidecar; guard that window.

    Resuming a newer array against an older sidecar would re-count files and
    inflate the coincidence total - worse than simply starting again.
    """
    reference = np.load(_counts_run(bin_folder, flush_every=1))

    real = _die_after(monkeypatch, 3)
    with pytest.raises(KeyboardInterrupt):
        _counts_run(bin_folder, flush_every=1, rewrite=True)
    monkeypatch.setattr(dt, "unpack", real)

    # Stand in for a crash between the two moves: the array no longer sums to
    # what the sidecar records.
    path = _counts_path(bin_folder)
    tampered = np.load(path)
    tampered[0, 0] += 1
    np.save(path, tampered)

    resumed = np.load(_counts_run(bin_folder, flush_every=1))

    np.testing.assert_array_equal(resumed, reference)


# --------------------------------------------------------------------------
# reading it back
# --------------------------------------------------------------------------


def test_load_pools_selected_pairs_and_reports_missing(bin_folder):
    counts_path = _counts_run(bin_folder)

    every, _, info = dt.load_delta_counts(bin_folder)
    per_pair, _, _ = dt.load_delta_counts(bin_folder, pool=False)
    one, _, one_info = dt.load_delta_counts(
        bin_folder, pairs=[LABELS[0], "9,9-9,9"]
    )

    assert info["labels"] == LABELS
    assert per_pair.shape == (len(LABELS), np.load(counts_path).shape[1])
    np.testing.assert_array_equal(every, per_pair.sum(axis=0))
    np.testing.assert_array_equal(one, per_pair[0])
    assert one_info["missing"] == ["9,9-9,9"]


def test_a_code_grid_is_scaled_to_ps_at_load_time(bin_folder):
    """The ps-per-code assumption must not be baked into the artifact."""
    _counts_run(bin_folder)

    _, default_centres, default_info = dt.load_delta_counts(bin_folder)
    _, doubled, doubled_info = dt.load_delta_counts(
        bin_folder, time_unit_ps=2 * dt._TS_CODE_PS
    )

    assert default_info["unit"] == "code"
    assert default_info["cell_ps"] == pytest.approx(dt._TS_CODE_PS)
    assert doubled_info["cell_ps"] == pytest.approx(2 * dt._TS_CODE_PS)
    np.testing.assert_allclose(doubled, 2 * default_centres)


def test_plot_driver_returns_counts_centres_and_a_fit(bin_folder):
    import matplotlib

    matplotlib.use("Agg")

    _counts_run(bin_folder)
    counts, centres, fit = dt.collect_and_plot_delta_counts(
        bin_folder,
        bin_width_ps=4 * dt._TS_CODE_PS,
        plot_window_ps=2400 * dt._TS_CODE_PS,
        label="test",
    )

    assert counts.shape == centres.shape
    assert fit["n"] == int(counts.sum())
    png = os.path.join(
        store.results_dir(bin_folder, "coincidences", create=False),
        f"{os.path.basename(bin_folder)}_test_delta_counts.png",
    )
    assert os.path.isfile(png), "the figure must land in results/"


# --------------------------------------------------------------------------
# subtract_background: the lag planes
# --------------------------------------------------------------------------


def test_without_subtraction_nothing_about_the_artifact_changes(bin_folder):
    """The default path must write the same 2D array and the same sidecar.

    'subtract_background' is a superset, and a parameter that quietly changed
    the shape or the resume signature of every existing artifact would not be
    one.
    """
    _counts_run(bin_folder)
    counts = np.load(_counts_path(bin_folder))
    meta = store.read_meta(_counts_path(bin_folder))

    assert counts.shape == (len(LABELS), meta["n_bins"])
    assert "lags" not in meta
    assert "lags" not in meta["signature"]
    assert "frames_used" not in meta


def test_lag_zero_plane_is_the_unsubtracted_histogram(tmp_path):
    """Plane 0 must be bin-for-bin the array a plain run writes.

    This is what lets one artifact answer both questions, and what makes
    'subtract_background' safe to switch on: the ordinary measurement is still
    in there, untouched.
    """
    rng = np.random.default_rng(20260803)
    plain, lagged = tmp_path / "plain", tmp_path / "lagged"
    for folder in (plain, lagged):
        folder.mkdir()
    data = [rng.bytes(io.BYTES_PER_FRAME * NFRAMES) for _ in range(4)]
    for folder in (plain, lagged):
        for i, blob in enumerate(data):
            (folder / f"ORT_{i}.bin").write_bytes(blob)

    _counts_run(str(plain), support_ps=WIDE_PS)
    _counts_run(
        str(lagged),
        support_ps=WIDE_PS,
        subtract_background=True,
        background_lags=(1, 2, 3),
    )

    flat = np.load(_counts_path(str(plain)))
    lags = np.load(_counts_path(str(lagged)))

    assert lags.shape == (4, *flat.shape)
    np.testing.assert_array_equal(lags[0], flat)


def test_frames_used_is_recorded_per_lag(bin_folder):
    """A lag of k loses the last k frames of every file, and says so."""
    _counts_run(
        bin_folder, support_ps=WIDE_PS, subtract_background=True,
        background_lags=(1, 4),
    )
    meta = store.read_meta(_counts_path(bin_folder))

    assert meta["lags"] == [0, 1, 4]
    assert meta["frames_used"] == [6 * NFRAMES, 6 * (NFRAMES - 1), 6 * (NFRAMES - 4)]


@pytest.mark.parametrize(
    ("lags", "match"),
    [((0, 1), "background_lags"), ((1, 1), "duplicates"), ((), "empty")],
)
def test_bad_background_lags_are_rejected(bin_folder, lags, match):
    with pytest.raises(ValueError, match=match):
        _counts_run(bin_folder, subtract_background=True, background_lags=lags)


def test_a_bare_int_is_taken_as_one_lag(bin_folder):
    _counts_run(
        bin_folder, support_ps=WIDE_PS, subtract_background=True,
        background_lags=1,
    )
    assert store.read_meta(_counts_path(bin_folder))["lags"] == [0, 1]


def test_load_keeps_the_lag_axis_and_pools_pairs_under_it(bin_folder):
    _counts_run(
        bin_folder, support_ps=WIDE_PS, subtract_background=True,
        background_lags=(1, 2),
    )
    pooled, centres, info = dt.load_delta_counts(bin_folder)
    per_pair, _, _ = dt.load_delta_counts(bin_folder, pool=False)
    one_pair, _, _ = dt.load_delta_counts(bin_folder, pairs=[LABELS[0]])

    assert pooled.shape == (3, centres.size)
    assert per_pair.shape == (3, len(LABELS), centres.size)
    assert one_pair.shape == (3, centres.size)
    np.testing.assert_array_equal(pooled, per_pair.sum(axis=1))
    np.testing.assert_array_equal(one_pair, per_pair[:, 0])
    assert info["lags"] == [0, 1, 2]


def test_subtracting_needs_an_artifact_with_lag_planes(bin_folder):
    """Asking for it on a plain artifact must say what to do about it."""
    import matplotlib

    matplotlib.use("Agg")

    _counts_run(bin_folder, support_ps=WIDE_PS)
    with pytest.raises(ValueError, match="lag planes"):
        dt.collect_and_plot_delta_counts(
            bin_folder, subtract_background=True, label="test"
        )


def test_subtracting_refuses_a_triangle_background(bin_folder):
    """There is no triangle left to fit once it has been subtracted away."""
    import matplotlib

    matplotlib.use("Agg")

    _counts_run(
        bin_folder, support_ps=WIDE_PS, subtract_background=True,
        background_lags=(1, 2),
    )
    with pytest.raises(ValueError, match="no triangle left"):
        dt.collect_and_plot_delta_counts(
            bin_folder,
            subtract_background=True,
            background="triangle",
            label="test",
        )


def test_the_subtracted_driver_returns_the_residual_and_writes_both_figures(
    bin_folder,
):
    """The residual is what gets fitted, and the diagnostic lands beside it."""
    import matplotlib

    matplotlib.use("Agg")

    _counts_run(
        bin_folder, support_ps=WIDE_PS, subtract_background=True,
        background_lags=(1, 2, 3),
    )
    counts, centres, fit = dt.collect_and_plot_delta_counts(
        bin_folder,
        bin_width_ps=4 * dt._TS_CODE_PS,
        plot_window_ps=200 * dt._TS_CODE_PS,
        subtract_background=True,
        peak_window_ps=10 * dt._TS_CODE_PS,
        label="test",
    )

    assert counts.shape == centres.shape
    # The residual is a difference of histograms, so it is signed - the plain
    # path's "counts are non-negative" does not hold here.
    assert counts.dtype.kind == "f"
    np.testing.assert_allclose(
        counts, fit["subtraction"]["residual"], rtol=0, atol=0
    )
    assert fit["n"] == int(round(fit["model_free"]["n"]))

    results = store.results_dir(bin_folder, "coincidences", create=False)
    name = os.path.basename(bin_folder)
    assert os.path.isfile(
        os.path.join(results, f"{name}_test_delta_counts_sub.png")
    ), "the fit figure must be named apart from the unsubtracted one"
    assert os.path.isfile(
        os.path.join(results, f"{name}_test_background_subtraction.png")
    ), "the three-panel diagnostic must land in results/ too"
    assert fit["subtraction_png"].endswith("_background_subtraction.png")


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
    width = 3 * dt._TS_CODE_PS
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
    counts_npy = _counts_run(bin_folder, support_ps=WIDE_PS, **kwargs)

    _, _, from_feather = dt.collect_and_plot_timestamp_differences(
        feather_path=feather,
        bin_width_ps=width,
        plot_window_ps=20 * width,
        support_ps=WIDE_PS,
        peak_window_ps=2 * width,
        subtract_background=True,
        label="feather",
    )
    _, _, from_counts = dt.collect_and_plot_delta_counts(
        counts_path=counts_npy,
        bin_width_ps=width,
        plot_window_ps=20 * width,
        peak_window_ps=2 * width,
        subtract_background=True,
        snap_to_grid=False,
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
