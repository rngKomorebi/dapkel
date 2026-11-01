"""The delta-t pipeline: 'process_coincidences' bins, 'plot_coincidences' fits.

Stage 1 bins every pair's differences onto a fixed native grid inside the
acquisition loop and keeps only the counts. That is only worth having if the
grid is exactly the histogram of the differences, so the central test here
recomputes it independently from the decoded codes and demands equality bin for
bin. The rest pin what makes the trade safe: nothing is dropped silently, the
artifact does not grow with the run, a run that dies resumes without
double-counting, and the frame-shift background subtraction recovers a known
number of injected pairs.

The '.bin' files are seeded random bytes. 'unpack' is a pure decode, so they
carry garbage physics but drive a perfectly deterministic pipeline.
"""

from __future__ import annotations

import inspect
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from dapkel.core import io, store, timing  # noqa: E402
from dapkel.functions import coincidences as co  # noqa: E402

# Two pixels per group, so 'all_pairs' gives four pair rows.
PIXELS = [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]
LABELS = ["0,0-1,0", "0,0-1,1", "0,1-1,0", "0,1-1,1"]
NFRAMES = 50
CODE_S = timing.CODE_S
CODE_PS = timing.to_ps(CODE_S)


@pytest.fixture(autouse=True)
def _no_rewrite_countdown(monkeypatch):
    """Skip 'confirm_rewrite's five-second pause."""
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


def _run(path, **kwargs):
    """Run stage 1 uncalibrated, with the arguments these tests share."""
    kwargs.setdefault("apply_TDC_calibration", False)
    return co.process_coincidences(path, NFRAMES, PIXELS, **kwargs)


def _counts_path(path, tag="ort"):
    return os.path.join(
        store.processed_dir(path, create=False),
        f"{os.path.basename(path)}_{tag}_coincidences.npy",
    )


# --------------------------------------------------------------------------
# stage 1: the grid is exactly the histogram of the differences
# --------------------------------------------------------------------------


def test_the_grid_is_exactly_the_histogram_of_the_code_differences(bin_folder):
    """The whole claim, checked against an independent recomputation.

    Uncalibrated, so each difference is an integer number of codes and the
    one-code grid is a re-encoding, not a summary: decoding the files again
    and bincounting ``t_b - t_a`` per pair has to give the saved array exactly.
    """
    counts = np.load(_run(bin_folder))
    meta = store.read_meta(_counts_path(bin_folder))

    want = np.zeros_like(counts)
    for fp in io.find_bin_files(bin_folder, "ORT"):
        ts, _ = co.unpack(fp, NFRAMES, compute_time_series=True)
        for row, label in enumerate(LABELS):
            (ra, ca), (rb, cb) = (
                tuple(int(v) for v in p.split(",")) for p in label.split("-")
            )
            a, b = ts[ra, ca], ts[rb, cb]
            both = (a > 0) & (b > 0)
            d = (b[both] - a[both]).astype(np.int64)
            np.add.at(want[row], d + meta["zero_index"], 1)

    np.testing.assert_array_equal(counts, want)
    assert counts.sum() > 0, "the fixture produced no coincidences"
    assert meta["labels"] == LABELS


def test_zero_is_a_cell_centre_and_the_grid_is_symmetric(bin_folder):
    meta = store.read_meta(_run(bin_folder))

    assert meta["n_bins"] % 2 == 1
    assert meta["zero_index"] == meta["n_bins"] // 2
    assert meta["unit"] == "ps"
    assert meta["grid_ps"] == pytest.approx(CODE_PS)


def test_the_grid_is_sized_from_the_data_so_nothing_falls_outside(bin_folder):
    """The support is measured from the run, and wide enough to clip nothing."""
    meta = store.read_meta(_run(bin_folder))

    assert meta["max_code"] > 0
    assert meta["support_s"] >= timing.full_scale_from_max_code(meta["max_code"])
    assert meta["n_outside"] == 0
    assert meta["total_in_grid"] == meta["total"] > 0


def test_a_calibrated_run_uses_the_board_taps_on_a_finer_grid(bin_folder):
    """The packaged compact calibration applies at any clock; no LUT to pick."""
    meta = store.read_meta(
        _run(
            bin_folder,
            apply_TDC_calibration=True,
            daughterboard_number="D0",
            motherboard_number="M0",
        )
    )
    assert meta["calibration"] == "D0_M0"
    assert meta["grid_ps"] == pytest.approx(CODE_PS / co._GRID_SUBDIV)
    assert meta["n_outside"] == 0


def test_calibration_without_a_board_is_refused(bin_folder):
    with pytest.raises(ValueError, match="no board"):
        _run(bin_folder, apply_TDC_calibration=True)


def test_artifact_size_is_set_by_the_grid_not_the_run_length(bin_folder):
    """Six files produce exactly the same bytes as two."""
    two_bytes = np.load(_run(bin_folder, max_files=2)).nbytes
    two_total = store.read_meta(_counts_path(bin_folder))["total"]

    six_path = _run(bin_folder, max_files=6, rewrite=True)

    assert np.load(six_path).nbytes == two_bytes
    assert store.read_meta(six_path)["total"] > two_total


def test_a_complete_artifact_is_not_silently_overwritten(bin_folder):
    _run(bin_folder)
    with pytest.raises(FileExistsError, match="covers all"):
        _run(bin_folder)


def test_different_inputs_will_not_overwrite_without_rewrite(bin_folder):
    _run(bin_folder)
    with pytest.raises(FileExistsError, match="different inputs"):
        co.process_coincidences(
            bin_folder, NFRAMES, [[(2, 2)], [(3, 3)]], apply_TDC_calibration=False
        )


def test_two_programs_in_one_folder_keep_separate_artifacts(bin_folder):
    """The tag is in the file name, so a second program does not collide."""
    rng = np.random.default_rng(20261030)
    for i in range(3):
        with open(os.path.join(bin_folder, f"S0T_{i}.bin"), "wb") as f:
            f.write(rng.bytes(io.BYTES_PER_FRAME * NFRAMES))

    ort = _run(bin_folder)
    s0t = _run(bin_folder, tag="S0T")

    assert ort != s0t
    assert os.path.basename(s0t).endswith("_s0t_coincidences.npy")
    assert store.read_meta(ort)["n_files"] == 6
    assert store.read_meta(s0t)["n_files"] == 3


def _die_after(monkeypatch, n_files):
    """Make 'unpack' raise once it has decoded ``n_files`` files."""
    real = co.unpack
    calls = {"n": 0}

    def dying(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > n_files:
            raise KeyboardInterrupt("simulated dead run")
        return real(*args, **kwargs)

    monkeypatch.setattr(co, "unpack", dying)
    return real


def test_an_interrupted_run_resumes_without_double_counting(
    bin_folder, monkeypatch
):
    monkeypatch.setattr(co, "_FLUSH_EVERY", 1)
    reference = np.load(_run(bin_folder))
    reference_total = store.read_meta(_counts_path(bin_folder))["total"]

    # 4, not 3: the first file is read once up front to measure the code range
    # the grid has to span, so the first *data* unpack is the second call.
    real = _die_after(monkeypatch, 4)
    with pytest.raises(KeyboardInterrupt):
        _run(bin_folder, rewrite=True)

    partial = store.read_meta(_counts_path(bin_folder))
    assert partial["complete"] is False
    assert partial["files_done"] == 3

    monkeypatch.setattr(co, "unpack", real)
    resumed_path = _run(bin_folder)
    np.testing.assert_array_equal(np.load(resumed_path), reference)
    assert store.read_meta(resumed_path)["total"] == reference_total
    assert store.read_meta(resumed_path)["complete"] is True


def test_a_checkpoint_inconsistent_with_its_sidecar_is_not_resumed(
    bin_folder, monkeypatch
):
    """Resuming a newer array against an older sidecar would count files twice."""
    monkeypatch.setattr(co, "_FLUSH_EVERY", 1)
    reference = np.load(_run(bin_folder))

    real = _die_after(monkeypatch, 4)
    with pytest.raises(KeyboardInterrupt):
        _run(bin_folder, rewrite=True)
    monkeypatch.setattr(co, "unpack", real)

    path = _counts_path(bin_folder)
    tampered = np.load(path)
    tampered[0, 0] += 1
    np.save(path, tampered)

    np.testing.assert_array_equal(np.load(_run(bin_folder)), reference)


# --------------------------------------------------------------------------
# frame-shifted (lag) planes
# --------------------------------------------------------------------------


def test_without_subtraction_the_artifact_has_no_lag_axis(bin_folder):
    counts = np.load(_run(bin_folder))
    meta = store.read_meta(_counts_path(bin_folder))

    assert counts.shape == (len(LABELS), meta["n_bins"])
    assert "lags" not in meta
    assert "frames_used" not in meta


def test_lag_zero_plane_is_the_unsubtracted_histogram(tmp_path):
    """Plane 0 is bin-for-bin what a plain run writes."""
    rng = np.random.default_rng(20260803)
    data = [rng.bytes(io.BYTES_PER_FRAME * NFRAMES) for _ in range(4)]
    for folder in ("plain", "lagged"):
        (tmp_path / folder).mkdir()
        for i, blob in enumerate(data):
            (tmp_path / folder / f"ORT_{i}.bin").write_bytes(blob)

    flat = np.load(_run(str(tmp_path / "plain")))
    lags = np.load(
        _run(str(tmp_path / "lagged"), subtract_background=True, background_lags=(1, 2, 3))
    )

    assert lags.shape == (4, *flat.shape)
    np.testing.assert_array_equal(lags[0], flat)


def test_a_lag_shifts_group_b_by_whole_frames(tmp_path, monkeypatch):
    """Lag k pairs frame i of A with frame i + k of B - checked to the code."""
    ts = np.zeros((32, 32, 4))
    ts[0, 0] = [10, 0, 30, 40]  # pixel A; 0 = did not fire
    ts[1, 0] = [0, 100, 200, 300]  # pixel B
    monkeypatch.setattr(co, "unpack", lambda fp, n, compute_time_series=True: (ts, None))
    (tmp_path / "ORT_0.bin").write_bytes(b"")

    counts = np.load(
        co.process_coincidences(
            str(tmp_path),
            4,
            [(0, 0), (1, 0)],
            apply_TDC_calibration=False,
            subtract_background=True,
            background_lags=(1, 9),
        )
    )
    zero = store.read_meta(_counts_path(str(tmp_path)))["zero_index"]

    def hits(plane):
        return sorted((np.nonzero(plane[0])[0] - zero).tolist())

    assert hits(counts[0]) == [170, 260]  # same frame: 200-30, 300-40
    assert hits(counts[1]) == [90, 270]  # lag 1: 100-10, 300-30
    assert counts[2].sum() == 0  # a lag longer than the run yields nothing


def test_frames_used_is_recorded_per_lag(bin_folder):
    _run(bin_folder, subtract_background=True, background_lags=(1, 4))
    meta = store.read_meta(_counts_path(bin_folder))

    assert meta["lags"] == [0, 1, 4]
    assert meta["frames_used"] == [6 * NFRAMES, 6 * (NFRAMES - 1), 6 * (NFRAMES - 4)]


@pytest.mark.parametrize("lags", [(0, 1), (1, 1), (), (-2,)])
def test_bad_background_lags_are_rejected(bin_folder, lags):
    with pytest.raises(ValueError, match="background_lags"):
        _run(bin_folder, subtract_background=True, background_lags=lags)


def test_a_bare_int_is_taken_as_one_lag(bin_folder):
    _run(bin_folder, subtract_background=True, background_lags=1)
    assert store.read_meta(_counts_path(bin_folder))["lags"] == [0, 1]


# --------------------------------------------------------------------------
# stage 2: rebinning
# --------------------------------------------------------------------------


def test_rebinning_conserves_every_count_inside_the_window():
    grid = np.arange(-100, 101) * 10.0
    counts = np.arange(grid.size, dtype=np.int64)

    binned, centres = co._rebin(counts, grid, 50.0e-12, 500.0e-12)

    inside = np.abs(grid) < centres.max() + 25.0
    assert binned.sum() == counts[inside].sum()
    assert binned.size == centres.size


def test_rebinning_onto_a_multiple_of_the_grid_is_a_plain_regrouping():
    """A 50 ps bin over a 10 ps grid is the sum of each run of five."""
    grid = np.arange(-50, 51) * 10.0
    binned, _ = co._rebin(np.ones(grid.size, dtype=np.int64), grid, 50e-12, 250e-12)

    assert set(np.unique(binned[1:-1])) == {5}


def test_lag_planes_rebin_onto_shared_bins():
    """The lags are only subtractable if they share their bins."""
    grid = np.arange(-50, 51) * 10.0
    planes = np.vstack([np.ones(grid.size), 2 * np.ones(grid.size)])

    binned, centres = co._rebin(planes, grid, 50e-12, 250e-12)

    assert binned.shape == (2, centres.size)
    np.testing.assert_array_equal(binned[1], 2 * binned[0])


def _continuous_grid(deltas, cell):
    """Bin continuous (calibrated) differences onto a native grid, as stage 1 does."""
    zero = int(np.ceil(200e3 / cell))
    counts = np.bincount(np.rint(deltas / cell).astype(np.int64) + zero, minlength=2 * zero + 1)
    return counts, (np.arange(2 * zero + 1) - zero) * cell


@pytest.mark.parametrize("cells", [9, 11, 13])
def test_an_odd_number_of_cells_per_bin_rebins_exactly(cells):
    """Odd cells per bin puts the bin edges between cells, so nothing splits."""
    deltas = np.random.default_rng(3).normal(scale=400.0, size=50_000)
    cell = CODE_PS / co._GRID_SUBDIV
    counts, centres = _continuous_grid(deltas, cell)

    width = cells * cell
    got, bins = co._rebin(counts, centres, width * 1e-12, 60 * width * 1e-12)
    edges = np.r_[bins - width / 2, bins[-1] + width / 2]

    np.testing.assert_array_equal(got, np.histogram(deltas, bins=edges)[0])


def test_an_even_number_of_cells_per_bin_does_not():
    """The counterexample, pinned so the odd-cell rule is not lore."""
    deltas = np.random.default_rng(3).normal(scale=400.0, size=50_000)
    cell = CODE_PS / co._GRID_SUBDIV
    counts, centres = _continuous_grid(deltas, cell)

    width = 10 * cell
    got, bins = co._rebin(counts, centres, width * 1e-12, 60 * width * 1e-12)
    want = np.histogram(deltas, bins=np.r_[bins - width / 2, bins[-1] + width / 2])[0]

    assert got.sum() == want.sum(), "counts are still conserved"
    assert np.abs(got - want).sum() > 0, "an even width must split cells"


def test_snapping_picks_an_odd_cell_count_and_a_whole_window(capsys):
    cell = CODE_PS / co._GRID_SUBDIV

    width, window = co._snap_binning(71.0e-12, 10e-9, cell * 1e-12)

    assert width * 1e12 / cell == pytest.approx(9.0)
    assert window / width == pytest.approx(round(window / width))
    assert "9 cells" in capsys.readouterr().out


def test_snapping_moves_an_even_request_off_the_even_count():
    width, _ = co._snap_binning(10 * 7.7e-12, 1000.0e-12, 7.7e-12)
    assert round(width / 7.7e-12) % 2 == 1


def test_snapping_never_returns_a_zero_width():
    width, window = co._snap_binning(0.1e-12, 1.0e-12, 7.7e-12)
    assert width == pytest.approx(7.7e-12)
    assert window > 0


# --------------------------------------------------------------------------
# the subtraction itself, on synthetic data with a known answer
# --------------------------------------------------------------------------


def _injected(nframes, n_pairs, seed, lags):
    """Two uncorrelated pixels plus ``n_pairs`` same-frame pairs at delta 0.

    Codes are uniform over one oscillator range, so the accidentals form the
    same triangle real data does. Returns ``(n_lags, n_bins)`` code counts.
    """
    rng = np.random.default_rng(seed)
    ta = np.where(rng.random(nframes) < 0.30, rng.integers(1, 1347, nframes), np.nan)
    tb = np.where(rng.random(nframes) < 0.25, rng.integers(1, 1347, nframes), np.nan)
    where = rng.choice(nframes, size=n_pairs, replace=False)
    ta[where] = tb[where] = rng.integers(1, 1347, n_pairs)

    support = 1400
    counts = np.zeros((len(lags), 2 * support + 1), dtype=np.int64)
    for li, lag in enumerate(lags):
        d = tb[lag:] - ta[: nframes - lag]
        d = d[np.isfinite(d)].astype(np.int64)
        counts[li] = np.bincount(d + support, minlength=2 * support + 1)
    return counts, (np.arange(2 * support + 1) - support) * 1.0


def test_subtraction_recovers_the_injected_pairs(tmp_path):
    """The residual integrates to the number of pairs actually injected."""
    nframes, n_pairs, lags = 200_000, 400, [0, 1, 2, 3, 4]
    counts, centres = _injected(nframes, n_pairs, seed=7, lags=lags)

    sub = co._subtract_background(
        counts,
        centres,
        [nframes - k for k in lags],
        name="synthetic",
        results_dir=str(tmp_path),
        bin_width_s=1e-12,
        zoom_window_s=1400e-12,
        fit_window_s=1e-12,
        label="test",
    )
    peak = sub["model_free"]

    assert abs(peak["n"] - n_pairs) < 4 * peak["n_err"]
    far = np.abs(sub["centers"]) > 50
    far_sum = sub["residual"][far].sum()
    assert abs(far_sum) < 4 * np.sqrt((sub["error"][far] ** 2).sum())
    assert os.path.isfile(sub["png_path"])
    plt.close("all")


def test_a_lag_that_saw_fewer_frames_is_scaled_up_not_averaged_in():
    counts = np.array([[0, 0, 100, 0, 0], [0, 0, 50, 0, 0]])
    out = co._subtract(counts, np.arange(5.0) - 2, [1000, 500])

    assert out["background"][2] == pytest.approx(100.0)
    assert out["residual"][2] == pytest.approx(0.0)


def test_the_background_is_the_mean_of_the_lags_not_the_sum():
    """K lags estimate one background, so its variance enters divided by K."""
    counts = np.zeros((5, 3), dtype=np.int64)
    counts[0, 1] = 500
    counts[1:, 1] = 400
    out = co._subtract(counts, np.array([-1.0, 0.0, 1.0]), [1000] * 5)

    assert out["k"] == 4
    assert out["residual"][1] == pytest.approx(100.0)
    assert out["error"][1] == pytest.approx(np.sqrt(500 + 400 / 4))


def test_the_model_free_count_sums_the_window_and_adds_errors_in_quadrature(
    tmp_path,
):
    counts = np.zeros((3, 7), dtype=np.int64)
    counts[0] = [10, 10, 40, 60, 40, 10, 10]
    counts[1:] = 10
    centres = (np.arange(7) - 3) * 100.0  # -300..300 ps

    sub = co._subtract_background(
        counts,
        centres,
        [1000] * 3,
        name="synthetic",
        results_dir=str(tmp_path),
        bin_width_s=100e-12,
        zoom_window_s=300e-12,
        fit_window_s=100e-12,
        label="test",
    )
    got = sub["model_free"]

    assert got["n_bins"] == 3
    assert got["n"] == pytest.approx(110.0)  # 30 + 50 + 30 above 10 per bin
    assert got["n_background"] == pytest.approx(30.0)
    assert got["car"] == pytest.approx(110.0 / 30.0)
    assert got["n_err"] == pytest.approx(np.sqrt((sub["error"][2:5] ** 2).sum()))
    plt.close("all")


# --------------------------------------------------------------------------
# stage 2: the plot driver
# --------------------------------------------------------------------------


def test_plot_returns_counts_centres_and_a_fit(bin_folder):
    _run(bin_folder)
    counts, centres, fit = co.plot_coincidences(
        bin_folder, bin_width_s=4 * CODE_S, plot_window_s=2400 * CODE_S, label="test"
    )

    assert counts.shape == centres.shape
    assert fit["n"] == int(counts.sum())
    assert os.path.isfile(
        os.path.join(
            store.results_dir(bin_folder, "coincidences", create=False),
            f"{os.path.basename(bin_folder)}_ort_test_coincidences.png",
        )
    )
    plt.close("all")


def test_plot_pools_only_the_requested_pairs(bin_folder, capsys):
    grid = np.load(_run(bin_folder))
    kwargs = dict(bin_width_s=3 * CODE_S, plot_window_s=300 * CODE_S, label="test")

    every, _, _ = co.plot_coincidences(bin_folder, **kwargs)
    one, centres, _ = co.plot_coincidences(bin_folder, pairs=[LABELS[0], "9,9-9,9"], **kwargs)
    meta = store.read_meta(_counts_path(bin_folder))
    cells = (np.arange(grid.shape[1]) - meta["zero_index"]) * meta["grid_ps"]
    want, _ = co._rebin(grid[0], cells, 3 * CODE_S, 300 * CODE_S)

    np.testing.assert_array_equal(one, want)
    assert every.sum() >= one.sum()
    assert "9,9-9,9" in capsys.readouterr().out
    plt.close("all")


def test_a_lagged_artifact_plots_lag_zero_without_a_subtraction(bin_folder):
    """Lag planes must not leak into a plain plot."""
    grid = np.load(_run(bin_folder, subtract_background=True, background_lags=(1, 2)))
    meta = store.read_meta(_counts_path(bin_folder))

    counts, _, _ = co.plot_coincidences(
        bin_folder, bin_width_s=3 * CODE_S, plot_window_s=200 * CODE_S, label="test"
    )
    cells = (np.arange(grid.shape[-1]) - meta["zero_index"]) * meta["grid_ps"]
    lag0, _ = co._rebin(grid[0].sum(axis=0), cells, 3 * CODE_S, 200 * CODE_S)

    np.testing.assert_array_equal(counts, lag0)
    plt.close("all")


def test_subtracting_needs_an_artifact_with_lag_planes(bin_folder):
    _run(bin_folder)
    with pytest.raises(ValueError, match="lag planes"):
        co.plot_coincidences(bin_folder, subtract_background=True, label="test")


def test_the_fit_model_follows_the_histogram_and_is_not_a_parameter(
    bin_folder, monkeypatch
):
    """Triangle on raw data, flat on a residual - without being told.

    The model is read off the call, not off the fit result: the fixture is
    random bytes with no peak, and whether a fit to noise converges is down to
    the platform's floating point (it did not on ubuntu / py3.11).
    """
    _run(bin_folder, subtract_background=True, background_lags=(1, 2))
    kwargs = dict(bin_width_s=4 * CODE_S, plot_window_s=200 * CODE_S, label="test")
    half_bases = []
    real_fit = co._fit_peak

    def spy(x, y, *, half_base=None, **kw):
        half_bases.append(half_base)
        return real_fit(x, y, half_base=half_base, **kw)

    monkeypatch.setattr(co, "_fit_peak", spy)

    co.plot_coincidences(bin_folder, **kwargs)
    _, _, residual = co.plot_coincidences(bin_folder, subtract_background=True, **kwargs)

    assert half_bases[0] is not None  # raw: the triangle
    assert all(hb is None for hb in half_bases[1:])  # residual: zero baseline
    assert "model_free" in residual
    assert "background" not in inspect.signature(co.plot_coincidences).parameters
    plt.close("all")


def test_the_subtracted_plot_returns_the_residual_and_writes_both_figures(
    bin_folder,
):
    _run(bin_folder, subtract_background=True, background_lags=(1, 2, 3))
    counts, centres, fit = co.plot_coincidences(
        bin_folder,
        bin_width_s=4 * CODE_S,
        plot_window_s=200 * CODE_S,
        subtract_background=True,
        fit_window_s=10 * CODE_S,
        label="test",
    )

    assert counts.shape == centres.shape
    assert counts.dtype.kind == "f"  # a difference of histograms is signed
    np.testing.assert_array_equal(counts, fit["subtraction"]["residual"])
    assert fit["n"] == int(round(fit["model_free"]["n"]))

    results = store.results_dir(bin_folder, "coincidences", create=False)
    name = os.path.basename(bin_folder)
    assert os.path.isfile(os.path.join(results, f"{name}_ort_test_coincidences_sub.png"))
    assert os.path.isfile(
        os.path.join(results, f"{name}_ort_test_background_subtraction.png")
    )
    plt.close("all")


def test_a_gaussian_on_a_triangle_is_recovered():
    """The raw-histogram model, on data drawn from it."""
    x = np.arange(-100_000.0, 100_001.0, 70.0)
    truth = dict(amp=400.0, mu=30.0, sigma=600.0, bkg=900.0, hb=100_000.0)
    y = truth["amp"] * np.exp(-((x - truth["mu"]) ** 2) / (2 * truth["sigma"] ** 2))
    y = y + truth["bkg"] * np.clip(1 - np.abs(x) / truth["hb"], 0, None)
    y = np.random.default_rng(5).poisson(y).astype(float)

    fit = co._fit_peak(x, y, half_base=95_000.0)

    assert fit["ok"]
    assert fit["sigma"] == pytest.approx(truth["sigma"], rel=0.1)
    assert fit["half_base"] == pytest.approx(truth["hb"], rel=0.02)
    assert fit["fwhm"] == pytest.approx(2.355 * fit["sigma"])


def test_the_subtracted_fit_holds_the_baseline_at_zero():
    """A free constant would climb onto the tails of a window narrower than them."""
    x = np.arange(-10_000.0, 10_001.0, 150.0)
    core = 3600 * np.exp(-(x**2) / (2 * 450.0**2))
    tails = 600 * np.exp(-(x**2) / (2 * 1500.0**2))
    y = core + tails + np.random.default_rng(3).normal(0.0, 30.0, x.size)

    fit = co._fit_peak(x, y, fit_window=1000.0)

    assert fit["ok"]
    assert fit["bkg"] == 0.0
    assert fit["fit_range"][0] >= -1000.0 - 150.0
    assert fit["fit_range"][1] <= 1000.0 + 150.0


def test_a_flat_fit_that_cannot_determine_a_width_says_so():
    """No curve is drawn for a failed fit, so it must warn rather than go quiet."""
    x = np.arange(-50.0, 51.0)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit = co._fit_peak(x, np.zeros_like(x), fit_window=20.0)
    assert not fit["ok"]
