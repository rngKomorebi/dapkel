"""ARCHIVED tests (branch modules-restructure, 2026-10-31): the (64, 64)
quadrant assembly, written against the 'S*C' wildcard of core.reduce.count_map
(commit f0f2a03), now archive/core/reduce.py:count_map_64. Not collected by
pytest, not maintained.
"""

import numpy as np

from dapkel.core import reduce
from dapkel.functions import hitmap_analysis

NFRAMES = 8


# --------------------------------------------------------------------------
# the full sensor: four quadrants, one exposure
# --------------------------------------------------------------------------


def test_the_quadrants_interleave_onto_the_full_sensor(tmp_path, monkeypatch):
    """Each SPAD's map lands on its own cell of every 2x2 macropixel."""

    def by_quadrant(fp, n, compute_time_series=True):
        spad = int(fp.split("_S")[1][0])
        return None, np.full((32, 32, n), float(spad + 1))

    monkeypatch.setattr("dapkel.core.reduce.unpack", by_quadrant)
    for spad in range(4):
        (tmp_path / f"run_S{spad}C_0.bin").write_bytes(b"")

    full, n_files = reduce.count_map(str(tmp_path), "S*C", nframes=NFRAMES)

    assert full.shape == (64, 64) and n_files == 1
    for spad, (drow, dcol) in reduce.SPAD_LAYOUT.items():
        assert (full[drow::2, dcol::2] == (spad + 1) * NFRAMES).all()


def test_a_missing_quadrant_is_nan_not_zero(tmp_path, monkeypatch):
    """Zero would read as a perfectly dark SPAD; NaN reads as no data."""
    monkeypatch.setattr(
        "dapkel.core.reduce.unpack",
        lambda fp, n, compute_time_series=True: (None, np.ones((32, 32, n))),
    )
    for spad in (0, 1, 2):
        (tmp_path / f"run_S{spad}C_0.bin").write_bytes(b"")

    full, _ = reduce.count_map(str(tmp_path), "S*C", nframes=NFRAMES)

    drow, dcol = reduce.SPAD_LAYOUT[3]
    assert np.isnan(full[drow::2, dcol::2]).all()
    assert np.isfinite(full[0::2, 0::2]).all()


def test_quadrants_are_cut_to_a_shared_exposure(tmp_path, monkeypatch):
    """A quadrant with more files must not get a longer exposure than the rest."""
    monkeypatch.setattr(
        "dapkel.core.reduce.unpack",
        lambda fp, n, compute_time_series=True: (None, np.ones((32, 32, n))),
    )
    for spad in range(4):
        for i in range(3 if spad == 0 else 2):
            (tmp_path / f"run_S{spad}C_{i}.bin").write_bytes(b"")

    full, n_files = reduce.count_map(str(tmp_path), "S*C", nframes=NFRAMES)

    assert n_files == 2
    assert np.unique(full).tolist() == [2.0 * NFRAMES]


def test_full_sensor_hitmap_from_the_quadrant_wildcard(tmp_path):
    path = _write(tmp_path, [f"run_S{s}C_0.bin" for s in range(4)])

    saved = hitmap_analysis.process_hitmap(path, NFRAMES, "S*C")
    rate = hitmap_analysis.plot_hitmap(path, "S*C")

    assert os.path.basename(saved).endswith("_sxc_hitmap.npy")
    assert rate.shape == (64, 64)
