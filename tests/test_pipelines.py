"""Every analysis, end to end: 'process_<x>' saves, 'plot_<x>' reads it back.

The physics of each reduction is pinned in its own test file. These pin the
contract that makes the package two calls per measurement: process writes its
artifact under ``processed/`` with a sidecar, plot finds it from the same
``(path, tag)`` alone - no '.bin' re-read - and writes its figures under
``results/``. Seeded random bytes stand in for data; 'unpack' is a pure decode.
"""

from __future__ import annotations

import glob
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from dapkel.core import io, store  # noqa: E402
from dapkel.functions import (  # noqa: E402
    crosstalk_analysis,
    data_quality,
    dcr_analysis,
    hitmap_analysis,
)

NFRAMES = 20


def _write(folder, names, seed=1):
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    for name in names:
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(rng.bytes(io.BYTES_PER_FRAME * NFRAMES))
    return str(folder)


def _figures(path, kind):
    return sorted(
        os.path.basename(f)
        for f in glob.glob(os.path.join(store.results_dir(path, kind), "*.png"))
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_hitmap_round_trip(tmp_path):
    path = _write(tmp_path, [f"ORT_{i}.bin" for i in range(3)])

    saved = hitmap_analysis.process_hitmap(path, NFRAMES, "ORT")
    counts, rate = hitmap_analysis.plot_hitmap(path, "ORT")

    saved_counts, meta = store.load_map(npy_path=saved)
    assert meta["total_frames"] == 3 * NFRAMES
    np.testing.assert_array_equal(counts, saved_counts)
    np.testing.assert_allclose(rate, counts / (3 * NFRAMES * 9e-6))
    name = os.path.basename(path)
    assert _figures(path, "hitmap") == [
        f"{name}_ort_hitmap.png",
        f"{name}_ort_hitmap_rate.png",
    ]


def test_dcr_round_trip(tmp_path):
    path = _write(tmp_path, [f"run_S0C_{i}.bin" for i in range(2)])

    dcr_analysis.process_dcr(path, NFRAMES, "S0C", open_shutter_time_s=1e-7)
    dcr = dcr_analysis.plot_dcr(path, "S0C")

    assert dcr.shape == (32, 32)
    name = os.path.basename(path)
    assert _figures(path, "dcr") == [
        f"{name}_s0c_dcr_distribution.png",
        f"{name}_s0c_dcr_heatmap.png",
    ]


def test_crosstalk_round_trip_for_one_pairing(tmp_path):
    path = _write(tmp_path, ["CT_S0S1_coinc_0.bin", "CT_S0S1_OR_0.bin"])

    crosstalk_analysis.process_crosstalk(path, NFRAMES)
    ct = crosstalk_analysis.plot_crosstalk(path)

    assert ct.shape == (32, 32)
    assert store.load_map(path, kind="crosstalk", tag="CT_")[1]["pair"] == "S0S1"
    assert len(_figures(path, "crosstalk")) == 2


def test_crosstalk_combines_a_parent_folder_of_pairings(tmp_path):
    """Each pairing lands on its own neighbour cell of the (64, 64) map."""
    for pair in ("S0S1", "S0S3", "S0S2"):
        _write(tmp_path / pair, [f"CT_{pair}_coinc_0.bin", f"CT_{pair}_OR_0.bin"])
    parent = str(tmp_path)

    crosstalk_analysis.process_crosstalk(parent, NFRAMES)
    combined = crosstalk_analysis.plot_crosstalk(parent)

    assert combined.shape == (64, 64)
    one, _ = store.load_map(str(tmp_path / "S0S3"), kind="crosstalk", tag="CT_")
    np.testing.assert_array_equal(combined[1::2, 0::2], one)  # S3 is vertical
    meta = store.load_map(parent, kind="crosstalk", tag="CT_")[1]
    assert set(meta["median_ct_percent"]) == {
        "horizontal",
        "vertical",
        "diagonal",
        "overall (mean)",
    }


def test_crosstalk_without_any_pairing_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="CT_"):
        crosstalk_analysis.process_crosstalk(str(tmp_path), NFRAMES)


def test_time_codes_round_trip(tmp_path):
    path = _write(tmp_path, [f"ORT_{i}.bin" for i in range(2)])

    saved = data_quality.process_time_codes(path, NFRAMES)
    hist = data_quality.plot_time_codes(path, (3, 4))

    full, _ = store.load_map(npy_path=saved)
    assert full.shape[:2] == (32, 32)
    np.testing.assert_array_equal(hist, full[3, 4])
    assert hist.sum() > 0
    assert len(_figures(path, "data_quality")) == 1


def test_plot_without_process_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="process_"):
        hitmap_analysis.plot_hitmap(str(tmp_path), "ORT")
