"""Data-quality check to run on a fresh Kelpie acquisition.

    process_time_codes   raw '.bin' -> processed/<name>_<tag>_time_codes.npy
    plot_time_codes      that array -> results/data_quality/ one pixel's codes

Nothing here contributes to a physics result - it answers one question before
you spend time on analysis: did the TDC actually record timing?

The failure mode is silent. When the TDC never stops, 'unpack' still returns a
full array of plausible integers and every downstream analysis still runs on
meaningless data. Healthy codes spread over the full code range of the clock in
use (~279 codes at 50 MHz, ~1397 at 10 MHz, ~6985 at 2 MHz); a broken run
collapses onto a handful of values.

See ``docs/guide/data_quality.md``.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import io, reduce, store, timing
from dapkel.core.pairs import Pixel

__all__ = ["process_time_codes", "plot_time_codes"]

#: Analysis name: the stage-1 file stem and results sub-folder.
_KIND = "time_codes"


def process_time_codes(
    path: str,
    nframes: int,
    tag: str = "ORT",
    *,
    max_files: int | None = None,
) -> str:
    """Histogram every pixel's TDC codes over all frames and save it.

    Saves the (32, 32, n_codes) per-pixel code histogram, which is a few tens
    of MB at the slowest clock - so any pixel can then be inspected without
    re-reading the run.

    Parameters
    ----------
    path : str
        Folder with the '.bin' data files.
    nframes : int
        Frames the acquisition recorded per '.bin' file.
    tag : str, optional
        File tag of a timestamping program. The default is ``'ORT'``.
    max_files : int | None, optional
        Use at most this many files - a handful is enough to see whether the
        timing is populated. The default is None (all).

    Returns
    -------
    str
        Path to the saved '.npy'.
    """
    files = io.find_bin_files(path, tag)[:max_files]
    counts = reduce.code_histogram(files, nframes=nframes, label=tag)
    return store.save_map(
        counts,
        path,
        kind=_KIND,
        tag=tag,
        meta={"n_files": len(files), "nframes": int(nframes)},
    )


def plot_time_codes(
    path: str, pixel: Pixel = (16, 16), tag: str = "ORT"
) -> np.ndarray:
    """Plot one pixel's TDC-code histogram from the saved per-pixel histogram.

    Mirrors the ``Kelpie_run.m`` timestamp inspection: the codes should span
    the whole code range of the reference clock and not collapse onto a few
    values. The title gives the span as a full scale in ns, the same
    measurement every other analysis takes off the data
    ('core.timing.full_scale_from_max_code').

    Parameters
    ----------
    path : str
        Data folder that 'process_time_codes' was run on.
    pixel : tuple[int, int], optional
        The ``(row, col)`` pixel to inspect. The default is ``(16, 16)``.
    tag : str, optional
        The tag it was run with. The default is ``'ORT'``.

    Returns
    -------
    np.ndarray
        The pixel's counts per code.
    """
    counts, _ = store.load_map(path, kind=_KIND, tag=tag)
    r, c = int(pixel[0]), int(pixel[1])
    hist = counts[r, c]
    used = np.nonzero(hist)[0]

    # The x axis stays in raw codes: whether the codes span the TDC's range is
    # the question, and the code range is the evidence.
    fig, ax = plt.subplots()
    ax.step(np.arange(hist.size), hist, where="mid")
    ax.set_xlabel("Timestamp  (TDC code)")
    ax.set_ylabel("Counts")
    span = (
        f"{timing.full_scale_from_max_code(int(used[-1])) * 1e9:.1f} ns"
        if used.size
        else "nothing"
    )
    ax.set_title(
        f"Pixel ({r},{c})  time codes  (n={hist.sum()}, {used.size} unique, "
        f"spanning {span})"
    )
    ax.grid(True)
    fig.tight_layout()

    name = os.path.basename(os.path.normpath(path))
    store.save_figure(
        fig,
        store.results_dir(path, "data_quality"),
        f"{store.map_file_name(name, _KIND, tag).removesuffix('.npy')}_{r}_{c}.png",
    )
    return hist
