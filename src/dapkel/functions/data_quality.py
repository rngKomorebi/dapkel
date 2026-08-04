"""Data-quality checks to run on a fresh Kelpie acquisition.

Nothing here contributes to a physics result - these answer one question before
you spend time on analysis: did the TDC actually record timing?

The failure mode is silent. When the TDC never stops, 'unpack' still returns a
full array of plausible integers and every downstream analysis still runs on
meaningless data. Healthy codes spread over the full ~0..1300 oscillator range;
a broken run collapses onto a handful of values.

See ``docs/guide/data_quality.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dapkel.core import io
from dapkel.functions import calc_diff as cd
from dapkel.functions.calc_diff import Pixel
from dapkel.functions.unpack import unpack

__all__ = [
    # gather raw TDC codes
    "collect_time_mat",
    # plot
    "plot_time_code_histogram",
]


def collect_time_mat(
    folder: str,
    pixels: Pixel | Sequence[Pixel],
    *,
    nframes: int,
    tag: str = "ORT",
    max_files: int | None = None,
) -> dict[Pixel, np.ndarray]:
    """Collect the per-frame TDC codes (``time_mat``) for given pixels.

    Returns, per requested pixel, the stack of its per-frame TDC codes over
    every file, keeping only frames with a valid timestamp - 'unpack' leaves
    non-fired slots at ``<= 0``, so that is ``time_series > 0``.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    pixels : tuple[int, int] | Sequence[tuple[int, int]]
        A single ``(row, col)`` pixel or a sequence of them.
    nframes : int
        Number of frames stored in each '.bin' file.
    tag : str, optional
        Filename fragment selecting the files. The default is ``'ORT'``.
    max_files : int | None, optional
        Process at most this many files. The default is None.

    Returns
    -------
    dict[tuple[int, int], np.ndarray]
        Mapping ``(row, col) -> 1D array`` of valid TDC codes.
    """
    pix = cd.as_pixel_list(pixels)
    files = io.find_bin_files(folder, tag)
    if max_files is not None:
        files = files[:max_files]

    chunks: dict[Pixel, list[np.ndarray]] = {p: [] for p in pix}
    for fp in tqdm(files, desc=tag or "time_mat"):
        ts, _ = unpack(fp, nframes, compute_time_series=True)
        for r, c in pix:
            code = ts[r, c]
            chunks[(r, c)].append(code[code > 0])

    return {
        p: (np.concatenate(v) if v else np.empty(0, dtype=np.float64))
        for p, v in chunks.items()
    }


def plot_time_code_histogram(
    folder: str,
    pixel: Pixel,
    *,
    nframes: int,
    tag: str = "ORT",
    bins: int | np.ndarray = 256,
    time_unit_ps: float | None = None,
    max_files: int | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Histogram one pixel's TDC codes over all frames (QA / jitter view).

    Mirrors the ``Kelpie_run.m`` timestamp inspection. Useful to confirm, on a
    fresh acquisition, that the timing is populated (codes should span the full
    0..~1300 oscillator range, not collapse onto a few values).

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    pixel : tuple[int, int]
        The ``(row, col)`` pixel to inspect.
    nframes : int
        Number of frames stored in each '.bin' file.
    tag : str, optional
        Filename fragment selecting the files. The default is ``'ORT'``.
    bins : int | np.ndarray, optional
        Bins for ``np.histogram``. The default is 256.
    time_unit_ps : float | None, optional
        Picoseconds per code for the x-axis; None (default) keeps raw codes.
    max_files : int | None, optional
        Process at most this many files. The default is None.

    Returns
    -------
    fig : plt.Figure
        The generated figure.
    codes : np.ndarray
        The pixel's valid TDC codes over all frames.
    """
    r, c = int(pixel[0]), int(pixel[1])
    codes = collect_time_mat(
        folder, (r, c), nframes=nframes, tag=tag, max_files=max_files
    )[(r, c)]

    scale = time_unit_ps if time_unit_ps else 1.0
    unit = "ps" if time_unit_ps else "TDC code"

    fig, ax = plt.subplots()
    if codes.size:
        ax.hist(codes * scale, bins=bins, alpha=0.75)
    ax.set_xlabel(f"Timestamp  ({unit})")
    ax.set_ylabel("Counts")
    ax.set_title(
        f"Pixel ({r},{c})  time codes  (n={codes.size}, "
        f"{np.unique(codes).size} unique)"
    )
    ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig, codes
