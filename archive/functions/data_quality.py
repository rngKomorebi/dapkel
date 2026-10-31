"""ARCHIVED from dapkel.functions.data_quality (branch modules-restructure, 2026-10-31).

Compute-only (no save) code collection. Now 'process_time_codes' saves the per-pixel code histogram and 'plot_time_codes' plots it.

Reference only: not packaged, not tested, not maintained. Copied verbatim
from the pre-restructure module (commit 6b49a4b), so it still refers to
helpers and names that no longer exist in 'src/'. To revive a piece, port it
into the live module rather than importing it from here.
"""

from __future__ import annotations
from collections.abc import Sequence
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dapkel.core import io, timing
from dapkel.functions import calc_diff as cd
from dapkel.functions.calc_diff import Pixel
from dapkel.functions.unpack import unpack


def collect_time_mat(
    path: str,
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
    path : str
        Path to the path with the '.bin' data files.
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
    files = io.find_bin_files(path, tag)
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
    path: str,
    pixel: Pixel,
    *,
    nframes: int,
    tag: str = "ORT",
    bins: int | np.ndarray = 256,
    max_files: int | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Histogram one pixel's TDC codes over all frames (QA / jitter view).

    Mirrors the ``Kelpie_run.m`` timestamp inspection. Useful to confirm, on a
    fresh acquisition, that the timing is populated: the codes should span the
    whole code range of the reference clock in use - ~279 codes at 50 MHz,
    ~1397 at 10 MHz, ~6985 at 2 MHz - and not collapse onto a few values. The
    span is printed in the title as a full scale in ns, which is the same
    measurement every other analysis takes off the data.

    Parameters
    ----------
    path : str
        Path to the path with the '.bin' data files.
    pixel : tuple[int, int]
        The ``(row, col)`` pixel to inspect.
    nframes : int
        Number of frames stored in each '.bin' file.
    tag : str, optional
        Filename fragment selecting the files. The default is ``'ORT'``.
    bins : int | np.ndarray, optional
        Bins for ``np.histogram``. The default is 256.
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
        path, (r, c), nframes=nframes, tag=tag, max_files=max_files
    )[(r, c)]

    # The x axis stays in raw codes: the question this plot answers is whether
    # the codes span the TDC's range, and the code range *is* the evidence. The
    # range they span is reported in the title, in ns, which is the run's own
    # full scale ('core.timing.full_scale_from_max_code') - the number every
    # other analysis now measures rather than being told.
    fig, ax = plt.subplots()
    if codes.size:
        ax.hist(codes, bins=bins, alpha=0.75)
    ax.set_xlabel("Timestamp  (TDC code)")
    ax.set_ylabel("Counts")
    span = (
        f"{timing.full_scale_from_max_code(int(codes.max())) * 1e9:.1f} ns"
        if codes.size
        else "nothing"
    )
    ax.set_title(
        f"Pixel ({r},{c})  time codes  (n={codes.size}, "
        f"{np.unique(codes).size} unique, spanning {span})"
    )
    ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig, codes
