"""Stage 1: folding a set of '.bin' files down to one sensor map.

Every count-like analysis in dapkel is the same loop - unpack each file, pull
one (32, 32) slab out of the frames, add it up - differing only in *which* slab
it pulls. That loop lives here once; the analyses supply the extraction.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from dapkel.core import io
from dapkel.functions.unpack import unpack

__all__ = [
    "accumulate_frames",
    "assemble_64",
]

#: An extraction: given one file's ``(time_series, photon_counts)`` arrays,
#: each shaped ``(32, 32, nframes)``, return the (32, 32) slab to accumulate.
Extract = Callable[[np.ndarray | None, np.ndarray], np.ndarray]


def accumulate_frames(
    files: list[str],
    extract: Extract,
    *,
    nframes: int | None = None,
    need_time_series: bool = True,
    label: str = "",
    shape: tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Unpack every file and sum an extracted slab over all of them.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files to unpack and accumulate.
    extract : callable
        ``extract(time_series, photon_counts) -> np.ndarray`` returning the
        (32, 32) contribution of one file. Both inputs are
        ``(32, 32, nframes)``; ``time_series`` is None when
        ``need_time_series`` is False. Typical bodies are
        ``lambda ts, pc: pc.sum(axis=2)`` for photon counts and
        ``lambda ts, pc: (ts > 0).sum(axis=2)`` for ORT occupancy.
    nframes : int | None, optional
        Frames per file. When None (the default) it is derived from each
        file's size, so files of differing length are handled correctly.
    need_time_series : bool, optional
        Whether ``extract`` uses the timestamps. Passing False lets 'unpack'
        skip the coarse/fine time decoding - roughly two thirds of the
        per-pixel work - so count-only analyses should set it. The default is
        True.
    label : str, optional
        Label used in the progress bar. The default is "".
    shape : tuple[int, int], optional
        Shape of the accumulator. The default is ``(32, 32)``.

    Returns
    -------
    np.ndarray
        The summed map, float64.
    """
    total = np.zeros(shape, dtype=np.float64)
    for fp in tqdm(files, desc=label or None):
        nf = nframes if nframes is not None else io.frames_in_file(fp)
        ts, pc = unpack(fp, nf, compute_time_series=need_time_series)
        total += extract(ts, pc)
    return total


def assemble_64(
    quadrants: dict[str, np.ndarray], layout: dict[str, tuple[int, int]]
) -> np.ndarray:
    """Interleave four (32, 32) quadrant maps into the (64, 64) sensor map.

    Each SPAD tag reads one micropixel per 2x2 macropixel, so the four
    quadrant maps interleave rather than tile: quadrant with offset
    ``(drow, dcol)`` occupies ``full[drow::2, dcol::2]``.

    Parameters
    ----------
    quadrants : dict[str, np.ndarray]
        Mapping of SPAD tag to its (32, 32) map. Tags absent from the mapping
        are left as zero, so a partial acquisition still assembles.
    layout : dict[str, tuple[int, int]]
        Mapping of SPAD tag to its ``(drow, dcol)`` offset inside the 2x2
        macropixel. See the ``_SPAD_LAYOUT`` of the calling module - the SPAD
        indices run clockwise (S0 S1 / S3 S2), not row-major.

    Returns
    -------
    np.ndarray
        The (64, 64) assembled map, float64.
    """
    full = np.zeros((64, 64), dtype=np.float64)
    for tag, (drow, dcol) in layout.items():
        if tag in quadrants:
            full[drow::2, dcol::2] = quadrants[tag]
    return full
