"""Stage 1: folding a set of '.bin' files down to one sensor map.

Every count-like analysis in dapkel is the same loop - unpack each file, pull
one (32, 32) slab out of the frames, add it up - differing only in *which* slab
it pulls. That loop lives here once; the analyses supply the extraction.

What a count *is* follows from the acquisition program, which the file tag
names by its last letter: ``...T`` programs (``ORT``, ``S0T``) timestamp, so a
count is a frame carrying a valid timestamp (at most one per pixel per frame);
``...C`` programs (``ORC``, ``S0C``) count, so the per-pixel photon counter is
summed. In timestamp mode that counter field does not exist - 'unpack' returns
low bits of the coarse timestamp there - so reading a ``T`` tag as counts would
overstate the rate ~50x. Deriving the mode from the tag makes that impossible.

Every program reads a (32, 32) array: ``ORT``/``ORC`` all four micropixels of a
macropixel at once, ``S<n>T``/``S<n>C`` one of them. (Assembling four ``S<n>``
quadrants into the (64, 64) micropixel grid is archived:
``archive/core/reduce.py``.)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from dapkel.core import io, timing
from dapkel.functions.unpack import unpack

__all__ = [
    "SPAD_LAYOUT",
    "accumulate_frames",
    "code_histogram",
    "count_map",
    "readout_mode",
]

#: SPAD (micropixel) index -> its ``(drow, dcol)`` offset inside one 2x2
#: macropixel. The indices run CLOCKWISE, not row-major::
#:
#:     S0 (0,0)   S1 (0,1)
#:     S3 (1,0)   S2 (1,1)
#:
#: so S1 is S0's horizontal neighbour, S3 its vertical one and S2 its diagonal.
#: The cross-talk directions read this.
SPAD_LAYOUT = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}

#: An extraction: given one file's ``(time_series, photon_counts)`` arrays,
#: each shaped ``(32, 32, nframes)``, return the (32, 32) slab to accumulate.
Extract = Callable[[np.ndarray | None, np.ndarray], np.ndarray]


def readout_mode(tag: str) -> str:
    """Return ``'timestamp'`` or ``'count'`` for a file tag, from its last letter.

    Raises
    ------
    ValueError
        Raised when the tag ends in neither ``T`` nor ``C`` (trailing digits
        ignored), so the program - and hence what a count is - is unknown.
    """
    letter = tag.upper().rstrip("0123456789_")[-1:]
    mode = {"T": "timestamp", "C": "count"}.get(letter)
    if mode is None:
        raise ValueError(
            f"tag {tag!r} names neither a timestamping ('...T', e.g. 'ORT', "
            "'S0T') nor a counting ('...C', e.g. 'ORC', 'S0C') program, so "
            "what a count is cannot be decided. Pass the acquisition's tag."
        )
    return mode


def accumulate_frames(
    files: list[str],
    extract: Extract,
    *,
    nframes: int,
    need_time_series: bool = True,
    label: str = "",
) -> np.ndarray:
    """Unpack every file and sum an extracted (32, 32) slab over all of them.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files to unpack and accumulate.
    extract : callable
        ``extract(time_series, photon_counts) -> np.ndarray`` returning the
        (32, 32) contribution of one file. ``time_series`` is None when
        ``need_time_series`` is False.
    nframes : int
        Frames per file - the acquisition setting. Never infer it from the
        file size: the exe dumps a fixed DDR3 region larger than the run.
    need_time_series : bool, optional
        Whether ``extract`` uses the timestamps. False lets 'unpack' skip the
        coarse/fine decoding, about two thirds of its work. The default is True.
    label : str, optional
        Label used in the progress bar. The default is "".

    Returns
    -------
    np.ndarray
        The summed (32, 32) map, float64.
    """
    total = np.zeros((32, 32), dtype=np.float64)
    for fp in tqdm(files, desc=label or None):
        ts, pc = unpack(fp, nframes, compute_time_series=need_time_series)
        total += extract(ts, pc)
    return total


def count_map(
    path: str, tag: str, *, nframes: int, max_files: int | None = None
) -> tuple[np.ndarray, int]:
    """Accumulate the (32, 32) per-pixel count map of one program's files.

    Parameters
    ----------
    path : str
        Folder with the '.bin' files.
    tag : str
        File tag, e.g. ``'ORT'``, ``'ORC'``, ``'S0C'``, ``'S2T'``. Its last
        letter decides what a count is (see the module docstring).
    nframes : int
        Frames per file.
    max_files : int | None, optional
        Use at most this many files. The default is None (all).

    Returns
    -------
    counts : np.ndarray
        The (32, 32) map, float64.
    n_files : int
        Files accumulated, so ``nframes * n_files`` frames per pixel.

    Raises
    ------
    FileNotFoundError
        Raised when no file matches ``tag``.
    """
    timestamp = readout_mode(tag) == "timestamp"
    if timestamp:
        extract = lambda ts, pc: (ts > 0).sum(axis=2)  # noqa: E731
    else:
        extract = lambda ts, pc: pc.sum(axis=2)  # noqa: E731
    files = io.find_bin_files(path, tag)[:max_files]
    counts = accumulate_frames(
        files, extract, nframes=nframes, need_time_series=timestamp, label=tag
    )
    return counts, len(files)


def code_histogram(
    files: list[str], *, nframes: int, label: str = ""
) -> np.ndarray:
    """Accumulate the (32, 32, n_codes) per-pixel TDC-code histogram.

    For every pixel, how often each integer code occurs over all frames with a
    valid timestamp (``time_series > 0``). The code axis is allocated at the
    widest the frame format can express ('timing.MAX_TS_CODE') and trimmed to
    the largest code any pixel produced, so the code count comes out of the
    run rather than out of a stated clock and nothing can overflow.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files.
    nframes : int
        Frames per file.
    label : str, optional
        Label used in the progress bar. The default is "".

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) int64 histogram.
    """
    n_alloc = timing.MAX_TS_CODE + 1
    # One bincount per file histograms all 1024 pixels at once:
    # linear index = pixel * n_alloc + code.
    pix_index = np.arange(1024, dtype=np.int64).reshape(32, 32, 1)
    counts = np.zeros(1024 * n_alloc, dtype=np.int64)
    for fp in tqdm(files, desc=label or None):
        ts, _ = unpack(fp, nframes, compute_time_series=True)
        code = ts.astype(np.int64)
        valid = (ts > 0) & (code < n_alloc)
        counts += np.bincount(
            (pix_index * n_alloc + code)[valid], minlength=1024 * n_alloc
        )
    counts = counts.reshape(32, 32, n_alloc)
    used = np.nonzero(counts.sum(axis=(0, 1)))[0]
    return counts[:, :, : int(used[-1]) + 1] if used.size else counts
