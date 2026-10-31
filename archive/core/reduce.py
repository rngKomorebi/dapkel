"""ARCHIVED from dapkel.core.reduce (branch modules-restructure, 2026-10-31).

The full-sensor (64, 64) assembly: the four SPAD quadrant programs (S0C..S3C, or
S0T..S3T) each read one micropixel per 2x2 macropixel, so their (32, 32) maps
interleave onto the 64 x 64 micropixel grid. It was the ``'S*C'`` / ``'S*T'``
wildcard of ``count_map`` in commit f0f2a03; every program in use is 32 x 32,
so it was retired. Written against the live 'core' (accumulate_frames,
readout_mode, SPAD_LAYOUT still exist there), so it should port back as is.

Reference only: not packaged, not tested, not maintained. To plot the full
sensor from four already-saved quadrant maps instead of from raw data, load
each tag's map and interleave with the same ``full[drow::2, dcol::2]`` rule.
"""

from __future__ import annotations

import numpy as np

from dapkel.core import io
from dapkel.core.reduce import SPAD_LAYOUT, accumulate_frames, readout_mode


def count_map_64(
    path: str, tag: str, *, nframes: int, max_files: int | None = None
) -> tuple[np.ndarray, int]:
    """Accumulate the four quadrants of ``'S*C'`` / ``'S*T'`` into (64, 64).

    Each quadrant is cut to the same number of files, so every pixel shares one
    exposure and one rate normalisation; a missing quadrant is NaN, not zero
    (zero would read as a perfectly dark SPAD).

    Returns
    -------
    tuple[np.ndarray, int]
        The (64, 64) count map and the files accumulated per quadrant.
    """
    timestamp = readout_mode(tag) == "timestamp"
    if timestamp:
        extract = lambda ts, pc: (ts > 0).sum(axis=2)  # noqa: E731
    else:
        extract = lambda ts, pc: pc.sum(axis=2)  # noqa: E731

    # Quadrant tags need the separator, or 'S0C' would also match any name that
    # merely contains those characters.
    quadrants: dict[int, list[str]] = {}
    for spad in SPAD_LAYOUT:
        try:
            quadrants[spad] = io.find_bin_files(
                path, tag.replace("*", str(spad)), require_separator=True
            )
        except FileNotFoundError:
            print(f"  (no '{tag.replace('*', str(spad))}' files - left as NaN)")
    if not quadrants:
        raise FileNotFoundError(f"No '{tag}' quadrant files in:\n  {path}")

    n_files = min(len(f) for f in quadrants.values())
    if max_files is not None:
        n_files = min(n_files, max_files)

    full = np.full((64, 64), np.nan)
    for spad, files in quadrants.items():
        drow, dcol = SPAD_LAYOUT[spad]
        full[drow::2, dcol::2] = accumulate_frames(
            files[:n_files],
            extract,
            nframes=nframes,
            need_time_series=timestamp,
            label=tag.replace("*", str(spad)),
        )
    return full, n_files
