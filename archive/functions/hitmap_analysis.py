"""ARCHIVED from dapkel.functions.hitmap_analysis (branch modules-restructure, 2026-10-31).

Compute-only (no save) entry points. Every hitmap is now saved: 'process_hitmap' / 'plot_hitmap'.

Reference only: not packaged, not tested, not maintained. Copied verbatim
from the pre-restructure module (commit 6b49a4b), so it still refers to
helpers and names that no longer exist in 'src/'. To revive a piece, port it
into the live module rather than importing it from here.
"""

from __future__ import annotations
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from dapkel.core import io, plots, reduce, store, timing


def compute_hitmap(
    path: str,
    nframes: int,
    mode: str = "timestamp",
    tag: str = "ORT",
    max_files: int | None = None,
) -> np.ndarray:
    """Unpack the binary data and compute the (32, 32) hitmap.

    Builds a single-readout hitmap from the '.bin' files matching ``tag``.

    Parameters
    ----------
    path : str
        Path to the path with the '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    mode : str, optional
        ``'timestamp'`` for the ORT occupancy hitmap (per pixel, the number
        of frames with a valid first-photon timestamp), or ``'count'`` for
        the accumulated photon-count hitmap (S*C / ORC count programs). The
        default is ``'timestamp'``.
    tag : str, optional
        Filename fragment selecting the files (e.g. ``'ORT'``, ``'ORC'``,
        or a SPAD quadrant tag ``'S0C'``). Use ``''`` to match every
        '.bin' file in the path. The default is ``'ORT'``.
    max_files : int | None, optional
        Process at most this many files (after natural sorting). Useful
        for a quick preview over a large batch. The default is None (all
        files).

    Returns
    -------
    np.ndarray
        The (32, 32) hitmap.
    """
    files = io.find_bin_files(path, tag)
    if max_files is not None:
        files = files[:max_files]
    return _accumulate_hitmap(
        files, mode, nframes=nframes, label=tag or mode
    )


def compute_hitmap_64(
    path: str,
    nframes: int,
    max_files: int | None = None,
) -> np.ndarray:
    """Unpack the four S*C tags and assemble the (64, 64) count hitmap.

    Accumulates the four SPAD tags' photon counts (``mode='count'``) and
    interleaves them onto the full sensor. Only the count programs (S*C)
    tile this way; ORT/ORC are (32, 32) and use 'compute_hitmap'.

    Parameters
    ----------
    path : str
        Path to the path with the S*C '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    max_files : int | None, optional
        Process at most this many files per tag. The default is None.

    Returns
    -------
    np.ndarray
        The (64, 64) accumulated-count hitmap for the full sensor.

    Raises
    ------
    FileNotFoundError
        Raised when no '.bin' files for any of the four SPAD tags are
        found in the path.
    """
    tag_files = {}
    for tag in _SPAD_LAYOUT:
        files = glob.glob(os.path.join(path, f"*_{tag}*.bin"))
        if files:
            files = sorted(
                files,
                key=lambda fp: int(
                    "".join(filter(str.isdigit, os.path.basename(fp))) or "0"
                ),
            )
            tag_files[tag] = files[:max_files] if max_files else files
    if not tag_files:
        raise FileNotFoundError(
            "No S0C/S1C/S2C/S3C .bin files found in:\n  " + path
        )

    quadrants = {
        tag: _accumulate_hitmap(files, "count", nframes=nframes, label=tag)
        for tag, files in tag_files.items()
    }
    return reduce.assemble_64(quadrants, _SPAD_LAYOUT)
