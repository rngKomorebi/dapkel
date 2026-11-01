"""Stage 0: locating and sizing the raw Kelpie '.bin' files.

Every analysis starts by picking the '.bin' files that belong to a measurement.
The frame count is never read off a file's size: the exe dumps a fixed DDR3
region larger than the acquisition, so it is the caller's ``nframes``.
"""

from __future__ import annotations

import glob
import os

__all__ = [
    "BYTES_PER_FRAME",
    "find_bin_files",
]

#: Bytes per frame in a Kelpie v2 '.bin' file: 4 * 64 * 8 (see 'unpack').
BYTES_PER_FRAME = 4 * 64 * 8


def find_bin_files(
    path: str, tag: str, *, require_separator: bool = False
) -> list[str]:
    """Find and naturally sort the '.bin' files for a readout tag.

    "Naturally" means ordered by the digits in the file name, so ``..._2.bin``
    sorts before ``..._10.bin``.

    Parameters
    ----------
    path : str
        Path to the folder with the '.bin' data files.
    tag : str
        Filename fragment selecting the files (e.g. ``'ORT'``, ``'S0C'``,
        ``'S2T'``), or ``''`` to match every '.bin' file in the path.
    require_separator : bool, optional
        When True the tag must be preceded by an underscore (pattern
        ``'*_<tag>*.bin'`` instead of ``'*<tag>*.bin'``). The DCR quadrant
        tags need this so that, for example, ``'S0C'`` cannot also match a
        file whose name merely contains those characters. The default is
        False.

    Returns
    -------
    list[str]
        Naturally sorted list of paths to the matching '.bin' files.

    Raises
    ------
    FileNotFoundError
        Raised when no '.bin' files match the tag in the path.
    """
    if tag:
        pattern = f"*_{tag}*.bin" if require_separator else f"*{tag}*.bin"
    else:
        pattern = "*.bin"

    files = sorted(
        glob.glob(os.path.join(path, pattern)),
        key=lambda fp: int(
            "".join(filter(str.isdigit, os.path.basename(fp))) or "0"
        ),
    )
    if not files:
        raise FileNotFoundError(
            f"No .bin files matching '{pattern}' found in:\n  {path}"
        )
    return files
