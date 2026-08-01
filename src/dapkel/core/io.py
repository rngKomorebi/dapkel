"""Stage 0: locating and sizing the raw Kelpie '.bin' files.

Every analysis starts by picking the '.bin' files that belong to a measurement
and working out how many frames each holds. Both were previously copied into
four modules; they live here now so the '.bin' format is described in exactly
one place.
"""

from __future__ import annotations

import glob
import os

__all__ = [
    "BYTES_PER_FRAME",
    "frames_in_file",
    "find_bin_files",
]

#: Bytes per frame in a Kelpie v2 '.bin' file: 4 * 64 * 8 (see 'unpack').
BYTES_PER_FRAME = 4 * 64 * 8


def frames_in_file(fp: str) -> int:
    """Return the number of frames stored in a '.bin' file from its size.

    Parameters
    ----------
    fp : str
        Path to the '.bin' file.

    Returns
    -------
    int
        Number of frames, ``filesize // BYTES_PER_FRAME``.

    Raises
    ------
    ValueError
        Raised when the file size is not a whole number of frames.
    """
    size = os.path.getsize(fp)
    if size == 0 or size % BYTES_PER_FRAME != 0:
        raise ValueError(
            f"{os.path.basename(fp)} is {size} bytes, not a whole number of "
            f"{BYTES_PER_FRAME}-byte frames."
        )
    return size // BYTES_PER_FRAME


def find_bin_files(
    folder: str, tag: str, *, require_separator: bool = False
) -> list[str]:
    """Find and naturally sort the '.bin' files for a readout tag.

    "Naturally" means ordered by the digits in the file name, so ``..._2.bin``
    sorts before ``..._10.bin``.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    tag : str
        Filename fragment selecting the files (e.g. ``'ORT'``, ``'S0C'``,
        ``'SPAD0_S0'``), or ``''`` to match every '.bin' file in the folder.
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
        Raised when no '.bin' files match the tag in the folder.
    """
    if tag:
        pattern = f"*_{tag}*.bin" if require_separator else f"*{tag}*.bin"
    else:
        pattern = "*.bin"

    files = sorted(
        glob.glob(os.path.join(folder, pattern)),
        key=lambda fp: int(
            "".join(filter(str.isdigit, os.path.basename(fp))) or "0"
        ),
    )
    if not files:
        raise FileNotFoundError(
            f"No .bin files matching '{pattern}' found in:\n  {folder}"
        )
    return files
