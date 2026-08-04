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
    """Return how many frames a '.bin' file has *room* for, from its size.

    !!! warning "Do not use this as the acquisition's frame count."
        The relation between the frame count an acquisition was *asked* for and
        the size of the '.bin' it produces is **not established**. Every sample
        file measured is a whole power-of-two number of frames - 32 MiB
        (16 384 frames) and 16 MiB (8 192) - which is what the exe's DDR3
        transfer size would give for a power-of-two ``nframes``, but not what a
        10 000-frame acquisition should produce. Either those files were taken
        at 16 384 and 8 192, or something rounds; a controlled sweep on the rig
        is what settles it.

        Both this package's 'unpack' and the reference ``kelpie_data_ddr3.m``
        read only the first ``nframes`` frames, so the acquisition setting is
        what every analysis takes as a required argument, and it is the number
        a rate must be normalised by. Use this only when the file size really
        is the only record left, and treat it as an upper bound.

    Parameters
    ----------
    fp : str
        Path to the '.bin' file.

    Returns
    -------
    int
        Frame capacity of the file, ``filesize // BYTES_PER_FRAME``.

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
