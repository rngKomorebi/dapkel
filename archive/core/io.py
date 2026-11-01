"""ARCHIVED from dapkel.core.io (branch modules-restructure, 2026-10-31).

Names no analysis uses any more. Reference only: not packaged, not
tested, not maintained (commit 6b49a4b).
"""


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
