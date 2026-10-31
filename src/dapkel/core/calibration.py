"""Applying the compact TDC calibration: a code in, picoseconds out, any clock.

The ring-oscillator TDC has ``TAPS_PER_LAP`` = 8 taps, and 'unpack' builds the
code as ``(coarse - 1) * 8 + (8 - fine)``: code ``k`` is tap ``k % 8`` of lap
``k // 8``. The coarse counter adds no non-linearity, so one calibration is just
the eight tap widths of every pixel - ``(32, 32, 8)`` per SPAD - and the time of
a code is

    t(k) = (k // 8) * lap + sum(taps[:k % 8]) + taps[k % 8] / 2

with ``lap = sum(taps)``. Nothing in it depends on the reference clock, which
only decides how many laps a run sweeps, so one table serves every clock and
there is no code range for it to fall short of. 'dapkel.functions.tdc_calibration'
builds the tables; the packaged ones live in ``dapkel/params/calibration_data``.
"""

from __future__ import annotations

import re
from importlib.resources import files as _resource_files

import numpy as np

__all__ = [
    "TAPS_PER_LAP",
    "board_taps",
    "codes_to_ps",
]

#: Taps on the ring oscillator, i.e. TDC codes per lap.
TAPS_PER_LAP = 8

# A timestamping tag reading one micropixel, 'S0T'...'S3T'. 'ORT' reads all four
# at once and cannot say which fired, so it gets the mean of the four tables.
_SPAD_TAG = re.compile(r"S([0-3])T", re.IGNORECASE)


def board_taps(
    daughterboard_number: str, motherboard_number: str, tag: str
) -> np.ndarray:
    """Load the packaged tap widths a run with this file tag should use.

    Reads ``{daughterboard}_{motherboard}_TDC_COMPACT_SPAD{n}_S{n}.npy`` from
    ``dapkel/params/calibration_data``. A single-micropixel tag (``'S2T'``)
    gets that SPAD's table; anything else (``'ORT'``) the mean of the four - as
    does a single-SPAD tag whose table is missing, with a note.

    Parameters
    ----------
    daughterboard_number : str
        Daughterboard id, e.g. ``'D0'``.
    motherboard_number : str
        Motherboard id, e.g. ``'M0'``.
    tag : str
        The file tag the run was selected with.

    Returns
    -------
    np.ndarray
        The (32, 32, 8) tap widths in picoseconds.

    Raises
    ------
    FileNotFoundError
        Raised when no compact calibration is packaged for the board.
    """
    board = f"{daughterboard_number}_{motherboard_number}"
    data_dir = _resource_files("dapkel.params").joinpath("calibration_data")
    tables = {}
    for entry in sorted(data_dir.iterdir(), key=lambda t: t.name):
        if entry.name.startswith(f"{board}_TDC_COMPACT_") and entry.name.endswith(
            ".npy"
        ):
            with entry.open("rb") as fh:
                tables[entry.name] = np.load(fh)
    if not tables:
        raise FileNotFoundError(
            f"No packaged compact TDC calibration for board {board} (looked for "
            f"{board}_TDC_COMPACT_*.npy in dapkel/params/calibration_data). Build "
            "one with 'tdc_calibration.process_tdc_calibration'."
        )

    spad = _SPAD_TAG.search(tag or "")
    if spad:
        mine = [n for n in tables if f"SPAD{spad.group(1)}_" in n]
        if len(mine) == 1:
            return tables[mine[0]]
        print(
            f"  NOTE: no compact calibration for SPAD{spad.group(1)} on board "
            f"{board}; using the mean of {len(tables)} SPAD table(s)."
        )
    return np.mean(list(tables.values()), axis=0)


def codes_to_ps(codes: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convert one pixel's TDC codes to picoseconds with its 8 tap widths.

    Each code maps to the *centre* of its tap - the bin-centre CDF of the
    density test, one value per code, reproduced from 8 numbers.

    Parameters
    ----------
    codes : np.ndarray
        Integer-valued TDC codes (any shape), all >= 0.
    taps : np.ndarray
        The pixel's ``(8,)`` tap widths in picoseconds.

    Returns
    -------
    np.ndarray
        Times in picoseconds, same shape as ``codes``.
    """
    taps = np.asarray(taps, dtype=np.float64)
    centre = np.cumsum(taps) - taps / 2.0
    codes = np.asarray(codes).astype(np.int64)
    return (codes // TAPS_PER_LAP) * taps.sum() + centre[codes % TAPS_PER_LAP]
