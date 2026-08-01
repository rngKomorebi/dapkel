"""Decode Kelpie v2 DDR3 binary ('.bin') data files.

The shared foundation for every analysis in the package: turns the raw bytes
written by ``Kelpie_v2.exe`` into per-frame, per-pixel photon counts and TDC
timestamps on the (32, 32) micropixel grid.

NOTE: ``photon_counts`` is a real count only for the ``S*C``/``ORC`` programs.
Under ``ORT`` those bits are the low bits of the coarse timestamp, so validity
must come from ``time_series``.

Frame format, the MATLAB equivalence and the bit layout: see
``docs/guide/unpack.md``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["unpack"]


def unpack(
    filepath: str,
    nframes: int,
    compute_time_series: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Read a Kelpie v2 binary file and return timestamps and photon counts.

    Parameters
    ----------
    filepath : str
        Path to the .bin file written by Kelpie_v2.exe.
    nframes : int
        Number of frames to decode from the file. Pass 1 for a single-frame
        preview even if more were captured.
    compute_time_series : bool, optional
        Switch for computing the 'time_series' output. Photon-count-only
        consumers (e.g. DCR analysis) can set this to False to skip the
        coarse/fine time decoding, which is roughly two thirds of the
        per-pixel work. The default is True.

    Returns
    -------
    time_series : ndarray, shape (32, 32, nframes), float64, or None
        None if 'compute_time_series' is False.
    photon_counts : ndarray, shape (32, 32, nframes), float64
    """
    n_bytes = 4 * nframes * 64 * 8
    raw_bytes = np.fromfile(filepath, dtype=np.uint8, count=n_bytes)

    # --- Steps 1-3: assemble big-endian 32-bit words, ordered [frame, chan, word] ---
    # MATLAB loop: for each frame kk and channel ii (0-indexed),
    #   ddr_3_mem_resh[(kk*8+ii)*64 : ...] = ddr3_mem[kk*64:..., ii*4:ii*4+4]
    # then assemble each word big-endian (column 0 = MSB, column 3 = LSB).
    ddr3_mem = raw_bytes.reshape(nframes * 64, 32).astype(np.uint32)
    ddr3_interleaved = (
        ddr3_mem.reshape(nframes, 64, 8, 4)  # split 32 cols into 8 groups of 4
        .transpose(0, 2, 1, 3)  # → (nframes, 8, 64, 4)
        .reshape(-1, 4)  # → (nframes*8*64, 4)
    )
    data_raw = (
        ddr3_interleaved[:, 3]
        + ddr3_interleaved[:, 2] * 256
        + ddr3_interleaved[:, 1] * 65536
        + ddr3_interleaved[:, 0] * 16777216
    )
    words = data_raw.reshape(nframes, 8, 64)  # (frame, channel, word)

    # --- Step 4: extract the packed pixel fields directly with shifts+masks ---
    # Only bits 27..0 of each word are used (MATLAB "bits 5:32"), and each word
    # packs *two* 14-bit pixels, MSB-first:
    #   14 bits/pixel = [9-bit count | 1 extra coarse bit | 4-bit fine time]
    #     count  = 9 bits (shares its bits with the coarse field)
    #     coarse = 10 bits (count's 9 bits + one more)
    #     fine   = 4 bits
    #   pixel A (high 14 bits): count=bits27:19  coarse=bits27:18  fine=bits17:14
    #   pixel B (low  14 bits): count=bits13:5   coarse=bits13:4   fine=bits3:0
    # This replaces the old "explode every bit to a 235 MB uint8 array and do
    # weighted sums" path (~6x faster) and is bit-for-bit identical to it.
    # int16 (signed): 'ts' below can legitimately go negative for empty slots.
    def _fields(
        shift_count: int, shift_coarse: int, shift_fine: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cnt = ((words >> shift_count) & 0x1FF).astype(np.int16)
        coarse = ((words >> shift_coarse) & 0x3FF).astype(np.int16)
        fine = ((words >> shift_fine) & 0xF).astype(np.int16)
        return cnt, coarse, fine

    cnt_a, coarse_a, fine_a = _fields(19, 18, 14)  # high 14 bits → pixel 2w
    cnt_b, coarse_b, fine_b = _fields(5, 4, 0)  # low 14 bits  → pixel 2w+1

    def _interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # (frame, chan, 64) x2 → (frame, chan, 128), pixel order [2w, 2w+1]
        return np.stack([a, b], axis=-1).reshape(nframes, 8, 128)

    # --- Steps 5-6: map (128 pixels, 8 channels) → 32×32 pixel grid ---
    # Channel jj occupies columns jj*4 .. jj*4+3; within a channel the 128
    # pixels fill 32 rows × 4 columns (row-major). Scatter with (pixel, chan)
    # index arrays; the pixel/channel axes move last so the frame axis stays.
    pixel_rows = np.arange(128) // 4
    pixel_cols_local = np.arange(128) % 4
    channel_offsets = np.arange(8) * 4
    rows_idx = np.broadcast_to(pixel_rows[:, np.newaxis], (128, 8))  # (128, 8)
    cols_idx = (
        pixel_cols_local[:, np.newaxis] + channel_offsets[np.newaxis, :]
    )  # (128, 8)

    photon_counts = np.zeros((32, 32, nframes))
    # (frame, chan, pixel) → (pixel, chan, frame) for the scatter assignment.
    photon_counts[rows_idx, cols_idx, :] = _interleave(cnt_a, cnt_b).transpose(
        2, 1, 0
    )

    if not compute_time_series:
        return None, photon_counts

    coarse = _interleave(coarse_a, coarse_b)
    fine = _interleave(fine_a, fine_b)
    coarse = coarse - (fine < 4)  # coarse correction when fine time < 4
    ts = (coarse - 1) * 8 + (8 - fine)  # (frame, chan, pixel)

    time_series = np.zeros((32, 32, nframes))
    time_series[rows_idx, cols_idx, :] = ts.transpose(2, 1, 0)

    return time_series, photon_counts
