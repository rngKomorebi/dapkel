import numpy as np


def unpack(filepath, nframes, compute_time_series=True):
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

    # --- Step 1: reshape file bytes into DDR3 memory rows (32 bytes each) ---
    # ddr3_mem[row, col] — (64*nframes rows, 32 columns)
    ddr3_mem = raw_bytes.reshape(nframes * 64, 32).astype(np.uint32)

    # --- Step 2: reorder columns to interleave 8 channels of 4 bytes ---
    # MATLAB loop: for each frame kk and channel ii (0-indexed),
    #   ddr_3_mem_resh[(kk*8+ii)*64 : (kk*8+ii+1)*64, :] = ddr3_mem[kk*64:..., ii*4:ii*4+4]
    # This is equivalent to the reshape/transpose below.
    ddr3_interleaved = (
        ddr3_mem.reshape(
            nframes, 64, 8, 4
        )  # split 32 columns into 8 groups of 4
        .transpose(0, 2, 1, 3)  # → (nframes, 8, 64, 4)
        .reshape(-1, 4)  # → (nframes*8*64, 4)
    )

    # --- Step 3: assemble big-endian uint32 words from 4 bytes ---
    # column 0 = MSB, column 3 = LSB  (matches MATLAB's bitshift assembly)
    data_raw = (
        ddr3_interleaved[:, 3]
        + ddr3_interleaved[:, 2] * 256
        + ddr3_interleaved[:, 1] * 65536
        + ddr3_interleaved[:, 0] * 16777216
    )  # shape: (nframes * 512,)

    # --- Step 4: extract bits 28..1 from each 32-bit word (MSB-first, 28 bits used) ---
    # MATLAB: bit_positions = fliplr(1:28) = [28,27,...,1] (1-indexed from LSB)
    # Python 0-indexed right-shift amounts: [27, 26, ..., 0]
    bit_shifts = np.arange(27, -1, -1, dtype=np.int32)  # (28,)

    # Vectorised extraction: group words as (nframes*8 channels, 64 words per channel)
    data_per_channel = data_raw.reshape(nframes * 8, 64)  # (nframes*8, 64)
    bits_raw = ((data_per_channel[:, :, np.newaxis] >> bit_shifts) & 1).astype(
        np.uint8
    )
    # shape: (nframes*8, 64, 28)

    # Flatten the (64 words × 28 bits) dimension → 1792 bits per frame/channel,
    # then rearrange to (nframes*1792, 8) matching the MATLAB bin_all layout.
    # MATLAB column-major reshape(bits_5_32', [1792,1]) == Python row-major ravel of bits_5_32,
    # which is what .reshape(nframes, 8, 64*28) gives after reading in row-major order.
    bin_all = (
        bits_raw.reshape(nframes, 8, 64 * 28)  # (nframes, 8, 1792)
        .transpose(0, 2, 1)  # (nframes, 1792, 8)
        .reshape(nframes * 1792, 8)  # (nframes*1792, 8)
    )

    # --- Step 5: decode 128 pixels × 14 bits per frame/channel ---
    # Layout: 14 bits/pixel = [9-bit count | 1 extra coarse bit | 4-bit fine time]
    #   bits 0..8  → photon count  (9 bits, MSB first)
    #   bits 0..9  → coarse time   (10 bits, MSB first, shares first 9 with count)
    #   bits 10..13 → fine time    (4 bits, MSB first)
    # Bits are already 0/1 in uint8 — no need to blow this up to
    # float64 (8x the memory) just to do a weighted sum below.
    bin_pixels = bin_all.reshape(nframes, 128, 14, 8)
    # shape: (nframes, 128 pixels, 14 bits, 8 channels)

    # int16 (not uint16): 'ts' below can legitimately go negative for
    # empty timestamp slots, so the weighted sums must stay signed.
    w_photon = np.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=np.int16)

    photons = (
        bin_pixels[:, :, 0:9, :]
        * w_photon[np.newaxis, np.newaxis, :, np.newaxis]
    ).sum(axis=2)
    # shape: (nframes, 128, 8)

    # --- Step 6: map (128 pixels, 8 channels) → 32×32 pixel grid ---
    # Channel jj occupies columns jj*4 .. jj*4+3.
    # Within each channel, 128 pixels fill 32 rows × 4 columns (row-major order).
    pixel_rows = np.arange(128) // 4  # (128,)
    pixel_cols_local = np.arange(128) % 4  # (128,)
    channel_offsets = np.arange(8) * 4  # (8,)

    rows_idx = np.broadcast_to(pixel_rows[:, np.newaxis], (128, 8))  # (128, 8)
    cols_idx = (
        pixel_cols_local[:, np.newaxis] + channel_offsets[np.newaxis, :]
    )  # (128, 8)

    photon_counts = np.zeros((32, 32, nframes))
    # photons shape: (nframes, 128, 8) → transpose to (128, 8, nframes) for assignment
    photon_counts[rows_idx, cols_idx, :] = photons.transpose(1, 2, 0)

    if not compute_time_series:
        return None, photon_counts

    w_coarse = np.array(
        [512, 256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=np.int16
    )
    w_fine = np.array([8, 4, 2, 1], dtype=np.int16)

    time_coarse = (
        bin_pixels[:, :, 0:10, :]
        * w_coarse[np.newaxis, np.newaxis, :, np.newaxis]
    ).sum(axis=2)
    time_fine = (
        bin_pixels[:, :, 10:14, :]
        * w_fine[np.newaxis, np.newaxis, :, np.newaxis]
    ).sum(axis=2)
    # shapes: (nframes, 128, 8)

    # Coarse-time correction when fine time < 4
    corrected_coarse = time_coarse.copy()
    corrected_coarse[time_fine < 4] -= 1
    ts = (corrected_coarse - 1) * 8 + (8 - time_fine)

    time_series = np.zeros((32, 32, nframes))
    time_series[rows_idx, cols_idx, :] = ts.transpose(1, 2, 0)

    return time_series, photon_counts
