# Decoding the raw `.bin` files

Every analysis in this package — DCR, cross-talk, hitmap, coincidence, TDC
calibration — is built on this one decoder. It turns the raw bytes written by
`Kelpie_v2.exe` into per-frame, per-pixel photon counts and TDC timestamps on
the (32, 32) micropixel grid.

## Frame format

Each frame is `4 * 64 * 8 = 2048` bytes. A pixel occupies 14 bits, packed as a
9-bit photon count (sharing its bits with a 10-bit coarse time) plus a 4-bit
fine time:

```
14 bits/pixel = [ 9-bit count | 1 extra coarse bit | 4-bit fine time ]
                  count  = 9 bits  (shares its bits with the coarse field)
                  coarse = 10 bits (count's 9 bits + one more)
                  fine   = 4 bits
```

Each 32-bit word packs *two* pixels, MSB-first, using only bits 27..0.

The fields are pulled straight out of the assembled 32-bit words with shifts and
masks. This reproduces the reference MATLAB decoder (`kelpie_data_ddr3.m`)
bit-for-bit while avoiding a large intermediate bit-array, which is what makes
it fast.

## `photon_counts` is mode-dependent

The `photon_counts` array means different things depending on which program
wrote the data, and this catches people out:

- **`S*C` / `ORC` programs** — a genuine per-pixel photon count.
- **`ORT` program** — **not a count.** Those bits are the low bits of the coarse
  timestamp. Validity must come from `time_series`, never from `photon_counts`.
  See [hitmaps](hitmap.md).

## Skipping the timestamp decode

`compute_time_series=False` skips the coarse/fine time decoding, which is
roughly two thirds of the per-pixel work. Count-only consumers (DCR, cross-talk)
pass it. Inside the package this is handled for you by
[`core.reduce.accumulate_frames`][dapkel.core.reduce.accumulate_frames] via its
`need_time_series` argument.

## API

::: dapkel.functions.unpack
