# Dark count rate (DCR)

The DCR map is the per-pixel dark-count rate in counts per second: accumulated
dark counts divided by the time the sensor was actually collecting.

```
DCR = sum(counts) / (nframes * n_files * live_per_frame)
```

## Counting mode vs timestamping mode

What "a count" *is* depends on the acquisition program, so `mode` is not
cosmetic — it decides which field of the unpacked data is read.

- **`mode="count"`** (default) — `S*C` / `ORC` data. Each pixel has a real
  9-bit photon counter per frame, and the map is its sum.
- **`mode="timestamp"`** — `S*T` / `ORT` data. There **is no counter**: the
  chip timestamps the first hit instead, and [`unpack`](unpack.md) returns the
  low bits of the coarse timestamp in the `photon_counts` slot. The count is
  therefore the number of frames carrying a valid timestamp,
  `(ts > 0).sum(axis=2)` — the same definition `hitmap_analysis` uses.

!!! danger "Summing the counter field on `*T` data inflates the DCR ~40-50×"
    Those bits are `coarse >> 1`, so a fired pixel contributes a few hundred
    instead of 1, and the result still looks like a plausible DCR map. On real
    24 V `S0T` data the median came out at 247 kcps — nearly twice the 125 kcps
    that one-hit-per-8-µs-frame physically allows. `mode` is cross-checked
    against the last letter of `tag`, so `tag="S0T"` with `mode="count"` is a
    `ValueError`, not a wrong number.

### Saturation and `pileup_correct`

Timestamping records at most **one** count per pixel per frame, so the raw rate
is really an occupancy and it saturates at `1 / open_shutter_time_s` — 125 kcps
at an 8 µs shutter. It under-reports well before the ceiling: at 50 % occupancy
it reads ~30 % low.

`pileup_correct=True` inverts the censoring under Poisson statistics, since a
frame fires with probability `p = 1 - exp(-r·w)`:

```
r = -ln(1 - hits/total_frames) / open_shutter_time_s
```

It is off by default — the raw occupancy rate is what the sensor measured — and
the drivers print a warning naming the ceiling whenever the hottest pixel gets
within 5 % of it. Counting mode has no such ceiling and ignores the flag.

## Live time is not frame time

The normalisation uses the **live** time per frame, which depends on the
firmware — see [`core.timing.resolve_live_time`][dapkel.core.timing.resolve_live_time].

- **`short_window`** (current firmware, the default) — each ~9 µs frame is mostly
  readout and only the user-set `open_shutter_time_s` (e.g. `200e-9` s) is
  photon-sensitive. You must pass `open_shutter_time_s`; `compute_dcr_32` raises
  without it rather than guessing.
- **`full_window`** (legacy) — the whole frame is live, so the per-frame time
  comes from `resolve_frame_time`: explicit `exp_time_s` + 9 µs overhead, else
  `frame_rate_cnt.txt`, else the 9 µs free-running fallback.

Under the legacy firmware the exposure argument passed to `Kelpie_v2.exe` maps
to an actual frame period with 9 µs of overhead: `0 → 9 µs`, `10 µs → 19 µs`.

## Quadrants and the full sensor

`compute_dcr_32` handles one SPAD quadrant tag — `S0C`...`S3C` in counting
mode, `S0T`...`S3T` in timestamping mode.
`compute_dcr_64` unpacks all four and interleaves them onto the full (64, 64)
sensor — each tag reads one micropixel per 2×2 macropixel, so the quadrant maps
interleave rather than tile.

The SPAD indices run **clockwise** (`S0 S1 / S3 S2`), not row-major; the layout
is documented under [cross-talk](crosstalk.md#spad-geometry-the-indices-run-clockwise)
and enforced by `tests/test_spad_geometry.py`.

!!! note "Quadrant tags are matched with a leading underscore"
    DCR globs `*_S0C*.bin`, not `*S0C*.bin`, so a tag cannot match a filename
    that merely happens to contain those characters. This is the
    `require_separator=True` argument to
    [`core.io.find_bin_files`][dapkel.core.io.find_bin_files].

## Pixel masking

Hot/warm pixel masking is **not implemented yet**. When per-board mask files
become available, masked pixels will read as `NaN` — which is why every
statistic in [`core.plots`][dapkel.core.plots] is already NaN-aware. Plain
`np.median` / `np.percentile` propagate NaN, and a single masked pixel would
otherwise turn `vmax` into `NaN` and flatten the entire heatmap to one colour
with no error raised.

## API

::: dapkel.functions.dcr_analysis
