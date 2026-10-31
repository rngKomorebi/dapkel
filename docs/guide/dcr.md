# Dark count rate (DCR)

The DCR map is the per-pixel dark-count rate in counts per second: accumulated
dark counts divided by the time the sensor was actually collecting.

```
DCR = sum(counts) / (nframes * n_files * live_per_frame)
```

Two calls, like every analysis:
[`process_dcr`][dapkel.functions.dcr_analysis.process_dcr] unpacks the dark
`.bin` files and saves the map in cps to `processed/<name>_<tag>_dcr.npy`;
[`plot_dcr`][dapkel.functions.dcr_analysis.plot_dcr] loads it, saves
`<name>_<tag>_dcr_heatmap.png` and `<name>_<tag>_dcr_distribution.png` to
`results/dcr/`, and returns the map.

```python
from dapkel.functions import dcr_analysis

# one quadrant, counting mode, 100 ns open shutter
dcr_analysis.process_dcr(path, 1000, "S0C", open_shutter_time_s=100e-9)
dcr = dcr_analysis.plot_dcr(path, "S0C")

# the full (64, 64) sensor, timestamping mode, Poisson-corrected
dcr_analysis.process_dcr(
    path, 1000, "S*T", open_shutter_time_s=8e-6, pileup_correct=True
)
dcr_analysis.plot_dcr(path, "S*T")
```

## Counting mode vs timestamping mode

What "a count" *is* depends on the acquisition program, and it decides which
field of the unpacked data is read. There is no `mode` argument: the mode comes
from the **last letter of the tag**
([`core.reduce.readout_mode`][dapkel.core.reduce.readout_mode]), so `tag` is
required.

- **Count mode** — `...C` tags (`S0C`, `ORC`). Each pixel has a real
  9-bit photon counter per frame, and the map is its sum.
- **Timestamp mode** — `...T` tags (`S0T`, `ORT`). There **is no counter**: the
  chip timestamps the first hit instead, and [`unpack`](unpack.md) returns the
  low bits of the coarse timestamp in the `photon_counts` slot. The count is
  therefore the number of frames carrying a valid timestamp,
  `(ts > 0).sum(axis=2)` — the same definition `hitmap_analysis` uses.

!!! danger "Summing the counter field on `*T` data inflates the DCR ~40-50×"
    Those bits are `coarse >> 1`, so a fired pixel contributes a few hundred
    instead of 1, and the result still looks like a plausible DCR map. On real
    24 V `S0T` data the median came out at 247 kcps — nearly twice the 125 kcps
    that one-hit-per-8-µs-frame physically allows. Because the mode is read off
    the tag rather than passed beside it, `S0T` data can no longer be summed as
    counts by mistake.

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
`process_dcr` prints a warning naming the ceiling whenever the hottest pixel
gets past 5 % of it. Counting mode has no such ceiling and ignores the flag.

## Live time is not frame time

The normalisation uses the **live** time per frame, which depends on the
firmware — see [`core.timing.resolve_live_time`][dapkel.core.timing.resolve_live_time].

- **`short_exposure`** (current firmware, the default) — each ~9 µs frame is
  mostly readout and only the user-set `open_shutter_time_s` is
  photon-sensitive. That window sits inside the cycle without extending it and is
  settable from ~50 ns up to the full 9 µs, so a wrong guess is a rate error of up
  to two decades. Pass the `open_shutter_time_s` the run used (the default is
  `500e-9`); `resolve_live_time` raises on `None` rather than guessing. Nothing in the data reveals it, which is why
  it is one of the few timescales still asked for — and note it is **not** the
  TDC full scale, which [the coincidence pipeline](coincidences.md) measures for
  itself.
- **`long_exposure`** (legacy) — the whole frame is live, so the per-frame time
  is `exp_time_s` + 9 µs overhead, or the 9 µs free-running cycle when
  `exp_time_s` is None.

Under the legacy firmware the exposure argument passed to `Kelpie_v2.exe` maps
to an actual frame period with 9 µs of overhead: `0 → 9 µs`, `10 µs → 19 µs`.

## Quadrants and the full sensor

A single SPAD quadrant tag — `S0C`...`S3C` in counting mode, `S0T`...`S3T` in
timestamping mode — gives a (32, 32) map. `tag="S*C"` or `"S*T"` unpacks all
four and interleaves them onto the full (64, 64) sensor — each tag reads one
micropixel per 2×2 macropixel, so the quadrant maps interleave rather than tile.
The quadrants are cut to the same number of files so every pixel shares one
exposure, and a missing quadrant stays `NaN`. The saved file writes the `*` as
`x` (`<name>_sxc_dcr.npy`).

The SPAD indices run **clockwise** (`S0 S1 / S3 S2`), not row-major; the layout
is documented under [cross-talk](crosstalk.md#spad-geometry-the-indices-run-clockwise)
and enforced by `tests/test_spad_geometry.py`.

!!! note "Quadrant tags are matched with a leading underscore"
    The full-sensor tags glob `*_S0C*.bin`, not `*S0C*.bin`, so a tag cannot match a filename
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
