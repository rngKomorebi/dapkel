# Dark count rate (DCR)

The DCR map is the per-pixel dark-count rate in counts per second: accumulated
photon counts divided by the time the sensor was actually collecting.

```
DCR = sum(counts) / (nframes * n_files * live_per_frame)
```

## Live time is not frame time

The normalisation uses the **live** time per frame, which depends on the
firmware — see [`core.timing.resolve_live_time`][dapkel.core.timing.resolve_live_time].

- **`short_window`** (current firmware, the default) — each ~9 µs frame is mostly
  readout and only the user-set `acq_window` (e.g. `200e-9` s) is
  photon-sensitive. You must pass `acq_window`; `compute_dcr_32` raises without
  it rather than guessing.
- **`full_window`** (legacy) — the whole frame is live, so the per-frame time
  comes from `resolve_frame_time`: explicit `exp_time` + 9 µs overhead, else
  `frame_rate_cnt.txt`, else the 9 µs free-running fallback.

Under the legacy firmware the exposure argument passed to `Kelpie_v2.exe` maps
to an actual frame period with 9 µs of overhead: `0 → 9 µs`, `10 µs → 19 µs`.

## Quadrants and the full sensor

`compute_dcr_32` handles one SPAD quadrant tag (`S0C`, `S1C`, `S2C`, `S3C`).
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
