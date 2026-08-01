# Hitmaps and photon rate

A *hitmap* is the per-pixel occupancy of the sensor — how often each macropixel
registered a photon over the acquisition. How that is measured depends on which
Kelpie program wrote the data, so there are two modes, matched to what
[`unpack`][dapkel.functions.unpack.unpack] actually returns.

## The two modes

### `mode="timestamp"` — the ORT program

Every frame, each macropixel reports the timestamp of the *first* photon it saw
(from one of its four micropixels); the later photons are not read out. This is
a single first-arrival time per macropixel. It is **not** an array-wide "who
fired first" ranking, and one macropixel's timestamp is not comparable to
another's.

The hitmap therefore counts, per pixel, the number of frames carrying a valid
timestamp (`time_series > valid_min`) — i.e. how often that macropixel fired.

!!! warning "`photon_counts` is not a photon count in ORT"
    In this mode `unpack`'s `photon_counts` is *not* a count — those bits are
    the low bits of the coarse timestamp (see [`unpack`](unpack.md)). Validity
    must come from `time_series`, never from `photon_counts`.

### `mode="count"` — the `S*C` / `ORC` programs

No timestamp is recorded; the data holds per-pixel photon counts, and the
hitmap sums those over every frame. This is the same quantity
[DCR](dcr.md) accumulates before dividing by the acquisition time.

## Photon rate

The raw map is counts (or occupancy). A *rate* is that divided by time — but
which time depends on the mode, and getting this wrong is the easiest way to
publish a number that is off by 45x.

### `count` mode — divide by the live window

Photons accumulate within the photon-sensitive acquisition window (e.g. 200 ns),
so the rate is

```
counts / (total_frames * acq_window)
```

in counts per second (cps). Under the current firmware each ~9 µs frame is
mostly readout and only `acq_window` is live, so the live time is
`total_frames * acq_window` — **not** the 9 µs period. Pass `acq_window` so the
rate can be formed at all.

### `timestamp` mode — divide by the frame period

Each macropixel reports at most ONE first-photon time per frame, so this is an
occupancy, and its rate ceiling is one firing per frame period:

```
counts / (total_frames * frame_period)
```

giving a firing rate in Hz that cannot exceed `1 / frame_period` (~9 µs →
~111 kHz).

Dividing occupancy by the (short) acquisition window instead would overstate the
rate by `frame_period / acq_window` — a factor of 45 for 9 µs / 200 ns — and
would exceed what a single-timestamp-per-frame readout can physically measure.

The frame period comes from `frame_rate_cnt.txt`, falling back to the 9 µs
free-running value. See [`core.timing.resolve_frame_time`][dapkel.core.timing.resolve_frame_time].

## Compute once, plot many times

Analysis and plotting are decoupled. `compute_and_save_hitmap` writes the array
to `processed/` with a `.meta.json` sidecar recording the frame count and
acquisition window; `collect_and_plot_hitmap` reads that back and never
re-unpacks the binary data. Retrying a colormap costs a load, not a full re-read
of the raw files. See [Adding an analysis](../adding_an_analysis.md) for the
stage rules.

Because the sidecar stores the timing, a saved hitmap can still be turned into a
rate months later without remembering the original call arguments.

## API

::: dapkel.functions.hitmap_analysis
