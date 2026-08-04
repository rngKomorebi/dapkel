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
timestamp — i.e. how often that macropixel fired. Validity is `time_series > 0`:
`unpack` leaves non-fired slots at `<= 0`, so the threshold follows from the
decoding and is not a tunable.

!!! warning "`photon_counts` is not a photon count in ORT"
    In this mode `unpack`'s `photon_counts` is *not* a count — those bits are
    the low bits of the coarse timestamp (see [`unpack`](unpack.md)). Validity
    must come from `time_series`, never from `photon_counts`.

### `mode="count"` — the `S*C` / `ORC` programs

No timestamp is recorded; the data holds per-pixel photon counts, and the
hitmap sums those over every frame. This is the same quantity
[DCR](dcr.md) accumulates before dividing by the acquisition time.

## Photon rate

The raw map is counts (or occupancy). The rate is that divided by the wall-clock
duration of the acquisition, the same way in both modes:

```
hits / (nframes * n_files * cycle)
```

in Hz — photons per second of experiment. Every frame is one acquisition cycle,
so `nframes * n_files * cycle` *is* the elapsed time of the run.

### The cycle depends on the firmware

Two versions, selected with `firmware_version`
([`core.timing.resolve_cycle_time`][dapkel.core.timing.resolve_cycle_time]):

| `firmware_version` | cycle | notes |
|---|---|---|
| `"short_exposure"` (default) | always **9 µs** | the photon-sensitive exposure inside the cycle is 50–500 ns, typically 100 ns |
| `"long_exposure"` | **`exp_time` + 9 µs** | you set X, the firmware adds 9 µs. `exp_time` is required |

`"short_window"` / `"full_window"` are accepted as aliases of the two, since
that is what [`resolve_live_time`][dapkel.core.timing.resolve_live_time] and
[DCR](dcr.md) call them.

!!! warning "`frame_rate_cnt.txt` is not used, on purpose"
    The acquisition exe writes a `frame_rate_cnt.txt` beside the data. Nothing
    in dapkel reads it: it is produced by an exe we do not control, so no
    analysis result is allowed to depend on it. The cycle length is a property
    of the firmware and the requested exposure, and `Kelpie_run.m` states it
    directly: `exp_time = 0 → 9 µs`, `1e-6 → 10 µs`, `2e-6 → 11 µs`.

    It is worth knowing what the counter says, though, because it disagrees
    with the nominal figure. It is a 200 MHz tick count over the whole run,
    latched by the firmware after the acquisition reports done, and across every
    sample folder on disk it fits

    ```
    ticks = nframes * N + 138 + window/5ns
    ```

    exactly — the `138` (690 ns) a fixed per-run overhead, the exposure window
    appearing once per run rather than per frame, and `N` the frame period. `N`
    comes out at **1937–1954 ticks = 9.685–9.770 µs**, so the real cycle is
    685–770 ns per frame longer than the nominal 9 µs, and a rate normalised by
    9 µs runs **~7.8 % high**. `N` does *not* track the window (a 50 → 500 ns
    window moves it by 14 ticks, not 90), which is the direct evidence that the
    window sits inside the cycle rather than extending it. What the extra
    ~700 ns per frame is has not been established.

### What the number means

Under `timestamp` mode only the *first* photon of a cycle is timestamped, so
the map saturates at one firing per cycle — 111 kHz at a 9 µs cycle — and
under-reports as it approaches that. It is a firing rate, not a corrected
photon flux.

`exposure_window` is recorded and reported as a duty cycle but does **not**
enter the rate. Dividing by it instead would give the rate *during* the
exposure, which is this figure times `cycle / exposure` — 90× for 9 µs / 100 ns.
Both are defensible quantities; this one is per second of experiment, which is
what the printed line and the colorbar say.

### `nframes` is yours to supply, not the file's

The rate is proportional to `1 / nframes`, and the frame count **cannot** be
trusted to the file. Every sample `.bin` is a whole power-of-two number of
frames — 32 MiB (16 384) or 16 MiB (8 192) — which is not what a 10 000-frame
acquisition should produce, so the relation between the number you enter and
the file you get is **not established**; see
[`io.frames_in_file`][dapkel.core.io.frames_in_file].

Both this package's `unpack` and the reference `kelpie_data_ddr3.m` read only
the first `nframes` frames, so passing the number you acquired with gives
exactly the frames MATLAB gives. That is why `nframes` is a required argument
everywhere and is stored in the sidecar: it is the acquisition setting, and
the rate is only right if it matches what the camera actually recorded.

## Compute once, plot many times

Analysis and plotting are decoupled. `compute_and_save_hitmap` writes the array
to `processed/` with a `.meta.json` sidecar recording the frame count, the
firmware version and the resolved cycle; `collect_and_plot_hitmap` reads that
back and never re-unpacks the binary data. Retrying a colormap costs a load, not
a full re-read of the raw files. See
[Adding an analysis](../adding_an_analysis.md) for the stage rules.

```python
from dapkel.functions import hitmap_analysis as hm

# short-exposure firmware: 9 µs cycle, 100 ns exposure inside it
hm.compute_and_save_hitmap(path, 10_000, "timestamp", "ORT",
                           exposure_window=100e-9)

# long-exposure firmware: you set X, the firmware adds 9 µs
hm.compute_and_save_hitmap(path, 10_000, "timestamp", "ORT",
                           firmware_version="long_exposure", exp_time=91e-6)

hm.collect_and_plot_hitmap(path, "timestamp", "ORT")   # counts + rate figures
```

Because the sidecar stores the frame count and the cycle, a saved hitmap can
still be turned into a rate months later without remembering the original call
arguments. A hitmap saved before the sidecar carried them gets the counts
figure and a note instead of a rate.

## API

::: dapkel.functions.hitmap_analysis
