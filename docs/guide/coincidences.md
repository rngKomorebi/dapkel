# Coincidences and timing jitter (delta-t)

A daplis-style pipeline for coincidence / timing-jitter analysis of the Kelpie
`ORT` (timestamp) program, built on three layers:

1. [`unpack`](unpack.md) decodes each `.bin` file into the per-frame, per-pixel
   TDC code (`time_series`, the MATLAB `time_mat`).
2. [`calc_diff.calculate_differences`][dapkel.functions.calc_diff.calculate_differences]
   turns structured per-pixel timestamps into per-window differences for pixel
   pairs.
3. `delta_t` drives those two over a folder of files, pools the differences and
   saves them to a `.feather` table, then histograms, fits and plots the
   coincidence peak.

## Timing model

Each acquisition window is ~9 µs, but the ring-oscillator TDC is a free-running
~10 MHz clock. The recorded `time_series` value is therefore the oscillator
*code* within one ~100 ns period (codes up to ~1300), and the timestamp in
picoseconds is

```
t_ps = code * time_unit_ps        with  time_unit_ps ≈ 100 ns / 1300 ≈ 77 ps
```

on average. The exact per-code bin widths come from a density-test LUT — pass it
as `time_lut` once you have one; see [TDC calibration](tdc_calibration.md).

There is one timestamp per photon, per window, per macropixel.

## Fitting the peak: the background is a triangle

This is the part that most affects the number you report.

The accidental-coincidence background under an ORT delta-t peak is
**triangular**, not flat, because the free-running oscillator has no cycle
counter. Fitting a Gaussian on a *flat* background over a wide window therefore
inflates the fitted sigma — you measure the background's slope as if it were
jitter.

Two models are provided:

| model | fit | when to use |
|---|---|---|
| `gaussian` / `fit_gaussian_peak` | Gaussian on flat background | valid only on a **narrow** window around the peak |
| `gaussian_plus_triangle` / `fit_gaussian_on_triangle` | Gaussian on triangular background | the correct model for ORT delta-t over a wide window |

The full derivation, with the demonstration that the triangle is intrinsic
rather than an artefact, is in
[the triangular background note](../ort_triangle_background.md).

## Before you analyse: check the data

Run [`data_quality.plot_time_code_histogram`][dapkel.functions.data_quality.plot_time_code_histogram]
on a pixel or two first. When the TDC never stops, `unpack` still returns a full
array of plausible integers and this whole pipeline still produces a
peak-shaped histogram — it is simply meaningless. See
[Data quality](data_quality.md).

## Large runs: the differences are streamed, not accumulated

`calculate_and_save_timestamp_differences` writes as it goes. Whenever the
buffered differences would exceed `part_size_mb` (default 64 MB) they are
flushed to a part file, and the parts are concatenated into the final feather
one Arrow record batch at a time:

```
processed/
  <dataset>_delta_t_parts/
    <dataset>_delta_t_0.feather      <- flushed during the run
    <dataset>_delta_t_1.feather
    ...
  <dataset>_delta_t.feather          <- the combined result
```

This matters at scale. Holding a whole 10 000-file run in memory and writing it
once fails twice over: the buffers exhaust RAM, and the single write then asks
pyarrow for a multi-GB contiguous allocation on top of what is already held.
Streaming makes the loop's memory ceiling *one part plus one unpacked file*,
whatever the file count.

Two consequences worth knowing:

- **A crashed run is recoverable.** Every part flushed before the crash is a
  complete, readable feather. Assemble them with
  [`combine_delta_t_parts`][dapkel.functions.delta_t.combine_delta_t_parts],
  or plot a single part directly via `feather_path`.
- **The parts are kept by default** (`keep_parts=True`) — they are the
  insurance, and they cost disk rather than memory. Pass `keep_parts=False`
  to have the run remove its own parts once the combined feather is written.

Why 64 MB and not 10: `unpack` already holds a `(32, 32, nframes)` float64
array — about 82 MB at 10 000 frames — so below that the part buffer is not the
binding constraint, and a smaller cap only multiplies the number of part files.
A smaller value is perfectly safe if you want the memory ceiling lower.

## Overwriting: the five-second countdown

Every `rewrite=True` in this module routes through
[`store.confirm_rewrite`][dapkel.core.store.confirm_rewrite]. On a fresh run it
does nothing at all. When something really is about to be destroyed it names
each target with its size and timestamp, then counts down five seconds before
proceeding:

```
!!  rewrite=True - the following data WILL BE OVERWRITTEN:
!!    .../processed/SPDC_delta_t.feather  (1.4 GB, written 2026-07-30 02:11)
!!    .../processed/SPDC_delta_t_parts/  (23 file(s), 1.4 GB)
!!  Press Ctrl-C within 5 s to abort and copy it somewhere safe.
```

The pause is a plain `time.sleep`, so Ctrl-C raises straight out of the
analysis with the data on disk untouched. This is aimed at the common accident:
a `rewrite=True` left in the script from the previous acquisition, silently
eating an overnight run. Set `store.REWRITE_DELAY_S = 0` in an unattended
pipeline that has its own guard.

Note the part folder is named in the warning as well — a rewrite clears the
previous run's parts, otherwise they would be pooled into the new output.

## Pooling runs

`combine_delta_t_feathers` pools the per-run delta-t feathers under a parent
folder (many acquisitions of the same measurement) into one combined `.feather`
in a `combined` sub-folder of `processed/`. It never folds a previous combined
output back into itself, and it streams batch-by-batch just as the part
combiner does, so pooling a hundred runs costs one batch of memory rather than
the whole pooled data set.

Part files are not picked up by the search: they end in `_<i>.feather`, which
the `*_delta_t.feather` glob does not match, so nothing is ever counted twice.

## API

::: dapkel.functions.delta_t

::: dapkel.functions.calc_diff
