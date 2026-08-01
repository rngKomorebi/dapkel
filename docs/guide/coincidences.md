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

## Pooling runs

`combine_delta_t_feathers` pools the per-run delta-t feathers under a parent
folder (many acquisitions of the same measurement) into one combined `.feather`
in a `combined` sub-folder of `processed/`. It never folds a previous combined
output back into itself.

## API

::: dapkel.functions.delta_t

::: dapkel.functions.calc_diff
