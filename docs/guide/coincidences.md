# Coincidences and timing jitter (delta-t)

A daplis-style pipeline for coincidence / timing-jitter analysis of the Kelpie
`ORT` (timestamp) program, in two calls like every analysis:

1. [`unpack`](unpack.md) decodes each `.bin` file into the per-frame, per-pixel
   TDC code (`time_series`, the MATLAB `time_mat`).
2. [`process_delta_t`][dapkel.functions.delta_t.process_delta_t] converts the
   codes of the chosen pixels to picoseconds, differences every pixel pair frame
   by frame, and bins the differences onto a fixed native grid as it finds them,
   saving only the counts to `processed/`.
3. [`plot_delta_t`][dapkel.functions.delta_t.plot_delta_t] rebins that grid in
   memory, fits the coincidence peak and saves the figure to
   `results/coincidences/`.

```python
from dapkel.functions import delta_t as dt

signal = [(6, 10), (6, 11), (7, 10)]     # blobs read off the hitmap
idler = [(21, 22), (21, 23), (22, 22)]

dt.process_delta_t(path, 10_000, [signal, idler],
                   daughterboard_number="D0", motherboard_number="M0")
counts, centers, fit = dt.plot_delta_t(path)
```

`pixels` is two groups `[group_a, group_b]`; a bare `(row, col)` is a one-pixel
group. With `mode="all_pairs"` (the default) every A × B combination is a pair —
use it when it is not known which pixel of a blob catches the photon; `"1v1"`
zips the groups element-wise. Each pair is one row of the saved grid, labelled
`"ra,ca-rb,cb"`.

## Timing model

Each acquisition window is ~9 µs, but the ring-oscillator TDC measures only
within one period of the **reference clock**. The recorded `time_series` value is
the oscillator *code* on that range, and the timestamp is

```
t = code * CODE_S             with  CODE_S ≈ 71.6 ps  (core.timing.CODE_S)
```

on average. The ring sets that code width, so it does **not** change with the
clock — only the number of codes does, as 1/f:

| clock | full scale | codes |
|---|---|---|
| 50 MHz | 20 ns | ~279 |
| 10 MHz | 100 ns | ~1397 |
| 5 MHz | 200 ns | ~2794 |
| 2 MHz | 500 ns | ~6985 |

That full scale bounds a difference, so it is also the accidental triangle's
half-base — see [the triangle background](../ort_triangle_background.md).

**No entry point asks you for it.** The largest code a run produced *is* the
range its TDC swept, so it is measured from the data
([`core.timing.full_scale_from_max_code`][dapkel.core.timing.full_scale_from_max_code]).
That replaced three parameters that used to ask for the same number under
different names — `open_shutter_time_s`, `full_scale_s` and `support_s` — one of
which was not even the right quantity: the open shutter time is how long the
SPAD waits for a photon, and an 8 µs shutter on a 10 MHz clock still yields
~100 ns timestamps, because the free-running oscillator has no cycle counter. It
belongs to rate normalisation ([DCR](dcr.md), [hitmaps](hitmap.md)) and nothing
else.

The actual bin widths come from the board's 8-tap calibration — see
[TDC calibration](tdc_calibration.md). With `apply_TDC_calibration=True` (the
default, which needs `daughterboard_number` and `motherboard_number`) every code
is converted with
[`core.calibration.codes_to_ps`][dapkel.core.calibration.codes_to_ps] before
differencing. The taps belong to the ring, not the clock, so the same table
serves a run at any clock and there is nothing to select: an `S0T`...`S3T` tag
gets that SPAD's table, `ORT` the mean of the four. `apply_TDC_calibration=False`
uses the nominal `CODE_S`.

### Units

Every time parameter in the package is in **seconds**, with an `_s` suffix.
There are three of them here, and they are all display or fit choices:
`bin_width_s`, `plot_window_s`, `fit_window_s`. Histogram bin centres, edges and
plot axes are in picoseconds — that is the scale the TDC lives on — and the
conversion happens on entry.

Differences are binned in **picoseconds** whether or not a calibration was
applied; uncalibrated codes are converted with the nominal `CODE_S` on the way
in. There is no raw-code artifact, because there would be nothing to do with
one: a calibration maps a *code* to a time, and a stored difference has lost
both of its codes, so it could never be calibrated after the fact.

There is one timestamp per photon, per window, per macropixel.

## Fitting the peak: the background is a triangle

This is the part that most affects the number you report.

The accidental-coincidence background under an ORT delta-t peak is
**triangular**, not flat, because the free-running oscillator has no cycle
counter. Fitting a Gaussian on a *flat* background over a wide window therefore
inflates the fitted sigma — you measure the background's slope as if it were
jitter.

**`plot_delta_t` picks the model for you, and takes no `background`
argument.** The pedestal is either still in the histogram, and then it is the
triangle, or `subtract_background=True` has measured it away and the residual is
flat about zero — one decision, not two:

| histogram | model | fitted over |
|---|---|---|
| raw (default) | Gaussian on a triangular background, the measured full scale seeding the half-base | the whole plotted window |
| lag-subtracted residual | Gaussian on a constant | `fit_window_s` (default 1 ns) around the peak |

A fit that fails — including one whose sigma the covariance cannot tell from
zero, which a wrong background model produces — comes back with `fit["ok"]`
False and a `RuntimeWarning`, so a plot without a curve always says why.

`plot_delta_t` returns `(counts, centers, fit)`: the fitted histogram, the bin
centres in picoseconds and a dict of `amp`, `mu`, `sigma`, `fwhm`,
`jitter_per_detector` (sigma / √2), each with its `_err`, the background level
`bkg`, `car` (peak-to-background, raw histogram only), `n` and `ok` — enough to
re-fit by hand with any model without touching disk. The fit models themselves
are private, and the two-Gaussian (narrow core on broad tails) model is retired
to the repository's `archive/` folder.

The full derivation, with the demonstration that the triangle is intrinsic
rather than an artefact, is in
[the triangular background note](../ort_triangle_background.md).

## Before you analyse: check the data

Run [`data_quality.process_time_codes`][dapkel.functions.data_quality.process_time_codes]
and [`plot_time_codes`][dapkel.functions.data_quality.plot_time_codes] on a
pixel or two first. When the TDC never stops, `unpack` still returns a full
array of plausible integers and this whole pipeline still produces a
peak-shaped histogram — it is simply meaningless. See
[Data quality](data_quality.md).

## Stage 1: histogram straight off the `.bin`

`process_delta_t` bins the differences onto a fixed native grid as it finds
them and keeps only the counts:

```
processed/
  <dataset>_<tag>_delta_counts.npy       <- (n_pairs, n_cells) int64, tag lower-cased
  <dataset>_<tag>_delta_counts.meta.json <- grid, pair labels, totals, progress
```

The size is set by the **grid**, not by the data. A 256-pair run on the default
grid is 106 MB whether it came from 100 files or 10 000 — where keeping every
difference took 3.22 GB for the same run, and 135 GB for a larger one. That is
what makes every pair plottable at once. The tag is in the name, so several
programs (`ORT`, `S0T`, ...) recorded into one folder keep their own artifact
and their own figures.

### Choosing the grid

One argument, `bin_width_s` (cell width):

| data | default cell | why |
|---|---|---|
| uncalibrated (`apply_TDC_calibration=False`) | one code, ~71.6 ps | the codes are **integers**, so counts-per-code is a re-encoding, not a summary — exactly lossless |
| calibrated | 1/10 code, ~7.2 ps | the per-pixel calibration smears the code lattice; 7.2 ps is ~5× finer than the jitter |

The half-range is not a parameter. It is the run's own TDC full scale — the
largest code in the first file — with a 20 % margin for the extreme-value bias
of reading that off one file. A difference cannot exceed the full scale, so that
covers the whole physical support at any clock without being told the clock; a
fixed 200 ns was 10× too wide at 50 MHz and 2.5× too narrow at 2 MHz. Anything
landing outside is reported as `n_outside` rather than dropped silently, so an
under-sized grid announces itself instead of quietly clipping the triangle.

### Re-binning is free, and exact

`plot_window_s` becomes a slice and `bin_width_s` a reduce, both in memory —
nothing is read but the one saved array:

```python
dt.plot_delta_t(
    path,
    pairs=["6,10-21,22"],
    bin_width_s=71e-12,
    plot_window_s=10e-9,        # omit for the whole measured support
)
```

The binning is always **snapped** onto the grid, and the snap is not
cosmetic:

- the width goes to a whole number of cells, or bins hold 9 then 10 then 9
  cells and a smooth distribution picks up an ~11% sawtooth;
- that number is made **odd**. Bins are centred on zero, so their edges sit at
  half-integer multiples of the width — odd puts those edges on cell
  *boundaries*, even puts them through cell *centres*, splitting a cell at
  every boundary. Measured on a real 3.22 GB run: **exactly zero** difference
  from histogramming every stored difference at 9, 11 and 13 cells per bin,
  ~1% at 8, 10 and 12;
- the window goes to a whole number of bins, so **zero is a bin centre whatever
  the window is**.

So this is not an approximation of a histogram of the raw differences — it is
the same histogram, in every bin.

The figure is `results/coincidences/<dataset>_<tag>_<label>_delta_counts.png`,
`label` (default `"spdc"`) also heading the title.

### What you give up

No per-event data survives, so there is nothing that needs individual
differences — time ordering, splitting by acquisition, higher-order
correlations. It also does not make stage 1 meaningfully faster: unpacking
dominates either way. The win is artifact size and replot latency. The retired
per-event (feather) path is in the repository's `archive/` folder.

### Crash insurance, and resuming

The grid is rewritten every 50 files (the module constant `_FLUSH_EVERY`, not
a parameter — checkpoint cadence cannot change the numbers), so a run that dies
leaves a valid artifact covering everything up to its last checkpoint, and the
next call resumes from it (`resume=True`, the default) when it was computed from
the same inputs — files, pairs, mode, grid, calibration and lags. There is only
ever one file, so two runs cannot interleave. `plot_delta_t` on an incomplete
artifact plots what is there and says how many files it covers.

The sidecar records `files_done` and `total_in_grid`. The array is moved into
place before the sidecar, so a crash between the two moves could pair a newer
array with an older sidecar; resuming on that would re-count files and inflate
the total. The resume path checks the array against `total_in_grid` and starts
over rather than continue from something inconsistent.

A *complete* artifact, or one computed from different inputs, is never touched
without `rewrite=True` — `process_delta_t` raises `FileExistsError` instead.

### Measuring the background instead of fitting it

`subtract_background=True` histograms the *frame-shifted* pixel pairs alongside
the real ones, one grid plane per lag, so the accidental triangle can be
subtracted bin-for-bin rather than fitted — see
[Removing the accidental background by shifting frames](background_subtraction.md).

```python
# the lags cost one small grid plane each; the .bin decoding is paid once
dt.process_delta_t(path, 10_000, [signal, idler],
                   daughterboard_number="D0", motherboard_number="M0",
                   subtract_background=True)
counts, centers, fit = dt.plot_delta_t(
    path, subtract_background=True, fit_window_s=2e-9)
```

## Overwriting: the five-second countdown

`rewrite=True` routes through
[`store.confirm_rewrite`][dapkel.core.store.confirm_rewrite]. On a fresh run it
does nothing at all. When something really is about to be destroyed it names
each target with its size and timestamp, then counts down five seconds before
proceeding:

```
!!  rewrite=True - the following data WILL BE OVERWRITTEN:
!!    .../processed/SPDC_ort_delta_counts.npy  (106.0 MB, written 2026-07-30 02:11)
!!  Press Ctrl-C within 5 s to abort and copy it somewhere safe.
```

The pause is a plain `time.sleep`, so Ctrl-C raises straight out of the
analysis with the data on disk untouched. This is aimed at the common accident:
a `rewrite=True` left in the script from the previous acquisition, silently
eating an overnight run. Set `store.REWRITE_DELAY_S = 0` in an unattended
pipeline that has its own guard.

## API

::: dapkel.functions.delta_t
