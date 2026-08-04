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
buffered differences would exceed the 64 MB part cap they are flushed to a part
file, and the parts are concatenated into the final feather one Arrow record
batch at a time:

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
- **The parts are always kept.** They cost disk rather than memory, they are
  the crash insurance, and — see [below](#plotting-without-the-combined-file) —
  the read side streams them just as happily as the combined file, so there is
  nothing to gain from deleting them automatically. Remove the folder by hand
  once you have checked the combined feather.

Why 64 MB and not 10: `unpack` already holds a `(32, 32, nframes)` float64
array — about 82 MB at 10 000 frames — so below that the part buffer is not the
binding constraint, and a smaller cap only multiplies the number of part files.
The cap is the module constant `_PART_MB`, not a parameter: sharding is a
memory strategy and cannot change the numbers, so no call site needs to tune
it.

## Reading it back: the histogram is streamed, not loaded

The write side streams, and so must the read side. A 10 000-file run's combined
feather reaches ~135 GB; loading it into a DataFrame and pooling the columns
needs it resident several times over, which simply does not happen.

[`compute_and_save_delta_histogram`][dapkel.functions.delta_t.compute_and_save_delta_histogram]
reads it one Arrow record batch at a time instead. A histogram over fixed edges
is **additive**, so the counts can be summed batch by batch with one batch
resident — and the result is bin-for-bin what histogramming the whole pooled
array at once would have given. This is the exact baseline: no sampling, no
re-binning, no rounding.

Nothing downstream ever needed the raw differences. Both fit models take
`(bin_centers, counts)`, and so does the plot; the pooled array was a pure
intermediate. So the counts are saved as a stage-1 artifact:

```
processed/
  <dataset>_delta_t.feather        <- ~135 GB, read once
  <dataset>_delta_hist.npy         <- the counts, a few MB
  <dataset>_delta_hist.meta.json   <- binning, source fingerprint, statistics
```

The sidecar records `bin_width_ps`, `plot_window_ps`, `time_unit_ps`, `pairs`
and a fingerprint of the source files. A saved histogram is reused only when
all of it matches, so changing the binning or rewriting the feather re-streams
automatically; pass `reuse=False` (or `reuse_histogram=False` to the plot
driver) to force the full pass. Re-fitting, switching between the flat and
triangular background, or redrawing therefore costs a small load rather than
another read of the feather.

`collect_and_plot_timestamp_differences` goes through this, and returns
`(counts, centers, fit)` — enough to re-fit by hand without touching disk:

```python
counts, centers, fit = dt.collect_and_plot_timestamp_differences(
    path, background="triangle", plot_window_ps=100_000
)
narrow = np.abs(centers) < 1000
dt.fit_gaussian_peak(centers[narrow], counts[narrow])
```

The `.meta.json` also reports `padding_fraction`: the share of the cells read
that were padding rather than data. The parts are written wide and padded to
the longest pair column, so an unbalanced blob pair inflates the feather —
that number says how much of the 135 GB is real.

### Plotting without the combined file

`feather_path` accepts a `*_delta_t_parts` folder as well as a `.feather`, and
a run whose combined feather is missing falls back to its parts on its own. The
parts are complete feathers, so the combine step is optional for plotting:

```python
dt.collect_and_plot_timestamp_differences(
    feather_path=".../processed/SPDC_delta_t_parts"
)
```

Parts or combined file is not a numerical choice — the histogram is a sum over
record batches either way, and a sum does not care which file a batch came out
of, so the counts are identical. It is only an I/O choice, and the parts win it:
histogramming them skips the combine pass, which reads and rewrites every byte
of the run to produce a second copy of it. Working from the parts is the reason
they are always kept. Keep the combined feather when you want one portable file
per run, or when the data set is small enough to read whole into a DataFrame.

## The other stage 1: histogram straight off the `.bin`

Everything above keeps every difference and histograms it later. That is the
baseline — nothing is thrown away — but the artifact grows with the run, and
the binning is still chosen at plot time, so every change of mind costs another
pass over hundreds of GB.

[`calculate_and_save_delta_counts`][dapkel.functions.delta_t.calculate_and_save_delta_counts]
runs the same loop and bins the differences onto a fixed native grid as it
finds them, keeping only the counts. It writes no feather at all:

```
processed/
  <dataset>_delta_counts.npy         <- (n_pairs, n_cells) int64
  <dataset>_delta_counts.meta.json   <- grid, pair labels, totals, progress
```

The size is set by the **grid**, not by the data. A 256-pair run on the default
grid is 106 MB whether it came from 100 files or 10 000 — measured against a
3.22 GB feather for the same run, and the same 106 MB for one that reaches
135 GB. That is what makes every pair plottable at once instead of whichever
columns you could afford to stream.

### Choosing the grid

Two arguments, `grid_ps` (cell width) and `support_ps` (half-range):

| data | default cell | why |
|---|---|---|
| raw codes (`apply_TDC_calibration=False`) | one code, ~77 ps | `delta_code` is an **integer**, so counts-per-code is a re-encoding, not a summary — exactly lossless |
| calibrated (`unit='ps'`) | 1/10 code, ~7.7 ps | the per-pixel LUT smears the code lattice; 7.7 ps is ~5× finer than the jitter |

`support_ps` defaults to 200 ns — one oscillator period, the entire range a
difference can physically occupy. Anything landing outside is reported as
`n_outside` rather than dropped silently; if that is not ~0, widen it.

### Re-binning is free, and exact

`plot_window_ps` becomes a slice and `bin_width_ps` a reduce, both in memory:

```python
dt.collect_and_plot_delta_counts(
    path, pairs=["6,10-21,22"], bin_width_ps=71,
    plot_window_ps=10e3, background="triangle", osc_period_ps=200e3,
)
```

By default the binning is **snapped** onto the grid, and the snap is not
cosmetic:

- the width goes to a whole number of cells, or bins hold 9 then 10 then 9
  cells and a smooth distribution picks up an ~11% sawtooth;
- that number is made **odd**. Bins are centred on zero, so their edges sit at
  half-integer multiples of the width — odd puts those edges on cell
  *boundaries*, even puts them through cell *centres*, splitting a cell at
  every boundary. Measured on a real 3.22 GB run: **exactly zero** difference
  from the feather path at 9, 11 and 13 cells per bin, ~1% at 8, 10 and 12;
- the window goes to a whole number of bins, so zero is a bin centre.

So with snapping on, this is not an approximation of the feather histogram —
it is the same histogram. Pass `snap_to_grid=False` to take your numbers
literally, and you will be warned about the sawtooth you are asking for.

The one honest disagreement is the outermost bin at each end. The feather path
cuts data at `abs(delta) <= plot_window_ps` but draws edges half a bin beyond
it, so its first and last bins can only ever be part filled; the counts path
fills them. That, not the method, is why fitted `sigma` differs by ~1%.

### What you give up

No per-event data survives, so there is no re-calibrating with a different LUT,
no re-cutting `delta_window`, and nothing that needs individual differences —
time ordering, splitting by acquisition, higher-order correlations. It also
does not make stage 1 meaningfully faster: unpacking dominates either way. The
win is artifact size and replot latency. Keep the feather until you are sure
you want none of that back.

### Crash insurance without a part folder

The grid is rewritten every `flush_every` files (default 50), so a run that
dies leaves a valid artifact covering everything up to its last checkpoint, and
the next call resumes from it. There is only ever one file, so two runs cannot
interleave — the failure mode `combine_delta_t_parts` exists to guard against
simply does not arise here.

The sidecar records `files_done` and `total_in_grid`. The array is moved into
place before the sidecar, so a crash between the two moves could pair a newer
array with an older sidecar; resuming on that would re-count files and inflate
the total. The resume path checks the array against `total_in_grid` and starts
over rather than continue from something inconsistent.

### Measuring the background instead of fitting it

`subtract_background=True` histograms the *frame-shifted* pixel pairs alongside
the real ones, so the accidental triangle can be subtracted bin-for-bin rather
than fitted. Both stage-1 paths take it, and both give the same residual — see
[Removing the accidental background by shifting frames](background_subtraction.md).

```python
# counts path: the lags cost one small grid plane each
dt.calculate_and_save_delta_counts(path, [signal, idler], nframes=10_000,
                                   subtract_background=True)
counts, centers, fit = dt.collect_and_plot_delta_counts(
    path, subtract_background=True, peak_window_ps=2_000.0)

# feather path: same call, but the lags cost one column set each - so ~9x the
# feather at the default eight lags. Use fewer, or use the counts path.
dt.calculate_and_save_timestamp_differences(path, [signal, idler], nframes=10_000,
                                            subtract_background=True,
                                            background_lags=(1, 2, 3, 4))
counts, centers, fit = dt.collect_and_plot_timestamp_differences(
    path, subtract_background=True, peak_window_ps=2_000.0)
```

The feather's lag columns are named `<pair>@lag<k>`; lag 0 keeps the bare pair
name, and pooling "every pair column" never picks up the shifted ones.

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
