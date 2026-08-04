# Background subtraction — removing the accidentals by shifting frames

The ORT accidental background is a **triangle** filling ±100 ns, and it sits
directly under the coincidence peak — [why](../ort_triangle_background.md).
Fitting it is possible (`fit_gaussian_on_triangle`), but then the coincidence
count you report depends on a model: on the triangle really being a triangle,
on the peak really being a Gaussian, and on the fitter separating the two.

`background_subtraction` measures the background instead of modelling it. You do
not call it directly for ordinary work — it is switched on with one parameter on
each half of the [delta-t pipeline](coincidences.md), and **both stage-1 paths
support it**: the counts grid and the feather.

## The idea

An SPDC pair is always inside **one frame**. So:

| | pairing | contains |
|---|---|---|
| **lag 0** | pixel A frame *i* ↔ pixel B frame *i* | SPDC + accidentals |
| **lag k** | pixel A frame *i* ↔ pixel B frame *i+k* | **accidentals only** |

Both see the same two pixels, at the same singles rates, over (almost) the same
number of frames — so their accidental content is the same in expectation. Only
the lag-0 histogram can hold a true pair. Subtract, and the triangle cancels
bin-for-bin.

```python
from dapkel.functions import delta_t as dt

signal = [(6, 10), (6, 11), (7, 10), (7, 11)]     # blobs off the hitmap
idler  = [(21, 22), (21, 23), (22, 22), (22, 23)]

# Stage 1: one pass over the '.bin' files, histogramming lag 0 and lags 1..8.
dt.calculate_and_save_delta_counts(
    path, [signal, idler],
    nframes=10_000, tag="S0T",
    apply_TDC_calibration=False,
    subtract_background=True,
)

# Stage 2: subtract, fit the residual, plot both figures.
counts, centers, fit = dt.collect_and_plot_delta_counts(
    path,
    subtract_background=True,
    bin_width_ps=231.0,
    plot_window_ps=10_000.0,
    peak_window_ps=2_000.0,
)
print(fit["model_free"])       # {'n': ..., 'n_err': ..., 'car': ..., ...}
```

This is not an ORT trick. It is the standard shift/rotation estimator used for
`g⁽²⁾` and coincidence counting; the only ORT-specific part is that the shift
unit is a *frame* rather than a laser period.

## Either stage 1 will do — but prefer the counts path

The feather path takes the same parameter:

```python
dt.calculate_and_save_timestamp_differences(
    path, [signal, idler], nframes=10_000, tag="S0T",
    apply_TDC_calibration=False,
    subtract_background=True, background_lags=(1, 2, 3, 4),
)
counts, centers, fit = dt.collect_and_plot_timestamp_differences(
    path, subtract_background=True, bin_width_ps=231.0,
    plot_window_ps=10_000.0, peak_window_ps=2_000.0,
)
```

The two give **bit-identical residuals** — same lags, same rescaling, same bins;
that is a test (`test_both_stage_one_paths_give_the_same_subtracted_residual`),
and it was checked on real data too. What differs is the cost:

| | artifact grows by | on a 1000-file run |
|---|---|---|
| counts grid | one grid plane per lag | 0.5 MB → 4.6 MB |
| feather | one *column set* per lag | 1 GB → ~9 GB |

The counts grid is sized by the grid, the feather by the data. With eight lags
that is a 9× multiplier on something already measured in gigabytes, so on the
feather path use fewer lags (`background_lags=(1, 2, 3, 4)` costs ~12% more
residual variance than eight and half the disk) — or use the counts path, which
is what `subtract_background` was built for.

On the read side both are one pass: the feather's lags are all histogrammed in
the *same* streaming pass, since that pass is the expensive part.

## It is a superset, never a substitute

`subtract_background=False` (the default) is the pipeline exactly as it was: a
2D `(n_pairs, n_bins)` artifact, the same resume signature, the same figure.

With it True:

* the **counts** artifact grows a leading lag axis, and plane 0 is bin-for-bin
  identical to what the plain run writes;
* the **feather** grows one column per pair per lag, named `<pair>@lag<k>`,
  while lag 0 keeps the bare `"r,c-r,c"` name. A reader that pools "every pair
  column" still gets exactly the lag-0 columns — the shifted ones are only
  reachable through an explicit lag request, or the coincidence histogram would
  silently inflate 9×.

Both identities are tests, not claims
(`test_lag_zero_plane_is_the_unsubtracted_histogram`,
`test_pooling_every_pair_column_ignores_the_lag_columns`). So one artifact
answers both questions, and switching the parameter on cannot change the
unsubtracted number you already reported. `@` cannot occur in a pair label —
digits, commas and one hyphen — so the two kinds of column can never be
confused.

## What it buys you, and what it does not

**It does not improve CAR.** The residual still carries the Poisson noise of
the accidentals it removed:

```
sigma_residual = sqrt( N_lag0 + N_bkg / K )        K = number of background lags
```

Contrast is set by the acquisition — singles rate, frame length, oscillator
period — not by the analysis. Nothing downstream of the camera can change it.

**It does give you two things worth having:**

* a **model-free coincidence count**. Sum the residual over a window
  (`peak_window_ps`, reported as `fit["model_free"]`). No Gaussian, no triangle,
  no fitter that can trade peak area against background slope;
* a **test of whether the pedestal is accidental at all**. If the residual is
  flat and zero away from the peak, it was. If it is not, something else is
  producing correlated counts and no fitted background was ever going to be
  right. This is the useful diagnostic when the background "looks too high" —
  it separates *"too many accidentals"* (a rate problem, fix the optics or the
  frame length) from *"a systematic"* (a detector or decoding problem).

That middle check is why the driver writes a **three-panel** figure whose middle
panel is the residual over the whole support rather than just around the peak,
and why it prints the check whether or not you look at the figure:

```
  control, |delta| beyond the peak: -187 +/- 605 (-0.3 sigma from zero)
```

Read that line before quoting anything from the peak.

## The fit model must be flat

Once the triangle has been subtracted there is no triangle left to fit, so
`subtract_background=True` with `background='triangle'` (or `'two_gaussians'`)
is refused rather than quietly fitted. The residual sits on a zero baseline,
which is precisely the case `fit_gaussian_peak` is valid for — and unlike on raw
data, the flat background is not an approximation over a narrow window, it is
the truth over the whole support.

Two numbers come out, and they answer different questions:

| number | where | means |
|---|---|---|
| `fit["model_free"]["n"]` | sum of the residual | coincidences, no model at all |
| `fit["amp"]`, `fit["sigma"]` | Gaussian on the residual | the peak's height and width |
| `fit["model_free"]["car"]` | residual / background in the window | coincidences per accidental |
| `fit["car"]` | fitted peak height / fitted baseline | peak-to-pedestal at Δ = 0 |

The last two will not agree, and neither is wrong — one is an area ratio over a
window, the other a height ratio at one point.

## Choosing the lags

`background_lags` defaults to `(1, …, 8)`. Two considerations:

* **more lags → quieter background.** Its Poisson noise enters the residual
  divided by `K`, so at `K = 8` the background contributes ~12% of the residual
  variance and the error is dominated by the lag-0 histogram itself. Going past
  that buys almost nothing;
* **check they agree with each other.** Lags 1 through 8 agreeing is evidence
  that there is no frame-to-frame correlation — no afterpulsing bleeding into
  the next frame, no readout artifact with a period. If lag 1 disagrees with
  lag 5, drop the near lags rather than averaging them in.

A bare int is one lag (`background_lags=1`). Lag 0 is always included and must
not appear in `background_lags`.

## Where things land

```
processed/
  <dataset>_delta_counts.npy         <- (n_lags, n_pairs, n_bins) int64, lag 0 first
  <dataset>_delta_counts.meta.json   <- lags, frames_used, labels, progress
  <dataset>_delta_t.feather          <- or: '<pair>' and '<pair>@lag<k>' columns
  <dataset>_delta_hist.npy           <- the streamed (n_lags, n_bins) histogram
results/coincidences/
  <dataset>_<label>_delta_counts_sub.png        <- the fitted residual (counts path)
  <dataset>_<label>_delta_t_sub.png             <- the fitted residual (feather path)
  <dataset>_<label>_background_subtraction.png  <- the three-panel diagnostic
```

The feather records its lag set and frame count in the Arrow schema metadata,
and that survives a `combine_delta_t_parts`. Note what is *not* in there: the
file count. The subtraction only ever uses the ratio
`frames[0] / frames[k] = nframes / (nframes - k)`, in which the file count
cancels — so a part folder left by a run that died is as usable as a complete
feather.

The grid is the counts path's own — `grid_ps` cells over ±`support_ps`, one TDC
code per cell for raw-code data (where that is *lossless*, since `delta_code` is
an integer) and a tenth of a code when calibrated. See
[Choosing the grid](coincidences.md#choosing-the-grid).

Holding the full support matters here: the flat-zero residual out at ±100 ns is
the evidence, and you cannot see it through a ±10 ns window. That is why the
feather path streams over `support_ps` (default 100 ns) rather than
`plot_window_ps` when subtracting — the fitted zoom is then a *slice* of that
histogram and the diagnostic's wide panels a regrouping of it, so widening the
plot window afterwards costs nothing and the panels cannot disagree about a bin
edge.

## Normalisation

A lag of `k` can only pair up `nframes - k` frames per file, so each background
lag is rescaled by `frames_used[0] / frames_used[k]` before averaging. At the
defaults that is a 0.01% correction and it makes no difference; it is applied
anyway because it is free and because it is what keeps the estimator correct if
anyone ever uses a lag comparable to the frame count.

`frames_used` is *derived*, not tracked — lag `k` loses the last `k` frames of
every file, which is deterministic — so it stays right across a resume.

Note what is **not** normalised: the totals. It is tempting to scale the
background so the two histograms have the same number of entries — that would
subtract the signal away along with the background. The excess in the total
count *is* the coincidence count.

## A real subtlety: the continuum is slightly over-subtracted

On real data the residual integrated over the whole ±100 ns comes out a little
**below** the peak excess — a deficit of ~0.2% of the accidental continuum.

That is physics, not a bug. ORT keeps **one timestamp per pixel per frame**, the
first photon's. In a frame where a true pair lands, both pixels report the
pair's arrival, so that frame contributes to Δ ≈ 0 *instead of* to some random
Δ elsewhere in the triangle. True pairs therefore partly **displace**
accidentals rather than adding to them, and the lag-0 continuum sits marginally
lower than the shifted one.

The size of it is the pair fraction, ~1% of the singles here, so it is far below
the peak and does not affect a windowed coincidence count. It does mean the
whole-support integral is not a second, independent estimate of the same number.

## API

::: dapkel.functions.background_subtraction
