# Background subtraction — removing the accidentals by shifting frames

The ORT accidental background is a **triangle** filling ± one TDC full scale
(one reference-clock period: 20 ns at 50 MHz, 100 ns at 10 MHz, 500 ns at
2 MHz), and it sits
directly under the coincidence peak — [why](../ort_triangle_background.md).
Fitting it is possible — it is what [`plot_coincidences`][dapkel.functions.coincidences.plot_coincidences]
does on a raw histogram — but then the coincidence count you report depends on a
model: on the triangle really being a triangle, on the peak really being a
Gaussian, and on the fitter separating the two.

Background subtraction measures the background instead of modelling it. It is
part of the [delta-t pipeline](coincidences.md), switched on with one parameter,
`subtract_background=True`, on each of its two functions:
[`process_coincidences`][dapkel.functions.coincidences.process_coincidences] histograms the
shifted frames, `plot_coincidences` subtracts them.

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
from dapkel.functions import coincidences as co

signal = [(6, 10), (6, 11), (7, 10), (7, 11)]     # blobs off the hitmap
idler  = [(21, 22), (21, 23), (22, 22), (22, 23)]

# Stage 1: one pass over the '.bin' files, histogramming lag 0 and lags 1..8.
co.process_coincidences(
    path, 10_000, [signal, idler],
    tag="S0T",
    apply_TDC_calibration=False,
    subtract_background=True,
)

# Stage 2: subtract, fit the residual, plot both figures.
counts, centers, fit = co.plot_coincidences(
    path,
    tag="S0T",
    subtract_background=True,
    bin_width_s=231e-12,
    plot_window_s=10e-9,
    fit_window_s=2e-9,
)
print(fit["model_free"])       # {'n': ..., 'n_err': ..., 'car': ..., ...}
```

This is not an ORT trick. It is the standard shift/rotation estimator used for
`g⁽²⁾` and coincidence counting; the only ORT-specific part is that the shift
unit is a *frame* rather than a laser period.

## What it costs

Each lag adds one grid plane to the stage-1 artifact — on a 1000-file run,
0.5 MB becomes 4.6 MB with the default eight — and one differencing pass per
file. The `.bin` decoding, which dominates, is paid once. (The retired feather
path also supported it, at one column set per lag; it is in the repository's
`archive/` folder.)

## It is a superset, never a substitute

`subtract_background=False` (the default) is the pipeline exactly as it was: a
2D `(n_pairs, n_bins)` artifact, the same resume signature, the same figure.

With it True the artifact grows a leading lag axis, and plane 0 is bin-for-bin
identical to what the plain run writes. That identity is a test, not a claim
(`test_lag_zero_plane_is_the_unsubtracted_histogram`). So one artifact answers
both questions: `plot_coincidences` on it without `subtract_background` fits lag 0
exactly as before, and switching the parameter on cannot change the unsubtracted
number you already reported.

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
  (`fit_window_s`, reported as `fit["model_free"]`). No Gaussian, no triangle,
  no fitter that can trade peak area against background slope;
* a **test of whether the pedestal is accidental at all**. If the residual is
  flat and zero away from the peak, it was. If it is not, something else is
  producing correlated counts and no fitted background was ever going to be
  right. This is the useful diagnostic when the background "looks too high" —
  it separates *"too many accidentals"* (a rate problem, fix the optics or the
  frame length) from *"a systematic"* (a detector or decoding problem).

That middle check is why `plot_coincidences` writes a **three-panel** figure whose
middle panel is the residual over the plotted window — the whole support unless
you narrow `plot_window_s` — rather than just around the peak,
and why it prints the check whether or not you look at the figure:

```
  control, |delta| beyond the peak: -187 +/- 605 (-0.3 sigma from zero)
```

Read that line before quoting anything from the peak.

## The fit model must be flat

Once the triangle has been subtracted there is no triangle left to fit — which
is why the model is not a parameter. `subtract_background` settles it: flat on
the residual, triangular on a raw histogram. The residual sits on a zero
baseline, which is precisely the case a Gaussian on a constant is valid for —
and unlike on raw data, the flat background is not an approximation over a
narrow window, it is the truth over the whole support. The fit runs within
`fit_window_s` of the peak.

Two numbers come out, and they answer different questions:

| number | where | means |
|---|---|---|
| `fit["model_free"]["n"]` | sum of the residual | coincidences, no model at all |
| `fit["amp"]`, `fit["sigma"]` | Gaussian on the residual | the peak's height and width |
| `fit["model_free"]["car"]` | residual / background in the window | coincidences per accidental |
| `fit["car"]` | fitted peak height / fitted baseline, raw histogram only | peak-to-pedestal at Δ = 0 |

The last two will not agree, and neither is wrong — one is an area ratio over a
window, the other a height ratio at one point. On the residual `fit["car"]` is
`NaN`: a ratio to a zero baseline is a division by noise.

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
  <dataset>_<tag>_coincidences.npy   <- (n_lags, n_pairs, n_bins) int64, lag 0 first
  <dataset>_<tag>_coincidences.meta.json <- lags, frames_used, labels, progress
results/coincidences/
  <dataset>_<tag>_<label>_coincidences_sub.png         <- the fitted residual
  <dataset>_<tag>_<label>_background_subtraction.png   <- the three-panel diagnostic
```

The sidecar records the lag set and `frames_used` per lag. The subtraction only
ever uses their ratio, `frames[0] / frames[k] = nframes / (nframes - k)`, in
which the file count cancels — so an artifact left by a run that died at a
checkpoint is as usable as a complete one.

The grid is the ordinary delta-t grid — `bin_width_s` cells over the run's measured
full scale, one TDC code per cell for uncalibrated data (where that is
*lossless*, since the codes are integers) and a tenth of a code when calibrated.
See [Choosing the grid](coincidences.md#choosing-the-grid).

Holding the full support matters here: the flat-zero residual out at the edge of
the full scale is the evidence, and you cannot see it through a ±10 ns window.
Nothing needs telling how far that is — the grid is sized from the run's own
largest code. The fitted zoom is then a *slice* of that
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

On real data the residual integrated over the whole full scale comes out a little
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

Both switches are parameters of the delta-t functions —
[`process_coincidences`][dapkel.functions.coincidences.process_coincidences] and
[`plot_coincidences`][dapkel.functions.coincidences.plot_coincidences]; see
[Coincidences](coincidences.md#api).
