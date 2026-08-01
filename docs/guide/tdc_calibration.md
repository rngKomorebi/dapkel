# TDC calibration — the statistical code density test

The Kelpie ring-oscillator TDC does not have uniform bin widths. The raw
`time_series` code produced by [`unpack`](unpack.md) advances in steps whose
true duration varies code-to-code (differential non-linearity, DNL). A flat
`code * 77 ps` conversion is therefore only an *average*; timing-jitter and
coincidence work needs the exact per-code bin widths.

Those come from a **statistical code density test**.

## Method

Under illumination that is uncorrelated with the free-running oscillator, the
phase at which a photon is timestamped is uniformly random within one oscillator
period. The probability of landing in a given code bin is then proportional to
that bin's width. So, for every pixel independently:

1. Histogram the valid TDC codes over all frames of all files →
   `counts[pixel, code]`.
2. The width of code `k` in picoseconds is `period_ps * counts[k] / total`.
3. The calibrated time of code `k` is the running integral up to the **centre**
   of its bin:

   ```
   LUT[k] = period_ps * (cumsum(counts)[k] - counts[k]/2) / total
   ```

   which is the minimum-quantisation-error estimator of the true time a code
   represents.

That `LUT` is exactly the `time_lut` (code → ps map) that
[the coincidence pipeline](coincidences.md) and `calc_diff` accept.

## Absolute scale

The density test supplies only the *relative* per-code widths. The absolute
scale is fixed by `period_ps`, the nominal ~100 ns ring-oscillator period, onto
which the full populated code range is mapped.

## Per-SPAD LUTs

The calibration data are recorded per micropixel — one `SPADn / Sn` tag per SPAD
of the 2×2 macropixel — so a separate `(32, 32, n_codes)` LUT is produced for
each of the four SPADs.

Because ORT does not record *which* of the four micropixels fired, the caller
must choose: name a specific SPAD when the optics illuminate a known one, or use
`"average"` to average the four.

## Code range and overflow

The oscillator completes ~1300–1400 codes within one ~100 ns window. The default
bound is 1536, which leaves headroom while keeping the working histogram small.
Codes at or above it are out-of-range artefacts and are dropped **with a
warning** rather than silently folded in — if that warning reports a large
fraction, raise `n_codes`.

Pixels with no counts (dead in the calibration run) fall back to a nominal
*linear* ramp over the same range, so downstream indexing never breaks.

## Where the LUTs live

`collect_and_save_luts` writes `TDC_LUT_<tag>.npy` into `processed/` (stage-1
data) with the QA plots going to `results/tdc_calibration/`. `load_lut` accepts
either the folder holding the LUTs directly or a data folder whose `processed/`
holds them.

`load_board_lut` instead reads the LUTs **shipped with the package** in
`dapkel/params/calibration_data`, selected by daughterboard / motherboard
number and resolved via `importlib.resources` so it works from a pip install.

## API

::: dapkel.functions.tdc_calibration
