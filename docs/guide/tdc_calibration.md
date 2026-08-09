# TDC calibration — the statistical code density test

The Kelpie ring-oscillator TDC does not have uniform bin widths. The raw
`time_series` code produced by [`unpack`](unpack.md) advances in steps whose
true duration varies code-to-code (differential non-linearity, DNL). A flat
`code * 71.6 ps` conversion is therefore only an *average*; timing-jitter and
coincidence work needs the exact per-code bin widths.

Those come from a **statistical code density test**.

## Method

Under illumination that is uncorrelated with the free-running oscillator, the
phase at which a photon is timestamped is uniformly random within one oscillator
period. The probability of landing in a given code bin is then proportional to
that bin's width. So, for every pixel independently:

1. Histogram the valid TDC codes over all frames of all files →
   `counts[pixel, code]`.
2. The width of code `k` is `full_scale * counts[k] / total`.
3. The calibrated time of code `k` is the running integral up to the **centre**
   of its bin:

   ```
   LUT[k] = full_scale * (cumsum(counts)[k] - counts[k]/2) / total
   ```

   which is the minimum-quantisation-error estimator of the true time a code
   represents.

That `LUT` is exactly the `time_lut` (code → ps map) that
[the coincidence pipeline](coincidences.md) and `calc_diff` accept.

## Absolute scale

The density test supplies only the *relative* per-code widths. The absolute
scale is fixed by `open_shutter_time_s` — one period of the reference clock —
onto which the full populated code range is mapped.

**A LUT belongs to one clock.** The ring oscillator sets the code width
(~71.6 ps, `core.timing.CODE_S`) and the reference clock sets the full scale, so
the number of codes scales as 1/f:

| clock | full scale | codes (measured) |
|---|---|---|
| 50 MHz | 20 ns | 275 |
| 10 MHz | 100 ns | 1379 |
| 5 MHz | 200 ns | 2761 |
| 2 MHz | 500 ns | 6914 |

Calibrate once per clock. Every loader takes `open_shutter_time_s` and checks it
against the LUT's own full scale — a LUT is self-describing, its largest entry
*is* the scale it was built for (`lut_full_scale_s`) — and raises rather than
silently mis-converting. `collect_and_save_luts` tags the file name with the
window (`TDC_LUT_SPAD0_S0_500ns.npy`) so per-clock tables coexist in one folder.

The LUTs shipped in `dapkel/params/calibration_data` are **10 MHz only**. Using
one on a 2 MHz run discards every code past 100 ns: 79 % of the singles, and
since a coincidence needs both arms, 96 % of the coincidences. That used to be
silent; it now raises at load time, and `_structure_pixel_timestamps` warns with
the discarded fraction if a LUT ever gets that far.

Note that the reference clock cannot go arbitrarily low. `unpack` builds the code
as `(coarse - 1) * 8 + (8 - fine)` from a 10-bit coarse field, so it saturates at
8184 codes ≈ 586 ns: below ~1.7 MHz the counter wraps and the timestamps are
meaningless. 2 MHz already uses 85 % of it.

## Per-SPAD LUTs

The calibration data are recorded per micropixel — one `SPADn / Sn` tag per SPAD
of the 2×2 macropixel — so a separate `(32, 32, n_codes)` LUT is produced for
each of the four SPADs.

Because ORT does not record *which* of the four micropixels fired, the caller
must choose: name a specific SPAD when the optics illuminate a known one, or use
`"average"` to average the four.

## Code range and overflow

The oscillator completes `full_scale / 71.6 ps` codes per measurement range, so
the code axis is sized from `open_shutter_time_s` via
`core.timing.codes_for_full_scale` — 10 % of headroom over the nominal count,
then trimmed to what is actually populated. Pass `n_codes` to override it.

Codes at or above the bound are out-of-range artefacts and are dropped **with a
warning** naming the fraction lost and the largest code seen. In practice that
warning means the code axis is too short for the clock in use, i.e. the wrong
`open_shutter_time_s` was given.

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
