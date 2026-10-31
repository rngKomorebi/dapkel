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
scale is fixed by `clock_period_s` — one period of the reference clock — onto
which the full populated code range is mapped.

**This is the only time parameter the package cannot derive from the data**, and
this module is the only place it appears. Everything downstream measures the full
scale from the largest code a run produced
([`core.timing.full_scale_from_max_code`][dapkel.core.timing.full_scale_from_max_code])
or leans on `CODE_S`, which was itself measured against known clocks.

**A LUT belongs to one clock.** The ring oscillator sets the code width
(~71.6 ps, `core.timing.CODE_S`) and the reference clock sets the full scale, so
the number of codes scales as 1/f:

| clock | full scale | codes (measured) |
|---|---|---|
| 50 MHz | 20 ns | 275 |
| 10 MHz | 100 ns | 1379 |
| 5 MHz | 200 ns | 2761 |
| 2 MHz | 500 ns | 6914 |

Calibrate once per clock. The loaders take `min_codes` — how many codes the data
being calibrated actually occupies — and check the LUT against *that*: a table is
self-describing, its largest entry *is* the scale it was built for
(`lut_full_scale_s`), so comparing the two raises rather than silently
mis-converting. Callers read `min_codes` off the run, so nothing has to be
restated; that is the check that matters, because it compares the table to the
data it is about to be applied to rather than to a remembered frequency.
`collect_and_save_luts` tags the file name with the clock *frequency*
(`TDC_LUT_S0T_2MHz.npy`) so per-clock tables coexist in one folder.

Covering the run is necessary but not sufficient: a 2 MHz table covers a 10 MHz
run five times over, so coverage alone leaves several tables in the running. The
loaders keep the *tightest* covering one — the calibration whose full scale the
run's own code range points at.

The penalty for getting this wrong is smaller than it looks, though, because the
code width barely moves with the clock: mean LSB comes out at 72.3 / 71.9 /
70.4 ps at 2 / 10 / 50 MHz, so cross-applying a table mis-scales by well under a
percent rather than by the ratio of full scales. Selecting by clock is hygiene,
not a rescue from a factor of five.

## Choosing between calibrations: `clock`

Coverage cannot separate two calibrations of the **same** clock, because they
cover identically. That is what `clock` is for — on `load_lut`, `load_board_lut`,
and as `lut_clock` on both `delta_t` entry points:

```python
dt.calculate_and_save_timestamp_differences(
    path, pixels, nframes=8_000,
    daughterboard_number="D0", motherboard_number="M0",
    lut_clock="2MHz",
)
```

The tag is whatever follows the SPAD fragment in the file name. An **exact**
match wins, so a plain `clock="10MHz"` keeps selecting the table named exactly
that even once a qualified one like `10MHz_2026-08-13` sits beside it; failing
an exact match the tag may be given in part, and resolves when exactly one
calibration answers to it. Naming a clock that is not installed raises and lists
the ones that are.

**One calibration per clock is installed, so `lut_clock` is not normally
needed** — the run's own code range picks the table. Ask for one by name when a
second calibration of a clock is added; the loader raises and names both rather
than blending them.

!!! tip "Prefer the compact form"
    The dense per-clock table is kept for comparison, but on real data the
    [compact 8-tap calibration](#the-compact-form-8-ring-taps) below is better on
    every axis — 2× narrower coincidence peak, 550× smaller, and no per-clock
    table to select. Pass `tdc_model="compact"`.

## The compact form: 8 ring taps

The density test measures one width per code, but the hardware has only
**eight** distinct widths. The designers' account of it: the TDC is 13 bits — a
10-bit coarse counter (`core.timing.COARSE_BITS`) times an 8-tap fine
interpolator, which `unpack` folds together as
`ts = (coarse - 1) * 8 + (8 - fine)`. The coarse counter counts whole laps of the
ring oscillator and contributes no non-linearity; *all* of it is in the eight
taps, and they repeat every lap for as long as the counter runs. So code `k` is
always tap `k % 8`.

```
t(code) = (code // 8) * ring_period + sum(taps[:code % 8]) + taps[code % 8] / 2
```

Nine numbers per pixel — eight tap widths, whose sum *is* the ring period —
instead of one per code. `histogram_to_compact` reduces the same histogram to
them, and `compact_to_lut` expands it onto whatever code range a run occupies:

```python
tc.collect_and_save_compact(path, nframes=8192, clock_period_s=500e-9,
                            tags=("S0T", "S1T", "S2T", "S3T"))
```

**Measured on this chip** (100 files/SPAD, 2026-08-13): ring period
**599.7 ps** — an LSB of 75.0 ps, matching the designers' nominal figure — with
a **1.16 %** spread across pixels and a tap pattern of **5.06 ps rms**. Built
independently at 2 MHz and at 10 MHz the two agree to **0.05 %** on the ring
period (per-pixel r = 0.98) and r = 0.992 on the tap shape, which is the
clock-independence claim measured rather than argued. The four SPADs agree to
0.014 %.

Why this is not merely smaller:

- **The per-code detail was noise.** The density test estimates each code from
  an empirical CDF, so its uncertainty is `0.5/sqrt(N) x full_scale` — **569 ps
  per pixel** at the 193 k events/pixel a 100-file run gives at 2 MHz, ~800 ps
  once two pixels are differenced. Averaging each tap over the ~865 laps a
  2 MHz run sweeps cuts that by `sqrt(n_codes/8)` ≈ 29.
- **No per-clock table.** The taps belong to the ring, so one calibration serves
  every clock and `clock` / `lut_clock` stop mattering.
- **No flat tail.** A dense LUT is a cumulative count, so above the last code a
  pixel ever produced it is pinned at the full scale — a flat region covering
  ~4 % of the table (up to 33 ns) where a code that *did* arrive reads as the
  end of the range. The tap model has no edge: laps keep accumulating.

Measured on the 2026.08.07 2 MHz SPDC run, 227 k coincidences over 54 adjacent
pixel pairs, pooled crosstalk peak:

| calibration | size/SPAD | FWHM | peak height |
|---|---|---|---|
| none (flat 71.58 ps/code) | – | 6300 ps | 160 |
| dense per-clock LUT | 27 MB | 2550 ps | 592 |
| **compact 8-tap** | **64 KB** | **1250 ps** | **937** |

(pairs whose time bases differ by more than 1 %; over all 54 pairs it is
2150 / 2450 / **1300** ps.)

!!! note "This supersedes an earlier reading of the same data"
    A stored calibration looked like it aged badly — a residual 0.3–0.4 % of
    gain that cost 1.5–2 ns at a 492 ns full scale, implying you had to
    recalibrate in the same session as the science run. That was an artifact of
    estimating the gain from the dense table's noisy per-code values. The
    compact form, built on 2026-08-13 and applied to a run taken six days
    earlier, matches what deriving the gain from the run's *own* singles
    achieves. Session-to-session drift is not the limitation; the dense table's
    counting noise was.

`dapkel/params/calibration_data` ships **50, 10 and 2 MHz** tables as `float32`,
four SPADs each, one calibration per clock. Superseded calibrations sit in a
subfolder beside them; the loader only reads `.npy` files at the top level, and
`pyproject.toml` packages `calibration_data/*.npy` non-recursively, so they stay
in the repo without shipping in a wheel. Which calibration a file came from is
recorded by the commit that added it, not by its name.

Using a 10 MHz LUT on a 2 MHz run would discard every code past 100 ns: 79 % of
the singles, and since a coincidence needs both arms, 96 % of the coincidences.
That used to be silent; it now raises at load time — `delta_t` reads the run's
code range off its first file and hands it to the loader — and
`_structure_pixel_timestamps` warns with the discarded fraction if a LUT ever
gets that far.

Note that the reference clock cannot go arbitrarily low. `unpack` builds the code
as `(coarse - 1) * 8 + (8 - fine)` from a 10-bit coarse field, so it saturates at
8184 codes ≈ 586 ns: below ~1.7 MHz the counter wraps and the timestamps are
meaningless. 2 MHz already uses 85 % of it.

## Per-SPAD LUTs

The calibration data are recorded per micropixel — one `SPADn / Sn` tag per SPAD
of the 2×2 macropixel — so a separate `(32, 32, n_codes)` LUT is produced for
each of the four SPADs.

Because ORT does not record *which* of the four micropixels fired, the average of
the four is the only honest table for an `ORT` run — while an `S0T`...`S3T` run
reads one micropixel per macropixel and gets that SPAD's own LUT.

The delta-t stage-1 functions therefore take no `spad` argument: the acquisition
program writes the tag into the file name, so the `tag` that selected the files
already answers the question (`delta_t._spad_from_tag`). Only
`load_board_lut`/`load_board_compact`, which know nothing about a run, still ask.

## Code range

The code axis takes no sizing at all. `compute_code_histogram` allocates the
widest range the frame format can express (`core.timing.MAX_TS_CODE`, ~67 MB) and
trims it to what the data populated, so the code count comes out of the run.

Nothing can overflow it, which removes a whole failure mode: there is no
dropped-codes warning to heed, and no way to lose 79 % of a calibration run to a
mis-stated frequency. The two parameters that used to size the axis —
`open_shutter_time_s` and `n_codes` — are gone.

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
