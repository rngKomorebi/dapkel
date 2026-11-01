# TDC calibration — the statistical code density test

The Kelpie ring-oscillator TDC does not have uniform bin widths. The raw
`time_series` code produced by [`unpack`](unpack.md) advances in steps whose
true duration varies code-to-code (differential non-linearity, DNL). A flat
`code * 71.6 ps` conversion is therefore only an *average*; timing-jitter and
coincidence work needs the actual bin widths.

Those come from a **statistical code density test**.

## Method

Under illumination that is uncorrelated with the free-running oscillator, the
phase at which a photon is timestamped is uniformly random within one oscillator
period. The probability of landing in a given code bin is then proportional to
that bin's width. So, for every pixel independently:

1. Histogram the valid TDC codes over all frames of all files →
   `counts[pixel, code]`
   ([`core.reduce.code_histogram`][dapkel.core.reduce.code_histogram]).
2. The width of code `k` is `full_scale * counts[k] / total`.
3. The calibrated time of code `k` is the running integral up to the **centre**
   of its bin — the minimum-quantisation-error estimator of the true time a code
   represents.

But the hardware does not have one width per code: it has **eight**, repeating
every lap of the ring. So the eight are what is estimated, each averaged over
every lap the run swept — see [the 8 ring taps](#the-8-ring-taps) below.

## Absolute scale

The density test supplies only the *relative* per-code widths. The absolute
scale is fixed by `clock_period_s` — one period of the reference clock the test
ran at — onto which each pixel's populated code range is mapped.

**This is the only time parameter the package cannot derive from the data**, and
this module is the only place it appears. Everything downstream measures the full
scale from the largest code a run produced
([`core.timing.full_scale_from_max_code`][dapkel.core.timing.full_scale_from_max_code])
or leans on `CODE_S`, which was itself measured against known clocks.

The ring oscillator sets the code width (~71.6 ps, `core.timing.CODE_S`) and the
reference clock sets the full scale, so the number of codes a run occupies
scales as 1/f:

| clock | full scale | codes (measured) |
|---|---|---|
| 50 MHz | 20 ns | 275 |
| 10 MHz | 100 ns | 1379 |
| 5 MHz | 200 ns | 2761 |
| 2 MHz | 500 ns | 6914 |

The clock only decides how many laps are swept, not what a lap is, so **one
calibration serves every clock**. A slow clock sweeps more laps per frame and so
measures the ring period best; the tap shape comes out the same at any clock.

## The 8 ring taps

The density test measures one width per code, but the hardware has only
**eight** distinct widths. The designers' account of it: the TDC is 13 bits — a
10-bit coarse counter (`core.timing.COARSE_BITS`) times an 8-tap fine
interpolator, which `unpack` folds together as
`ts = (coarse - 1) * 8 + (8 - fine)`. The coarse counter counts whole laps of the
ring oscillator and contributes no non-linearity; *all* of it is in the eight
taps, and they repeat every lap for as long as the counter runs. So code `k` is
always tap `k % 8` (`core.calibration.TAPS_PER_LAP`).

```
t(k) = (k // 8) * lap + sum(taps[:k % 8]) + taps[k % 8] / 2,   lap = sum(taps)
```

Eight numbers per pixel — whose sum *is* the ring period — instead of one per
code: the whole calibration is a `(32, 32, 8)` table per SPAD.
[`process_tdc_calibration`][dapkel.functions.tdc_calibration.process_tdc_calibration]
histograms the density-test files and reduces them to the taps;
[`core.calibration.codes_to_ps`][dapkel.core.calibration.codes_to_ps] applies
the formula above directly, to any code at any clock, so there is no code range
for a table to fall short of.

```python
from dapkel.functions import tdc_calibration as tc

tc.process_tdc_calibration(path, 8192, clock_period_s=500e-9)   # 2 MHz test
tables = tc.plot_tdc_calibration(path, pixel=(16, 16))          # QA figures
```

Each pixel is normalised over **its own** populated code range, not the
array-wide one, so its time-base gain survives: a pixel whose ring runs fast
reaches more codes in one clock period and gets narrower taps.

**Measured on this chip** (100 files/SPAD, 2026-08-13): ring period
**599.7 ps** — an LSB of 75.0 ps, matching the designers' nominal figure — with
a **1.16 %** spread across pixels and a tap pattern of **5.06 ps rms**. Built
independently at 2 MHz and at 10 MHz the two agree to **0.05 %** on the ring
period (per-pixel r = 0.98) and r = 0.992 on the tap shape, which is the
clock-independence claim measured rather than argued. The four SPADs agree to
0.014 %.

### Why 8 taps and not a per-code table

The package used to calibrate with a dense per-code, per-clock lookup table; it
is retired to the repository's `archive/` folder. The comparison is why:

- **The per-code detail was noise.** The density test estimates each code from
  an empirical CDF, so its uncertainty is `0.5/sqrt(N) x full_scale` — **569 ps
  per pixel** at the 193 k events/pixel a 100-file run gives at 2 MHz, ~800 ps
  once two pixels are differenced. Averaging each tap over the ~865 laps a
  2 MHz run sweeps cuts that by `sqrt(n_codes/8)` ≈ 29.
- **No per-clock table.** The taps belong to the ring, so one calibration serves
  every clock and there is no table to select.
- **No flat tail.** A dense LUT is a cumulative count, so above the last code a
  pixel ever produced it is pinned at the full scale — a flat region covering
  ~4 % of the table (up to 33 ns) where a code that *did* arrive reads as the
  end of the range. The tap model has no edge: laps keep accumulating.

Measured on the 2026.08.07 2 MHz SPDC run, 227 k coincidences over 54 adjacent
pixel pairs, pooled crosstalk peak:

| calibration | size/SPAD | FWHM | peak height |
|---|---|---|---|
| none (flat 71.58 ps/code) | – | 6300 ps | 160 |
| dense per-clock LUT (archived) | 27 MB | 2550 ps | 592 |
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

Note that the reference clock cannot go arbitrarily low. `unpack` builds the code
as `(coarse - 1) * 8 + (8 - fine)` from a 10-bit coarse field, so it saturates at
8184 codes ≈ 586 ns: below ~1.7 MHz the counter wraps and the timestamps are
meaningless. 2 MHz already uses 85 % of it.

## Per-SPAD tables

The calibration data are recorded per micropixel — one `S0T`...`S3T`
timestamping run per SPAD of the 2×2 macropixel, the default `tags` of
`process_tdc_calibration` — so a separate `(32, 32, 8)` table is produced for
each of the four SPADs. Only the program tag is fixed in a file name, so files
are matched as `*_<tag>*.bin` whatever the rest of the name is
(`density_test_S2T12.bin`). A counting tag (`S0C`, `ORC`) is refused: those
programs record no TDC codes.

Because ORT does not record *which* of the four micropixels fired, the average of
the four is the only honest table for an `ORT` run — while an `S0T`...`S3T` run
reads one micropixel per macropixel and gets that SPAD's own table.

The delta-t functions therefore take no `spad` argument: the acquisition program
writes the tag into the file name, so the `tag` that selected the files already
answers the question.
[`core.calibration.board_taps`][dapkel.core.calibration.board_taps] reads it —
a single-SPAD tag gets that SPAD's table (or the mean of the four, with a note,
if that table is missing), anything else the mean.

## Code range

The code axis takes no sizing at all. `core.reduce.code_histogram` allocates the
widest range the frame format can express (`core.timing.MAX_TS_CODE`) and trims
it to what the data populated, so the code count comes out of the run.

Nothing can overflow it, which removes a whole failure mode: there is no
dropped-codes warning to heed, and no way to lose most of a calibration run to a
mis-stated frequency. The clock the test was measured at is recorded in the
sidecar (`measured_at_clock_period_s`), not in the file name: the table belongs
to the ring, not to the clock.

Pixels with no counts (dead in the calibration run) fall back to a flat nominal
lap, so downstream conversion never breaks.

## Where the tables live

`process_tdc_calibration` writes `TDC_COMPACT_<tag>.npy` (with a `.meta.json`
sidecar) into `processed/` — stage-1 data — and returns the saved paths.
`plot_tdc_calibration` reads them back and saves one QA figure per SPAD to
`results/tdc_calibration/`: the chosen pixel's tap DNL against the array's
median and 5–95 % band, and the ring period across all pixels — the per-pixel
time-base gain the table carries.

To ship a calibration, copy the tables into `dapkel/params/calibration_data` as
`<daughterboard>_<motherboard>_TDC_COMPACT_<tag>.npy`, e.g.
`D0_M0_TDC_COMPACT_S2T.npy`. The packaged tables still use the older
`D0_M0_TDC_COMPACT_SPAD{n}_S{n}.npy` names; `board_taps` reads both.
[`board_taps`][dapkel.core.calibration.board_taps] reads them by daughterboard /
motherboard number, resolved via `importlib.resources` so it works from a pip
install, and [the coincidence pipeline](coincidences.md) applies them with
`apply_TDC_calibration=True`.

!!! warning "Recalibrate after the restructure"
    The packaged tables predate the current `process_tdc_calibration`. The
    board is due to be recalibrated with it, and the tables above replaced.

## API

::: dapkel.functions.tdc_calibration
