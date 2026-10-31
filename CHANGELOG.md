# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING: every analysis is two functions, `process_<x>` and `plot_<x>`.**
  `process_<x>(path, nframes, ...)` unpacks the raw '.bin' files and *always*
  saves its artifact under `processed/`; `plot_<x>(path, tag, ...)` loads it and
  writes the figures. Everything else is private, and code no pipeline used is
  moved, verbatim, to `archive/` at the repository root (reference only, not
  packaged or tested). `tests/test_api_surface.py` now enforces the pair, and
  caps each module at ten private functions. Entries further down this section
  describe the pre-restructure API.

  | module | was | is |
  | --- | --- | --- |
  | `hitmap_analysis` | `compute_hitmap(_64)`, `compute_and_save_hitmap(_64)`, `collect_and_plot_hitmap`, `plot_hitmap(array)` | `process_hitmap`, `plot_hitmap(path, tag)` |
  | `dcr_analysis` | `compute_dcr_32/64`, `compute_and_save_dcr_32/64`, `collect_and_plot_dcr_32/64`, `plot_heatmap`, `plot_distribution` | `process_dcr`, `plot_dcr` |
  | `crosstalk_analysis` | `compute_crosstalk`, `compute_and_save_crosstalk`, `collect_and_plot_crosstalk`, `combine_directional_crosstalk`, 4 × `plot_ct_*` | `process_crosstalk`, `plot_crosstalk` |
  | `data_quality` | `collect_time_mat`, `plot_time_code_histogram` | `process_time_codes`, `plot_time_codes` |
  | `tdc_calibration` | 12 functions over two calibration models | `process_tdc_calibration`, `plot_tdc_calibration` |
  | `delta_t` | 15 functions over two stage-1 paths | `process_delta_t`, `plot_delta_t` |

  Details worth knowing when porting a script:

  - **The tag decides what a count is** (`core.reduce.readout_mode`): `...T`
    programs count frames carrying a timestamp, `...C` programs sum the photon
    counter. The `mode` parameter is gone, and with it the tag/mode mismatch that
    inflated a timestamp-mode DCR ~50x. A tag ending in neither letter raises.
  - **32 or 64 is the tag, not a second function**: `'S*C'` / `'S*T'` assemble
    the four quadrants into the (64, 64) map (saved as `..._sxc_...`). The
    quadrants are cut to a common file count, so all pixels share one exposure,
    and a missing quadrant is NaN rather than zero.
  - **`process_delta_t(path, nframes, pixels, ...)`** - `nframes` is now the
    second positional argument, as in every `process_*`. `plot_delta_t(path,
    tag)` finds the grid from the tag; `counts_path`, `npy_path` and `reuse`
    are gone. Verified on 30 files of the 2026.09.14 SPDC run: the new grid is
    bin-for-bin identical to `calculate_and_save_delta_counts` (5 lag planes x
    81 pairs, raw and calibrated), and the fits and model-free count match to
    every printed digit.
  - **`plot_hitmap` returns the rate map** (Hz) rather than the counts.
  - **`process_crosstalk` on a parent folder** processes each pairing
    sub-folder and saves the combined (64, 64) directional map; `plot_crosstalk`
    draws whichever was saved. The artifact is named by the batch prefix
    (`<name>_ct_crosstalk.npy`), the pairing goes in the sidecar.
  - **`data_quality` saves** the (32, 32, n_codes) per-pixel code histogram,
    so any pixel can be inspected without re-reading the run.
  - **Background subtraction is unchanged in behaviour** but no longer a
    module: it is private to `delta_t`, still driven by `subtract_background`.
- **BREAKING: the TDC calibration is the compact 8-tap form only.** One
  (32, 32, 8) table per SPAD serves every clock, and is applied by the new
  `core.calibration.codes_to_ps` straight from the tap widths - there is no LUT
  to expand, so no code range to cover, no per-clock file to pick and no
  overflow warning. `lut_clock` and `tdc_model` are gone. `plot_tdc_calibration`
  draws each SPAD's tap DNL against the array and its ring-period spread. The
  packaged tables are to be rebuilt after this restructure.
- `core.timing.resolve_live_time(firmware_version, *, open_shutter_time_s,
  exp_time_s)` drops its unused `nframes` and raises when `short_exposure` has no
  shutter time, instead of returning None for every caller to check.

### Removed

Moved to `archive/` (code and its tests, verbatim from commit 6b49a4b):

- the delta-t **feather path** - `calculate_and_save_timestamp_differences`,
  `combine_delta_t_parts`, `combine_delta_t_feathers`,
  `compute_and_save_delta_histogram`, `collect_and_plot_timestamp_differences`
  and their part-file machinery. The binned-counts path gives the same
  histogram without the 3-135 GB intermediate; per-event analyses are the one
  thing it cannot do.
- the **dense per-clock LUT** - `histogram_to_lut`, `compute_tdc_lut`,
  `collect_and_save_luts`, `load_lut`, `load_board_lut`, `lut_full_scale_s` and
  the clock-tag selection. The packaged per-clock tables move to the untracked
  `archive/calibration_data/`.
- the **two-Gaussian peak model** (`two_gaussians_on_triangle`,
  `fit_two_gaussians_on_triangle`), which no pipeline called.
- `calc_diff` (within-frame differencing is the lag-0 case of `delta_t`'s
  frame-shifted differencing), `core.timing.resolve_cycle_time`,
  `resolve_frame_time`, `CLK_PERIOD`, `MAX_DELTA_S`, `core.pairs.pair_labels`
  and `core.io.frames_in_file`.

### Added

- **`clip_colorbar_to` sets the sensor-map colour-scale ceiling** — on
  `core.plots.sensor_map`, `hitmap_analysis.plot_hitmap` and
  `collect_and_plot_hitmap`. It is the percentile the scale stops at, `99` by
  default (the behaviour so far), and `None` for an unclipped bar running to the
  true maximum. The clip is why a bar can end well below the maximum quoted in
  the title: the title always reports the real `max`, while the scale is held
  down so a few hot pixels do not flatten the rest of the array into one shade.
- **A compact TDC calibration: 8 ring taps per pixel, one table for every
  clock.** `histogram_to_compact` reduces the density-test histogram to the
  eight tap widths the hardware actually has — the coarse counter contributes no
  non-linearity, so code `k` is always tap `k % 8` — and `compact_to_lut`
  expands them onto whatever code range a run occupies. Built by
  `collect_and_save_compact`, loaded by `load_board_compact`, and selected on
  both `delta_t` entry points with `tdc_model="compact"` (`"table"`, the dense
  per-clock LUT, stays the default for now).

  It is better than the dense table on every axis, not just smaller. On the
  2026.08.07 2 MHz SPDC run (227 k coincidences, 54 adjacent pairs) the pooled
  crosstalk peak goes 2550 → **1250 ps** FWHM with the peak 1.6× higher, from
  **64 KB** per SPAD instead of 27 MB. Three reasons: each tap is averaged over
  every lap the run swept, cutting the density test's 569 ps/pixel CDF noise by
  `sqrt(n_codes/8)`; the taps belong to the ring rather than the clock, so one
  table serves all of them; and there is no flat tail, where a dense LUT pins
  every code above a pixel's own range at the full scale (~4 % of the table, up
  to 33 ns off).

  Measured: ring period 599.7 ps (LSB 75.0 ps, the designers' nominal figure),
  1.16 % spread across pixels, 5.06 ps rms tap pattern. Built independently at
  2 MHz and 10 MHz the two agree to 0.05 % on the period and r = 0.992 on the
  tap shape.
- **TDC calibrations for 50, 10 and 2 MHz ship with the package.** Four SPADs
  per clock, stored as `float32` in `dapkel/params/calibration_data` and named
  `D0_M0_TDC_LUT_SPAD{n}_S{n}_{clock}.npy`. The 10 MHz set they supersede moved
  to a subfolder: the loader reads only top-level `.npy` files and the package
  data glob is non-recursive, so superseded calibrations stay in the repo
  without shipping in a wheel. Provenance is the commit that added them.
- **`clock` selects which calibration to load** — on
  `tdc_calibration.load_lut` and `load_board_lut`, and as `lut_clock` on both
  `delta_t` entry points. Not normally needed, since code coverage picks the
  table whenever one calibration per clock is installed; it is what resolves two
  calibrations of the *same* clock, which cover identically. The tag is what
  follows the SPAD fragment in the file name; an exact match wins, so a plain
  `clock="10MHz"` keeps meaning the table named exactly that, while a partial
  tag resolves when one calibration answers to it.

- **Every fitted width is reported with its 1 sigma error** — in the figure
  legend, in the console summary and in the returned dict. `delta_t` now derives
  `fwhm_err`, `jitter_per_detector_err` and, for `background="two_gaussians"`,
  `fwhm_n_err` / `fwhm_b_err` alongside the values they belong to, so a number
  read off a figure and the same number out of the dict cannot disagree. A fit
  with no covariance prints the width alone rather than `+- nan`.

### Fixed

- **A fitted width the data does not determine was reported as a measurement.**
  `curve_fit` converging is not the same as succeeding: point the flat model at
  a stretch with no peak in it and amplitude, width and baseline trade against
  each other freely, so it returns a width whose covariance says it is worth
  nothing. On a Ne-lamp HBT run whose lag subtraction left a triangle behind,
  that was `sigma = 39257 +/- 2665891 ps` — drawn as a curve, printed as a
  number, and legended with its own 68x error. Both fitters now reject
  `sigma_err >= sigma` the way they already reject a non-convergence, with the
  `RuntimeWarning` naming the likely cause (the background model).

- **The colour-bar label was cut off every sensor map.** `core.plots.sensor_map`
  draws a square image in a square figure, so the bar sits against the right
  edge of the canvas and its label - which hangs off the *outside* of the bar -
  was drawn a few pixels beyond it. Every hitmap, DCR and cross-talk map written
  since lost its unit. The figure now uses constrained layout, and
  `tests/test_plots.py` asserts the label's bounding box stays inside the
  canvas rather than trusting the eye: an unattended run writes its figures
  once, and nobody is watching to notice.

- **A lagged counts artifact plotted without a subtraction raised
  `IndexError`.** `collect_and_plot_delta_counts(..., subtract_background=False)`
  handed the whole `(n_lags, n_cells)` grid to `rebin_delta_counts`, which
  ravels its input: the planes flattened into one array twice as long as the
  cell centres it was binned against. It now takes lag 0 — the signal — exactly
  as the feather path does by ignoring the `@lag` columns, so an artifact
  written with lags plots either way. Had the mask not caught it, the histogram
  would have been the signal plus a full copy of the accidentals. The rebinner
  also names the mismatch now instead of failing inside a boolean index.

- **The flat-background peak fit failed on background-subtracted data, and said
  nothing.** `fit_gaussian_peak` seeds the flat level from the histogram tail
  median but bounded it at zero from below. A lag-subtracted residual has a
  zero-mean baseline, so that seed lands on or under its own bound, `curve_fit`
  raises `x0 is infeasible`, and the `except (RuntimeError, ValueError)` returned
  `ok=False` — leaving `collect_and_plot_timestamp_differences` to draw a
  histogram with no fitted curve and no explanation. This hit exactly the
  histograms `subtract_background=True` produces, and `'flat'` is the only model
  that combination allows. The level may now go negative, seeds are clamped
  strictly inside their bounds (as the triangle fitter already did), and **a fit
  that does not converge raises a `RuntimeWarning` naming the reason** instead of
  going quiet.
- **A LUT from the wrong clock could be selected even when the right one was
  installed.** Coverage narrowing only set a lower bound, so a 2 MHz table —
  which covers a 10 MHz run five times over — stayed in the running and could be
  applied, converting every code on the wrong scale. The loaders now keep the
  *tightest* covering table.
- **`spad="average"` failed on a real calibration set.** The four SPADs' code
  counts differ by one or two, since whether a SPAD ever hit its last code is
  chance, and `_average_luts` rejected any shape mismatch. Tables agreeing to
  within the coverage tolerance are now padded to the longest with each one's own
  full scale — the honest time for a code that SPAD never produced — while a
  wider disagreement still raises. Averaging across *different calibrations* now
  raises rather than silently blending them.

### Changed

- **BREAKING: the delta-t counts artifact carries its tag** —
  `calculate_and_save_delta_counts` now writes
  `processed/<dataset>_<tag>_delta_counts.npy` (tag lower-cased, as the hitmap
  and DCR arrays), so several programs recorded into one folder (`ORT`, `S0T`,
  ...) no longer overwrite each other. `load_delta_counts` and
  `collect_and_plot_delta_counts` take a matching `tag="ORT"`, and the figures
  pick the tag up too (`<dataset>_<tag>_<label>_delta_counts.png`). An existing
  untagged `<dataset>_delta_counts.npy` is not found by `path` any more: load it
  with `counts_path=`, or rename it.
- **BREAKING: `spad` is gone from both `delta_t` stage-1 functions** — the file
  `tag` already answers it. A timestamping run is read either through one
  micropixel per macropixel (`S0T`...`S3T`), where that SPAD's own LUT is the
  right one, or through all four at once (`ORT`), where the readout cannot say
  which fired and the average is the only honest table. The acquisition program
  writes the tag into the file name, so there was never a second choice to make
  — only a way to state it wrongly. See `delta_t._spad_from_tag`;
  `load_board_lut` and `load_board_compact`, which know nothing about a run,
  still take `spad`.

- **BREAKING: `background` is gone from both stage-2 drivers** — the fit model
  follows the histogram. `collect_and_plot_timestamp_differences` and
  `collect_and_plot_delta_counts` now fit `Gaussian + triangle` on a raw
  histogram and `Gaussian + constant` on a subtracted residual, and take no
  argument for it. The two were never independent choices: either the
  accidental pedestal is still in the data, and on ORT that pedestal is a
  *triangle*, or `subtract_background` has measured it away and what is left is
  flat about zero. Naming the model separately could only restate that or
  contradict it — and the contradiction was the old fixed `'flat'` default,
  which on raw ORT data fits a constant to a slope and inflates sigma. The
  combination that used to be refused at run time (`subtract_background=True`
  with a triangle) is now unsayable, so `_check_subtraction_model` is gone too.

  `'two_gaussians'` — a narrow core on broad tails — is a statement about the
  *peak* rather than the pedestal, so it moves out of the drivers with the
  parameter. Fit it on the `counts` and `centers` they return:
  `fit_two_gaussians_on_triangle(centers, counts, half_base=fit["half_base"])`,
  no disk read. The drivers no longer plot its two components.

- **`collect_and_save_luts` names files by clock frequency, not period** —
  `TDC_LUT_S0T_2MHz.npy`, not `TDC_LUT_S0T_500ns.npy`, so the name reads the way
  the clock is set on the generator. Both spellings are still recognised when
  loading, as is the older `D0_M0_500ns_TDC_LUT_...` layout.

- **BREAKING: the TDC full scale is measured from the data, not stated.** It was
  asked for three ways at once — `open_shutter_time_s`, `full_scale_s`,
  `support_s` — and one of them was the wrong quantity: the open shutter time is
  how long the SPAD waits for a photon, and an 8 µs shutter on a 10 MHz clock
  still yields ~100 ns timestamps, because the free-running oscillator has no
  cycle counter. The largest code a run produced *is* the range its TDC swept, so
  `core.timing.full_scale_from_max_code` resolves it and
  `resolve_tdc_full_scale_s` / `codes_for_full_scale` are gone.
  `open_shutter_time_s` survives only where it was always correct: rate
  normalisation in `dcr_analysis` and `hitmap_analysis`.
- **BREAKING: one clock parameter, in one module.** `tdc_calibration` takes
  `clock_period_s` (was `open_shutter_time_s`), because a density test fixes only
  *relative* per-code widths and something has to set the absolute scale. That is
  the only time the package now has to be told. Its loaders take `min_codes` —
  the code range the data occupies, read off the run by the caller — and check the
  LUT against that rather than against a restated frequency.
- **BREAKING: delta-t artifacts are always in picoseconds.** `time_unit_s` and
  the `delta_code` / raw-code path are gone from `delta_t`: a LUT maps a *code* to
  a time, and a stored difference has lost both operands, so an uncalibrated
  artifact could never have been calibrated afterwards — the documented reason for
  keeping codes was not achievable. Uncalibrated runs are converted with the
  nominal `CODE_S` on the way in. Feathers and counts artifacts written in
  `'code'` are refused with a message rather than read as ps (a silent ~14×
  error).
- **BREAKING: the delta-t histogram covers the whole physical support.**
  `compute_and_save_delta_histogram` no longer takes `plot_window_s`; it
  histograms ±`timing.MAX_DELTA_S` (~586 ns, ~131 kB at one code per bin) and
  reports the largest difference it saw as `support_s`. Every plot window is now a
  slice of one cached histogram, so changing it costs a load rather than another
  pass over the feather.
- **BREAKING: both stage-1 paths share one histogram lattice.**
  `_delta_hist_edges` uses the counts path's construction, so **zero is a bin
  centre whatever the window is**. The old `arange(-W - w/2, ...)` centred a bin
  on zero only when the window was a whole number of bins, and left the outermost
  bin at each end half-filled by the `abs(delta) <= W` cut; the two paths now
  agree in *every* bin, and `_whole_bins` is gone with the workaround it was.
- **BREAKING: parameters that the data can answer, or that were synonyms, are
  gone.** `delta_window_s` / `delta_window_native` (memory knob, default was
  already "keep all"), `support_s`, `full_scale_s`, `time_unit_s`, `n_codes`,
  `snap_to_grid` (the un-snapped path was documented as worse), `flush_every` and
  `wide_rebin` (now module constants — cadence and cosmetics cannot change the
  numbers). Renamed: `grid_s` → `bin_width_s`, `peak_window_s` → `fit_window_s`,
  `cmap_title` → `title`, `folder` → `path`, `file_prefix` → `tag`,
  `from_saved` → `reuse`. `collect_and_plot_timestamp_differences` goes from 17
  parameters to 11, `collect_and_plot_delta_counts` from 16 to 10,
  `calculate_and_save_delta_counts` from 19 to 15, `compute_code_histogram` from
  5 to 3.
- **BREAKING: one firmware vocabulary.** `firmware_version` takes
  `'short_exposure'` / `'long_exposure'` (`core.timing.FIRMWARE_VERSIONS`)
  everywhere. `'short_window'` / `'full_window'` are removed rather than aliased:
  `resolve_cycle_time` accepted both spellings while `resolve_live_time` accepted
  only the second pair, so the two functions disagreed about what was valid.
- `plot_window_s` and `fit_window_s` now default sensibly instead of being
  guessed at: `plot_window_s=None` means the measured full scale, which is what
  the triangle models want, and `fit_window_s` is the peak's half-width for both
  the flat fit and the model-free integration.
- `data_quality.plot_time_code_histogram` drops `time_unit_s` and always plots raw
  codes — the code range is the evidence it exists to show — and reports the span
  as a full scale in ns in the title.
- `compute_code_histogram` allocates the widest code axis the frame format can
  express and trims it to what the data populated, so nothing can overflow it and
  the dropped-codes warning is gone with the failure mode.
- **The short-exposure shutter range was documented far too narrow.**
  `core.timing.FIRMWARE_VERSIONS`, `resolve_live_time`, `hitmap_analysis`, and the
  hitmap and DCR guides all said the exposure inside the 9 µs `short_exposure`
  cycle is "50–500 ns, typically 100 ns". It is settable from ~50 ns up to the
  full 9 µs, and both ends are in use — 100 ns for DCR and crosstalk work, 8 µs
  for the 2026.08.10 Ne 703 nm HBT run, whose acquisition metadata records
  `open_shutter_clks = 1600`. The narrow claim was wrong in two ways that matter:
  it made `open_shutter_time_s` look guessable when a wrong guess is a rate error
  of up to two decades, and it hid the fact that an exposure much longer than the
  TDC full scale wraps the timestamps and dilutes any coincidence peak by
  `open_shutter_time_s / full_scale` (84× for that run: 8 µs over a 95 ns full
  scale). `resolve_cycle_time`'s note on the frame-rate counter dropped the stale
  range too. Released changelog entries are left as they were written.

### Added

- `core.timing.full_scale_from_max_code`, `core.timing.MAX_DELTA_S` and
  `core.timing.FIRMWARE_VERSIONS`.
- `tests/test_timing_and_calibration.py`: the measured full scale against the
  2026-08-07 rate-vs-clock set, the one firmware vocabulary, and LUT coverage
  checked against a run rather than a stated clock.
- `docs/adding_an_analysis.md` now states the shared parameter vocabulary and the
  rule behind it: if the data can answer it, it is not a parameter.

## [0.3.0] - 2026-08-09

### Changed

- **BREAKING: nothing assumes a 10 MHz reference clock any more.** The ring
  oscillator sets the TDC code width (~71.6 ps, `core.timing.CODE_S`) and the
  *reference clock* sets the full scale, so the number of codes scales as 1/f —
  measured on the 2026-08-07 rate-vs-clock set as 275 codes at 50 MHz, 1379 at
  10 MHz, 2761 at 5 MHz and 6914 at 2 MHz, i.e. 1/f to within 2 %. The old
  `_PERIOD_PS = 100_000.0`, `_MAX_CODES = 1536`, `osc_period_ps = 100000.0` and
  `_SUPPORT_PS = 200000.0` constants are gone. Both stage-1 entry points
  (`calculate_and_save_timestamp_differences`, `calculate_and_save_delta_counts`)
  and every `tdc_calibration` entry point now take **`open_shutter_time_s`** —
  one reference-clock period, in seconds — with no default, because the TDC's
  full scale *is* that window and nothing downstream is right without it.
- **BREAKING: every time parameter is in seconds, named with an `_s` suffix.**
  `bin_width_ps` → `bin_width_s`, `plot_window_ps` → `plot_window_s`,
  `fit_window_ps` → `fit_window_s`, `time_unit_ps` → `time_unit_s`,
  `support_ps` → `support_s`, `grid_ps` → `grid_s`,
  `peak_window_ps` → `peak_window_s`, `zoom_window_ps` → `zoom_window_s`,
  `window_ps` → `window_s`, `period_ps` → `open_shutter_time_s`,
  `acq_window` → `open_shutter_time_s`, `exp_time` → `exp_time_s`,
  `frame_time` → `frame_time_s`, `live_time` → `live_time_s`. There is no
  compatibility shim: an old keyword raises `TypeError` naming the new one,
  which is far safer than silently reading `10e3` as 10 000 seconds. Histogram
  bin centres, edges and plot axes stay in picoseconds — that is the scale the
  TDC lives on — and the conversion happens on entry.
- **BREAKING: `osc_period_ps` is replaced by `full_scale_s`** on both plot
  drivers, resolved through `core.timing.resolve_tdc_full_scale_s`: from the
  calibration LUT first (a LUT is self-describing — its largest entry *is* the
  full scale it was built for), then an explicit `full_scale_s`, then
  `open_shutter_time_s` with a warning. That last fallback is only right when
  the shutter is exactly one clock period; a longer shutter folds the timestamps
  into the clock period, which is why the 2026-08-05 run showed a ±100 ns
  triangle behind a 500 ns shutter.
- **BREAKING: `calc_diff.calculate_differences(delta_window=...)` is now
  `delta_window_native=...`**, and it is deliberately *not* in seconds: it is
  compared against `pixel_timestamps`, which are ps with a LUT and raw codes
  without, so the unit is the caller's to know. The drivers take
  `delta_window_s` and convert. Same for
  `background_subtraction.compute_lagged_differences`.
- **BREAKING: renamed keys in returned dicts and `.meta.json` sidecars** —
  `granularity_ps` → `granularity_s`, `cell_ps` → `cell_s`,
  `grid_ps` → `grid_s`, `support_ps` → `support_s`, `window_ps` → `window_s`,
  `bin_width_ps` → `bin_width_s`, and `time_unit_ps`/`bin_width_ps`/
  `plot_window_ps` inside the histogram signature. Every cached
  `*_delta_hist.npy` and `*_delta_counts.npy` is therefore invalidated and will
  re-stream once; that is the intended behaviour, since the values themselves
  changed unit.
- **`_TS_CODE_PS` corrected from 77.0 ps to the measured ~71.58 ps**
  (`core.timing.CODE_S`), and documented as clock-*independent*: 77.0 came from
  dividing 100 ns by a rounded "~1300 codes", and was 7.5 % high. This changes
  the time axis of any uncalibrated (`unit='code'`) histogram.
- LUTs written by `collect_and_save_luts` now carry the window in the file name
  (`TDC_LUT_SPAD0_S0_500ns.npy`) so per-clock calibrations coexist in one
  folder, and record `open_shutter_time_s` / `full_scale_s` in the sidecar.

### Fixed

- **`dcr_analysis` read timestamping data as if it were counting data**, which
  overstated the DCR by 40-50x. Every entry point summed the `photon_counts`
  field, but under the `S*T` / `ORT` programs there is no photon counter —
  `unpack` returns the low bits of the coarse timestamp in that slot (it says
  so, and `hitmap_analysis` already honoured it). On the 2026-08-07 `DCR_S0T`
  run at 24 V this gave a median of 247 kcps, nearly twice the 125 kcps that
  one hit per 8 µs frame physically allows; the correct occupancy reading is
  5.9 kcps. Nothing flagged it, because a map scaled by ~45 still looks like a
  DCR map.
- **A calibration LUT from the wrong reference clock was silently discarding
  most of the data.** `_structure_pixel_timestamps` dropped any code past the
  LUT's range to NaN with no count and no warning — the comment said "treat them
  as invalid rather than indexing out of bounds", which is safe but mute. On the
  2026-08-07 2 MHz data the packaged 10 MHz LUT keeps 21 % of the singles, and
  since a coincidence needs both arms, ~4 % of the coincidences; the surviving
  peak still looks entirely plausible. Now `load_lut` / `load_board_lut` raise at
  load time when the LUT's own full scale does not match `open_shutter_time_s`,
  and `_structure_pixel_timestamps` warns with the discarded fraction, the
  largest code seen and the implied coincidence loss if a mismatched LUT ever
  gets that far.
- **Reusing a saved histogram always reported "0 differences in the window".**
  The message read `meta.get('n', 0)`, but `n` lives at `meta["stats"]["n"]`.
  Cosmetic — the `info` dict two lines later was already correct.

### Added

- **`mode` on every `dcr_analysis` entry point** — `'count'` (default, `S*C` /
  `ORC`) sums the photon counter, `'timestamp'` (`S*T` / `ORT`) counts the
  frames carrying a valid timestamp, matching `hitmap_analysis`. It also
  selects which quadrant files `compute_dcr_64` globs (`S0T`...`S3T` vs
  `S0C`...`S3C`), and is recorded in the `.meta.json` sidecar.
- **A tag/mode cross-check.** `tag='S0T'` with `mode='count'` is now a
  `ValueError` naming the correct mode, rather than a plausible-looking map
  with the wrong numbers on it.
- **`pileup_correct` for timestamp mode.** Timestamping records at most one
  count per frame, so the raw rate saturates at `1 / open_shutter_time_s`
  (125 kcps at an 8 µs shutter) and under-reports well before that — the
  hottest pixel of the 24 V run sits at 60 % of the ceiling and reads 76 kcps
  instead of 117 kcps. `pileup_correct=True` inverts the Poisson censoring,
  `r = -ln(1 - p) / open_shutter_time_s`. Off by default; the drivers print a
  warning naming the ceiling whenever the map gets within 5 % of it.
- `tests/test_dcr_modes.py` — pins the field each mode reads, the tag/mode
  check, and the pile-up inversion against known Poisson rates.
- **`core.timing` is now the home of the ORT timing scale**, alongside the
  live-time helpers: `CODE_S` (the ring's ~71.6 ps LSB, with the measurement it
  comes from), `COARSE_BITS` / `MAX_TS_CODE`, `to_ps` / `to_s`,
  `codes_for_full_scale` and `resolve_tdc_full_scale_s`. It also documents the
  distinction that caused the confusion: the **open shutter time** is the
  photon-sensitive window, the **TDC full scale** is one clock period, and they
  coincide only when the shutter is one period.
- **The reference clock has a floor.** `unpack` builds the code as
  `(coarse - 1) * 8 + (8 - fine)` from a 10-bit coarse field, so it saturates at
  8184 codes ≈ 586 ns: below ~1.7 MHz the counter wraps and the timestamps are
  meaningless. 2 MHz already uses 85 % of it. `codes_for_full_scale` warns when
  a requested full scale exceeds this.
- `tdc_calibration.lut_full_scale_s(lut)` — read a LUT's own full scale back out
  of it, which is the trustworthy source for the triangle half-base.
- `compute_code_histogram` sizes its code axis from `open_shutter_time_s` and its
  overflow warning now names the fraction lost and the largest code seen, rather
  than only the bound it exceeded.

## [0.2.0] - 2026-08-05

### Added

- **`functions.background_subtraction`** — accidental-background removal by
  frame shifting. An SPDC pair is always in one frame, so pairing pixel A of
  frame *i* with pixel B of frame *i+k* samples the accidentals *only*;
  subtracting that from the same-frame histogram cancels the ORT triangle
  bin-for-bin, with no fit model. Five functions and no I/O:
  `compute_lagged_differences`, `frames_per_lag`, `subtract_background`,
  `integrate_residual` (the model-free coincidence count) and
  `plot_background_subtraction` (three panels, the middle one being the
  evidence that the pedestal really was accidental). See
  `docs/guide/background_subtraction.md`.
- **`subtract_background` on all four delta-t entry points**, feather path and
  counts path alike. Stage 1 (`calculate_and_save_delta_counts`,
  `calculate_and_save_timestamp_differences`) also histograms/records the
  frame-shifted pairs; stage 2 (`collect_and_plot_delta_counts`,
  `collect_and_plot_timestamp_differences`) fits and plots the *residual*
  instead of the raw histogram, writes a three-panel diagnostic beside it, and
  reports `fit["model_free"]` — a coincidence count with no peak or background
  model in it. Default False, which leaves every artifact, resume signature and
  figure exactly as they were. Only valid with `background='flat'`: there is no
  triangle left to fit, and that is checked before the expensive pass rather
  than after it.

  The two stage-1 paths produce **bit-identical residuals** (tested, and
  verified on a real 12-file ORT run), so the choice is only about cost: the
  counts artifact grows one grid plane per lag (0.5 MB → 4.6 MB on a 1000-file
  run), the feather grows one column set per lag (1 GB → ~9 GB). Lag 0 stays the
  plain measurement in both — plane 0 of the grid, and the bare `"r,c-r,c"`
  columns of the feather, where the shifted ones are `"<pair>@lag<k>"` and are
  never pooled into the coincidence histogram. `compute_and_save_delta_histogram`
  gained the parameter too and histograms every lag in one streaming pass.
- **`core.pairs`** — `pair_list`, `pair_labels` and `pair_label`, the pixel-pair
  enumeration and the `"ra,ca-rb,cb"` label format that every coincidence
  artifact is indexed by. `delta_t` had it twice (once inline in the feather
  path, once as `_pair_labels`); both now call this.
- **`core.timing.resolve_cycle_time`** — the length of one acquisition cycle,
  from the firmware setting rather than from a file on disk. Two versions:
  `'short_exposure'` (the cycle is always 9 µs; the photon-sensitive exposure
  inside it is 50–500 ns, typically 100 ns) and `'long_exposure'` (the caller
  sets X and the firmware adds 9 µs, so the cycle is `exp_time + 9 µs`). The
  older `'short_window'` / `'full_window'` names are accepted as aliases.

### Changed

- **The hitmap rate is now a photon rate per wall-clock second, in Hz, for both
  modes**: `hits / (nframes * n_files * cycle)`. Every frame is one acquisition
  cycle, so that product is the elapsed time of the run. `count` mode no longer
  divides by the exposure window (which gave counts-per-live-second, a number
  ~90x larger), and `timestamp` mode no longer resolves the period from
  `frame_rate_cnt.txt`. `compute_and_save_hitmap` / `compute_and_save_hitmap_64`
  take `firmware_version` (`'short_exposure'` default, or `'long_exposure'` with
  `exp_time`) and record the resolved cycle in the sidecar; the figure is now
  produced unconditionally, since `nframes` is required and the cycle always
  resolves. `acq_window` is renamed `exposure_window` and is recorded and
  reported as a duty cycle only — it does not enter the rate. The in-exposure
  rate is the plotted figure times `cycle / exposure`, and that factor is
  printed.
- **`frame_rate_cnt.txt` is no longer read anywhere.** It is written by the
  acquisition exe, which is not ours, so no analysis result may depend on it.
  `core.timing.resolve_frame_time` no longer opens it — the frame period now
  comes from `exp_time + 9 µs` or the 9 µs free-running base — and with the file
  read gone, `resolve_frame_time` no longer needs a `folder` argument and
  `resolve_live_time` no longer needs one either (both signatures changed;
  `dcr_analysis` updated accordingly). The file itself is left on disk
  untouched.

  For the record, the counter is *self-consistent*, and an earlier draft of this
  entry wrongly called it unreliable on the grounds that several folders share a
  value. They share it because they share settings. It is a 200 MHz tick count
  over the whole run and every value on disk fits
  `ticks = nframes * N + 138 + window/5ns` exactly, which independently confirms
  `nframes = 10 000`, reports the exposure register back (`(ticks mod nframes)
  − 138 = window / 5 ns`, matching the folder names), and flags runs that died
  early. It also says the **real cycle is 9.685–9.770 µs, not 9.000 µs**, so the
  nominal normalisation used here runs ~7.8 % high — a deliberate,
  documented choice rather than a measurement. Note the sample scripts
  `extract_DCR.m`, `extract_dcr.py` and `dcr_hitmap.py` read it as ticks *per
  frame* with no division, which is wrong by a factor of `nframes`.

- **`nframes` is now required wherever `.bin` files are unpacked.** It was
  optional, and `None` meant "derive it from the file size". A frame count is
  an acquisition setting, not a property of a file that happens to be on disk:
  deriving it silently turns a truncated or still-being-written file into a
  different (wrong) normalisation, and every rate the package reports divides
  by it. It is now a required argument on `hitmap_analysis.compute_hitmap` /
  `compute_hitmap_64` / `compute_and_save_hitmap` / `compute_and_save_hitmap_64`
  (second positional, matching `dcr_analysis`), on
  `crosstalk_analysis.compute_crosstalk` / `compute_and_save_crosstalk` /
  `collect_and_plot_crosstalk` / `combine_directional_crosstalk`, and as a
  required keyword on `core.reduce.accumulate_frames`,
  `data_quality.collect_time_mat` / `plot_time_code_histogram`,
  `tdc_calibration.compute_code_histogram` / `compute_tdc_lut` /
  `collect_and_save_luts` and
  `delta_t.calculate_and_save_timestamp_differences` /
  `calculate_and_save_delta_counts`. `core.io.frames_in_file`
  is unchanged and still there for the case where a file's size really is the
  only record left — it is now the caller's explicit choice rather than a
  silent default, and its docstring warns that the relation between the frame
  count an acquisition is asked for and the size of the '.bin' it produces is
  not established (every sample file is a whole power-of-two number of frames,
  which a 10 000-frame run should not produce). `hitmap_analysis.collect_and_plot_hitmap` no longer recovers
  a missing frame count from the `.bin` sizes either: a stage-1 `.npy` whose
  sidecar has no `total_frames` gets the counts figure but no rate figure,
  rather than a rate normalised by a guess. Recompute it to get one.
- The `nframes` used is part of the resume signature for the checkpointed
  counts pass (`delta_t.calculate_and_save_delta_counts`), replacing `valid_min`
  there. It decides how much of each file is read, so a checkpoint written with
  one value must not be continued with another. Existing partial artifacts will
  not resume across this change; they restart rather than mixing.

### Removed

- **`valid_min`.** Validity of an ORT timestamp is `time_series > 0` — `unpack`
  leaves non-fired slots at `<= 0` — so this was a knob whose only correct
  value was its default, and a wrong one silently drops or invents photons. The
  threshold is now written into the code at each site, with the reason. Dropped
  from `hitmap_analysis`, `data_quality.collect_time_mat`, `tdc_calibration`
  and `delta_t` (both stage-1 paths).
- **`part_size_mb` and `keep_parts`** on
  `delta_t.calculate_and_save_timestamp_differences`. The part cap is now the
  module constant `_PART_MB` (still 64 MB): sharding is a memory strategy that
  cannot change the numbers, so no call site needs to tune it. The parts are
  now always kept. They are the crash insurance, and the read side streams a
  part folder exactly as it streams the combined feather — a histogram over
  fixed edges is a sum over record batches, and a sum does not care which file
  a batch came from — so working from the parts gives identical counts while
  skipping the combine pass, which reads and rewrites every byte of the run.
  Delete the folder by hand once the combined feather has been checked.

- **`TODO.md`**, beside this file and published as *Open items* — the questions
  that have deliberately **not** been decided, with the trade-off and what
  evidence would settle each. This file records what was decided; that one
  records what was left open, so a known gap cannot read as an oversight. It
  opens with the delta-t feather keeping both its parts and its combined copy,
  which doubles the disk a finished run occupies.

### Fixed

- The README's quick start passed `acq_window=1000` to `dcr_analysis`, which
  takes **seconds** — a live time seven orders of magnitude too long, and so a
  DCR quietly reported as ~1e-7 of the truth. It now reads `100e-9`, with the
  unit spelled out. Nothing in the package changed; the example was wrong.
- The changelog's link block had no `[0.1.4]` entry and still pointed
  `[Unreleased]` at `v0.1.3`.

## [0.1.4] - 2026-08-02

### Added

- **A second stage 1 that writes no feather at all.**
  `delta_t.calculate_and_save_delta_counts` runs the same acquisition loop but
  bins the differences onto a fixed native grid as it finds them, saving only
  the counts to `processed/<name>_delta_counts.npy`. The artifact is sized by
  the grid rather than the data: 106 MB for 256 pairs whether the run is 100
  files or 10 000, against a 3.22 GB feather for the same run and ~135 GB for
  a long one. Every pair is then plottable at once, and `bin_width_ps` /
  `plot_window_ps` stop being compute-time decisions. For raw codes it is
  exactly lossless — `delta_code` is an integer, so counts-per-code is a
  re-encoding. It is offered *beside* the feather path, not instead of it: no
  per-event data survives, so re-calibrating with a different LUT, re-cutting
  `delta_window` and anything needing individual differences are all gone.
- `delta_t.load_delta_counts` and `delta_t.rebin_delta_counts` read the grid
  back and reduce it onto any coarser histogram in memory, and
  `delta_t.collect_and_plot_delta_counts` is the matching plot driver. It
  returns `(counts, centers, fit)` and shares its fitting and drawing code
  with `collect_and_plot_timestamp_differences`, so a difference between the
  two paths can only come from the histogram.
- Crash insurance for the counts path is one file rather than a part folder:
  the grid is rewritten every `flush_every` files and the next call resumes.
  Two runs cannot interleave, so the part-mixing hazard does not arise. The
  sidecar carries `total_in_grid`, and a checkpoint that does not sum to it —
  a crash between writing the array and writing the sidecar — is discarded
  rather than resumed, which would otherwise re-count files.

### Fixed

- **The "streaming" feather read was not streaming a single-batch feather.**
  `pa.ipc.open_file(path)` materialises a whole record batch the moment one is
  asked for. That is invisible on a feather assembled from 64 MB parts, but
  every feather written before the parts rewrite - and anything
  `_write_delta_feather` writes directly - is a *single* batch, so the read
  loaded the entire file: measured at +794 MB for a 0.79 GB feather, meaning a
  9.42 GB one asked for 9.42 GB in one allocation, on a 16.8 GB machine. Every
  batch-reading site now memory-maps instead, so `get_batch` costs nothing and
  the OS pages in only the columns touched. The same 9.42 GB, 320-column
  feather now histograms in 18 s with **no** resident growth.
- **Re-binning onto an incommensurate width put a sawtooth in the histogram.**
  A 71 ps bin over a 7.7 ps grid holds 9 or 10 cells depending on where it
  lands, so a smooth distribution acquires an ~11% bin-to-bin ripple; on real
  data it moved the peak bin by 8%. `collect_and_plot_delta_counts` now snaps
  the width to an **odd** whole number of cells and the window to a whole
  number of bins. Odd matters: bins are centred on zero, so their edges lie at
  half-integer multiples of the width, which for an odd cell count fall on
  cell boundaries and for an even one fall through cell centres — splitting a
  cell at every boundary. Verified bin-for-bin on a 3.22 GB run: zero
  difference from the feather path at 9, 11 and 13 cells per bin, ~1% at 8, 10
  and 12. Snapping also puts zero exactly on a bin centre, which the feather
  path's `arange(-W - w/2, ...)` edges only manage when the window happens to
  be a whole multiple of the width.

### Known

- The feather path's outermost bin at each end is under-filled: it cuts data
  at `abs(delta) <= plot_window_ps` but draws edges half a bin beyond it. The
  counts path fills those bins properly, which is most of the ~1% difference
  in fitted `sigma` between the two. Not changed here — it would alter every
  existing feather-path result.

## [0.1.3] - 2026-08-02

### Fixed

- **Delta-t parts from two runs could be pooled into one feather.** Part
  numbering restarts at 0 on every run, so a run that died at part 300 and was
  restarted for 120 parts had its first 120 overwritten and its last 180 left
  in place — and `combine_delta_t_parts` globbed the whole folder, pooling all
  300. The combined feather then counted a slice of the run twice, inflating
  both the file and the coincidence total, and skewing the peak against the
  accidental background. Each run now tags its parts with its own id and
  records them in a manifest (rewritten on every flush, so a dead run is still
  described), `combine_delta_t_parts` refuses to pool several runs and takes
  `run=` to pick one, and
  `calculate_and_save_timestamp_differences` refuses to start while another
  run's parts are in the folder rather than interleaving with them. Nothing is
  deleted on that refusal — the parts are the crash insurance.

### Changed

- Delta-t parts are now named `<name>_delta_t_<run>_<i>.feather` and are
  accompanied by a `<name>_delta_t_<run>_manifest.json`. Folders written
  before this keep working: their untagged parts read as a single `legacy`
  run.

## [0.1.2] - 2026-08-02

### Added

- `delta_t.compute_and_save_delta_histogram` — streams a delta-t feather down
  to its histogram one Arrow record batch at a time, so a combined feather far
  larger than RAM (a 10 000-file run reaches ~135 GB) can be read at all. The
  counts are saved to `processed/<name>_delta_hist.npy` with a `.meta.json`
  sidecar recording the binning and a fingerprint of the sources, so replotting
  or changing the background model costs a small load instead of another full
  pass. The sidecar also reports what share of the cells read were padding.
- `feather_path` may now be a `*_delta_t_parts` folder, and a run whose
  combined feather is missing falls back to its parts automatically — the
  giant combined file is no longer needed to plot.

### Changed

- `delta_t.collect_and_plot_timestamp_differences` now returns
  `(counts, centers, fit)` instead of `(deltas_ps, fit)`, and never
  materialises the differences. **Breaking**: returning the pooled array was
  the reason the function could not open a full-size feather. The histogram is
  bin-for-bin identical to what the old load-everything path produced — both
  fit models and the plot only ever consumed `(bin_centers, counts)`.
- Promoting `[Unreleased]` to a numbered release is a **manual edit** of
  `CHANGELOG.md`, documented step by step in the README. Nothing has ever
  written the changelog automatically; the removed script only did that edit
  for you.
- `tools/changelog.py` is now usable from a Jupyter interactive window, so the
  release notes can be previewed without a terminal. Running the file there
  defines its functions instead of firing the CLI — `__name__` really is
  `"__main__"` in a kernel, so argparse used to parse ipykernel's own argv and
  die with `SystemExit: 2`. `main()` also takes an explicit `argv`, and
  `notes_for` raises `ChangelogError` instead of calling `sys.exit`. Only
  `main()` converts that to a non-zero exit, so the gate `publish.yml` relies
  on is unchanged.
- The "no section for this version" failure now spells out the heading and
  link edits to make, and warns that deleting a release leaves its tag behind.
  That message is what the person cutting a failed release actually reads.

### Removed

- `tools/release.py` and its tests. It automated one two-line edit to
  `CHANGELOG.md` and was the only part of the release process that wanted a
  terminal. `tools/changelog.py` stays: `publish.yml` runs it **on the
  runner** to refuse a tag whose version has no changelog section, and to fill
  the release body.

### Fixed

- The changelog's link block still pointed `[Unreleased]` at a `v0.2.0` that
  was never tagged, carried a `[0.2.0]` link with no matching section, and had
  no `[0.1.1]` entry.

## [0.1.1] - 2026-08-02

### Added

- `core.store.confirm_rewrite` — the guard every `rewrite=True` now routes
  through. Silent on a fresh run; when data really is about to be destroyed it
  names each target with its size and timestamp, then counts down
  `store.REWRITE_DELAY_S` (5 s) so Ctrl-C can still save an overnight
  acquisition. `delta_t`'s two `rewrite` parameters use it.
- `delta_t.combine_delta_t_parts` — assemble a run's part files into its
  combined feather. Recovers a run that died partway through: every part
  flushed before the crash is complete and readable.
- `delta_t.calculate_and_save_timestamp_differences` gained `part_size_mb`
  (default 64 MB) and `keep_parts` (default True).
- `tools/release.py` — promotes the changelog's `[Unreleased]` section to a
  numbered, dated, linked release. The publish workflow reads the changelog as
  it was *at the tagged commit*, so notes left under `[Unreleased]` can never
  ship; forgetting to rename the section meant deleting and recreating the tag.

### Changed

- **Delta-t stage 1 streams to disk instead of accumulating the whole run.**
  Differences are flushed to
  `processed/<dataset>_delta_t_parts/<dataset>_delta_t_<i>.feather` as they are
  found and concatenated one Arrow record batch at a time, so the loop's memory
  ceiling is one part plus one unpacked file regardless of file count. Over
  ~10 000 files the old path failed twice over — the buffers exhausted RAM, and
  the single final write then asked pyarrow for a multi-GB contiguous
  allocation on top of what was already held.
- `combine_delta_t_feathers` pools batch-by-batch rather than reading every
  feather into a DataFrame first, so combining many runs no longer needs the
  whole pooled data set resident twice.
- Delta-t feathers pad short pair columns with Arrow nulls rather than NaN
  values. Both read back as NaN through `to_pandas`, so this is invisible
  downstream and older feathers still read correctly; it just keeps padding
  distinguishable from data when parts are counted.

## [0.1.0] - 2026-08-01

The package grows from "unpack and plot DCR" into the full Kelpie analysis
set, on a shared three-stage structure: **raw `.bin` -> `processed/` ->
`results/`**. Stage 2 never re-reads raw data, so retrying a colormap costs a
load instead of a full unpack.

### Added

- `dapkel.core`, the shared substrate every analysis is built from:
    - `core.io` - locate and size the raw `.bin` files (`find_bin_files`,
      `frames_in_file`, `BYTES_PER_FRAME`).
    - `core.timing` - frame period and photon-sensitive live time
      (`resolve_frame_time`, `resolve_live_time`, `CLK_PERIOD`,
      `FREE_RUNNING_US`).
    - `core.reduce` - fold files into one sensor map (`accumulate_frames`) and
      interleave four quadrants into the full sensor (`assemble_64`).
    - `core.plots` - the two figure shapes the package draws (`sensor_map`,
      `sorted_distribution`) plus NaN-aware `map_stats`.
    - `core.store` - the single place that decides where artifacts land
      (`processed_dir`, `results_dir`, `save_map`, `load_map`, `save_figure`).
- `crosstalk_analysis` - per-direction and combined optical cross-talk maps.
- `hitmap_analysis` - occupancy and photon-rate maps, per quadrant and full
  sensor.
- `tdc_calibration` - statistical code density test producing per-pixel TDC
  lookup tables, with DNL diagnostics. Board LUTs ship with the wheel under
  `dapkel.params` and load through `load_board_lut`.
- `calc_diff` and `delta_t` - timestamp differences between pixel groups
  (`1v1` and `all_pairs`), written to Feather, with Gaussian and
  Gaussian-plus-triangle peak fits for jitter.
- `data_quality` - run this on a fresh acquisition. A TDC that never stopped
  produces data that looks entirely normal to every other analysis;
  `plot_time_code_histogram` makes that failure visible.
- Save-and-reload paths for every analysis, so figures can be redrawn from
  `processed/` without touching the raw files.
- Documentation site (mkdocs-material + mkdocstrings) under `docs/`, with a
  guide page per analysis carrying the physics derivations, and API pages
  generated from the docstrings.
- Test suite: structural guards on the public API surface (`__all__`
  completeness, no duplicated helpers across analyses, no cross-module private
  imports, no hand-built artifact paths) and geometry tests tying the SPAD
  layout, cross-talk directions and quadrant interleave together.
- `dapkel.__version__`, a `py.typed` marker (PEP 561), and ruff configuration.
- CI: tests and lint on Linux and Windows across Python 3.10 and 3.13, a
  strict docs build, and GitHub Pages deployment.
- Automated releases. Publishing a GitHub release validates the tag, refuses to
  ship a version with no changelog entry, checks that the built version matches
  the tag, uploads to PyPI, and fills the release body from `CHANGELOG.md`.
  `tools/changelog.py` renders those notes locally before you tag.

### Changed

- **Artifact layout.** Stage-1 data now goes to `processed/` and figures to
  `results/<kind>/`, replacing the ad-hoc `senpop_data/`, `delta_ts_data/` and
  `results/tdc_calibration/` folders. Existing data is not migrated
  automatically - move it by hand if you want old runs picked up.
- `unpack` is roughly 6x faster: packed pixel fields are extracted with shifts
  and masks instead of exploding every bit into a large `uint8` array. Output
  is bit-for-bit identical to the previous implementation and to the MATLAB
  reference.
- `matplotlib` is no longer pinned to an exact version (`==3.10.8` ->
  `>=3.8`), which was making dapkel uninstallable alongside anything else that
  needed a different matplotlib.
- Docstrings are shorter at the call site; the physics derivations moved to the
  documentation site. Full `Parameters` / `Returns` sections are kept, so
  `help()` still answers "what does this argument take".
- Figure sizing follows one rule: square sensor maps are 16x16, everything else
  takes `figure.figsize` from the active style.

### Fixed

- **Micropixel geometry.** SPADs are indexed clockwise inside a 2x2
  macropixel (`S0 S1 / S3 S2`), not row-major. The full-sensor interleave in
  `dcr_analysis` had `S2C` and `S3C` swapped, so quadrant data landed on the
  wrong cells of every `(64, 64)` map. Cross-talk was affected twice over: the
  vertical and diagonal measurements were both mislabelled and placed on each
  other's cells.
- `map_stats` and the heatmaps are NaN-aware. A single non-finite cell used to
  drive `vmax` to NaN, collapsing a whole sensor map to two colours and
  printing `median nan` in the title.
- `delta_t` derived its figure directory one level too shallow, nesting
  `results/` inside the data folder. Both `delta_t` and `hitmap_analysis` now
  route through `core.store`.
- `requirements.txt` was missing `komorebi_mpl` and pinned versions that
  contradicted `pyproject.toml`.

## [0.0.1] - 2026-07-05

Initial commit.

### Added

- Functions for unpacking raw, binary data and analyzing dark count rate.

[Unreleased]: https://github.com/rngKomorebi/dapkel/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rngKomorebi/dapkel/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/rngKomorebi/dapkel/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/rngKomorebi/dapkel/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/rngKomorebi/dapkel/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/rngKomorebi/dapkel/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rngKomorebi/dapkel/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/rngKomorebi/dapkel/releases/tag/v0.0.1
