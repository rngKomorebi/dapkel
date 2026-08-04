# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
