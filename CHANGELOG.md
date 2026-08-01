# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/rngKomorebi/dapkel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rngKomorebi/dapkel/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/rngKomorebi/dapkel/releases/tag/v0.0.1
