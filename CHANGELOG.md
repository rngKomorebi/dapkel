# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[Unreleased]: https://github.com/rngKomorebi/dapkel/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/rngKomorebi/dapkel/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rngKomorebi/dapkel/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/rngKomorebi/dapkel/releases/tag/v0.0.1
