# archive

Code retired from `dapkel` in the module restructure (branch `modules-restructure`,
2026-10-31), kept for reference. Nothing here is packaged, imported, linted or
tested, and none of it is maintained: every file is copied verbatim from commit
`6b49a4b` and still refers to names that no longer exist in `src/`. To bring a piece
back, port it into the live module rather than importing it from here.

The layout mirrors the package:

| path | what | superseded by |
| --- | --- | --- |
| `functions/delta_t.py` | the feather stage-1 path (every difference to a `.feather`, streamed into a histogram, part files and manifests), the two-Gaussian peak model, the dense-table calibration branch | `delta_t.process_delta_t` / `plot_delta_t` (binned counts) |
| `functions/tdc_calibration.py` | the dense per-clock code→ps LUT and its selection by clock tag and code coverage | the compact 8-tap calibration, `core.calibration` |
| `functions/calc_diff.py` | within-frame differencing (whole module) | lag 0 of `delta_t`'s frame-shifted differencing |
| `functions/hitmap_analysis.py`, `dcr_analysis.py`, `crosstalk_analysis.py`, `data_quality.py` | compute-only (no save) entry points and the separate 32/64 variants | each module's `process_*` / `plot_*` |
| `core/timing.py`, `core/pairs.py`, `core/io.py` | `resolve_cycle_time`, `resolve_frame_time`, `CLK_PERIOD`, `MAX_DELTA_S`, `pair_labels`, `frames_in_file` | — (unused) |
| `tests/` | the tests of the above, including the counts-vs-feather bin-for-bin comparison and the dense-vs-compact calibration comparison | — |
| `calibration_data/` | the packaged per-clock dense LUTs (~160 MB). **Untracked** (`.gitignore`), on disk only | the compact tables in `src/dapkel/params/calibration_data` |

The full pre-restructure package is also in git history: `git show 6b49a4b:<path>`.
