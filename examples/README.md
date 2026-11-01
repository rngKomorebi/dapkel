# Examples

Worked examples of using dapkel, as Jupyter notebooks — one per analysis, each
starting from a folder of raw `.bin` files. The paths and acquisition settings
in the first code cell are the only thing a reader has to change.

| notebook | analysis | data it needs |
|---|---|---|
| [01_data_quality](01_data_quality.ipynb) | did the TDC record timing? — run first | any timestamping run (`ORT`, `S0T`...) |
| [02_tdc_calibration](02_tdc_calibration.ipynb) | per-pixel 8-tap TDC calibration | code density test (`S0T`...`S3T`) |
| [03_dcr](03_dcr.ipynb) | dark count rate, in cps | dark run (`S0C`, `ORC`, `S0T`...) |
| [04_crosstalk](04_crosstalk.ipynb) | optical cross-talk per direction | coinc/OR runs, one folder per pairing |
| [05_hitmap](05_hitmap.ipynb) | occupancy and photon rate | any illuminated run |
| [06_coincidences](06_coincidences.ipynb) | coincidence peak, timing jitter, background subtraction | timestamping run with correlated photons (SPDC) |

The `plot_*` functions save their figures under `results/` and close them; the
notebooks that would otherwise show nothing display the saved PNGs inline.

What does not belong here: bench scripts with hardcoded paths, or functions
pasted inline to be stepped through. Those are development scratch and live
outside the repo.

The [documentation](https://rngkomorebi.github.io/dapkel/) explains each
analysis in full, and the [quick start](../README.md#quick-start) is the
shortest path from a folder of `.bin` files to a figure.
