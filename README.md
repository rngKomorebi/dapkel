# Data Analysis Package for KELpie (DAPKEL)

[![Tests](https://github.com/rngKomorebi/dapkel/actions/workflows/tests.yml/badge.svg)](https://github.com/rngKomorebi/dapkel/actions/workflows/tests.yml)
[![Docs](https://github.com/rngKomorebi/dapkel/actions/workflows/docs.yml/badge.svg)](https://rngkomorebi.github.io/dapkel/)
[![PyPI - Version](https://img.shields.io/pypi/v/dapkel)](https://pypi.org/project/dapkel/)
[![PyPI - License](https://img.shields.io/pypi/l/dapkel)](LICENSE)

Unpacking and analysis of the binary data written by the Kelpie SPAD camera:
dark count rate, optical cross-talk, hitmaps, TDC calibration and photon
coincidences.

**Documentation: <https://rngkomorebi.github.io/dapkel/>**

## The detector

The Kelpie detector was developed at EPFL by Dr. Tommaso Milanese. It features
a 64x64 Single-Photon Avalanche Diode (SPAD) sensor built from 2x2 macropixels.
It is fully reprogrammable, with high PDE across the whole visible spectrum
peaking at 780 nm, 40 ps (rms) jitter, low dark count rate and reasonable
cross-talk.

This package was derived from the original MATLAB functions written by
Dr. Milanese for offline unpacking and analysis of the detector's data.

## Installation

A fresh virtual environment is recommended
([how](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)).

```bash
pip install dapkel
```

To work on the package itself, clone the repo and install it in editable mode
with the test and documentation extras:

```bash
git clone https://github.com/rngKomorebi/dapkel.git
cd dapkel
pip install -e ".[dev,docs]"
```

`requirements.txt` lists the runtime dependencies alone, for the
`pip install -r requirements.txt` path; `pyproject.toml` is the source of truth.

## How it works

Every analysis is the same pipeline with a different reduction in the middle:

```
folder of .bin  ->  discover files  ->  fold frames into a sensor map
                ->  scale to physical units  ->  save
                ->  load  ->  plot
```

| stage | what it does | on disk |
|---|---|---|
| **0. raw** | the camera writes `.bin`; dapkel decodes it | `*.bin` (read-only) |
| **1. process** | fold frames into an array, scale to units, save | `processed/` |
| **2. plot** | load the saved array, render figures | `results/<kind>/` |

The rule that keeps this honest: **`processed/` holds data, `results/` holds
pictures, and stage 2 never re-unpacks `.bin` files.** Retrying a colormap
costs a load, not a re-read of the raw data.

## Quick start

> On a fresh acquisition, run the data-quality check first. A TDC that never
> stopped produces data that looks completely normal to every other analysis in
> this package.

```python
from dapkel.functions import data_quality, dcr_analysis

# 0. Did the TDC actually record timing? Codes should spread over 0..~1300,
#    not collapse onto a handful of values.
data_quality.plot_time_code_histogram("path/to/data", pixel=(16, 16))

# 1 + 2. Compute the full-sensor DCR map and save both figures.
dcr_analysis.collect_and_plot_dcr_64("path/to/data", nframes=1000, acq_window=1000)
```

Stages 1 and 2 are also separable, so a saved map can be re-plotted without
re-reading the raw files:

```python
from dapkel.core import store
from dapkel.functions import dcr_analysis

dcr_analysis.compute_and_save_dcr_64("path/to/data", 1000, acq_window=1000)

dcr, meta = store.load_map("path/to/data", kind="dcr", tag="64")
fig = dcr_analysis.plot_heatmap(dcr)
```

Each driver can also skip straight to stage 2 from what is already in
`processed/`: `dcr_analysis` and `crosstalk_analysis` take `from_saved=True`,
`hitmap_analysis` takes `npy_path=`, `delta_t` takes `feather_path=`, and TDC
lookup tables come back through `tdc_calibration.load_lut`.

## What's in the package

`dapkel.functions` holds one module per measurement. Each declares its public
surface in `__all__` - read that list to see what you can call.

| module | what it measures |
|---|---|
| [`unpack`](https://rngkomorebi.github.io/dapkel/guide/unpack/) | decodes the raw `.bin` files; the shared foundation |
| [`data_quality`](https://rngkomorebi.github.io/dapkel/guide/data_quality/) | is this acquisition usable at all? |
| [`dcr_analysis`](https://rngkomorebi.github.io/dapkel/guide/dcr/) | per-pixel dark count rate, in cps |
| [`crosstalk_analysis`](https://rngkomorebi.github.io/dapkel/guide/crosstalk/) | optical cross-talk, per direction and combined |
| [`hitmap_analysis`](https://rngkomorebi.github.io/dapkel/guide/hitmap/) | occupancy and photon rate |
| [`tdc_calibration`](https://rngkomorebi.github.io/dapkel/guide/tdc_calibration/) | per-pixel TDC LUTs from the code density test |
| [`calc_diff`, `delta_t`](https://rngkomorebi.github.io/dapkel/guide/coincidences/) | timestamp differences, coincidences and jitter |

Anything shared between two analyses lives in
[`dapkel.core`](https://rngkomorebi.github.io/dapkel/api/core/) rather than
being copied - a test fails if a function name is defined in two analysis
modules.

A standalone app for starting acquisitions and plotting the hitmap in real
time is at [dapkel-rtp](https://github.com/rngKomorebi/dapkel-rtp).

## Plot styling

Importing `dapkel` applies its matplotlib house style through `komorebi_mpl`,
so plots look consistent out of the box. Your choice always wins - the plotting
functions never touch `rcParams` themselves, so whatever style is active when
they draw is what you get:

```python
import dapkel                     # applies the default style on import
import komorebi_mpl

komorebi_mpl.use("night_wave")    # or any registered style, or "default"
```

## Development

```bash
pytest              # test suite, including the API-surface guards
ruff check .        # lint
mkdocs serve        # preview the documentation at localhost:8000
```

### Releasing

There is no version number to bump anywhere: `setuptools_scm` derives it from
the git tag, so **creating the tag is the version bump**.

1. Add a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md` and merge it into
   `main`. Preview what the release will say with
   `python tools/changelog.py X.Y.Z`.
2. On GitHub, *Releases -> Draft a new release*, create the tag `vX.Y.Z` there
   and publish.

Publishing runs `publish.yml`, which validates the tag, refuses to ship a
version with no changelog entry, runs the tests, checks the built version
matches the tag, uploads to PyPI, and rewrites the release body from the
changelog. PyPI versions cannot be reused, so a bad release costs the next
number.

### Contributing

`main` is the release branch: tested, and what PyPI is cut from. Work on a
feature branch and open a pull request against it. Contributions are welcome;
please follow [PEP 8](https://peps.python.org/pep-0008/) and
[PEP 257](https://peps.python.org/pep-0257/). If you are adding a new
measurement, start from
[Adding an analysis](https://rngkomorebi.github.io/dapkel/adding_an_analysis/) -
it describes the contract the tests enforce.

## License and contact

MIT - see [LICENSE](LICENSE). To get in touch, write to
sergei.kulkov23@gmail.com.
