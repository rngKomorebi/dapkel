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

Python 3.10 to 3.15; the test suite runs on every one of them, on Windows and
Ubuntu, on every push.

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

Every analysis is two calls: `process_<x>(path, nframes, ...)` unpacks the raw
`.bin` files, **always saves** the result under `processed/` and returns the
saved path; `plot_<x>(path, ...)` loads that and writes the figures to
`results/<kind>/`.

> On a fresh acquisition, run the data-quality check first. A TDC that never
> stopped produces data that looks completely normal to every other analysis in
> this package.

```python
from dapkel.functions import data_quality, dcr_analysis

# 0. Did the TDC actually record timing? Codes should spread over the whole
#    range for the clock in use (~1397 codes at 10 MHz, ~6985 at 2 MHz),
#    not collapse onto a handful of values.
data_quality.process_time_codes("path/to/data", nframes=10_000, max_files=5)
data_quality.plot_time_codes("path/to/data", pixel=(16, 16))

# 1. Compute the DCR map of one SPAD quadrant and save it to processed/.
#    'open_shutter_time_s' is the photon-sensitive part of each frame, in
#    SECONDS. It sets the rate normalisation and nothing else - it is not
#    the TDC full scale, which every analysis measures from the data.
dcr_analysis.process_dcr(
    "path/to/data", 1000, "S0C", open_shutter_time_s=100e-9
)

# 2. Load it and save the heatmap and sorted distribution to results/dcr/.
dcr = dcr_analysis.plot_dcr("path/to/data", "S0C")
```

Stage 2 only ever reads what stage 1 saved, so re-plotting - another colormap,
another bin width - never re-reads the raw files. The saved array is also
reachable directly:

```python
from dapkel.core import store

dcr, meta = store.load_map("path/to/data", kind="dcr", tag="S0C")
```

## What's in the package

`dapkel.functions` holds one module per measurement. Each has two
public functions, `process_<x>` and `plot_<x>`, declared in its `__all__`.

| module | what it measures |
|---|---|
| [`unpack`](https://rngkomorebi.github.io/dapkel/guide/unpack/) | decodes the raw `.bin` files; the shared foundation |
| [`data_quality`](https://rngkomorebi.github.io/dapkel/guide/data_quality/) | is this acquisition usable at all? |
| [`dcr_analysis`](https://rngkomorebi.github.io/dapkel/guide/dcr/) | per-pixel dark count rate, in cps |
| [`crosstalk_analysis`](https://rngkomorebi.github.io/dapkel/guide/crosstalk/) | optical cross-talk, per direction and combined |
| [`hitmap_analysis`](https://rngkomorebi.github.io/dapkel/guide/hitmap/) | occupancy and photon rate |
| [`tdc_calibration`](https://rngkomorebi.github.io/dapkel/guide/tdc_calibration/) | per-pixel 8-tap TDC calibration from the code density test |
| [`coincidences`](https://rngkomorebi.github.io/dapkel/guide/coincidences/) | timestamp differences, coincidences and jitter, with optional [frame-shift background subtraction](https://rngkomorebi.github.io/dapkel/guide/background_subtraction/) |

Anything shared between two analyses lives in
[`dapkel.core`](https://rngkomorebi.github.io/dapkel/api/core/) rather than
being copied - a test fails if a function name is defined in two analysis
modules. Retired code (the feather delta-t path, the dense per-clock TDC LUTs,
...) is kept for reference, unmaintained, in [`archive/`](archive/).

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

The changelog is edited by hand, and it has to be edited *before* the tag
exists - the workflow reads `CHANGELOG.md` as it was at the tagged commit, so
notes still sitting under `[Unreleased]` cannot ship.

1. In `CHANGELOG.md`, rename the `## [Unreleased]` heading, leaving an empty
   `[Unreleased]` above it for the next cycle:

   ```markdown
   ## [Unreleased]

   ## [0.2.0] - 2026-08-02
   ```

   and update the link definitions at the foot of the file:

   ```markdown
   [Unreleased]: https://github.com/rngKomorebi/dapkel/compare/v0.2.0...HEAD
   [0.2.0]: https://github.com/rngKomorebi/dapkel/compare/v0.1.1...v0.2.0
   ```

2. Merge that into `main`.
3. On GitHub, *Releases -> Draft a new release*, create the tag `v0.2.0`
   **there** - it must point at the commit from step 2 - and publish.

Publishing runs `publish.yml`, which validates the tag, refuses to ship a
version with no changelog entry, runs the tests, checks the built version
matches the tag, uploads to PyPI, and rewrites the release body from the
changelog.

If the changelog entry is missing, the run fails at the first job, so nothing
is built and nothing reaches PyPI - no version number is lost. Fix the
changelog, push, then **delete the tag as well as the release** before
retrying: deleting a release leaves its tag behind, and re-creating one with
the same name silently reuses that tag and its old commit. PyPI versions
themselves cannot be reused, so a release that *does* upload is final.

### Contributing

`main` is the release branch: tested, and what PyPI is cut from. Work on a
feature branch and open a pull request against it.
[`TODO.md`](TODO.md) is the list of open questions — what is known to need
deciding and what would settle it. Start there if you are looking for
something to pick up, and add to it rather than leaving a decision undocumented. Contributions are welcome;
please follow [PEP 8](https://peps.python.org/pep-0008/) and
[PEP 257](https://peps.python.org/pep-0257/). If you are adding a new
measurement, start from
[Adding an analysis](https://rngkomorebi.github.io/dapkel/adding_an_analysis/) -
it describes the contract the tests enforce.

## License and contact

MIT - see [LICENSE](LICENSE). To get in touch, write to
sergei.kulkov23@gmail.com.
