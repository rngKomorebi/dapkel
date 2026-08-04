# dapkel

Data Analysis Package for the Kelpie v2 SPAD camera.

Every analysis here is the same pipeline with a different reduction in the
middle:

```
folder of .bin  →  discover files  →  fold frames into a sensor map
                →  scale to physical units  →  save
                →  load  →  plot
```

## The three stages

| stage | what it does | on disk |
|---|---|---|
| **0. raw** | the camera writes `.bin`; dapkel decodes it | `*.bin` (read-only) |
| **1. process** | fold frames into an array, scale to units, save | `processed/` |
| **2. plot** | load the saved array, render figures | `results/<kind>/` |

The rule that keeps this honest: **`processed/` holds data, `results/` holds
pictures, and stage 2 never re-unpacks `.bin` files.** Retrying a colormap costs
a load, not a full re-read of the raw data.

## Start here

**On a fresh acquisition, run [the data-quality check](guide/data_quality.md)
first.** A TDC that never stopped produces data that looks completely normal to
every other analysis in this package.

## The analyses

- [Decoding `.bin` files](guide/unpack.md) — the shared foundation
- [Dark count rate](guide/dcr.md) — per-pixel DCR maps in cps
- [Optical cross-talk](guide/crosstalk.md) — including the clockwise SPAD geometry
- [Hitmaps](guide/hitmap.md) — occupancy and photon rate
- [TDC calibration](guide/tdc_calibration.md) — the code density test
- [Coincidences](guide/coincidences.md) — delta-t and timing jitter
- [Background subtraction](guide/background_subtraction.md) — measure the
  accidental background by shifting frames instead of fitting it
- [Data quality](guide/data_quality.md) — check the run before trusting it

## Reference

- [Adding an analysis](adding_an_analysis.md) — the contract for new measurements
- [The triangular delta-t background](ort_triangle_background.md) — why a flat
  fit inflates the jitter
- [`dapkel.core`](api/core.md) — the shared substrate
- [Open items](todo.md) — what is known to need deciding, and why it has not been

## Plot styling

Importing `dapkel` applies its matplotlib house style through `komorebi_mpl`,
so plots look consistent out of the box. Your choice always wins:

```python
import dapkel                     # applies the default style on import
import komorebi_mpl
from dapkel.functions import hitmap_analysis

komorebi_mpl.use("night_wave")    # or any registered style, or "default"
hitmap_analysis.collect_and_plot_hitmap(path)   # stage 2: reads the saved map
```

The plotting functions never touch `rcParams` themselves, so whatever style is
active when they draw is what you get — set it once, after the imports.

Figure sizing follows one rule: **square sensor maps are 16×16 in code**
(`core.plots.sensor_map` does this), **everything else takes `figure.figsize`
from the active style**.
