# Adding a new analysis

This page is the contract for adding a measurement to dapkel. Follow it and the
new module is ~80 lines. **Do not start by copying an existing analysis module** —
that is how the package accumulated four copies of `_frames_in_file` and three
copies of the heatmap plotting skeleton.

## The three stages

Every analysis in dapkel is the same pipeline, and the stage boundaries are
where the reusable code lives:

| stage | what it does | where the code lives |
|---|---|---|
| **0. raw** | camera writes `.bin`; dapkel decodes it | `dapkel.functions.unpack`, `dapkel.core.io` — never reimplement |
| **1. process** | fold frames into an array, scale to physical units, **save** | your `process_<x>` + `dapkel.core.reduce`, `dapkel.core.store` |
| **2. plot** | load the saved array, render figures | your `plot_<x>` + `dapkel.core.plots` |

> **Status.** All of `core.io`, `core.timing`, `core.calibration`, `core.pairs`,
> `core.reduce`, `core.plots` and `core.store` exist, and every analysis writes
> stage-1 artifacts to `processed/` and figures to `results/<kind>/`. This page
> describes the code as it is.

## Figure sizing

One rule, and the dapkel style is built around it:

- **Square sensor maps set `figsize=(16, 16)` in code** — `core.plots.sensor_map`
  does this for you, so never pass a figure size to it.
- **Everything else inherits `figure.figsize` from the active style** (the dapkel
  style ships `16, 10`). Do not set a figure size on distributions, histograms,
  bar charts or multi-panel figures.

The rule that makes this real:

> **`processed/` holds data. `results/` holds pictures. Stage 2 obtains data only
> by loading a stage-1 artifact — never by re-unpacking `.bin` files.**

So a colormap tweak costs a `load_map` and a redraw, not a full re-unpack of the
raw data. Layout under a data folder:

```
<data folder>/
  *.bin                          stage 0 — raw, read-only, dapkel never writes here
  processed/                     stage 1 — arrays + metadata
    <name>_<tag>_<kind>.npy  + .meta.json
  results/<kind>/                stage 2 — figures only
```

## What you must supply

Only the parts that are genuinely specific to your measurement:

1. **An extract function** — `(ts, pc) -> np.ndarray`, the per-frame reduction.
   Photon counts are `pc.sum(axis=2)`; ORT occupancy is `(ts > 0).sum(axis=2)`
   (`unpack` leaves non-fired slots at `<= 0`, so validity is not a tunable).
2. **A file tag / glob** identifying which `.bin` files belong to this measurement.
3. **A unit conversion**, if the raw fold is not already in physical units
   (e.g. DCR divides by live time to get cps). Keep this *separate* from the
   accumulation — it is a scaling, not part of the fold.
4. **Display strings** — title, colorbar label, unit, and a format spec.

## What you must NOT write

Every item here already exists in `dapkel.core`. Reimplementing any of it will
fail `tests/test_api_surface.py`:

- file discovery → `core.io.find_bin_files` (the frame count is never read off
  a file's size — it is the caller's `nframes`)
- the accumulate-over-frames loop → `core.reduce.accumulate_frames`
- a per-pixel count map of one tag, or of the four `S*C` / `S*T` quadrants
  assembled into (64, 64) → `core.reduce.count_map`; what a count *is* for a tag
  → `core.reduce.readout_mode`; the SPAD offsets inside a macropixel →
  `core.reduce.SPAD_LAYOUT`
- the per-pixel TDC-code histogram → `core.reduce.code_histogram`
- live time resolution, units and the measured TDC full scale → `core.timing`
  (`resolve_live_time`, `to_ps` / `to_s`, `full_scale_from_max_code`)
- converting TDC codes to picoseconds with the tap calibration →
  `core.calibration` (`board_taps`, `codes_to_ps`)
- enumerating pixel pairs, and the `"ra,ca-rb,cb"` label every coincidence
  artifact is indexed by → `core.pairs.pair_list`, `core.pairs.pair_label`
- the sensor heatmap figure → `core.plots.sensor_map`
- the sorted-percentile distribution figure → `core.plots.sorted_distribution`
- saving/loading arrays with metadata, and saving figures → `core.store`
  (`save_map` / `load_map` / `save_figure`, and `processed_dir` / `results_dir`
  for the paths — never hand-build `os.path.join(path, "results", ...)`)
- the warning before an overwrite → `core.store.confirm_rewrite`

That last one is a rule, not just a convenience: **if your analysis takes a
`rewrite` parameter, it must call `store.confirm_rewrite` with every artifact
that parameter is about to destroy** before writing anything. It is silent on a
fresh run and only costs time when there is genuinely something to lose. The
accident it exists for is a `rewrite=True` left in the script from the previous
acquisition, quietly eating an overnight measurement.

If you need something close to but not quite one of these, **extend the core
function with a parameter** rather than writing a variant next to it. Variants
are how the existing three heatmap plotters drifted into three different figure
sizes and two different NaN policies.

## Two public functions, always save

An analysis module has **exactly two public functions**, and its `__all__` is
exactly those two:

| function | signature | I/O |
|---|---|---|
| `process_<x>` | `(path, nframes, ...)`: raw `.bin` in → path to the saved stage-1 artifact | reads `.bin`, **always** writes `processed/<name>_<tag>_<kind>.npy` + `.meta.json` |
| `plot_<x>` | `(path, tag, ...)`: data folder in → the data it plotted | reads `processed/`, writes `results/<kind>/` |

`process_<x>` always saves — there is no in-memory-only variant and no `reuse=`
switch, because stage 2 reads the artifact and nothing else. A caller who only
wants the figures makes the two calls in a row; one who wants the array takes
it from `plot_<x>` (which returns it) or from `core.store.load_map`.

Everything else in the module is private (`_helper`), and there are at most
**ten** private top-level functions per module. A module that needs more is
doing several jobs, or has helpers that belong in `dapkel.core`. Code that is
retired rather than refactored moves to the repository's `archive/` folder,
which is kept for reference only and not maintained.

### Parameter names are shared, not per-module

One concept, one name, everywhere. A new analysis picks from this list rather
than inventing a synonym — `folder`, `file_prefix`, `from_saved`, `grid_s` and
`cmap_title` were all synonyms of entries in it, and every one of them made a
caller stop and check which module they were in.

| name | meaning |
|---|---|
| `path` | the data folder (never `folder`) |
| `tag` | filename fragment selecting the files (never `file_prefix`) |
| `nframes` | frames per `.bin`, as the acquisition was set |
| `max_files` | process at most this many files |
| `exp_time_s`, `open_shutter_time_s`, `firmware_version` | the live-time trio, for rate normalisation **only** |
| `bin_width_s`, `plot_window_s`, `fit_window_s` | histogram/fit ranges |
| `rewrite`, `resume`, `quiet` | housekeeping (never `from_saved`) |
| `label`, `title`, `cmap` | figure text and colours (never `cmap_title`) |

And the harder rule: **if the data can answer it, do not add a parameter.** The
TDC full scale used to be asked for three ways at once; it is now measured from
the largest code a run produced. Ask yourself whether the caller could get the
number wrong — if so, they will, and the code should read it instead. The one
survivor is `clock_period_s` in `tdc_calibration`, because a density test fixes
only *relative* code widths and something has to set the absolute scale.

## Docstrings

Keep in-code docstrings short: a one-line summary, then `Parameters`, `Returns`,
`Raises`. That is the reference someone needs at the call site when they have
forgotten what an argument takes.

**Long-form explanation belongs in this `docs/` tree, not in the docstring** —
physics background, derivations, worked examples, figures. Link to it from the
docstring with a one-line pointer, and link back from the page to the API entry.
`mkdocstrings` pulls the numpydoc docstrings into the generated API reference, so
the two halves stay side by side for readers.

## Checklist

- [ ] Exactly two public functions, `process_<x>` and `plot_<x>`, for the same `<x>`
- [ ] `__all__` is exactly `["process_<x>", "plot_<x>"]`
- [ ] At most ten private top-level functions
- [ ] Module added to `ANALYSIS_MODULES` in `tests/test_api_surface.py`
- [ ] Module added to `dapkel/functions/__init__.py` and to its `__all__`
- [ ] No file discovery, accumulate loop, or figure skeleton reimplemented
- [ ] No import of another analysis module's `_private` names
- [ ] Stage 1 writes to `processed/`; stage 2 reads it back and writes only to `results/`
- [ ] Stage-1 artifact has a `.meta.json` recording frame count and timing
- [ ] Docstrings short; narrative added under `docs/`
- [ ] `pytest tests/` green

## Worked example

A minimal analysis, end to end. Note how little of it is not measurement-specific.

```python
"""Per-pixel afterpulsing probability map.

    process_afterpulsing   raw '.bin' -> processed/<name>_<tag>_afterpulsing.npy
    plot_afterpulsing      that array -> results/afterpulsing/ figures
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import io, plots, reduce, store

__all__ = ["process_afterpulsing", "plot_afterpulsing"]

#: Analysis name: the stage-1 file stem and the results sub-folder.
_KIND = "afterpulsing"


def process_afterpulsing(path: str, nframes: int, tag: str = "AP") -> str:
    """Unpack the '.bin' files, compute the afterpulsing map and save it.

    Parameters
    ----------
    path : str
        Data folder holding the '.bin' files.
    nframes : int
        Frames the acquisition recorded per '.bin' file.
    tag : str, optional
        Filename fragment selecting the files. The default is ``'AP'``.

    Returns
    -------
    str
        Path to the saved '.npy'.
    """
    files = io.find_bin_files(path, tag)
    counts = reduce.accumulate_frames(
        files, lambda ts, pc: pc.sum(axis=2), nframes=nframes
    )
    ap = 100.0 * counts / (nframes * len(files))
    return store.save_map(
        ap,
        path,
        kind=_KIND,
        tag=tag,
        meta={"n_files": len(files), "nframes": int(nframes), "unit": "%"},
    )


def plot_afterpulsing(
    path: str, tag: str = "AP", *, cmap: str | None = None
) -> np.ndarray:
    """Load the saved map and write its figures to ``results/afterpulsing``.

    Parameters
    ----------
    path : str
        Data folder that 'process_afterpulsing' was run on.
    tag : str, optional
        The tag it was run with. The default is ``'AP'``.
    cmap : str | None, optional
        Matplotlib colormap name. The default is None - the active style's.

    Returns
    -------
    np.ndarray
        The afterpulsing map, in percent.
    """
    ap, _ = store.load_map(path, kind=_KIND, tag=tag)
    name = store.map_file_name(
        os.path.basename(os.path.normpath(path)), _KIND, tag
    ).removesuffix(".npy")
    results_dir = store.results_dir(path, _KIND)

    fig = plots.sensor_map(ap, title="Afterpulsing", clabel="AP (%)", cmap=cmap)
    store.save_figure(fig, results_dir, f"{name}_heatmap.png")
    plt.close(fig)

    fig = plots.sorted_distribution(
        ap, ylabel="AP (%)", title="Sorted afterpulsing", fmt=".2f", unit="%"
    )
    store.save_figure(fig, results_dir, f"{name}_distribution.png")
    plt.close(fig)
    return ap
```

## Why the tests matter

`tests/test_api_surface.py` enforces the rules most likely to be broken under
time pressure:

- an analysis module's `__all__` is exactly `process_<x>` and `plot_<x>`
- it defines at most ten private top-level functions
- `__all__` matches the module's actual public names
- no function name is defined in two analysis modules
- no module imports another dapkel module's `_private` names
- artifact paths come from `core.store`, never hand-built `os.path.join`s

The fourth is the anti-copy-paste guard. Together they are the reason this
contract is more than a suggestion — a checklist CI checks is a different object
from a style note. They only apply to a module listed in `ANALYSIS_MODULES` in
that test, so add yours there.
