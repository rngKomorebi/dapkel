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
| **1. process** | fold frames into an array, scale to physical units, **save** | your `compute_*` + `dapkel.core.reduce`, `dapkel.core.store` |
| **2. plot** | load the saved array, render figures | your `plot_*` + `dapkel.core.plots` |

> **Status.** All of `core.io`, `core.timing`, `core.pairs`, `core.reduce`,
> `core.plots` and `core.store` exist, and every analysis writes stage-1
> artifacts to `processed/` and figures to `results/<kind>/`. This page
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

- file discovery or `.bin` frame counting → `core.io.find_bin_files`, `core.io.frames_in_file`
- the accumulate-over-frames loop → `core.reduce.accumulate_frames`
- assembling four `S*C` quadrants into a (64, 64) map → `core.reduce.assemble_64`
- frame period / live time resolution → `core.timing`
- enumerating pixel pairs, and the `"ra,ca-rb,cb"` label every coincidence
  artifact is indexed by → `core.pairs.pair_labels`, `core.pairs.pair_list`
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

## Naming convention

The package is consistent about this, and `__all__` groups follow it:

| prefix | signature | I/O |
|---|---|---|
| `compute_*` | raw files or an array in → array out | none |
| `plot_*` | array in → `plt.Figure` out | none |
| `compute_and_save_*` | raw files in → path to a stage-1 artifact | writes `processed/` |
| `collect_and_*` | folder in → paths to figures | reads `processed/`, writes `results/` |
| `load_*` | path in → array out | reads |

Keeping `plot_*` free of I/O is what lets you iterate on a figure in a notebook
without touching disk.

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

- [ ] Public functions follow the `compute_*` / `plot_*` / `collect_and_*` / `load_*` convention
- [ ] `__all__` declared, grouped by stage, listing every public name
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
"""Per-pixel afterpulsing probability map."""

from __future__ import annotations

import numpy as np

from dapkel.core import io, plots, reduce, store

__all__ = [
    # stage 1
    "compute_afterpulsing",
    "compute_and_save_afterpulsing",
    # stage 2
    "plot_afterpulsing",
    # driver
    "collect_and_plot_afterpulsing",
]

KIND = "afterpulsing"
_UNIT = "%"


def compute_afterpulsing(folder: str, nframes: int, tag: str = "AP") -> np.ndarray:
    """Compute the (32, 32) afterpulsing probability map.

    Parameters
    ----------
    folder : str
        Data folder holding the '.bin' files.
    nframes : int
        Number of frames stored in each '.bin' file.
    tag : str
        Filename fragment selecting the files. The default is ``'AP'``.

    Returns
    -------
    np.ndarray
        (32, 32) afterpulsing probability in percent.
    """
    files = io.find_bin_files(folder, tag)
    counts = reduce.accumulate_frames(
        files, lambda ts, pc: pc.sum(axis=2), nframes=nframes
    )
    total = reduce.accumulate_frames(
        files, lambda ts, pc: np.full((32, 32), pc.shape[2]), nframes=nframes
    )
    return 100.0 * counts / np.where(total > 0, total, np.nan)


def compute_and_save_afterpulsing(
    folder: str, nframes: int, tag: str = "AP"
) -> str:
    """Compute the map and save it under ``processed/``; return the path."""
    ap = compute_afterpulsing(folder, nframes, tag)
    return store.save_map(ap, folder, kind=KIND, tag=tag, unit=_UNIT)


def plot_afterpulsing(ap: np.ndarray, cmap: str | None = None):
    """Render an afterpulsing map. Returns the Figure; writes nothing."""
    return plots.sensor_map(
        ap, title="Afterpulsing", clabel=f"AP ({_UNIT})", fmt=".2f", cmap=cmap
    )


def collect_and_plot_afterpulsing(folder: str, tag: str = "AP") -> list[str]:
    """Load the saved map and write its figures to ``results/afterpulsing``."""
    ap, meta = store.load_map(folder, kind=KIND, tag=tag)
    return [
        store.save_figure(plot_afterpulsing(ap), folder, KIND, f"{tag}_heatmap.png"),
        store.save_figure(
            plots.sorted_distribution(ap, ylabel=f"AP ({_UNIT})"),
            folder,
            KIND,
            f"{tag}_distribution.png",
        ),
    ]
```

## Why the tests matter

`tests/test_api_surface.py` enforces the two rules most likely to be broken
under time pressure:

- `__all__` matches the module's actual public names
- no function name is defined in two analysis modules

The second is the anti-copy-paste guard. It is the reason this contract is more
than a suggestion — a checklist CI checks is a different object from a style
note. Two of its assertions are currently `xfail(strict=True)`, tracking the
pending `dapkel.core` extraction; when that lands they start passing, the strict
marker turns them red, and the marker gets deleted.
