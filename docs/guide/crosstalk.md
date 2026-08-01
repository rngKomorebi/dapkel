# Optical cross-talk

Cross-talk is measured on the Kelpie v2 sensor by reading, for each 2×2
macropixel, two on-chip counters over the same set of frames:

- **`coinc`** — frames in which *both* micropixels of the pair fired (logical AND)
- **`OR`** — frames in which *at least one* of the pair fired (logical union)

The per-macropixel cross-talk metric is the ratio of the accumulated counts:

```
CT = sum(coinc) / sum(OR)
```

which is dimensionless (a probability), so the frame time cancels and no
exposure time is needed.

!!! note "This slightly overestimates true optical cross-talk"
    The raw ratio includes *accidental* coincidences from independent dark
    counts landing in the same frame.

Macropixels where `OR = 0` are undefined and come back as `NaN`. Everything in
[`core.plots`][dapkel.core.plots] is NaN-aware, so those pixels are dropped from
the statistics rather than poisoning them.

## SPAD geometry — the indices run clockwise

This is the single most important fact in this module, and it is not the
convention you would guess. Inside one 2×2 macropixel the four SPADs are
indexed **clockwise**, not row-major:

```
S0 (0,0)   S1 (0,1)
S3 (1,0)   S2 (1,1)
```

So relative to S0:

| pairing | neighbour | direction | cell offset |
|---|---|---|---|
| `S0S1` | right | **horizontal** | `(0, 1)` |
| `S0S3` | below | **vertical** | `(1, 0)` |
| `S0S2` | corner | **diagonal** | `(1, 1)` |

Note that `S0S2` is the *diagonal* pairing and `S0S3` the *vertical* one — the
reverse of what a row-major reading gives. Vertical and diagonal cross-talk are
physically different magnitudes (diagonal SPADs sit further apart), so swapping
them silently mislabels two real measurements.

This layout is asserted in `tests/test_spad_geometry.py`, which ties
`crosstalk_analysis._DIRECTIONS` to the `_SPAD_LAYOUT` used by
[DCR](dcr.md) and [hitmap](hitmap.md) for their (64, 64) assembly, so the three
encodings cannot drift apart.

## Data layout

One folder per micropixel pairing, e.g. `S0S1`, `S0S2`, `S0S3`. Each folder holds
many repeated `.bin` acquisitions per counter, named like
`CT_coincS0S1<n>.bin` / `CT_ORS0S1<n>.bin`; the counts are summed across every
acquisition for statistics.

The `01` preliminary batches use a `CT01_` / `CT02_` prefix and are excluded by
the default `file_prefix="CT_"`, so the two acquisitions are never silently
mixed.

## The combined 64×64 map

`combine_directional_crosstalk` scans a parent folder holding one sub-folder per
pairing, computes each (32, 32) directional map, and lays them onto a single
(64, 64) micropixel grid. Within each 2×2 block the neighbour cells carry S0's
cross-talk in that direction, and the S0 cell carries the mean of the available
directions:

```
mean   | horizontal
vertical | diagonal
```

## API

::: dapkel.functions.crosstalk_analysis
