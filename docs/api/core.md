# `dapkel.core`

The shared substrate every analysis is built on. Anything copied into two
analysis modules belongs here instead — `tests/test_api_surface.py` fails if a
function name is defined twice.

See [Adding an analysis](../adding_an_analysis.md) for the contract.

## `core.io` — finding and sizing raw files

::: dapkel.core.io

## `core.timing` — frame period and live time

::: dapkel.core.timing

## `core.pairs` — which pixel pairs, and what they are called

::: dapkel.core.pairs

## `core.reduce` — folding frames into a map

::: dapkel.core.reduce

## `core.plots` — the two figure shapes

::: dapkel.core.plots

## `core.store` — the stage 1 / stage 2 boundary

::: dapkel.core.store
