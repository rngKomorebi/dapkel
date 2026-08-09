# Data quality — check before you analyse

Nothing in this module contributes to a physics result. These are the checks you
run *first*, on a fresh acquisition, to answer one question: **did the TDC
actually record timing?**

## The failure mode is silent

When the ring-oscillator TDC never stops — or the readout is misconfigured —
[`unpack`](unpack.md) still returns a full `(32, 32, nframes)` array of
plausible-looking integers. Every downstream analysis still runs. The delta-t
histogram still has a shape. It is simply meaningless.

There is no exception, no warning, and no obviously wrong number. The only way
to catch it is to look at the raw code distribution.

## What healthy data looks like

A working pixel's TDC codes spread over the full range for the reference clock
in use — ~279 codes at 50 MHz, ~1397 at 10 MHz, ~2794 at 5 MHz, ~6985 at
2 MHz, since the ring sets the ~71.6 ps code width and the clock sets the
full scale. A
broken run collapses onto a handful of values — often a single code repeated
across every frame.

```python
from dapkel.functions import data_quality

fig, codes = data_quality.plot_time_code_histogram(
    folder, pixel=(16, 16), nframes=10_000
)
print(codes.size, "valid codes,", len(set(codes)), "unique")
```

A few unique values out of thousands of frames means the run is unusable. This
mirrors the timestamp inspection in `Kelpie_run.m`.

Run it on a pixel or two before spending time on
[coincidences](coincidences.md).

## API

::: dapkel.functions.data_quality
