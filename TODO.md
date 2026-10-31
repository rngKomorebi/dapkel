# Open items

Things this package is known to need, kept in the open next to the changelog.

The changelog says what was decided. This file says what has **not** been, and
why — so a question that was deliberately left open cannot be mistaken for one
nobody noticed, and so it can be picked up cold months later. Every item names
the choice, the trade-off, and what evidence would settle it.

## Decisions not yet made

None open at the moment. Three items used to sit here and under *Known and not fixed*: whether delta-t
stage 1 should keep its feather parts, the combined feather or both; the ~9×
disk cost of `subtract_background` on the feather path; and the feather path's
under-filled outermost bins. All three were about the per-event feather path,
which is now retired to `archive/`; `process_delta_t`'s counts grid has none of
them — its checkpoint *is* its artifact, a lag costs one small grid plane, and
its bins are centred on zero whatever the window.

## Known and not fixed

### Under `long_exposure`, hitmap and DCR rates use a nominal 9 µs overhead

Under the current `short_exposure` firmware the rate divides by
`open_shutter_time_s` and the cycle does not enter. Under `long_exposure` the
whole `exp_time_s + 9 µs` cycle is the live time, and the real overhead is not
9 µs: the real short-exposure cycle is **9.685–9.770 µs** — that is what the
acquisition's own 200 MHz tick count says, and every value on disk fits
`ticks = nframes * N + 138 + window/5 ns` exactly, so the counter is
self-consistent. Using 9.000 µs therefore runs a long-exposure rate high, by up
to ~7.8% at `exp_time_s = 0`.
It is a deliberate, documented choice, not a measurement: the alternative is to
depend on `frame_rate_cnt.txt`, which is written by the acquisition exe rather
than by us, and no analysis result should rest on a file we do not control.
Resolving it properly means measuring the cycle independently, or getting the
firmware to report it.

### The frame count of a `.bin` rests on an unestablished relation

Every sample `.bin` holds a whole power-of-two number of frames, which a
10 000-frame acquisition should not produce. So the mapping from "frames the
acquisition was asked for" to "bytes the file contains" is not understood.
`nframes` is a required argument everywhere for this reason, and the old
`frames_in_file` size-based guess is retired to `archive/`. Until the relation
is pinned down, a frame count read off a file's size is a guess that happens to
be well-formed.

## Physics still open

### The coincidence peak is not one Gaussian

A single Gaussian on the ORT triangle reports ~741 ps sigma for the 2026.07.31
SPDC run, against the ~40 ps rms per SPAD the designers measure with a focused
laser — a factor ~13, and their own two quoted numbers (40 ps rms per SPAD,
140 ps FWHM coincidence) agree with each other. The gap is most likely a
mixture rather than a wrong number: photons absorbed in the SPAD depletion
region are timed by the avalanche, photons absorbed deeper arrive by diffusion,
and one Gaussian returns the area-weighted average of the two.

Before reading anything into a two-component fit, rule out the mundane
explanation: `all_pairs` pools pairs whose true delay differs, so a spread in
per-pair `mu` would broaden the pooled peak on its own. Fit the pairs
separately first (`plot_delta_t(..., pairs=[...])`) and compare their centres.
The package's two-Gaussian (narrow core on broad tails) fit is retired to
`archive/`; fit the `counts` and `centers` `plot_delta_t` returns by hand.
