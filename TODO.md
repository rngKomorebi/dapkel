# Open items

Things this package is known to need, kept in the open next to the changelog.

The changelog says what was decided. This file says what has **not** been, and
why — so a question that was deliberately left open cannot be mistaken for one
nobody noticed, and so it can be picked up cold months later. Every item names
the choice, the trade-off, and what evidence would settle it.

## Decisions not yet made

### Delta-t stage 1 keeps the parts *and* the combined feather

`calculate_and_save_timestamp_differences` streams 64 MB parts to
`processed/<dataset>_delta_t_parts/`, then combines them into
`processed/<dataset>_delta_t.feather` at the end of a successful run — and does
not delete the parts, because they are the crash insurance. So a finished run
occupies **twice** its own size on disk: 6.4 GB for the 3.22 GB run, ~270 GB
for a 10 000-file one.

Only one of the two is needed. The read side does not care which it gets:
`feather_path` takes either a `.feather` or a `_parts` folder and streams
record batches from it, and a histogram over fixed edges is a sum over
batches, so the counts are identical either way.

| keep | costs |
|---|---|
| the combined feather (delete parts after it is verified) | the combine pass — a full read and rewrite of every byte of the run |
| the parts (stop combining) | the artifact becomes a folder plus a manifest, so anything that expects one file (`pandas.read_feather`, other tools, handing a run to a collaborator) has to change |

Note what does *not* change: the **peak** disk requirement is ~2× the run
either way, because the combine needs both at once. The question is only what
survives afterwards.

What would settle it: whether the combined feather is ever opened by something
that is not dapkel. If it is not, keeping the parts is strictly better — it
saves the pass as well as the space. Until someone knows, both are kept, which
is the safe direction to be wrong in.

The counts path (`calculate_and_save_delta_counts`) does not have this problem:
its checkpoint *is* its artifact.

### `subtract_background` on the feather path costs 9× the disk

One column set per lag, and the default is eight lags, so a 1 GB feather
becomes ~9 GB for a residual the counts path produces bit-identically from a
4.6 MB grid. `background_lags=(1, 2, 3, 4)` halves it for ~12% more residual
variance. Options: leave it (documented, and the user chose the path), lower
the default for the feather path only, or refuse past some lag count. Splitting
a default between two paths that are meant to agree is its own hazard, so this
is not obviously an improvement.

## Known and not fixed

### The feather path under-fills its outermost bin

It cuts data at `abs(delta) <= plot_window_ps` but draws edges half a bin
beyond that, so the first and last bin of the histogram are missing part of
their range. The counts path fills them properly, which is most of the ~1%
difference in fitted `sigma` between the two paths. Fixing it would change
every existing feather-path result, so it wants to happen together with a
decision about how such a change is versioned.

### The hitmap and DCR rates normalise by a nominal 9 µs cycle

The real cycle is **9.685–9.770 µs** — that is what the acquisition's own
200 MHz tick count says, and every value on disk fits
`ticks = nframes * N + 138 + window/5 ns` exactly, so the counter is
self-consistent. Using 9.000 µs therefore runs every reported rate ~7.8% high.
It is a deliberate, documented choice, not a measurement: the alternative is to
depend on `frame_rate_cnt.txt`, which is written by the acquisition exe rather
than by us, and no analysis result should rest on a file we do not control.
Resolving it properly means measuring the cycle independently, or getting the
firmware to report it.

### `core.io.frames_in_file` rests on an unestablished relation

Every sample `.bin` holds a whole power-of-two number of frames, which a
10 000-frame acquisition should not produce. So the mapping from "frames the
acquisition was asked for" to "bytes the file contains" is not understood.
`nframes` is a required argument everywhere for this reason, and
`frames_in_file` survives only for the case where a file's size is genuinely
the last record left. Until the relation is pinned down, its answer is a guess
that happens to be well-formed.

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
separately first and compare their centres.
