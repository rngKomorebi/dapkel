"""Per-pixel TDC calibration via the statistical code density test.

The ring-oscillator TDC has non-uniform bin widths (DNL), so a flat
``code * 71.6 ps`` conversion is only an average. Under illumination uncorrelated
with the oscillator, each code's hit probability is proportional to its width,
which turns a per-pixel code histogram into a code -> ps lookup table:

    LUT[k] = full_scale * (cumsum(counts)[k] - counts[k]/2) / total

One (32, 32, n_codes) LUT per SPAD; feed it to 'delta_t' as ``time_lut``.

``clock_period_s`` - one period of the reference clock - is **the only time this
package ever has to be told**. The density test measures relative code widths
only, so an externally known duration has to set the absolute scale; every other
timescale downstream is measured from the data or leans on 'CODE_S'. Nothing
outside this module takes it.

**A LUT is specific to the reference clock it was taken with.** The ring sets the
code width (~71.6 ps, see 'dapkel.core.timing.CODE_S') and the clock sets the
full scale, so the number of codes scales as 1/f: ~279 codes at 50 MHz, ~1397 at
10 MHz, ~2794 at 5 MHz, ~6985 at 2 MHz. Feeding a 10 MHz LUT to a 2 MHz run
discards every code past 100 ns - 79 % of the singles and 96 % of the
coincidences - so the loaders validate coverage against ``min_codes``, the
largest code the *data* actually holds. That is the check that matters: it
compares the table to the run it is about to be applied to, rather than to a
clock frequency recalled by hand. Calibrate once per clock.

Every time parameter is in seconds, per 'dapkel.core.timing'.
See ``docs/guide/tdc_calibration.md``.
"""

from __future__ import annotations

import glob
import os
import re
from importlib.resources import files as _resource_files

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dapkel.core import io, store, timing
from dapkel.functions.unpack import unpack

__all__ = [
    # stage 1 - build the per-pixel code -> ps LUT from raw '.bin' files
    "compute_code_histogram",
    "histogram_to_lut",
    "compute_tdc_lut",
    # stage 2 - plot from an already computed LUT
    "plot_tdc_calibration",
    # the compact form - 8 ring taps per pixel, valid at every clock
    "TAPS_PER_LAP",
    "histogram_to_compact",
    "compact_to_lut",
    # drivers - folder in, LUTs and QA figures on disk
    "collect_and_save_luts",
    "collect_and_save_compact",
    # load saved / shipped LUTs
    "load_lut",
    "load_board_lut",
    "load_board_compact",
    # introspection - a LUT describes its own full scale
    "lut_full_scale_s",
]

#: Analysis name: the results sub-folder for the QA plots.
_KIND = "tdc_calibration"

# The four SPAD (micropixel) calibration tags, one per 2x2 macropixel corner.
_SPAD_TAGS = ("SPAD0_S0", "SPAD1_S1", "SPAD2_S2", "SPAD3_S3")

# Fraction of a run's own code range a LUT may fall short of before it is
# treated as the wrong calibration. The populated range stops a few codes short
# of full scale in any finite sample (1378 of ~1397 at 10 MHz, i.e. ~1.4 %), and
# the run being calibrated is a different sample from the one the LUT was built
# on, so a little slack either way is normal. The clocks in use differ by factors
# of 2 to 5, so this is nowhere near admitting the wrong one.
_CODE_COVERAGE_RTOL = 0.05


def lut_full_scale_s(time_lut: np.ndarray) -> float:
    """Read a LUT's own full scale, in seconds.

    A code -> ps LUT is self-describing: it was built by mapping the populated
    code range onto one TDC full scale, so its largest entry *is* that full
    scale. This is the trustworthy source for the accidental background's
    triangle half-base - more so than any clock setting recalled by hand.

    Parameters
    ----------
    time_lut : np.ndarray
        A ``(n_codes,)`` or ``(32, 32, n_codes)`` code -> ps lookup table.

    Returns
    -------
    float
        The full scale in seconds.
    """
    return timing.to_s(float(np.asarray(time_lut).max()))


def _clock_tag(clock_period_s: float) -> str:
    """File-name fragment identifying the clock a calibration belongs to.

    Formats the reference-clock *frequency* in MHz with ``'p'`` for the decimal
    point, so LUTs for several clocks coexist in one folder and the name reads
    the way the clock is actually set on the generator: ``50MHz``, ``10MHz``,
    ``2MHz``, ``1p7MHz``.

    Parameters
    ----------
    clock_period_s : float
        One reference-clock period, in seconds.

    Returns
    -------
    str
        The tag, e.g. ``'10MHz'``.
    """
    return f"{1.0 / clock_period_s / 1e6:g}MHz".replace(".", "p")


# A clock token in a LUT file name: '2MHz', '51p02MHz', or the legacy period
# form '500ns'. Anything trailing it (a date, a board revision) is part of the
# tag too, so two calibrations of the same clock can coexist.
_CLOCK_TOKEN = r"\d+(?:p\d+)?(?:MHz|kHz|Hz|ns|us|ps)"
_LEADING_CLOCK = re.compile(rf"_({_CLOCK_TOKEN})_TDC_LUT_", re.IGNORECASE)
# '(?!TDC_LUT)' keeps the trailing form from swallowing the older layout, where
# the clock sits *before* 'TDC_LUT' and the rest of the name would otherwise be
# read as part of the tag.
_TRAILING_CLOCK = re.compile(
    rf"_({_CLOCK_TOKEN}(?:_(?!TDC_LUT)[^_]+)*)\.npy$", re.IGNORECASE
)


def _lut_clock_tag(name: str) -> str:
    """The clock tag carried by a LUT file name, or ``''`` if it has none.

    Both documented layouts are read - the tag after the SPAD fragment
    (``..._SPAD0_S0_2MHz_2026-08-13.npy`` -> ``'2MHz_2026-08-13'``) and the
    older one before ``TDC_LUT`` (``D0_M0_500ns_TDC_LUT_SPAD0_S0.npy`` ->
    ``'500ns'``). A name with neither is untagged and matches no explicit
    request, which is what keeps a legacy table from being taken for a
    calibration it is not.

    Parameters
    ----------
    name : str
        LUT file name (not a full path).

    Returns
    -------
    str
        The tag, e.g. ``'10MHz'`` or ``'2MHz_2026-08-13'``, or ``''``.
    """
    for pattern in (_LEADING_CLOCK, _TRAILING_CLOCK):
        found = pattern.search(name)
        if found:
            return found.group(1)
    return ""


def _filter_by_clock(
    luts: dict[str, np.ndarray], clock: str | None, source: str
) -> dict[str, np.ndarray]:
    """Narrow a named LUT set to the one calibration ``clock`` asks for.

    An exact tag match wins, so ``clock='10MHz'`` keeps taking the table named
    exactly that even once ``10MHz_2026-08-13`` sits beside it. Failing that,
    ``clock`` may name just the clock and let the rest of the tag follow
    (``'2MHz'`` -> ``'2MHz_2026-08-13'``), which resolves only when one
    calibration answers to it.

    Parameters
    ----------
    luts : dict[str, np.ndarray]
        Mapping file name -> LUT.
    clock : str | None
        Clock tag to select, or None to keep everything.
    source : str
        Human-readable origin, used in error messages.

    Returns
    -------
    dict[str, np.ndarray]
        The surviving subset (all of ``luts`` when ``clock`` is None).

    Raises
    ------
    ValueError
        Raised when no table carries the tag, or when several distinct
        calibrations match a partial one.
    """
    if clock is None:
        return luts

    tags = {name: _lut_clock_tag(name) for name in luts}
    wanted = clock.lower()

    exact = {n: lut for n, lut in luts.items() if tags[n].lower() == wanted}
    if exact:
        return exact

    partial = {
        n: lut
        for n, lut in luts.items()
        if tags[n].lower().startswith(f"{wanted}_")
    }
    distinct = {tags[n] for n in partial}
    if len(distinct) > 1:
        raise ValueError(
            f"clock={clock!r} matches {len(distinct)} calibrations in {source}: "
            f"{sorted(distinct)}. Name one of them in full."
        )
    if partial:
        return partial

    available = sorted({t for t in tags.values() if t}) or ["(none tagged)"]
    raise ValueError(
        f"No LUT tagged clock={clock!r} in {source}. Available: {available}. "
        "Calibrate this clock with "
        "'collect_and_save_luts(..., clock_period_s=...)'."
    )


def _check_code_coverage(
    lut: np.ndarray, min_codes: int | None, source: str
) -> np.ndarray:
    """Verify a LUT covers the code range the data being calibrated occupies.

    This compares the table against the *run*, not against a stated clock
    frequency: ``min_codes`` is the largest code the data actually holds, so a
    LUT that is too short is caught by the only evidence that can prove it.

    Parameters
    ----------
    lut : np.ndarray
        The loaded code -> ps lookup table.
    min_codes : int | None
        Number of codes the data needs covered, i.e. ``max_code + 1``, or None
        to skip the check (nothing is known about the data yet).
    source : str
        Human-readable origin, used in the error message.

    Returns
    -------
    np.ndarray
        The unchanged ``lut``, once validated.

    Raises
    ------
    ValueError
        Raised when the LUT is more than 5 % short of the data's own code range,
        which means it belongs to a different reference clock.
    """
    if min_codes is None:
        return lut
    n_codes = int(lut.shape[-1])
    if n_codes < min_codes * (1.0 - _CODE_COVERAGE_RTOL):
        raise ValueError(
            f"The LUT in {source} has {n_codes} codes "
            f"({lut_full_scale_s(lut) * 1e9:.2f} ns) but the data reaches code "
            f"{min_codes - 1} "
            f"({timing.full_scale_from_max_code(min_codes - 1) * 1e9:.2f} ns). "
            "A LUT is specific to its reference clock: using this one would "
            "silently drop every code past its full scale. Calibrate this clock "
            "with 'collect_and_save_luts(..., clock_period_s=...)'."
        )
    return lut


def compute_code_histogram(
    files: list[str],
    *,
    nframes: int,
    label: str = "",
) -> np.ndarray:
    """Accumulate the (32, 32, n_codes) per-pixel TDC-code histogram.

    Unpacks each file's ``time_series`` (the raw TDC code) and, for every
    pixel independently, counts how often each integer code occurs across
    all frames. Only frames in which the pixel carried a valid timestamp
    contribute - 'unpack' leaves non-fired slots at ``<= 0``, so that is
    ``time_series > 0``.

    The code axis takes no sizing. It is allocated at the widest the frame
    format can express ('core.timing.MAX_TS_CODE', ~67 MB) and trimmed to what
    the data populated, so the code count comes out of the run rather than out
    of a stated clock. Nothing can overflow it, so there is no dropped-codes
    warning to heed and no way to lose 79 % of a run to a mis-stated frequency.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files to unpack and accumulate.
    nframes : int
        Number of frames stored in each '.bin' file.
    label : str, optional
        Label used in the progress printout. The default is "".

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) int64 per-pixel code histogram, trimmed to the
        largest code any pixel produced.
    """
    n_alloc = timing.MAX_TS_CODE + 1

    # Linear pixel index 0..1023 for a single per-file np.bincount that
    # histograms all 1024 pixels at once: lin = pixel * n_alloc + code.
    pix_index = np.arange(1024, dtype=np.int64).reshape(32, 32)
    counts = np.zeros(1024 * n_alloc, dtype=np.int64)

    for fp in tqdm(files, desc=label or None):
        ts, _ = unpack(fp, nframes, compute_time_series=True)  # (32,32,nframes)

        code = ts.astype(np.int64)
        valid = (ts > 0) & (code < n_alloc)

        lin = pix_index[:, :, np.newaxis] * n_alloc + code
        counts += np.bincount(lin[valid], minlength=1024 * n_alloc)

    return _trim_trailing_zero_codes(counts.reshape(32, 32, n_alloc))


def histogram_to_lut(
    counts: np.ndarray, *, clock_period_s: float
) -> np.ndarray:
    """Convert a per-pixel code histogram into a code -> ps lookup table.

    ``LUT[k] = clock_period * (cumsum(counts)[k] - counts[k]/2) / total``.
    Dead pixels (no counts) fall back to a linear ramp so downstream indexing
    never divides by zero. Method: ``docs/guide/tdc_calibration.md``.

    Parameters
    ----------
    counts : np.ndarray
        The (32, 32, n_codes) per-pixel code histogram from
        'compute_code_histogram'.
    clock_period_s : float
        One reference-clock period, in seconds - the full scale onto which the
        populated code range is mapped. The density test only fixes the
        *relative* per-code widths, so this is the single constant that sets the
        absolute scale, and it is what 'lut_full_scale_s' reads back out of the
        finished table. The one irreducible clock parameter in the package.

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) float64 code -> ps lookup table. ``LUT[r, c]``
        maps an integer TDC code to picoseconds for pixel ``(r, c)``.
    """
    period_ps = timing.to_ps(clock_period_s)
    counts = counts.astype(np.float64)
    n_codes = counts.shape[-1]
    total = counts.sum(axis=-1)  # (32, 32)

    # Bin-centre integral: cumulative count up to and including code k, minus
    # half of code k's own count -> the centre of bin k in count units.
    centre_cum = np.cumsum(counts, axis=-1) - counts / 2.0

    lut = np.empty_like(counts)
    live = total > 0
    lut[live] = period_ps * centre_cum[live] / total[live][:, np.newaxis]

    # Dead pixels: nominal linear ramp (code centre k+0.5 over full scale).
    if not live.all():
        ramp = period_ps * (np.arange(n_codes) + 0.5) / n_codes
        lut[~live] = ramp

    return lut


def _trim_trailing_zero_codes(counts: np.ndarray) -> np.ndarray:
    """Trim trailing codes that are empty for every pixel.

    The code axis is allocated with headroom over the nominal
    ``full_scale / CODE_S``, so its top is all zeros. Trimming keeps the LUT
    compact without changing any populated bin.

    Parameters
    ----------
    counts : np.ndarray
        The (32, 32, n_codes) per-pixel code histogram.

    Returns
    -------
    np.ndarray
        The histogram trimmed to (32, 32, max_populated_code + 1).
    """
    per_code = counts.sum(axis=(0, 1))
    nonzero = np.nonzero(per_code)[0]
    if nonzero.size == 0:
        return counts
    return counts[:, :, : int(nonzero[-1]) + 1]


# --------------------------------------------------------------------------
# the compact form: 8 tap widths per pixel, valid at every clock
# --------------------------------------------------------------------------

#: Taps on the ring oscillator, i.e. TDC codes per lap. 'unpack' builds the code
#: as ``(coarse - 1) * 8 + (8 - fine)``, so ``code % TAPS_PER_LAP`` *is* the tap
#: the photon landed on and the fine interpolator's non-linearity repeats with
#: this period for as long as the coarse counter runs.
TAPS_PER_LAP = 8


def histogram_to_compact(
    counts: np.ndarray, *, clock_period_s: float
) -> np.ndarray:
    """Reduce a per-pixel code histogram to its 8 ring-oscillator tap widths.

    The density test measures one width per code, but the hardware only has
    ``TAPS_PER_LAP`` distinct widths: the coarse counter counts whole laps of
    the ring and contributes no non-linearity, so code ``k`` is always tap
    ``k % 8`` and every lap repeats the same eight. Averaging each tap over all
    the laps a run swept therefore estimates the *same* quantity from ~n/8 times
    as many events, which is the whole point - see
    ``docs/guide/tdc_calibration.md``.

    Each pixel is normalised over **its own** populated code range, not the
    array-wide one, so the per-pixel time-base gain survives: a pixel whose ring
    runs fast reaches more codes in one clock period and comes out with
    proportionally narrower taps.

    Parameters
    ----------
    counts : np.ndarray
        The (32, 32, n_codes) per-pixel code histogram from
        'compute_code_histogram'.
    clock_period_s : float
        One reference-clock period, in seconds - the absolute scale, exactly as
        in 'histogram_to_lut'. Only the *populated range* is tied to it, and the
        tap widths that come out are a property of the ring rather than of this
        clock, which is why one compact table serves every clock.

    Returns
    -------
    np.ndarray
        The (32, 32, 8) float64 tap widths in picoseconds. Their sum is the
        pixel's ring period; dead pixels get a flat nominal lap.
    """
    period_ps = timing.to_ps(clock_period_s)
    counts = counts.astype(np.float64)
    n_codes = counts.shape[-1]
    phase = np.arange(n_codes) % TAPS_PER_LAP

    # Where each pixel's own ruler ends: the last code it ever produced. Beyond
    # it the pixel contributes nothing, and counting those codes as if it had
    # would flatten exactly the gain spread we are trying to keep.
    populated = counts > 0
    last = np.where(populated.any(axis=-1), populated.shape[-1] - 1
                    - np.argmax(populated[:, :, ::-1], axis=-1), -1)
    n_used = last + 1  # (32, 32)

    # Per tap: total counts, and how many codes of that tap the pixel swept.
    totals = np.stack(
        [counts[:, :, phase == j].sum(axis=-1) for j in range(TAPS_PER_LAP)],
        axis=-1,
    )
    code_no = np.arange(n_codes)
    laps = np.stack(
        [
            ((code_no % TAPS_PER_LAP == j) & (code_no < n_used[..., None])).sum(-1)
            for j in range(TAPS_PER_LAP)
        ],
        axis=-1,
    ).astype(np.float64)

    # Mean counts per occurrence of the tap == its width, up to one scale that
    # the clock period fixes: sum(laps * width) spans one clock period.
    with np.errstate(invalid="ignore", divide="ignore"):
        per_lap = np.where(laps > 0, totals / laps, 0.0)
    total = counts.sum(axis=-1)  # == (laps * per_lap).sum(-1)

    taps = np.empty(counts.shape[:2] + (TAPS_PER_LAP,), dtype=np.float64)
    live = total > 0
    taps[live] = period_ps * per_lap[live] / total[live][:, np.newaxis]

    # Dead pixels: a flat lap over the array-wide range, matching the linear
    # ramp 'histogram_to_lut' gives them.
    if not live.all():
        taps[~live] = period_ps / n_codes

    return taps


def compact_to_lut(taps_ps: np.ndarray, n_codes: int) -> np.ndarray:
    """Expand 8 tap widths per pixel into a code -> ps lookup table.

    Rebuilds exactly what 'histogram_to_lut' returns - bin *centres*, so the two
    are interchangeable as ``time_lut`` - from ``code = 8 * lap + tap``:

        ``t(k) = (k // 8) * lap_ps + sum(taps[:k % 8]) + taps[k % 8] / 2``

    ``n_codes`` comes from the run being calibrated rather than from a clock, so
    one compact table generates a table of whatever length the data needs.

    Parameters
    ----------
    taps_ps : np.ndarray
        The (32, 32, 8) tap widths in picoseconds, from
        'histogram_to_compact' or 'load_board_compact'.
    n_codes : int
        Length of the table to generate, i.e. the run's largest code + 1.

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) float64 code -> ps lookup table.

    Raises
    ------
    ValueError
        Raised when ``taps_ps`` is not (32, 32, 8) or ``n_codes`` is not
        positive.
    """
    taps_ps = np.asarray(taps_ps, dtype=np.float64)
    if taps_ps.shape[-1] != TAPS_PER_LAP or taps_ps.ndim != 3:
        raise ValueError(
            f"taps must be (32, 32, {TAPS_PER_LAP}), got {taps_ps.shape}."
        )
    if n_codes <= 0:
        raise ValueError(f"n_codes must be positive, got {n_codes}.")

    lap_ps = taps_ps.sum(axis=-1)  # (32, 32) ring period
    # Offset from the start of a lap to the centre of each tap.
    start = np.concatenate(
        [np.zeros(taps_ps.shape[:2] + (1,)), np.cumsum(taps_ps, axis=-1)[..., :-1]],
        axis=-1,
    )
    centre = start + taps_ps / 2.0  # (32, 32, 8)

    code = np.arange(n_codes)
    return (
        (code // TAPS_PER_LAP) * lap_ps[..., np.newaxis]
        + centre[:, :, code % TAPS_PER_LAP]
    )


def compute_tdc_lut(
    path: str,
    tag: str,
    *,
    nframes: int,
    clock_period_s: float,
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one SPAD's per-pixel code histogram and code -> ps LUT.

    Accumulates the per-pixel code histogram for one SPAD tag (already trimmed
    to the codes the data populated) and integrates it into the code -> ps LUT.

    Parameters
    ----------
    path : str
        Path to the folder with the calibration '.bin' data files.
    tag : str
        Filename fragment selecting one SPAD's files (e.g. ``'SPAD0_S0'``).
    nframes : int
        Number of frames stored in each '.bin' file.
    clock_period_s : float
        One reference-clock period, in seconds. Sets the LUT's full scale.
    max_files : int | None, optional
        Process at most this many files (after natural sorting). The default
        is None (all files).

    Returns
    -------
    counts : np.ndarray
        The (32, 32, n_used) per-pixel code histogram (trimmed).
    lut : np.ndarray
        The (32, 32, n_used) code -> ps lookup table.
    """
    files = io.find_bin_files(path, tag)
    if max_files is not None:
        files = files[:max_files]

    counts = compute_code_histogram(files, nframes=nframes, label=tag)
    lut = histogram_to_lut(counts, clock_period_s=clock_period_s)
    return counts, lut


def plot_tdc_calibration(
    counts: np.ndarray,
    lut: np.ndarray,
    pixel: tuple[int, int] = (16, 16),
) -> plt.Figure:
    """QA plot of one pixel's density test: histogram, transfer curve, DNL.

    The full scale for the ideal-line reference is read off the LUT itself - a
    table built by 'histogram_to_lut' is self-describing, so there is nothing to
    pass and nothing that can disagree with the table being plotted.

    Parameters
    ----------
    counts : np.ndarray
        The (32, 32, n_codes) per-pixel code histogram.
    lut : np.ndarray
        The (32, 32, n_codes) code -> ps lookup table.
    pixel : tuple[int, int], optional
        The ``(row, col)`` pixel to inspect. The default is (16, 16).

    Returns
    -------
    plt.Figure
        The generated three-panel figure.
    """
    period_ps = timing.to_ps(lut_full_scale_s(lut))
    r, c = int(pixel[0]), int(pixel[1])
    h = counts[r, c].astype(np.float64)
    lut_pix = lut[r, c]
    n_codes = h.size
    codes = np.arange(n_codes)
    total = h.sum()

    # DNL in LSB: (bin width / mean bin width) - 1 = counts/mean_counts - 1.
    mean_counts = total / n_codes if n_codes else 0.0
    dnl = (h / mean_counts - 1.0) if mean_counts else np.zeros_like(h)
    mean_lsb_ps = period_ps / n_codes if n_codes else 0.0

    # Non-square, so the figure size comes from the active style (per the
    # package rule; only square sensor maps set their own).
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

    ax0.bar(codes, h, width=1.0, alpha=0.7)
    ax0.set_ylabel("Counts")
    ax0.set_title(
        f"Pixel ({r},{c})  code density test  "
        f"(n={total:.0f}, mean LSB {mean_lsb_ps:.1f} ps)"
    )
    ax0.set_xlim(0, n_codes)

    ax1.plot(codes, lut_pix, linewidth=1.5, label="LUT (density test)")
    ax1.plot(
        codes,
        codes * mean_lsb_ps,
        "--",
        linewidth=1.0,
        label="ideal (uniform)",
    )
    ax1.set_ylabel("Calibrated time  (ps)")
    ax1.legend()
    ax1.set_xlim(0, n_codes)

    ax2.plot(codes, dnl, linewidth=0.8)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_ylabel("DNL  (LSB)")
    ax2.set_xlabel("TDC code")
    ax2.set_xlim(0, n_codes)

    for ax in (ax0, ax1, ax2):
        ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig



def collect_and_save_luts(
    path: str,
    *,
    nframes: int,
    clock_period_s: float,
    tags: tuple[str, ...] = _SPAD_TAGS,
    max_files: int | None = None,
    save_qa: bool = True,
    qa_pixel: tuple[int, int] = (16, 16),
) -> dict[str, np.ndarray]:
    """Compute and save one per-pixel code -> ps LUT per SPAD.

    For every SPAD tag it unpacks the calibration files, builds the
    (32, 32, n_used) code -> ps lookup table, and saves it as
    ``TDC_LUT_<tag>_<clock>.npy`` in the ``processed`` folder created inside
    ``path``, where ``<clock>`` is the reference-clock period (e.g. ``100ns``).
    A QA plot for one pixel is saved per SPAD when ``save_qa`` is True.

    The clock is part of the file name because a LUT is only valid for the
    reference clock it was taken with, so several must be able to sit side by
    side without one shadowing another.

    Parameters
    ----------
    path : str
        Path to the folder with the calibration '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    clock_period_s : float
        One reference-clock period, in seconds (e.g. ``100e-9`` at 10 MHz).
        Sets the LUT's full scale and tags the file name.
    tags : tuple[str, ...], optional
        SPAD filename fragments to process. The default is the four Kelpie
        micropixel tags ``('SPAD0_S0', 'SPAD1_S1', 'SPAD2_S2', 'SPAD3_S3')``.
    max_files : int | None, optional
        Process at most this many files per tag. The default is None.
    save_qa : bool, optional
        Save a per-SPAD QA plot (histogram, transfer curve, DNL). The
        default is True.
    qa_pixel : tuple[int, int], optional
        Pixel shown in the QA plot. The default is (16, 16).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``tag -> (32, 32, n_used)`` code -> ps LUT for each SPAD
        found in the folder.

    Raises
    ------
    FileNotFoundError
        Raised when no files match any of ``tags``.
    """
    # LUTs are stage-1 data ('processed/'); the QA plots are stage-2 figures.
    results_dir = store.results_dir(path, _KIND, create=False)
    period_ps = timing.to_ps(clock_period_s)
    clock = _clock_tag(clock_period_s)
    luts: dict[str, np.ndarray] = {}

    for tag in tags:
        try:
            files = io.find_bin_files(path, tag)
        except FileNotFoundError:
            print(f"  (no files for tag '{tag}', skipping)")
            continue
        if max_files is not None:
            files = files[:max_files]

        print(
            f"\n> > > Density-test LUT for '{tag}' "
            f"({len(files)} files, {clock} clock) < < <"
        )
        counts = compute_code_histogram(files, nframes=nframes, label=tag)
        lut = histogram_to_lut(counts, clock_period_s=clock_period_s)

        n_used = counts.shape[-1]
        n_dead = int((counts.sum(axis=-1) == 0).sum())
        per_pixel_n = counts.sum(axis=-1)
        print(
            f"  {tag}: {n_used} codes  mean LSB {period_ps / n_used:.1f} ps  "
            f"events/pixel min/median/max "
            f"{per_pixel_n.min():.0f}/{np.median(per_pixel_n):.0f}/"
            f"{per_pixel_n.max():.0f}  dead pixels {n_dead}"
        )

        # Keep the established 'TDC_LUT_<tag>' stem - 'load_lut' globs it, and
        # the LUTs shipped in dapkel/params use the same naming - with the
        # clock appended so per-clock calibrations do not collide.
        out_path = store.save_map(
            lut,
            path,
            kind=_KIND,
            tag=tag,
            file_name=f"TDC_LUT_{tag}_{clock}.npy",
            meta={
                "n_codes": int(n_used),
                "clock_period_s": float(clock_period_s),
                "full_scale_s": lut_full_scale_s(lut),
                "code_s_mean": float(clock_period_s / n_used),
                "n_files": len(files),
                "dead_pixels": int(n_dead),
            },
            quiet=True,
        )
        print(f"  saved LUT -> {out_path}")

        if save_qa:
            fig = plot_tdc_calibration(counts, lut, qa_pixel)
            store.save_figure(
                fig, results_dir, f"TDC_LUT_{tag}_{clock}_qa.png"
            )
            plt.close(fig)

        luts[tag] = lut

    if not luts:
        raise FileNotFoundError(
            "No files for any SPAD tag "
            f"{tags} found in:\n  {path}"
        )
    return luts


def _narrow_to_covering(
    luts: dict[str, np.ndarray], min_codes: int | None
) -> dict[str, np.ndarray]:
    """Keep the tables whose full scale actually fits the run's code range.

    Covering the run is necessary but not sufficient: a 2 MHz table covers a
    10 MHz run five times over, and using it would convert every code on the
    wrong scale. So among the tables that cover, only those close to the
    *tightest* one are kept - the calibration whose full scale the run's own
    code range points at.

    Parameters
    ----------
    luts : dict[str, np.ndarray]
        Mapping file name -> LUT.
    min_codes : int | None
        Codes the run occupies, or None to keep everything.

    Returns
    -------
    dict[str, np.ndarray]
        The surviving subset, or all of ``luts`` when nothing covers the run -
        the coverage check downstream then reports it properly.
    """
    if min_codes is None:
        return luts
    covering = {
        name: lut
        for name, lut in luts.items()
        if lut.shape[-1] >= min_codes * (1.0 - _CODE_COVERAGE_RTOL)
    }
    if not covering:
        return luts
    tightest = min(lut.shape[-1] for lut in covering.values())
    return {
        name: lut
        for name, lut in covering.items()
        if lut.shape[-1] <= tightest * (1.0 + _CODE_COVERAGE_RTOL)
    }


def _average_luts(
    luts: dict[str, np.ndarray], source: str
) -> np.ndarray:
    """Average a set of named LUTs, guarding against shape mismatches.

    Parameters
    ----------
    luts : dict[str, np.ndarray]
        Mapping name -> (32, 32, n_codes) LUT.
    source : str
        Human-readable origin, used only in error messages.

    One SPAD's calibration commonly stops a code or two short of another's -
    the top codes are the sparsest, so whether the last one was ever hit is
    chance. Tables that agree to within ``_CODE_COVERAGE_RTOL`` are therefore
    padded up to the longest with each one's own final value, which is its full
    scale: the honest time for a code that SPAD never produced. A wider
    disagreement means different reference clocks, and that still raises.

    Returns
    -------
    np.ndarray
        The element-wise mean LUT.

    Raises
    ------
    ValueError
        Raised when the LUTs span more than one calibration.
    """
    tags = {_lut_clock_tag(name) for name in luts}
    if len(tags) > 1:
        raise ValueError(
            f"Cannot average across {len(tags)} calibrations in {source}: "
            f"{sorted(tags)}. Averaging is meant for the four SPADs of one "
            "calibration, not for several. Name one with 'clock=...'."
        )

    lengths = {lut.shape[-1] for lut in luts.values()}
    longest = max(lengths)
    if min(lengths) < longest * (1.0 - _CODE_COVERAGE_RTOL):
        raise ValueError(
            f"Cannot average LUTs with differing shapes "
            f"{ {lut.shape for lut in luts.values()} } in {source}: they span "
            "more than one reference clock. Select one with 'clock=...'."
        )
    if len(lengths) == 1:
        return np.mean(list(luts.values()), axis=0)

    padded = [
        np.pad(lut, ((0, 0), (0, 0), (0, longest - lut.shape[-1])), mode="edge")
        for lut in luts.values()
    ]
    return np.mean(padded, axis=0)


def _select_lut(
    luts: dict[str, np.ndarray],
    spad: int | str,
    source: str,
    *,
    fallback_average: bool = False,
) -> np.ndarray:
    """Pick one SPAD's LUT from a named set, or the average.

    Parameters
    ----------
    luts : dict[str, np.ndarray]
        Mapping file name -> (32, 32, n_codes) LUT.
    spad : int | str
        ``'average'`` for the mean, an int ``0..3`` for ``SPAD{n}``, or a
        tag fragment (e.g. ``'SPAD0_S0'``, ``'S3'``) matched against the
        names.
    source : str
        Human-readable origin, used in warnings/errors.
    fallback_average : bool, optional
        When True and a specific ``spad`` matches no file, warn and fall
        back to the average instead of raising. The default is False.

    Returns
    -------
    np.ndarray
        The selected (or averaged) LUT.

    Raises
    ------
    ValueError
        Raised when ``spad`` does not resolve to exactly one file (and no
        average fallback applies), or when averaging hits mixed shapes.
    """
    if isinstance(spad, str) and spad.lower() == "average":
        return _average_luts(luts, source)

    key = f"SPAD{int(spad)}" if isinstance(spad, (int, np.integer)) else str(spad)
    key = key.lower()
    matches = [name for name in luts if key in name.lower()]
    if len(matches) == 1:
        return luts[matches[0]]
    if len(matches) == 0 and fallback_average:
        print(
            f"  WARNING: no LUT for spad={spad!r} in {source}; "
            "using the average of the available SPADs."
        )
        return _average_luts(luts, source)
    hint = (
        " Several clocks are installed - narrow them with 'clock=...' (or let "
        "min_codes do it)."
        if len({_lut_clock_tag(name) for name in matches}) > 1
        else ""
    )
    raise ValueError(
        f"spad={spad!r} matched {len(matches)} file(s) in {source}; expected "
        f"exactly one.{hint} Available: {sorted(luts)}"
    )


def collect_and_save_compact(
    path: str,
    *,
    nframes: int,
    clock_period_s: float,
    tags: tuple[str, ...] = _SPAD_TAGS,
    max_files: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute and save one per-pixel 8-tap calibration per SPAD.

    The compact counterpart of 'collect_and_save_luts': same code histogram,
    reduced to the ring's ``TAPS_PER_LAP`` widths instead of one width per code,
    and saved as ``TDC_COMPACT_<tag>.npy`` in ``path``'s ``processed`` folder.

    **No clock in the file name, on purpose.** The tap widths belong to the
    ring oscillator, not to the reference clock, so one table serves every
    clock; 'compact_to_lut' generates a code -> ps table of whatever length the
    run being calibrated needs. The clock it was *measured* at is recorded in
    the sidecar, because that is what set the absolute scale.

    Parameters
    ----------
    path : str
        Path to the folder with the calibration '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    clock_period_s : float
        One reference-clock period, in seconds. Sets the absolute scale. A slow
        clock measures the ring period best - it sweeps far more laps per
        acquisition - while the tap *shape* comes out the same at any clock.
    tags : tuple[str, ...], optional
        SPAD filename fragments to process. The default is the four Kelpie
        micropixel tags.
    max_files : int | None, optional
        Process at most this many files per tag. The default is None.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``tag -> (32, 32, 8)`` tap widths in picoseconds.

    Raises
    ------
    FileNotFoundError
        Raised when no files match any of ``tags``.
    """
    compact: dict[str, np.ndarray] = {}

    for tag in tags:
        try:
            files = io.find_bin_files(path, tag)
        except FileNotFoundError:
            print(f"  (no files for tag '{tag}', skipping)")
            continue
        if max_files is not None:
            files = files[:max_files]

        print(
            f"\n> > > Compact 8-tap calibration for '{tag}' "
            f"({len(files)} files, measured at {_clock_tag(clock_period_s)}) < < <"
        )
        counts = compute_code_histogram(files, nframes=nframes, label=tag)
        taps = histogram_to_compact(counts, clock_period_s=clock_period_s)

        n_used = counts.shape[-1]
        n_dead = int((counts.sum(axis=-1) == 0).sum())
        lap = taps.sum(axis=-1)
        spread = taps / taps.mean(axis=-1, keepdims=True) - 1.0
        print(
            f"  {tag}: ring period {lap.mean():.2f} ps "
            f"(+/- {lap.std() / lap.mean() * 100:.2f} % across pixels)  "
            f"tap spread {spread.std() * taps.mean():.2f} ps rms  "
            f"dead pixels {n_dead}"
        )

        out_path = store.save_map(
            taps,
            path,
            kind=_KIND,
            tag=tag,
            file_name=f"TDC_COMPACT_{tag}.npy",
            meta={
                "taps_per_lap": TAPS_PER_LAP,
                "measured_at_clock_period_s": float(clock_period_s),
                "measured_over_n_codes": int(n_used),
                "ring_period_ps_mean": float(lap.mean()),
                "n_files": len(files),
                "dead_pixels": int(n_dead),
            },
            quiet=True,
        )
        print(f"  saved compact calibration -> {out_path}")
        compact[tag] = taps

    if not compact:
        raise FileNotFoundError(
            f"No '.bin' files matching any of {tags} in:\n  {path}"
        )
    return compact


def _load_compact_set(
    files: dict[str, np.ndarray], spad: int | str, source: str
) -> np.ndarray:
    """Pick one SPAD's tap widths from a named set, or average the four.

    Averaging is unconditional here, unlike the dense tables: every compact
    calibration is (32, 32, 8) whatever clock it came from, so there are no
    ragged code counts to reconcile.
    """
    if isinstance(spad, str) and spad.lower() == "average":
        return np.mean(list(files.values()), axis=0)
    key = (
        f"SPAD{int(spad)}" if isinstance(spad, (int, np.integer)) else str(spad)
    ).lower()
    matches = [name for name in files if key in name.lower()]
    if len(matches) == 1:
        return files[matches[0]]
    if not matches:
        print(
            f"  WARNING: no compact calibration for spad={spad!r} in {source}; "
            "using the average of the available SPADs."
        )
        return np.mean(list(files.values()), axis=0)
    raise ValueError(
        f"spad={spad!r} matched {len(matches)} file(s) in {source}; expected "
        f"exactly one. Available: {sorted(files)}"
    )


def load_board_compact(
    daughterboard_number: str,
    motherboard_number: str,
    spad: int | str = "average",
) -> np.ndarray:
    """Load the packaged 8-tap calibration for a given board.

    Reads ``{daughterboard}_{motherboard}_TDC_COMPACT_SPAD{n}_S{n}.npy`` from
    ``dapkel/params/calibration_data``. There is no clock to choose: pass the
    result to 'compact_to_lut' with the run's own code count.

    Parameters
    ----------
    daughterboard_number : str
        Daughterboard id, e.g. ``'D0'``.
    motherboard_number : str
        Motherboard id, e.g. ``'M0'``.
    spad : int | str, optional
        ``'average'`` (default) for the mean of the board's four SPADs, an int
        ``0..3``, or a tag fragment.

    Returns
    -------
    np.ndarray
        The (32, 32, 8) tap widths in picoseconds.

    Raises
    ------
    FileNotFoundError
        Raised when no packaged compact calibration exists for the board.
    """
    board = f"{daughterboard_number}_{motherboard_number}"
    data_dir = _resource_files("dapkel.params").joinpath("calibration_data")
    entries = [
        t
        for t in data_dir.iterdir()
        if t.name.endswith(".npy")
        and t.name.startswith(f"{board}_")
        and "_TDC_COMPACT_" in t.name
    ]
    if not entries:
        raise FileNotFoundError(
            f"No packaged compact TDC calibration for board "
            f"{daughterboard_number}/{motherboard_number} (looked for "
            f"{board}_*TDC_COMPACT_*.npy in dapkel/params/calibration_data). "
            "Build one with 'collect_and_save_compact'."
        )

    files: dict[str, np.ndarray] = {}
    for t in sorted(entries, key=lambda t: t.name):
        with t.open("rb") as fh:
            files[t.name] = np.load(fh)
    return _load_compact_set(
        files, spad, f"packaged board {board} (dapkel/params/calibration_data)"
    )


def load_lut(
    path: str,
    spad: int | str = "average",
    *,
    min_codes: int | None = None,
    clock: str | None = None,
) -> np.ndarray:
    """Load a per-pixel code -> ps LUT from a folder of '.npy' files.

    Reads the ``*TDC_LUT_*.npy`` files in a folder (as written by
    'collect_and_save_luts') and returns a single (32, 32, n_codes) lookup
    table ready to pass to 'delta_t' as ``time_lut``. For the LUTs *shipped
    with the package*, use 'load_board_lut' instead.

    Because the Kelpie ORT program stores one timestamp per *macropixel* and
    does not record which of the four micropixels fired, the caller must
    decide which SPAD's calibration to trust: pick a specific SPAD when the
    optics illuminate a known micropixel, or use ``'average'`` (the four
    SPADs differ by ~1 LSB, so the mean is the best single estimate and the
    residual spread is an irreducible systematic).

    Parameters
    ----------
    path : str
        Either the folder holding the ``*TDC_LUT_*.npy`` files (e.g.
        ``.../results/tdc_calibration``) or a direct path to a single
        ``.npy`` LUT file (in which case ``spad`` is ignored).
    spad : int | str, optional
        Which LUT to return when ``path`` is a folder:

        * ``'average'`` (default) - mean of all SPAD LUTs found;
        * an int ``0..3`` - the ``SPAD{n}`` file;
        * a tag fragment such as ``'SPAD0_S0'``, ``'SPAD2'`` or ``'S3'`` -
          the single file whose name contains it.
    min_codes : int | None, optional
        Number of codes the data being calibrated occupies, i.e. its largest
        code + 1. Not a setting to remember - callers read it off the run (see
        'delta_t'). When given, a folder holding several clocks' calibrations is
        narrowed to the tables that cover the run, and the selected one is
        validated against it, which is what stops a 10 MHz LUT silently
        discarding 79 % of a 2 MHz run. The default is None (no filtering, no
        check).
    clock : str | None, optional
        Select the calibration by the clock tag in its file name, e.g.
        ``'2MHz'``. Names the table outright rather than inferring it from code
        coverage, which is the only way to choose between two calibrations of
        the *same* clock. Ignored when ``path`` is a single ``.npy`` file. The
        default is None (coverage decides).

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) code -> ps lookup table.

    Raises
    ------
    FileNotFoundError
        Raised when no ``*TDC_LUT_*.npy`` files are found in the folder.
    ValueError
        Raised when ``spad`` does not select exactly one file, when ``clock``
        matches no table or several calibrations, when the LUTs have mismatched
        shapes and cannot be averaged, or when the selected LUT does not cover
        ``min_codes``.
    """
    if os.path.isfile(path) and path.endswith(".npy"):
        return _check_code_coverage(np.load(path), min_codes, path)

    # Accept either the folder holding the LUTs directly, or a data folder
    # whose 'processed/' holds them (where collect_and_save_luts writes).
    files = sorted(
        glob.glob(os.path.join(path, "*TDC_LUT_*.npy"))
        or glob.glob(
            os.path.join(path, store.PROCESSED_DIR, "*TDC_LUT_*.npy")
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No *TDC_LUT_*.npy files found in:\n  {path}\n"
            f"  (nor in {os.path.join(path, store.PROCESSED_DIR)})\n"
            "Run tdc_calibration.collect_and_save_luts first."
        )

    luts = {os.path.basename(f): np.load(f) for f in files}

    # An explicit clock is the caller's decision and outranks the inference
    # below; coverage then still has to agree with the data.
    luts = _filter_by_clock(luts, clock, path)

    # Several clocks may be calibrated into one folder. Keep the tables that
    # actually fit the data - read off the LUTs themselves rather than a file
    # name, so a renamed or untagged legacy LUT is judged on its contents.
    luts = _narrow_to_covering(luts, min_codes)

    return _check_code_coverage(_select_lut(luts, spad, path), min_codes, path)


def load_board_lut(
    daughterboard_number: str,
    motherboard_number: str,
    spad: int | str = "average",
    *,
    min_codes: int | None = None,
    clock: str | None = None,
) -> np.ndarray:
    """Load the packaged per-pixel code -> ps LUT for a given board.

    Resolves the calibration '.npy' files shipped inside the library
    (``dapkel/params/calibration_data``) for the requested board and returns
    a single (32, 32, n_codes) lookup table for 'delta_t'. Three naming forms
    are accepted, the data directory being located via ``importlib.resources``
    so this works both from source and from a pip-installed wheel:

    * ``{daughterboard}_{motherboard}_TDC_LUT_SPAD{n}_S{n}_{clock}.npy`` -
      the current form, e.g. ``D0_M0_TDC_LUT_SPAD0_S0_2MHz_2026-08-13.npy``.
      Anything after the frequency (here a date) is part of the tag, so two
      calibrations of one clock can sit side by side;
    * ``{daughterboard}_{motherboard}_{clock}_TDC_LUT_SPAD{n}_S{n}.npy`` -
      the older per-clock layout, e.g. ``D0_M0_500ns_TDC_LUT_SPAD0_S0.npy``;
    * ``{daughterboard}_{motherboard}_TDC_LUT_SPAD{n}_S{n}.npy`` - untagged
      legacy tables. They were taken at 10 MHz (100 ns full scale), so they are
      **not** valid for another clock; the ``min_codes`` check catches that.

    Without ``clock`` the table is chosen by code coverage, not by the file
    name, so a table only has to be honest about its own contents. Pass
    ``clock`` when coverage cannot decide - two calibrations of the same clock
    cover identically.

    Parameters
    ----------
    daughterboard_number : str
        Daughterboard id, e.g. ``'D0'`` (the ``{daughterboard}`` prefix).
    motherboard_number : str
        Motherboard id, e.g. ``'M0'`` (the ``{motherboard}`` prefix).
    spad : int | str, optional
        ``'average'`` (default) for the mean of the board's four SPAD LUTs,
        an int ``0..3`` for a specific SPAD, or a tag fragment. If a specific
        SPAD is requested but its file is missing, the average is used with a
        warning.
    min_codes : int | None, optional
        Number of codes the data being calibrated occupies, i.e. its largest
        code + 1, read off the run by the caller. Narrows a multi-clock set to
        the tables that cover the data and validates the choice. The default is
        None (no filtering, no check).
    clock : str | None, optional
        Clock tag naming the calibration to use: ``'2MHz'``, ``'10MHz'`` or
        ``'50MHz'`` as shipped. An exact tag match wins, so a plain ``'10MHz'``
        keeps selecting the table named exactly that even with a qualified one
        beside it; otherwise the tag may be given in part and is resolved when
        exactly one calibration answers to it. The default is None (coverage
        decides).

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) code -> ps lookup table for the board.

    Raises
    ------
    FileNotFoundError
        Raised when no packaged LUTs exist for the requested board.
    ValueError
        Raised when ``spad`` is ambiguous, when ``clock`` matches no table or
        several calibrations, when the LUTs have mixed shapes, or when the
        selected LUT does not cover ``min_codes``.
    """
    board = f"{daughterboard_number}_{motherboard_number}"
    data_dir = _resource_files("dapkel.params").joinpath("calibration_data")
    names = [t for t in data_dir.iterdir() if t.name.endswith(".npy")]

    # Any file for this board, per-clock or untagged: '_TDC_LUT_' sits after the
    # board prefix in both forms, so one filter covers them.
    entries = [
        t
        for t in names
        if t.name.startswith(f"{board}_") and "_TDC_LUT_" in t.name
    ]

    if not entries:
        raise FileNotFoundError(
            f"No packaged TDC LUTs for board "
            f"{daughterboard_number}/{motherboard_number} "
            f"(looked for {board}_*TDC_LUT_*.npy in "
            "dapkel/params/calibration_data). Available: "
            f"{sorted(t.name for t in names)}"
        )

    luts: dict[str, np.ndarray] = {}
    for t in sorted(entries, key=lambda t: t.name):
        with t.open("rb") as fh:
            luts[t.name] = np.load(fh)

    source_dir = f"dapkel/params/calibration_data (board {board})"
    luts = _filter_by_clock(luts, clock, source_dir)

    # Prefer the tables that fit the run; the check below still has the last
    # word when none of them does.
    luts = _narrow_to_covering(luts, min_codes)

    source = f"packaged board {board} (dapkel/params/calibration_data)"
    return _check_code_coverage(
        _select_lut(luts, spad, source, fallback_average=True),
        min_codes,
        source,
    )
