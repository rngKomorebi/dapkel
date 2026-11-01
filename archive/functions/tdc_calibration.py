"""ARCHIVED from dapkel.functions.tdc_calibration (branch modules-restructure, 2026-10-31).

The dense per-clock code -> ps LUT: one width per code, one table per reference clock, selected by clock tag and code coverage. Superseded by the compact 8-tap calibration, which serves every clock.

Reference only: not packaged, not tested, not maintained. Copied verbatim
from the pre-restructure module (commit 6b49a4b), so it still refers to
helpers and names that no longer exist in 'src/'. To revive a piece, port it
into the live module rather than importing it from here.
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
