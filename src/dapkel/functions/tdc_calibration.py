"""Per-pixel TDC calibration via the statistical code density test.

The ring-oscillator TDC has non-uniform bin widths (DNL), so a flat
``code * 71.6 ps`` conversion is only an average. Under illumination uncorrelated
with the oscillator, each code's hit probability is proportional to its width,
which turns a per-pixel code histogram into a code -> ps lookup table:

    LUT[k] = full_scale * (cumsum(counts)[k] - counts[k]/2) / total

One (32, 32, n_codes) LUT per SPAD; feed it to 'delta_t' as ``time_lut``.

**A LUT is specific to the reference clock it was taken with.** The ring sets the
code width (~71.6 ps, see 'dapkel.core.timing.CODE_S') and the clock sets the
full scale, so the number of codes scales as 1/f: ~279 codes at 50 MHz, ~1397 at
10 MHz, ~2794 at 5 MHz, ~6985 at 2 MHz. Feeding a 10 MHz LUT to a 2 MHz run
therefore discards every code past 100 ns - 79 % of the singles and 96 % of the
coincidences - so every entry point here takes ``open_shutter_time_s`` and every
loader validates it against the LUT's own full scale (a LUT is self-describing:
its largest entry *is* its full scale). Calibrate once per clock.

Every time parameter is in seconds, per 'dapkel.core.timing'.
See ``docs/guide/tdc_calibration.md``.
"""

from __future__ import annotations

import glob
import os
import warnings
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
    # drivers - folder in, LUTs and QA figures on disk
    "collect_and_save_luts",
    # load saved / shipped LUTs
    "load_lut",
    "load_board_lut",
    # introspection - a LUT describes its own full scale
    "lut_full_scale_s",
]

#: Analysis name: the results sub-folder for the QA plots.
_KIND = "tdc_calibration"

# The four SPAD (micropixel) calibration tags, one per 2x2 macropixel corner.
_SPAD_TAGS = ("SPAD0_S0", "SPAD1_S1", "SPAD2_S2", "SPAD3_S3")

# How far a LUT's own full scale may sit from the one a caller asks for before
# it is treated as the wrong calibration. The populated code range stops a few
# codes short of full scale in any finite sample (1378 of ~1397 at 10 MHz, i.e.
# ~1.4 %), so the tolerance has to absorb that without admitting a whole
# different clock - the clocks in use differ by factors of 2 to 5.
_FULL_SCALE_RTOL = 0.05


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


def _shutter_tag(open_shutter_time_s: float) -> str:
    """File-name fragment identifying the clock a calibration belongs to.

    Formats the window in nanoseconds with ``'p'`` for the decimal point, so
    LUTs for several clocks coexist in one folder: ``20ns``, ``100ns``,
    ``500ns``, ``19p6ns``.

    Parameters
    ----------
    open_shutter_time_s : float
        Open shutter time (one reference-clock period), in seconds.

    Returns
    -------
    str
        The tag, e.g. ``'100ns'``.
    """
    return f"{open_shutter_time_s * 1e9:g}ns".replace(".", "p")


def _check_full_scale(
    lut: np.ndarray, open_shutter_time_s: float | None, source: str
) -> np.ndarray:
    """Verify a loaded LUT was built for the requested clock.

    Parameters
    ----------
    lut : np.ndarray
        The loaded code -> ps lookup table.
    open_shutter_time_s : float | None
        The full scale the caller expects, in seconds, or None to skip the
        check.
    source : str
        Human-readable origin, used in the error message.

    Returns
    -------
    np.ndarray
        The unchanged ``lut``, once validated.

    Raises
    ------
    ValueError
        Raised when the LUT's own full scale differs from the requested one by
        more than 5 %, which means it belongs to a different reference clock.
    """
    if open_shutter_time_s is None:
        return lut
    got = lut_full_scale_s(lut)
    if abs(got - open_shutter_time_s) > _FULL_SCALE_RTOL * open_shutter_time_s:
        raise ValueError(
            f"The LUT in {source} spans {got * 1e9:.2f} ns but "
            f"open_shutter_time_s={open_shutter_time_s * 1e9:.2f} ns was "
            f"requested ({lut.shape[-1]} codes). A LUT is specific to its "
            "reference clock: using this one would silently drop every code "
            "past its full scale. Calibrate this clock with "
            "'collect_and_save_luts(..., open_shutter_time_s=...)'."
        )
    return lut
def compute_code_histogram(
    files: list[str],
    *,
    nframes: int,
    open_shutter_time_s: float | None = None,
    n_codes: int | None = None,
    label: str = "",
) -> np.ndarray:
    """Accumulate the (32, 32, n_codes) per-pixel TDC-code histogram.

    Unpacks each file's ``time_series`` (the raw TDC code) and, for every
    pixel independently, counts how often each integer code occurs across
    all frames. Only frames in which the pixel carried a valid timestamp
    contribute - 'unpack' leaves non-fired slots at ``<= 0``, so that is
    ``time_series > 0``.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files to unpack and accumulate.
    nframes : int
        Number of frames stored in each '.bin' file.
    open_shutter_time_s : float | None, optional
        Open shutter time, i.e. one reference-clock period, in seconds. Sizes
        the code axis via 'core.timing.codes_for_full_scale'. Required unless
        ``n_codes`` is given. The default is None.
    n_codes : int | None, optional
        Number of code bins (0 .. n_codes-1), overriding the size derived from
        ``open_shutter_time_s``. Codes at or above this are dropped with a
        warning. The default is None.
    label : str, optional
        Label used in the progress printout. The default is "".

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) int64 per-pixel code histogram.

    Raises
    ------
    ValueError
        Raised when neither ``open_shutter_time_s`` nor ``n_codes`` is given.

    Warns
    -----
    UserWarning
        Raised when any timestamp falls at or above ``n_codes``, which means
        the code axis is too short for the reference clock in use.
    """
    if n_codes is None:
        if open_shutter_time_s is None:
            raise ValueError(
                "compute_code_histogram needs open_shutter_time_s (one "
                "reference-clock period, in seconds) to size the code axis, "
                "or an explicit n_codes."
            )
        n_codes = timing.codes_for_full_scale(open_shutter_time_s)

    # Linear pixel index 0..1023 for a single per-file np.bincount that
    # histograms all 1024 pixels at once: lin = pixel * n_codes + code.
    pix_index = np.arange(1024, dtype=np.int64).reshape(32, 32)
    counts = np.zeros(1024 * n_codes, dtype=np.int64)
    n_overflow = 0
    n_valid = 0
    code_max = 0

    for fp in tqdm(files, desc=label or None):
        ts, _ = unpack(fp, nframes, compute_time_series=True)  # (32,32,nframes)

        code = ts.astype(np.int64)
        hit = ts > 0
        valid = hit & (code < n_codes)
        n_overflow += int((hit & (code >= n_codes)).sum())
        n_valid += int(hit.sum())
        if hit.any():
            code_max = max(code_max, int(code[hit].max()))

        lin = pix_index[:, :, np.newaxis] * n_codes + code
        counts += np.bincount(lin[valid], minlength=1024 * n_codes)

    if n_overflow:
        warnings.warn(
            f"Dropped {n_overflow} of {n_valid} timestamps "
            f"({100 * n_overflow / max(n_valid, 1):.1f} %) with code >= "
            f"{n_codes}; the largest code seen was {code_max}, i.e. a full "
            f"scale of ~{code_max * timing.CODE_S * 1e9:.0f} ns. The code axis "
            "is too short for this reference clock - pass the matching "
            "open_shutter_time_s (or a larger n_codes).",
            UserWarning,
            stacklevel=2,
        )
    return counts.reshape(32, 32, n_codes)


def histogram_to_lut(
    counts: np.ndarray, *, open_shutter_time_s: float
) -> np.ndarray:
    """Convert a per-pixel code histogram into a code -> ps lookup table.

    ``LUT[k] = full_scale * (cumsum(counts)[k] - counts[k]/2) / total``.
    Dead pixels (no counts) fall back to a linear ramp so downstream indexing
    never divides by zero. Method: ``docs/guide/tdc_calibration.md``.

    Parameters
    ----------
    counts : np.ndarray
        The (32, 32, n_codes) per-pixel code histogram from
        'compute_code_histogram'.
    open_shutter_time_s : float
        Full scale, in seconds, onto which the populated code range is mapped:
        one reference-clock period. The density test only fixes the *relative*
        per-code widths, so this is what sets the absolute scale - and it is
        what 'lut_full_scale_s' reads back out of the finished table.

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) float64 code -> ps lookup table. ``LUT[r, c]``
        maps an integer TDC code to picoseconds for pixel ``(r, c)``.
    """
    period_ps = timing.to_ps(open_shutter_time_s)
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


def compute_tdc_lut(
    folder: str,
    tag: str,
    *,
    nframes: int,
    open_shutter_time_s: float,
    n_codes: int | None = None,
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one SPAD's per-pixel code histogram and code -> ps LUT.

    Accumulates the per-pixel code histogram for one SPAD tag, trims the
    empty top codes, and integrates it into the code -> ps LUT.

    Parameters
    ----------
    folder : str
        Path to the folder with the calibration '.bin' data files.
    tag : str
        Filename fragment selecting one SPAD's files (e.g. ``'SPAD0_S0'``).
    nframes : int
        Number of frames stored in each '.bin' file.
    open_shutter_time_s : float
        Open shutter time, i.e. one reference-clock period, in seconds. Sizes
        the code axis and sets the LUT's full scale.
    n_codes : int | None, optional
        Number of code bins to allocate before trimming, overriding the size
        derived from ``open_shutter_time_s``. The default is None.
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
    files = io.find_bin_files(folder, tag)
    if max_files is not None:
        files = files[:max_files]

    counts = compute_code_histogram(
        files,
        nframes=nframes,
        open_shutter_time_s=open_shutter_time_s,
        n_codes=n_codes,
        label=tag,
    )
    counts = _trim_trailing_zero_codes(counts)
    lut = histogram_to_lut(counts, open_shutter_time_s=open_shutter_time_s)
    return counts, lut


def plot_tdc_calibration(
    counts: np.ndarray,
    lut: np.ndarray,
    pixel: tuple[int, int] = (16, 16),
    *,
    open_shutter_time_s: float | None = None,
) -> plt.Figure:
    """QA plot of one pixel's density test: histogram, transfer curve, DNL.

    Parameters
    ----------
    counts : np.ndarray
        The (32, 32, n_codes) per-pixel code histogram.
    lut : np.ndarray
        The (32, 32, n_codes) code -> ps lookup table.
    pixel : tuple[int, int], optional
        The ``(row, col)`` pixel to inspect. The default is (16, 16).
    open_shutter_time_s : float | None, optional
        Full scale, in seconds, used to build the LUT, for the ideal-line
        reference. The default is None, which reads it back off the LUT with
        'lut_full_scale_s'.

    Returns
    -------
    plt.Figure
        The generated three-panel figure.
    """
    if open_shutter_time_s is None:
        open_shutter_time_s = lut_full_scale_s(lut)
    period_ps = timing.to_ps(open_shutter_time_s)
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
    folder: str,
    *,
    nframes: int,
    open_shutter_time_s: float,
    tags: tuple[str, ...] = _SPAD_TAGS,
    n_codes: int | None = None,
    max_files: int | None = None,
    save_qa: bool = True,
    qa_pixel: tuple[int, int] = (16, 16),
) -> dict[str, np.ndarray]:
    """Compute and save one per-pixel code -> ps LUT per SPAD.

    For every SPAD tag it unpacks the calibration files, builds the
    (32, 32, n_used) code -> ps lookup table, and saves it as
    ``TDC_LUT_<tag>_<window>.npy`` in the ``processed`` folder created inside
    ``folder``, where ``<window>`` is the open shutter time (e.g. ``100ns``).
    A QA plot for one pixel is saved per SPAD when ``save_qa`` is True.

    The window is part of the file name because a LUT is only valid for the
    reference clock it was taken with, so several must be able to sit side by
    side without one shadowing another.

    Parameters
    ----------
    folder : str
        Path to the folder with the calibration '.bin' data files.
    nframes : int
        Number of frames stored in each '.bin' file.
    open_shutter_time_s : float
        Open shutter time, i.e. one reference-clock period, in seconds. Sizes
        the code axis, sets the LUT's full scale, and tags the file name.
    tags : tuple[str, ...], optional
        SPAD filename fragments to process. The default is the four Kelpie
        micropixel tags ``('SPAD0_S0', 'SPAD1_S1', 'SPAD2_S2', 'SPAD3_S3')``.
    n_codes : int | None, optional
        Number of code bins to allocate before trimming, overriding the size
        derived from ``open_shutter_time_s``. The default is None.
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
    """
    # LUTs are stage-1 data ('processed/'); the QA plots are stage-2 figures.
    results_dir = store.results_dir(folder, _KIND, create=False)
    period_ps = timing.to_ps(open_shutter_time_s)
    window = _shutter_tag(open_shutter_time_s)
    luts: dict[str, np.ndarray] = {}

    for tag in tags:
        try:
            files = io.find_bin_files(folder, tag)
        except FileNotFoundError:
            print(f"  (no files for tag '{tag}', skipping)")
            continue
        if max_files is not None:
            files = files[:max_files]

        print(
            f"\n> > > Density-test LUT for '{tag}' "
            f"({len(files)} files, {window} window) < < <"
        )
        counts = compute_code_histogram(
            files,
            nframes=nframes,
            open_shutter_time_s=open_shutter_time_s,
            n_codes=n_codes,
            label=tag,
        )
        counts = _trim_trailing_zero_codes(counts)
        lut = histogram_to_lut(counts, open_shutter_time_s=open_shutter_time_s)

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
        # window appended so per-clock calibrations do not collide.
        out_path = store.save_map(
            lut,
            folder,
            kind=_KIND,
            tag=tag,
            file_name=f"TDC_LUT_{tag}_{window}.npy",
            meta={
                "n_codes": int(n_used),
                "open_shutter_time_s": float(open_shutter_time_s),
                "full_scale_s": lut_full_scale_s(lut),
                "code_s_mean": float(open_shutter_time_s / n_used),
                "n_files": len(files),
                "dead_pixels": int(n_dead),
            },
            quiet=True,
        )
        print(f"  saved LUT -> {out_path}")

        if save_qa:
            fig = plot_tdc_calibration(
                counts, lut, qa_pixel, open_shutter_time_s=open_shutter_time_s
            )
            store.save_figure(
                fig, results_dir, f"TDC_LUT_{tag}_{window}_qa.png"
            )
            plt.close(fig)

        luts[tag] = lut

    if not luts:
        raise FileNotFoundError(
            "No files for any SPAD tag "
            f"{tags} found in:\n  {folder}"
        )
    return luts


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

    Returns
    -------
    np.ndarray
        The element-wise mean LUT.

    Raises
    ------
    ValueError
        Raised when the LUTs do not all share one shape.
    """
    shapes = {lut.shape for lut in luts.values()}
    if len(shapes) != 1:
        raise ValueError(
            f"Cannot average LUTs with differing shapes {shapes} in {source}."
        )
    return np.mean(list(luts.values()), axis=0)


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
    raise ValueError(
        f"spad={spad!r} matched {len(matches)} file(s) in {source}; expected "
        f"exactly one. Available: {sorted(luts)}"
    )


def load_lut(
    path: str,
    spad: int | str = "average",
    *,
    open_shutter_time_s: float | None = None,
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

        * ``'average'`` (default) — mean of all SPAD LUTs found;
        * an int ``0..3`` — the ``SPAD{n}`` file;
        * a tag fragment such as ``'SPAD0_S0'``, ``'SPAD2'`` or ``'S3'`` —
          the single file whose name contains it.
    open_shutter_time_s : float | None, optional
        Open shutter time, i.e. one reference-clock period, in seconds. When
        given, only LUTs tagged with that window are considered (falling back
        to untagged files), and the selected table's own full scale is checked
        against it. **Pass this.** A LUT is only valid for the clock it was
        taken with, and the failure mode without the check is silent: every
        code past the LUT's full scale is discarded downstream. The default is
        None (no filtering, no check).

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) code -> ps lookup table.

    Raises
    ------
    FileNotFoundError
        Raised when no ``*TDC_LUT_*.npy`` files are found in the folder.
    ValueError
        Raised when ``spad`` does not select exactly one file, when the LUTs
        have mismatched shapes and cannot be averaged, or when the selected
        LUT's full scale does not match ``open_shutter_time_s``.
    """
    if os.path.isfile(path) and path.endswith(".npy"):
        return _check_full_scale(np.load(path), open_shutter_time_s, path)

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

    # Several clocks may be calibrated into one folder. Prefer the files whose
    # name carries the requested window; only fall back to the whole set when
    # none does, so an untagged legacy LUT still loads (and is then validated).
    if open_shutter_time_s is not None:
        window = _shutter_tag(open_shutter_time_s)
        tagged = [f for f in files if window in os.path.basename(f)]
        if tagged:
            files = tagged

    luts = {os.path.basename(f): np.load(f) for f in files}
    return _check_full_scale(
        _select_lut(luts, spad, path), open_shutter_time_s, path
    )


def load_board_lut(
    daughterboard_number: str,
    motherboard_number: str,
    spad: int | str = "average",
    *,
    open_shutter_time_s: float | None = None,
) -> np.ndarray:
    """Load the packaged per-pixel code -> ps LUT for a given board.

    Resolves the calibration '.npy' files shipped inside the library
    (``dapkel/params/calibration_data``) for the requested board and returns
    a single (32, 32, n_codes) lookup table for 'delta_t'. Two naming forms are
    accepted, the data directory being located via ``importlib.resources`` so
    this works both from source and from a pip-installed wheel:

    * ``{daughterboard}_{motherboard}_{window}_TDC_LUT_SPAD{n}_S{n}.npy`` —
      per-clock, e.g. ``D0_M0_500ns_TDC_LUT_SPAD0_S0.npy``; preferred when
      ``open_shutter_time_s`` is given;
    * ``{daughterboard}_{motherboard}_TDC_LUT_SPAD{n}_S{n}.npy`` — untagged,
      which is what the LUTs shipped so far use. They were taken at 10 MHz
      (100 ns full scale), so they are **not** valid for another clock; the
      ``open_shutter_time_s`` check is what catches that.

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
    open_shutter_time_s : float | None, optional
        Open shutter time, i.e. one reference-clock period, in seconds. Selects
        the per-clock LUTs when they exist and validates the full scale either
        way. **Pass this**: the shipped tables cover 100 ns only, and without
        the check a run on another clock silently loses every code past that.
        The default is None (no filtering, no check).

    Returns
    -------
    np.ndarray
        The (32, 32, n_codes) code -> ps lookup table for the board.

    Raises
    ------
    FileNotFoundError
        Raised when no packaged LUTs exist for the requested board.
    ValueError
        Raised when ``spad`` is ambiguous, the LUTs have mixed shapes, or the
        selected LUT's full scale does not match ``open_shutter_time_s``.
    """
    board = f"{daughterboard_number}_{motherboard_number}"
    data_dir = _resource_files("dapkel.params").joinpath("calibration_data")
    names = [t for t in data_dir.iterdir() if t.name.endswith(".npy")]

    # Prefer a calibration taken at this window; fall back to the untagged
    # form, which '_check_full_scale' then accepts or rejects on its own merits.
    prefixes = [f"{board}_TDC_LUT_"]
    if open_shutter_time_s is not None:
        prefixes.insert(
            0, f"{board}_{_shutter_tag(open_shutter_time_s)}_TDC_LUT_"
        )

    entries: list = []
    for prefix in prefixes:
        entries = [t for t in names if t.name.startswith(prefix)]
        if entries:
            break

    if not entries:
        raise FileNotFoundError(
            f"No packaged TDC LUTs for board "
            f"{daughterboard_number}/{motherboard_number} "
            f"(looked for {' or '.join(repr(p + '*.npy') for p in prefixes)} "
            "in dapkel/params/calibration_data). Available: "
            f"{sorted(t.name for t in names)}"
        )

    luts: dict[str, np.ndarray] = {}
    for t in sorted(entries, key=lambda t: t.name):
        with t.open("rb") as fh:
            luts[t.name] = np.load(fh)

    source = f"packaged board {board} (dapkel/params/calibration_data)"
    return _check_full_scale(
        _select_lut(luts, spad, source, fallback_average=True),
        open_shutter_time_s,
        source,
    )
