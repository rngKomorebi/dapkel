"""Per-pixel TDC calibration: the ring's 8 tap widths, valid at every clock.

    process_tdc_calibration   density-test '.bin' -> processed/TDC_COMPACT_<tag>.npy
    plot_tdc_calibration      those tables -> results/tdc_calibration/ QA figures

The ring-oscillator TDC has non-uniform bin widths (DNL), so a flat
``code * 71.6 ps`` is only an average. Under illumination uncorrelated with the
oscillator - the statistical code density test - each code's hit probability is
proportional to its width. And the hardware only has ``TAPS_PER_LAP`` = 8
distinct widths: code ``k`` is always tap ``k % 8`` and the coarse counter adds
none, so averaging each tap over every lap a run swept estimates the same eight
numbers from ~n/8 times as many events. That is the whole calibration,
``(32, 32, 8)`` per SPAD, and it serves every reference clock, which only sets
how many laps are swept. 'core.calibration' applies it.

``clock_period_s`` - one period of the reference clock the density test ran at
- is **the only time this package ever has to be told**: the test measures
relative widths only, so a known duration sets the absolute scale. Every other
timescale downstream is measured from the data.

Copy the tables into ``dapkel/params/calibration_data`` as
``<daughterboard>_<motherboard>_TDC_COMPACT_<tag>.npy`` to ship them.
See ``docs/guide/tdc_calibration.md``.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import io, reduce, store, timing
from dapkel.core.calibration import TAPS_PER_LAP

__all__ = ["process_tdc_calibration", "plot_tdc_calibration"]

#: Analysis name: the results sub-folder for the QA plots.
_KIND = "tdc_calibration"

# The four single-micropixel timestamping programs, one per 2x2 macropixel
# corner. Only the program tag is fixed in a file name; the rest of it varies
# from data set to data set.
_SPAD_TAGS = ("S0T", "S1T", "S2T", "S3T")


def _histogram_to_taps(counts: np.ndarray, clock_period_s: float) -> np.ndarray:
    """Reduce a (32, 32, n_codes) code histogram to (32, 32, 8) tap widths.

    Each pixel is normalised over **its own** populated code range, not the
    array-wide one, so its time-base gain survives: a pixel whose ring runs
    fast reaches more codes in one clock period and gets narrower taps. Dead
    pixels get a flat nominal lap.
    """
    period_ps = timing.to_ps(clock_period_s)
    counts = counts.astype(np.float64)
    n_codes = counts.shape[-1]
    tap_of = np.arange(n_codes) % TAPS_PER_LAP

    # Where each pixel's own ruler ends: past its last code it contributes
    # nothing, and counting those codes would flatten the gain spread.
    populated = counts > 0
    n_used = np.where(
        populated.any(axis=-1), n_codes - np.argmax(populated[..., ::-1], axis=-1), 0
    )
    totals = np.stack(
        [counts[..., tap_of == j].sum(-1) for j in range(TAPS_PER_LAP)], -1
    )
    laps = np.stack(
        [
            ((tap_of == j) & (np.arange(n_codes) < n_used[..., None])).sum(-1)
            for j in range(TAPS_PER_LAP)
        ],
        -1,
    ).astype(np.float64)

    # Mean counts per occurrence of a tap is its width, up to the one scale the
    # clock fixes: the swept codes span exactly one clock period.
    with np.errstate(invalid="ignore", divide="ignore"):
        per_lap = np.where(laps > 0, totals / laps, 0.0)
    total = counts.sum(axis=-1)
    taps = np.full(counts.shape[:2] + (TAPS_PER_LAP,), period_ps / max(n_codes, 1))
    live = total > 0
    taps[live] = period_ps * per_lap[live] / total[live][:, None]
    return taps


def process_tdc_calibration(
    path: str,
    nframes: int,
    clock_period_s: float,
    *,
    tags: tuple[str, ...] = _SPAD_TAGS,
    max_files: int | None = None,
) -> list[str]:
    """Build and save one (32, 32, 8) tap-width table per SPAD.

    For every SPAD tag found it histograms the codes of the density-test files
    and reduces them to the tap widths, saved as ``TDC_COMPACT_<tag>.npy`` in
    ``processed/``. The clock it was measured at goes in the sidecar, not the
    file name: the table belongs to the ring, not to the clock.

    Parameters
    ----------
    path : str
        Folder with the density-test '.bin' files.
    nframes : int
        Frames the acquisition recorded per '.bin' file.
    clock_period_s : float
        One reference-clock period of the density test, in seconds (``500e-9``
        at 2 MHz). Sets the absolute scale. A slow clock sweeps more laps per
        frame and so measures the ring period best; the tap shape comes out the
        same at any clock.
    tags : tuple[str, ...], optional
        Program tags to process, matched as ``*_<tag>*.bin`` whatever the rest
        of the file name is. The default is the four single-micropixel
        timestamping programs, ``('S0T', 'S1T', 'S2T', 'S3T')``. ``'ORT'``
        works too, but mixes the four micropixels into one table.
    max_files : int | None, optional
        Use at most this many files per tag. The default is None (all).

    Returns
    -------
    list[str]
        Paths to the saved tables.

    Raises
    ------
    ValueError
        Raised when a tag is not a timestamping (``...T``) program: counting
        programs record no TDC codes.
    FileNotFoundError
        Raised when no files match any of ``tags``.
    """
    for tag in tags:
        if reduce.readout_mode(tag) != "timestamp":
            raise ValueError(
                f"tag {tag!r} is a counting program, which records no TDC "
                "codes. The density test needs a timestamping one ('S0T', 'ORT')."
            )
    saved = []
    for tag in tags:
        try:
            files = io.find_bin_files(path, tag, require_separator=True)[:max_files]
        except FileNotFoundError:
            print(f"  (no files for tag '{tag}', skipping)")
            continue
        print(
            f"\n> > > Tap widths for '{tag}' ({len(files)} files, "
            f"{1e-6 / clock_period_s:g} MHz clock) < < <"
        )
        counts = reduce.code_histogram(files, nframes=nframes, label=tag)
        taps = _histogram_to_taps(counts, clock_period_s)

        lap = taps.sum(axis=-1)
        n_dead = int((counts.sum(axis=-1) == 0).sum())
        print(
            f"  ring period {lap.mean():.2f} ps (+/- "
            f"{lap.std() / lap.mean() * 100:.2f} % across pixels)  "
            f"dead pixels {n_dead}"
        )
        saved.append(
            store.save_map(
                taps,
                path,
                kind=_KIND,
                file_name=f"TDC_COMPACT_{tag}.npy",
                meta={
                    "tag": tag,
                    "taps_per_lap": TAPS_PER_LAP,
                    "measured_at_clock_period_s": float(clock_period_s),
                    "measured_over_n_codes": int(counts.shape[-1]),
                    "ring_period_ps_mean": float(lap.mean()),
                    "n_files": len(files),
                    "dead_pixels": n_dead,
                },
            )
        )
    if not saved:
        raise FileNotFoundError(
            f"No '.bin' files matching '*_<tag>*.bin' for any of {tags} in:\n  {path}"
        )
    return saved


def plot_tdc_calibration(
    path: str, pixel: tuple[int, int] = (16, 16)
) -> dict[str, np.ndarray]:
    """Save a QA figure per saved SPAD table: tap DNL and ring-period spread.

    Top panel: one pixel's tap widths as DNL against the mean tap
    (``width / mean - 1``, in LSB), over the array's median and 5-95 % band, so
    an outlier pixel stands out. Bottom panel: the ring period (the sum of the
    taps) across all pixels, the per-pixel time-base gain the table carries.

    Parameters
    ----------
    path : str
        Data folder that 'process_tdc_calibration' was run on.
    pixel : tuple[int, int], optional
        The ``(row, col)`` pixel drawn in the top panel. The default is
        ``(16, 16)``.

    Returns
    -------
    dict[str, np.ndarray]
        SPAD tag -> its (32, 32, 8) tap widths in picoseconds.

    Raises
    ------
    FileNotFoundError
        Raised when no saved table exists.
    """
    processed = store.processed_dir(path, create=False)
    found = sorted(glob.glob(os.path.join(processed, "TDC_COMPACT_*.npy")))
    if not found:
        raise FileNotFoundError(
            f"No TDC_COMPACT_*.npy in {processed}. Run process_tdc_calibration first."
        )
    r, c = int(pixel[0]), int(pixel[1])
    tables = {}
    for fp in found:
        tag = os.path.basename(fp).removeprefix("TDC_COMPACT_").removesuffix(".npy")
        taps, _ = store.load_map(npy_path=fp)
        tables[tag] = taps

        dnl = taps / taps.mean(axis=-1, keepdims=True) - 1.0
        lap = taps.sum(axis=-1).ravel()
        fig, (ax0, ax1) = plt.subplots(2, 1)
        x = np.arange(TAPS_PER_LAP)
        lo, mid, hi = np.percentile(dnl.reshape(-1, TAPS_PER_LAP), [5, 50, 95], axis=0)
        ax0.fill_between(x, lo, hi, step="mid", alpha=0.3, label="array 5-95 %")
        ax0.step(x, mid, where="mid", label="array median")
        ax0.plot(x, dnl[r, c], "o-", label=f"pixel ({r},{c})")
        ax0.axhline(
            0,
            color=plt.rcParams["axes.edgecolor"],
            linewidth=plt.rcParams["axes.linewidth"],
        )
        ax0.set_xlabel("Tap")
        ax0.set_ylabel("DNL  (LSB)")
        ax0.set_title(f"{tag}  tap widths, mean {taps.mean():.1f} ps")
        ax0.legend()

        ax1.hist(lap, bins=64)
        ax1.set_xlabel("Ring period  (ps)")
        ax1.set_ylabel("Pixels")
        ax1.set_title(
            f"Ring period: mean {lap.mean():.1f} ps, spread "
            f"{lap.std() / lap.mean() * 100:.2f} % rms"
        )
        for ax in (ax0, ax1):
            ax.grid(True)
        fig.tight_layout()
        store.save_figure(
            fig, store.results_dir(path, _KIND), f"TDC_COMPACT_{tag}_qa.png"
        )
        plt.close(fig)
    return tables
