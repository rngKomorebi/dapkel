"""Measure the accidental background instead of modelling it: frame shifting.

The ORT accidental background is a *triangle* filling +/-100 ns and it sits
directly under the coincidence peak, so the number you report depends on how
you model it (see ``docs/ort_triangle_background.md``). This module removes it
without a model.

An SPDC pair is always in **one frame**. So

    lag 0   pixel A and pixel B in the *same* frame     SPDC + accidentals
    lag k   pixel A in frame i, pixel B in frame i+k    accidentals ONLY

Both histograms see the same two pixels at the same rates over (almost) the
same number of frames, so their accidental content is identical in expectation
- but only the lag-0 one can hold a true pair. Subtract and the triangle
cancels bin-for-bin, leaving the coincidence peak on a zero baseline. No fit
model, no assumption that the pedestal is triangular; if it *is* accidental it
subtracts, and if it does not subtract it was never accidental.

Two things this does and does not buy you:

* it does **not** improve CAR. The residual still carries the Poisson noise of
  the accidentals it removed, ``sqrt(N_sig + N_bkg / K)`` for ``K`` background
  lags. Contrast is set by the acquisition, not by the analysis;
* it **does** give a model-free coincidence count (sum the residual) and a
  direct test of whether the pedestal is accidental at all - which is the
  useful diagnostic when the background looks too high.

These are the pieces of the technique and nothing else: the histogramming,
re-binning, fitting and artifact handling all belong to 'delta_t', which drives
this module through its ``subtract_background`` parameter. Nothing here reads
or writes a file.

See ``docs/guide/background_subtraction.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import pairs
from dapkel.functions.calc_diff import Pixel

__all__ = [
    # the default lag set, exposed so 'delta_t' can default to it
    "BACKGROUND_LAGS",
    # stage 1 - the lag-shifted differencing
    "compute_lagged_differences",
    "frames_per_lag",
    # stage 2 - subtract, integrate, look
    "subtract_background",
    "integrate_residual",
    "plot_background_subtraction",
]

#: Frame offsets used to estimate the accidentals. More lags mean a quieter
#: background estimate (its Poisson noise enters the residual divided by K),
#: at one extra histogram plane each. Eight is enough that the background
#: contributes ~12% of the residual variance, so the error is dominated by the
#: signal histogram itself and adding more lags buys almost nothing.
BACKGROUND_LAGS = (1, 2, 3, 4, 5, 6, 7, 8)


def compute_lagged_differences(
    pixel_timestamps: dict[Pixel, np.ndarray],
    pixels: Sequence[Sequence[Pixel]],
    delta_window: float | None = None,
    mode: str = "all_pairs",
    *,
    frame_lag: int = 0,
) -> dict[str, np.ndarray]:
    """Difference two pixel groups with group B offset by whole frames.

    ``frame_lag=0`` is an ordinary within-frame coincidence and reproduces
    'calc_diff.calculate_differences' exactly - same keys, same values, same
    order. ``frame_lag=k > 0`` pairs pixel A in frame ``i`` with pixel B in
    frame ``i + k``, which no SPDC pair can satisfy, so the result is a pure
    accidental sample.

    Parameters
    ----------
    pixel_timestamps : dict[tuple[int, int], np.ndarray]
        Mapping ``(row, col) -> 1D array`` of per-frame timestamps, NaN where
        the pixel did not fire. All arrays share the same length, as
        'delta_t' structures them.
    pixels : Sequence[Sequence[tuple[int, int]]]
        Two groups ``[group_a, group_b]``; a bare ``(row, col)`` is accepted
        for a one-pixel group.
    delta_window : float | None, optional
        Keep only differences with ``abs(delta) <= delta_window``, in the same
        units as the timestamps. The default is None (keep all).
    mode : str, optional
        ``'all_pairs'`` (default) or ``'1v1'``, as
        'calc_diff.calculate_differences'.
    frame_lag : int, optional
        Whole-frame offset applied to group B. The default is 0.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``"ra,ca-rb,cb" -> 1D array`` of differences. A lag longer
        than the run gives empty arrays rather than raising.

    Raises
    ------
    ValueError
        Raised on a negative ``frame_lag``, an unknown ``mode``, or unequal
        groups under ``'1v1'``.
    """
    if frame_lag < 0:
        raise ValueError(f"frame_lag must be >= 0, got {frame_lag}")

    out: dict[str, np.ndarray] = {}
    for a, b in pairs.pair_list(pixels, mode):
        ta = pixel_timestamps[a]
        tb = pixel_timestamps[b]
        hi = len(ta) - frame_lag
        if hi <= 0:
            out[pairs.pair_label(a, b)] = np.empty(0, dtype=np.float64)
            continue
        ta_w = ta[:hi]
        tb_w = tb[frame_lag:]
        both = ~np.isnan(ta_w) & ~np.isnan(tb_w)
        d = tb_w[both] - ta_w[both]
        if delta_window is not None:
            d = d[np.abs(d) <= delta_window]
        out[pairs.pair_label(a, b)] = d
    return out


def frames_per_lag(
    nframes: int, lags: Sequence[int], n_files: int = 1
) -> np.ndarray:
    """Frames each lag can pair up, given ``nframes`` per file.

    A lag of ``k`` loses the last ``k`` frames of every file - there is no
    frame ``i + k`` to pair them with - so it samples marginally fewer
    opportunities than lag 0 and its histogram must be rescaled before
    subtracting. This is what 'subtract_background' wants for ``frames_used``.

    Parameters
    ----------
    nframes : int
        Frames stored in each '.bin' file.
    lags : Sequence[int]
        The frame offsets, lag 0 first.
    n_files : int, optional
        Number of files the histogram covers. The default is 1.

    Returns
    -------
    np.ndarray
        ``(n_lags,)`` frame counts, int64. A lag longer than ``nframes``
        contributes 0 rather than a negative count.
    """
    per_file = np.maximum(int(nframes) - np.asarray(lags, dtype=np.int64), 0)
    return per_file * int(n_files)


def subtract_background(
    counts: np.ndarray,
    centers: np.ndarray,
    frames_used: Sequence[int] | np.ndarray,
) -> dict:
    """Subtract the shifted-frame accidentals from the lag-0 histogram.

    Each background lag is first rescaled by ``frames_used[0] /
    frames_used[k]`` (see 'frames_per_lag'). The correction is ~0.01% at
    typical frame counts and is applied anyway because it is free, and because
    it is the only thing standing between this and a wrong answer if someone
    uses a lag comparable to the frame count.

    Binning is the caller's business: pass whatever histogram you want the
    answer on ('delta_t.rebin_delta_counts' produces it), the same bins for
    every lag.

    Parameters
    ----------
    counts : np.ndarray
        ``(n_lags, n_bins)`` counts, lag 0 first.
    centers : np.ndarray
        ``(n_bins,)`` bin centres, in ps. Carried through so the result is a
        complete, plottable object.
    frames_used : Sequence[int] | np.ndarray
        ``(n_lags,)`` frames each lag paired up.

    Returns
    -------
    dict
        ``centers``, ``signal`` (lag 0), ``background`` (mean of the rescaled
        shifted lags), ``residual``, ``error`` (1 sigma, Poisson),
        ``bin_width_ps``, ``k`` (number of background lags) and ``scale`` (the
        per-lag frame-count correction applied).

    Raises
    ------
    ValueError
        Raised when ``counts`` is not 2D, has fewer than two lags, or does not
        match ``centers`` / ``frames_used``.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(
            f"counts must be (n_lags, n_bins); got shape {counts.shape}. "
            "Pool the pair axis first."
        )
    if counts.shape[0] < 2:
        raise ValueError(
            "need at least one background lag beside lag 0; got "
            f"{counts.shape[0]} lag(s)."
        )
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape[0] != counts.shape[1]:
        raise ValueError(
            f"centers has {centers.size} entries but counts has "
            f"{counts.shape[1]} bins."
        )
    frames = np.asarray(frames_used, dtype=np.float64)
    if frames.shape[0] != counts.shape[0]:
        raise ValueError(
            f"frames_used has {frames.size} entries but counts has "
            f"{counts.shape[0]} lags."
        )
    if np.any(frames <= 0):
        raise ValueError(f"frames_used must all be positive; got {frames}")

    scale = frames[0] / frames
    signal = counts[0]
    k = counts.shape[0] - 1
    background = (counts[1:] * scale[1:, None]).mean(axis=0)
    # Variance of the mean of K independent, rescaled Poisson histograms.
    # Written out rather than approximated as bkg/K so an unequal-frame lag
    # set stays correct.
    var_bkg = (counts[1:] * scale[1:, None] ** 2).sum(axis=0) / k**2

    cell_ps = float(np.median(np.diff(centers))) if centers.size > 1 else 1.0
    return {
        "centers": centers,
        "signal": signal,
        "background": background,
        "residual": signal - background,
        "error": np.sqrt(signal + var_bkg),
        "bin_width_ps": cell_ps,
        "k": k,
        "scale": scale,
    }


def integrate_residual(sub: dict, *, window_ps: float) -> dict:
    """Model-free coincidence count: sum the residual over a window.

    This is the payoff of the subtraction. No peak shape is assumed, so it is
    the number to quote when the fitted background is in doubt - but it is only
    meaningful once the residual has been *seen* to return to zero outside the
    window ('plot_background_subtraction', middle panel).

    Parameters
    ----------
    sub : dict
        Output of 'subtract_background'.
    window_ps : float
        Half-range summed over, in ps. Widen it until the answer stops
        growing; a window narrower than the peak undercounts.

    Returns
    -------
    dict
        ``n`` (coincidences), ``n_err`` (1 sigma), ``significance``
        (``n / n_err``), ``n_background`` (accidentals removed from the same
        window), ``car`` (``n / n_background``), ``window_ps`` and ``n_bins``.
    """
    keep = np.abs(sub["centers"]) <= window_ps
    n = float(sub["residual"][keep].sum())
    n_err = float(np.sqrt((sub["error"][keep] ** 2).sum()))
    n_bkg = float(sub["background"][keep].sum())
    return {
        "n": n,
        "n_err": n_err,
        "significance": n / n_err if n_err else np.inf,
        "n_background": n_bkg,
        # Coincidences over accidentals in the window - NOT the peak-height
        # ratio 'delta_t' reports as 'car' from a fit. The two answer
        # different questions and will not agree.
        "car": n / n_bkg if n_bkg else np.inf,
        "window_ps": float(window_ps),
        "n_bins": int(np.count_nonzero(keep)),
    }


def _legend_outside(ax: plt.Axes) -> None:
    """Put a legend in a column to the right of ``ax``, top-aligned.

    At the house 'legend.fontsize' these captions are wide enough to cover
    the peak wherever they are placed inside a panel, and the peak sits at
    delta = 0 with the triangle filling both upper corners. 'tight_layout'
    measures the legend, so the column it needs is reserved automatically -
    do not also pass a 'rect', that squeezes the axes twice.
    """
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        handlelength=1.4,
        labelspacing=0.5,
    )


def plot_background_subtraction(
    result: dict,
    *,
    full: dict | None = None,
    name: str = "",
    label: str = "spdc",
) -> plt.Figure:
    """Draw the subtraction: overlay, full-range residual, and the peak.

    Three panels, because the middle one is the evidence and it is the one
    people skip: if the residual is *not* flat and zero out at +/-100 ns, the
    pedestal was not accidental and no number from the third panel means
    anything.

    Parameters
    ----------
    result : dict
        A 'subtract_background' result over the zoomed window (bottom panel).
    full : dict | None, optional
        The same over the whole support, for the top two panels. The default
        is None, which uses ``result`` for all three.
    name : str, optional
        Dataset name, for the title. The default is "".
    label : str, optional
        Short measurement label. The default is ``'spdc'``.

    Returns
    -------
    plt.Figure
        The figure. Writes nothing.
    """
    wide = full if full is not None else result
    fig, axes = plt.subplots(3, 1)

    x = wide["centers"] / 1000.0
    ax = axes[0]
    ax.plot(x, wide["signal"], lw=1.2, label="same frame (lag 0)")
    ax.plot(
        x,
        wide["background"],
        lw=1.2,
        label=f"shifted frames (mean of {wide['k']} lags)",
    )
    ax.set_ylabel(f"per {wide['bin_width_ps']:.0f} ps")
    ax.set_title(f"{label.upper()} delta-t, {name}")
    _legend_outside(ax)
    ax.grid(True, lw=0.5, alpha=0.6)

    ax = axes[1]
    ax.fill_between(
        x, -wide["error"], wide["error"], alpha=0.35, label="+/- 1 sigma"
    )
    ax.plot(x, wide["residual"], lw=1.0, label="lag 0 - accidentals")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(f"excess / {wide['bin_width_ps']:.0f} ps")
    ax.set_title("Residual over the full support")
    _legend_outside(ax)
    ax.grid(True, lw=0.5, alpha=0.6)

    xz = result["centers"] / 1000.0
    ax = axes[2]
    ax.fill_between(
        xz, -result["error"], result["error"], alpha=0.35, label="+/- 1 sigma"
    )
    ax.step(xz, result["residual"], where="mid", lw=1.0, label="subtracted")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(f"excess / {result['bin_width_ps']:.0f} ps")
    ax.set_title("Coincidence peak on a zero baseline")
    _legend_outside(ax)
    ax.grid(True, lw=0.5, alpha=0.6)

    for ax in axes:
        ax.set_xlabel("Timestamp difference  (ns)")
    fig.tight_layout()
    return fig
