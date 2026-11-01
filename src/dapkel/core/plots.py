"""Stage 2: the two figure shapes every analysis draws.

A sensor map (a 2D image of the array) and a sorted-percentile distribution
cover every plot in the package bar the TDC QA panel. Both used to be copied
per analysis, which is how three heatmaps ended up with three different figure
sizes and two different NaN policies.

Figure sizing follows the package rule: **square sensor maps set 16x16 in
code, everything else takes ``figure.figsize`` from the active style** (the
dapkel style ships 16x10). So 'sensor_map' is explicit and
'sorted_distribution' is not.

All statistics here are **NaN-aware**. Plain ``np.median``/``np.percentile``
propagate NaN, so a single masked pixel would turn ``vmax`` into NaN and
flatten the whole image to one colour with no error - see
'docs/adding_an_analysis.md'.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "SENSOR_MAP_FIGSIZE",
    "map_stats",
    "sensor_map",
    "sorted_distribution",
]

#: Square sensor maps are drawn at this size; non-square figures inherit
#: ``figure.figsize`` from the active matplotlib style instead.
SENSOR_MAP_FIGSIZE = (16, 16)

#: Percentile used as the colour-scale ceiling, so a few hot pixels do not
#: compress the rest of the map into one shade.
_VMAX_PERCENTILE = 99


def map_stats(
    data: np.ndarray, *, scale: float = 1.0
) -> tuple[float, float, float]:
    """Return the NaN-aware ``(median, max, total)`` of a sensor map.

    Parameters
    ----------
    data : np.ndarray
        The map.
    scale : float, optional
        Multiplied into the data first, for unit conversion (e.g. 100.0 to
        turn a ratio into a percentage). The default is 1.0.

    Returns
    -------
    tuple[float, float, float]
        Median, maximum and sum, all ignoring NaN. An all-NaN map gives
        ``(nan, nan, 0.0)`` rather than raising.
    """
    scaled = data * scale
    if not np.any(np.isfinite(scaled)):
        return float("nan"), float("nan"), 0.0
    return (
        float(np.nanmedian(scaled)),
        float(np.nanmax(scaled)),
        float(np.nansum(scaled)),
    )


def sensor_map(
    data: np.ndarray,
    *,
    title: str,
    clabel: str | None = None,
    cmap: str | None = None,
    scale: float = 1.0,
    xlabel: str = "Column",
    ylabel: str = "Row",
    clip_colorbar_to: float | None = _VMAX_PERCENTILE,
) -> plt.Figure:
    """Render a sensor map as a square 2D image with a colour bar.

    By default the colour scale is clipped at the 99th percentile so a handful
    of hot pixels cannot wash out the rest of the array. The clip only affects
    the colour scale - callers title their figures from 'map_stats', which
    always reports the true maximum, so a clipped bar ending below the titled
    max is expected.

    Parameters
    ----------
    data : np.ndarray
        The (rows, cols) map to draw.
    title : str
        Full figure title. Callers build this themselves - the wording is
        measurement-specific - typically from 'map_stats'.
    clabel : str | None, optional
        Colour-bar label. The default is None (no label).
    cmap : str | None, optional
        Matplotlib colormap name. The default is None - use the active
        style's ``image.cmap``.
    scale : float, optional
        Multiplied into the data before drawing. The default is 1.0.
    xlabel, ylabel : str, optional
        Axis labels. The defaults are ``'Column'`` and ``'Row'``.
    clip_colorbar_to : float | None, optional
        Percentile (0-100) used as the colour-scale ceiling. The default is
        99 - the historical behaviour. Pass None to let the scale run to the
        true maximum, which shows the hot pixels at the cost of compressing
        everything else.

    Returns
    -------
    plt.Figure
        The figure. Nothing is written to disk.
    """
    scaled = data * scale
    vmax = (
        float(np.nanpercentile(scaled, clip_colorbar_to))
        if clip_colorbar_to is not None and np.any(np.isfinite(scaled))
        else None
    )

    # Square figure for the sensor map (kept explicit, per the package rule);
    # fonts and colormap come from the active matplotlib style.
    #
    # Constrained layout, because a square image inside a non-square figure
    # leaves the colour bar sitting against the right edge of the canvas: its
    # label was drawn *outside* and every saved map lost it. Constrained rather
    # than 'tight_layout' because the bar is the case tight layout handles
    # worst - it measures the axes it knows about and the bar is added after -
    # and it sizes the figure once here rather than at save time, so the
    # written '.png' is the size this module asked for.
    fig, ax = plt.subplots(
        figsize=SENSOR_MAP_FIGSIZE, layout="constrained"
    )

    im = ax.imshow(
        scaled, cmap=cmap, aspect="equal", vmax=vmax, origin="lower"
    )
    # 'fraction' is what keeps the bar proportional to the image rather than to
    # the figure; the label then needs 'labelpad' to clear the tick numbers.
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if clabel:
        bar.set_label(clabel, labelpad=10)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def sorted_distribution(
    data: np.ndarray,
    *,
    ylabel: str,
    title: str,
    xlabel: str = "Pixel percentile  (%)",
    logy: bool = False,
    scale: float = 1.0,
    fmt: str = ".3g",
    unit: str = "",
) -> plt.Figure:
    """Plot per-pixel values sorted low-to-high against their percentile.

    Non-finite values are dropped before sorting, so masked or undefined
    pixels neither plot nor skew the median and mean lines.

    Parameters
    ----------
    data : np.ndarray
        The map whose values are ranked.
    ylabel : str
        Y-axis label, including the unit (e.g. ``'DCR  (cps)'``).
    title : str
        Figure title.
    xlabel : str, optional
        X-axis label. The default is ``'Pixel percentile  (%)'``.
    logy : bool, optional
        Use a logarithmic y-axis, appropriate when the values span decades
        (DCR does). The default is False.
    scale : float, optional
        Multiplied into the data first. The default is 1.0.
    fmt : str, optional
        Format spec for the median/mean values in the legend. The default is
        ``'.3g'``.
    unit : str, optional
        Unit appended to the legend entries (e.g. ``'cps'``, ``'%'``). The
        default is "".

    Returns
    -------
    plt.Figure
        The figure. Nothing is written to disk.
    """
    values = (data * scale).ravel()
    values = values[np.isfinite(values)]

    # Figure size comes from the active style (non-square), per the package
    # rule; only the square sensor maps set their own.
    fig, ax = plt.subplots()

    suffix = f" {unit}" if unit else ""
    if values.size:
        median = float(np.median(values))
        mean = float(values.mean())
        ordered = np.sort(values)
        percentile = np.linspace(0, 100, ordered.size)

        plot = ax.semilogy if logy else ax.plot
        plot(
            percentile, ordered, ".", markersize=2, alpha=0.7,
            label="_nolegend_",
        )
        ax.axhline(
            median, linestyle="--", linewidth=2,
            label=f"Median  {median:{fmt}}{suffix}",
        )
        ax.axhline(
            mean, linestyle="--", linewidth=1,
            label=f"Mean    {mean:{fmt}}{suffix}",
        )
        ax.legend()

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig
