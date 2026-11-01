"""Guards on the shared sensor-map figure in 'core.plots'.

Layout is not decoration here. A figure is written to disk once, unattended, at
the end of a run that may have taken hours - nobody is watching to notice that
a label fell off the canvas, and the '.png' is what gets read months later. So
the one thing these tests pin is that everything drawn is actually *inside* the
saved image.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after the backend is fixed)

from dapkel.core import plots  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opened; pyplot keeps them alive otherwise."""
    yield
    plt.close("all")


def _map(rows: int = 32, cols: int = 32) -> np.ndarray:
    """Build a sensor map with a few hot pixels, as every real one has."""
    rng = np.random.default_rng(20260818)
    data = rng.gamma(0.3, 500.0, size=(rows, cols))
    data[7, 21] = 8.3e4
    return data


@pytest.mark.parametrize(
    "clabel",
    ["Photon rate  (Hz)", "DCR  (cps)", "Cross-talk probability  (%)"],
)
def test_the_colour_bar_label_is_inside_the_canvas(clabel):
    """Pin the label that used to be drawn past the right edge.

    The square image sits in a square figure, so the colour bar ends up against
    the canvas edge and its label - which hangs off the *outside* of the bar -
    had nowhere to go. Every map written lost its unit.
    """
    fig = plots.sensor_map(_map(), title="Test map", clabel=clabel)
    fig.canvas.draw()

    label = fig.axes[-1].yaxis.label
    assert label.get_text() == clabel
    box = label.get_window_extent()

    assert box.x1 <= fig.bbox.x1, "the colour-bar label runs off the right edge"
    assert box.x0 >= fig.bbox.x0
    assert box.y0 >= fig.bbox.y0 and box.y1 <= fig.bbox.y1


def test_every_artist_of_a_sensor_map_is_inside_the_canvas():
    """Hold the same for the title and the axis labels, not just the bar."""
    fig = plots.sensor_map(
        _map(),
        title="Photon rate  32x32\nmedian 186  max 8.3e+04  [Hz]",
        clabel="Photon rate  (Hz)",
    )
    fig.canvas.draw()

    for artist in (
        fig.axes[0].title,
        fig.axes[0].xaxis.label,
        fig.axes[0].yaxis.label,
        fig.axes[-1].yaxis.label,
    ):
        box = artist.get_window_extent()
        assert fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1, artist
        assert fig.bbox.y0 <= box.y0 and box.y1 <= fig.bbox.y1, artist


def test_the_map_is_still_square_and_the_figure_still_its_declared_size():
    """Check the geometry did not pay for fitting the label in."""
    fig = plots.sensor_map(_map(), title="Test map", clabel="x")
    fig.canvas.draw()

    assert tuple(fig.get_size_inches()) == plots.SENSOR_MAP_FIGSIZE
    box = fig.axes[0].get_window_extent()
    assert box.width == pytest.approx(box.height, rel=1e-6)


def test_a_map_with_no_label_still_draws():
    """Keep ``clabel`` optional - None must not become the string 'None'."""
    fig = plots.sensor_map(_map(), title="Test map")
    fig.canvas.draw()

    assert fig.axes[-1].yaxis.label.get_text() == ""
