"""Guards on the SPAD (micropixel) layout inside a 2x2 macropixel.

The four SPADs of a macropixel are indexed CLOCKWISE, not row-major::

    S0 (0,0)   S1 (0,1)
    S3 (1,0)   S2 (1,1)

so S1 is S0's horizontal neighbour, S3 its vertical neighbour, and S2 its
diagonal one. This is encoded once, in 'core.reduce.SPAD_LAYOUT', which the
cross-talk directions read. A swap of S2 and S3 is silent - the arrays keep
their shape and the plots still render, but vertical and diagonal cross-talk
get each other's labels and cells - so the layout is pinned here against an
independent spelling of it.
"""

from __future__ import annotations

from dapkel.core import reduce
from dapkel.functions import crosstalk_analysis

#: The layout, spelled out independently of the source.
EXPECTED_LAYOUT = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}

#: Which neighbour direction each offset is, from the geometry alone.
OFFSET_TO_DIRECTION = {(0, 1): "horizontal", (1, 0): "vertical", (1, 1): "diagonal"}


def test_spad_layout_is_clockwise() -> None:
    assert reduce.SPAD_LAYOUT == EXPECTED_LAYOUT, (
        f"SPAD_LAYOUT is {reduce.SPAD_LAYOUT}, expected the clockwise layout "
        f"{EXPECTED_LAYOUT} (S0 S1 / S3 S2)"
    )


def test_crosstalk_directions_agree_with_the_layout() -> None:
    """S0S<n> is labelled by where SPAD n actually sits relative to S0."""
    for index, name in crosstalk_analysis._DIRECTIONS.items():
        offset = EXPECTED_LAYOUT[index]
        assert name == OFFSET_TO_DIRECTION[offset], (
            f"S0S{index} has offset {offset}, which is "
            f"{OFFSET_TO_DIRECTION[offset]}, but it is labelled {name!r}"
        )


def test_all_three_directions_are_distinct() -> None:
    assert sorted(crosstalk_analysis._DIRECTIONS.values()) == [
        "diagonal",
        "horizontal",
        "vertical",
    ]
