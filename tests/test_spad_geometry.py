"""Guards on the SPAD (micropixel) layout inside a 2x2 macropixel.

The four SPADs of a macropixel are indexed CLOCKWISE, not row-major::

    S0 (0,0)   S1 (0,1)
    S3 (1,0)   S2 (1,1)

so S1 is S0's horizontal neighbour, S3 its vertical neighbour, and S2 its
diagonal one. This single fact is encoded in three places - the '_SPAD_LAYOUT'
of 'dcr_analysis' and 'hitmap_analysis' (used to interleave four (32, 32)
quadrant maps into the full (64, 64) sensor) and the '_DIRECTIONS' table of
'crosstalk_analysis' (used both to *label* a pairing horizontal/vertical/
diagonal and to place its map on the (64, 64) grid).

Because it is hand-copied, the three can drift apart, and a swap of S2 and S3
is silent: the arrays still have the right shape and the plots still render,
but vertical and diagonal cross-talk - physically different magnitudes - get
each other's labels and cells. These tests tie the three encodings to one
another so that cannot happen again.
"""

from __future__ import annotations

import numpy as np
import pytest

from dapkel.functions import crosstalk_analysis, dcr_analysis, hitmap_analysis

#: The layout, spelled out independently of the source so a copy-paste error in
#: the package cannot make this test agree with it by construction.
EXPECTED_LAYOUT = {
    "S0C": (0, 0),
    "S1C": (0, 1),
    "S2C": (1, 1),
    "S3C": (1, 0),
}

#: Which neighbour direction each offset represents, from the geometry alone.
OFFSET_TO_DIRECTION = {
    (0, 1): "horizontal",
    (1, 0): "vertical",
    (1, 1): "diagonal",
}


@pytest.mark.parametrize("module", [dcr_analysis, hitmap_analysis])
def test_spad_layout_is_clockwise(module) -> None:
    """Both quadrant-assembling modules must use the clockwise layout."""
    assert module._SPAD_LAYOUT == EXPECTED_LAYOUT, (
        f"{module.__name__}._SPAD_LAYOUT is {module._SPAD_LAYOUT}, expected the "
        f"clockwise layout {EXPECTED_LAYOUT} (S0 S1 / S3 S2)"
    )


def test_spad_layout_agrees_between_modules() -> None:
    """The hand-copied layouts must stay identical."""
    assert dcr_analysis._SPAD_LAYOUT == hitmap_analysis._SPAD_LAYOUT


def test_crosstalk_directions_agree_with_spad_layout() -> None:
    """Cross-talk offsets must be the layout offsets of the same SPAD.

    This is the invariant that the S2/S3 swap violated: '_DIRECTIONS[n]' is the
    offset of SPAD n relative to S0, so it has to equal '_SPAD_LAYOUT["S<n>C"]'.
    """
    for index, (name, offset) in crosstalk_analysis._DIRECTIONS.items():
        layout_offset = EXPECTED_LAYOUT[f"S{index}C"]
        assert offset == layout_offset, (
            f"crosstalk _DIRECTIONS[{index}] places S{index} at {offset}, but the "
            f"SPAD layout puts it at {layout_offset}"
        )
        assert name == OFFSET_TO_DIRECTION[offset], (
            f"S0S{index} has offset {offset}, which is "
            f"{OFFSET_TO_DIRECTION[offset]}, but it is labelled {name!r}"
        )


def test_all_three_directions_are_distinct() -> None:
    """Each of horizontal / vertical / diagonal must appear exactly once."""
    names = [name for name, _ in crosstalk_analysis._DIRECTIONS.values()]
    assert sorted(names) == ["diagonal", "horizontal", "vertical"]

    offsets = [offset for _, offset in crosstalk_analysis._DIRECTIONS.values()]
    assert len(set(offsets)) == 3, f"duplicate cell offsets: {offsets}"


def test_quadrant_assembly_places_each_spad_correctly() -> None:
    """Interleaving four constant quadrant maps must land on the right cells.

    Uses a distinct constant per SPAD so the assembled (64, 64) map can be read
    back cell by cell - this checks the layout as *used*, not just as declared.
    """
    values = {"S0C": 10.0, "S1C": 20.0, "S2C": 30.0, "S3C": 40.0}
    full = np.zeros((64, 64))
    for tag, (drow, dcol) in dcr_analysis._SPAD_LAYOUT.items():
        full[drow::2, dcol::2] = values[tag]

    # Top-left 2x2 block should read S0 S1 / S3 S2 clockwise.
    assert full[0, 0] == values["S0C"]
    assert full[0, 1] == values["S1C"]
    assert full[1, 1] == values["S2C"]
    assert full[1, 0] == values["S3C"]

    # And the pattern must tile: every macropixel block is identical.
    assert np.array_equal(full[0:2, 0:2], full[40:42, 18:20])
