"""Which pixel pairs an analysis is about, and what they are called.

Every coincidence analysis starts by turning two detector groups into an
ordered list of pairs and a label per pair. That enumeration decides the row
order of every stage-1 artifact, so two analyses are only comparable if they
build it the *same* way - which is why it lives here rather than in each of
them. 'delta_t' had it twice: once inline in the feather path, once as a
private helper for the counts path.

The label format ``"ra,ca-rb,cb"`` is a stored contract: it is the key
'calc_diff.calculate_differences' returns, the column name in the delta-t
feather, and the ``labels`` entry in every counts sidecar. Do not reformat it.
"""

from __future__ import annotations

from collections.abc import Sequence

from dapkel.functions.calc_diff import Pixel, as_pixel_list

__all__ = [
    "pair_label",
    "pair_list",
    "pair_labels",
]


def pair_label(a: Pixel, b: Pixel) -> str:
    """Return the canonical ``"ra,ca-rb,cb"`` name of one pixel pair."""
    return f"{a[0]},{a[1]}-{b[0]},{b[1]}"


def pair_list(
    pixels: Sequence[Sequence[Pixel]], mode: str = "all_pairs"
) -> list[tuple[Pixel, Pixel]]:
    """Enumerate the ordered ``(pixel_a, pixel_b)`` pairs of two groups.

    Parameters
    ----------
    pixels : Sequence[Sequence[tuple[int, int]]]
        Two groups ``[group_a, group_b]``; a bare ``(row, col)`` is accepted
        for a one-pixel group.
    mode : str, optional
        ``'all_pairs'`` (default) for every A-B combination, or ``'1v1'`` to
        zip the groups element-wise.

    Returns
    -------
    list[tuple[tuple[int, int], tuple[int, int]]]
        The pairs, in the order every artifact stores them in.

    Raises
    ------
    ValueError
        Raised on an unknown ``mode``, or unequal groups under ``'1v1'``.
    """
    if mode not in ("all_pairs", "1v1"):
        raise ValueError(f"mode must be 'all_pairs' or '1v1', got {mode!r}")

    group_a = as_pixel_list(pixels[0])
    group_b = as_pixel_list(pixels[1])
    if mode == "1v1":
        if len(group_a) != len(group_b):
            raise ValueError(
                "mode='1v1' needs equal-length groups "
                f"(got {len(group_a)} and {len(group_b)})."
            )
        return list(zip(group_a, group_b, strict=True))
    return [(a, b) for a in group_a for b in group_b]


def pair_labels(
    pixels: Sequence[Sequence[Pixel]], mode: str = "all_pairs"
) -> tuple[list[str], list[Pixel]]:
    """Pair labels and the de-duplicated pixels they mention.

    The two things a stage-1 loop needs: what to call each output row, and
    which pixels to decode out of the frames.

    Parameters
    ----------
    pixels : Sequence[Sequence[tuple[int, int]]]
        Two groups ``[group_a, group_b]``.
    mode : str, optional
        ``'all_pairs'`` (default) or ``'1v1'``.

    Returns
    -------
    labels : list[str]
        ``"ra,ca-rb,cb"`` per pair, in 'pair_list' order.
    all_pixels : list[tuple[int, int]]
        Every pixel mentioned, de-duplicated, first-seen order.

    Raises
    ------
    ValueError
        Raised on an unknown ``mode``, or unequal groups under ``'1v1'``.
    """
    pl = pair_list(pixels, mode)
    labels = [pair_label(a, b) for a, b in pl]
    all_pixels = list(
        dict.fromkeys(as_pixel_list(pixels[0]) + as_pixel_list(pixels[1]))
    )
    return labels, all_pixels
