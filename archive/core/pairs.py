"""ARCHIVED from dapkel.core.pairs (branch modules-restructure, 2026-10-31).

Names no analysis uses any more. Reference only: not packaged, not
tested, not maintained (commit 6b49a4b).
"""


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
