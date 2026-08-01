"""Per-window timestamp differences for pairs of pixels.

The low-level, daplis-style counterpart of ``daplis.functions.calc_diff``: it
takes an already-structured mapping of per-pixel, per-frame timestamps and
returns the differences for the requested pixel pairs. Unpacking, structuring
and saving are handled one level up in 'dapkel.functions.delta_t'.

A coincidence is a *within-window* event: in a frame where both pixels of a pair
hold a valid TDC code, the difference of their codes is one entry of the delta-t
distribution. Correlated pairs (e.g. SPDC signal/idler) pile up at delta = 0.

See ``docs/guide/coincidences.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    # type alias
    "Pixel",
    # helpers
    "as_pixel_list",
    # compute
    "calculate_differences",
]

Pixel = tuple[int, int]


def as_pixel_list(group: Pixel | Sequence[Pixel]) -> list[Pixel]:
    """Normalise a detector spec into a list of ``(row, col)`` pixels.

    Parameters
    ----------
    group : tuple[int, int] | Sequence[tuple[int, int]]
        A single ``(row, col)`` pixel or a sequence of them.

    Returns
    -------
    list[tuple[int, int]]
        List of ``(row, col)`` pixels.
    """
    if (
        len(group) == 2
        and isinstance(group[0], (int, np.integer))
        and isinstance(group[1], (int, np.integer))
    ):
        return [(int(group[0]), int(group[1]))]
    return [(int(r), int(c)) for r, c in group]


def calculate_differences(
    pixel_timestamps: dict[Pixel, np.ndarray],
    pixels: Sequence[Sequence[Pixel]],
    delta_window: float | None = None,
    mode: str = "all_pairs",
) -> dict[str, np.ndarray]:
    """Compute per-window timestamp differences for pairs of pixels.

    For each requested pixel pair ``(a, b)`` the difference ``t_b - t_a``
    is taken in every acquisition window (frame) where *both* pixels hold a
    valid timestamp. The per-pixel arrays must be aligned by frame index
    (same length, NaN where the pixel had no photon that frame), which is
    how 'delta_t' structures them.

    Parameters
    ----------
    pixel_timestamps : dict[tuple[int, int], np.ndarray]
        Mapping ``(row, col) -> 1D array`` of that pixel's per-frame
        timestamps (in TDC code units), with NaN where the pixel did not
        register a photon in that frame. All arrays share the same length
        (the number of frames).
    pixels : Sequence[Sequence[tuple[int, int]]]
        Two groups ``[group_a, group_b]``, each a sequence of
        ``(row, col)`` pixels (e.g. the signal and idler blobs read off the
        hitmap). A single ``(row, col)`` is accepted for a one-pixel group.
    delta_window : float | None, optional
        If given, keep only differences with ``abs(delta) <= delta_window``
        (same units as the timestamps). The default is None (keep all).
    mode : str, optional
        ``'all_pairs'`` (default) forms the full cross product of
        ``group_a`` x ``group_b`` — use this for SPDC/HBT when it is not
        known which pixel of a blob catches a given photon. ``'1v1'`` zips
        ``group_a[i]`` with ``group_b[i]`` (requires equal-length groups).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``"ra,ca-rb,cb" -> 1D array`` of timestamp differences for
        each pixel pair.

    Raises
    ------
    ValueError
        Raised when ``mode`` is invalid, or when ``mode='1v1'`` and the two
        groups differ in length.
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
        pair_iter = list(zip(group_a, group_b, strict=True))
    else:
        pair_iter = [(a, b) for a in group_a for b in group_b]

    deltas_all: dict[str, np.ndarray] = {}
    for a, b in pair_iter:
        ta = pixel_timestamps[a]
        tb = pixel_timestamps[b]
        both = ~np.isnan(ta) & ~np.isnan(tb)
        d = tb[both] - ta[both]
        if delta_window is not None:
            d = d[np.abs(d) <= delta_window]
        deltas_all[f"{a[0]},{a[1]}-{b[0]},{b[1]}"] = d

    return deltas_all
