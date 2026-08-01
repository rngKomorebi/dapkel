"""Stage 1: resolving how much wall-clock time an acquisition represents.

Counts become rates only once you know how long the sensor was actually
collecting, and that depends on the firmware. These helpers previously lived in
'dcr_analysis' as private functions that 'hitmap_analysis' imported across the
module boundary; they are shared API, so they live here.
"""

from __future__ import annotations

import os

__all__ = [
    "CLK_PERIOD",
    "FREE_RUNNING_US",
    "resolve_frame_time",
    "resolve_live_time",
]

#: Board clock period: 200 MHz.
CLK_PERIOD = 5e-9

#: Base frame period when ``exp_time=0`` is sent to the acquisition exe, in µs.
FREE_RUNNING_US = 9.0


def resolve_frame_time(
    folder: str, explicit: float | None, nframes: int
) -> tuple[float, str]:
    """Return the per-frame period and a description of where it came from.

    Mirrors the GUI's ``_run_analysis`` logic exactly:

    1. explicit ``exp_time`` supplied -> ``exp_time`` + 9 µs overhead
    2. ``frame_rate_cnt.txt`` present -> ``ticks * CLK_PERIOD / nframes``
    3. fallback                       -> 9 µs free-running base

    Parameters
    ----------
    folder : str
        Data folder, searched for ``frame_rate_cnt.txt``.
    explicit : float | None
        Explicit exposure time in seconds, or None to fall through to the
        counter file.
    nframes : int
        Frames per file, used to turn the tick count into a period.

    Returns
    -------
    tuple[float, str]
        The frame period in seconds and a human-readable source string.
    """
    if explicit is not None:
        ft = explicit + FREE_RUNNING_US * 1e-6
        return (
            ft,
            f"exp_time={explicit * 1e6:.1f} µs + {FREE_RUNNING_US:.0f} µs overhead",
        )

    cnt_file = os.path.join(folder, "frame_rate_cnt.txt")
    try:
        ticks = int(open(cnt_file).read().strip())
        if ticks > 0:
            ft = ticks * CLK_PERIOD / nframes
            return ft, f"frame_rate_cnt.txt -> {ft * 1e6:.4f} µs"
    except (OSError, ValueError):
        pass

    ft = FREE_RUNNING_US * 1e-6
    return ft, f"fallback free-running ({FREE_RUNNING_US} µs)"


def resolve_live_time(
    folder: str,
    firmware_version: str,
    nframes: int,
    *,
    acq_window: float | None = None,
    exp_time: float | None = None,
) -> tuple[float | None, str]:
    """Return the live seconds per frame for the rate normalisation.

    The photon rate is ``counts / (nframes * n_files * live_per_frame)``. How
    much of each frame is photon-sensitive depends on the firmware:

    * ``'short_window'`` - the current firmware: each ~9 µs frame is mostly
      readout and only the user-set acquisition window (``acq_window``, e.g.
      ``200e-9`` s, ``< 9 µs``) is live. Returns ``(None, ...)`` when
      ``acq_window`` is not supplied, so the caller can decide whether that is
      an error (DCR) or just "no rate available" (hitmap).

    * ``'full_window'`` - the legacy firmware: the whole frame is live, so the
      per-frame time comes from 'resolve_frame_time'.

    Parameters
    ----------
    folder : str
        Data folder (only used by ``'full_window'`` for
        ``frame_rate_cnt.txt``).
    firmware_version : str
        ``'short_window'`` or ``'full_window'``.
    nframes : int
        Frames per file (used by the ``'full_window'`` resolution).
    acq_window : float | None, optional
        Photon-sensitive window per frame, in seconds (``'short_window'``
        only). The default is None.
    exp_time : float | None, optional
        Legacy exposure time, in seconds (``'full_window'`` only). The default
        is None.

    Returns
    -------
    tuple[float | None, str]
        The live seconds per frame (None if unresolved) and a human-readable
        source string.

    Raises
    ------
    ValueError
        Raised when ``firmware_version`` is not one of the two known values.
    """
    if firmware_version == "short_window":
        if acq_window is None:
            return None, "short_window (no acq_window given)"
        return acq_window, f"short_window: {acq_window * 1e9:.1f} ns window"
    if firmware_version == "full_window":
        return resolve_frame_time(folder, exp_time, nframes)
    raise ValueError(
        "firmware_version must be 'short_window' or 'full_window', got "
        f"{firmware_version!r}"
    )
