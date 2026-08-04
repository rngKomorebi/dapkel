"""Stage 1: resolving how much wall-clock time an acquisition represents.

Counts become rates only once you know how long the sensor was actually
collecting, and that depends on the firmware. These helpers previously lived in
'dcr_analysis' as private functions that 'hitmap_analysis' imported across the
module boundary; they are shared API, so they live here.
"""

from __future__ import annotations

__all__ = [
    "CLK_PERIOD",
    "FREE_RUNNING_US",
    "resolve_cycle_time",
    "resolve_frame_time",
    "resolve_live_time",
]

#: Board clock period: 200 MHz.
CLK_PERIOD = 5e-9

#: Base frame period when ``exp_time=0`` is sent to the acquisition exe, in µs.
#: The firmware adds this to whatever exposure is requested, so it is both the
#: whole cycle under short-exposure firmware and the overhead under long.
FREE_RUNNING_US = 9.0

#: Firmware versions, by how they set the length of one acquisition cycle:
#:
#:     * ``'short_exposure'`` - the cycle is always 9 µs. The photon-sensitive
#:       exposure inside it is 50-500 ns (typically 100 ns) and does not change
#:       the cycle length.
#:     * ``'long_exposure'`` - the caller sets X and the firmware adds 9 µs, so
#:       the cycle is ``X + 9 µs`` and the exposure is essentially the whole of
#:       it.
#:
#: The ``'short_window'`` / ``'full_window'`` names 'resolve_live_time' and
#: 'dcr_analysis' use are accepted as aliases of the two.
_CYCLE_ALIASES = {
    "short_exposure": "short_exposure",
    "short_window": "short_exposure",
    "long_exposure": "long_exposure",
    "full_window": "long_exposure",
}


def resolve_cycle_time(
    firmware_version: str, exp_time: float | None = None
) -> tuple[float, str]:
    """Return the length of one acquisition cycle and where it came from.

    This is the wall-clock repetition period of a frame, which is what a
    photon *rate* divides by: ``nframes * n_files * cycle`` is the elapsed
    time of a run.

    Computed from the firmware setting, and **never** from the
    ``frame_rate_cnt.txt`` the acquisition exe writes beside the data: that exe
    is not ours, so no analysis here is allowed to depend on its output. The
    cycle length is a property of the firmware and the requested exposure, and
    ``Kelpie_run.m`` states it directly: ``exp_time = 0 -> 9 us``,
    ``1e-6 -> 10 us``, ``2e-6 -> 11 us``.

    For the record, the counter *is* self-consistent where it exists - it is a
    200 MHz tick count over the whole run, latched by the firmware, and across
    every sample folder it fits ``ticks = nframes * N + 138 + window/5ns``
    exactly. What it says is that the real cycle is **9.685-9.770 µs**, i.e.
    685-770 ns per frame longer than the nominal 9 µs used here, and that the
    50-500 ns exposure window sits *inside* the cycle rather than extending it.
    So a rate normalised by the nominal 9 µs runs ~7.8% high. Whether to prefer
    the nominal or a measured cycle is a decision for the experiment, not for
    this function to make silently off a file it happens to find on disk.

    Parameters
    ----------
    firmware_version : str
        ``'short_exposure'`` (the cycle is always 9 µs) or ``'long_exposure'``
        (the cycle is ``exp_time`` + 9 µs). ``'short_window'`` and
        ``'full_window'`` are accepted as aliases of the two.
    exp_time : float | None, optional
        The exposure X requested of long-exposure firmware, in seconds. The
        firmware adds 9 µs to it. Required for ``'long_exposure'``, ignored for
        ``'short_exposure'``. The default is None.

    Returns
    -------
    tuple[float, str]
        The cycle length in seconds and a human-readable source string.

    Raises
    ------
    ValueError
        Raised on an unknown ``firmware_version``, or when
        ``'long_exposure'`` is given without ``exp_time``.
    """
    try:
        version = _CYCLE_ALIASES[firmware_version]
    except KeyError:
        raise ValueError(
            "firmware_version must be 'short_exposure' or 'long_exposure' "
            f"(aliases 'short_window' / 'full_window'), got "
            f"{firmware_version!r}"
        ) from None

    base = FREE_RUNNING_US * 1e-6
    if version == "short_exposure":
        return base, f"short_exposure: {FREE_RUNNING_US:.0f} µs cycle"
    if exp_time is None:
        raise ValueError(
            "firmware_version='long_exposure' needs exp_time (the exposure X "
            "requested, in seconds; the firmware adds "
            f"{FREE_RUNNING_US:.0f} µs to it)."
        )
    return (
        exp_time + base,
        f"long_exposure: {exp_time * 1e6:.3f} µs + "
        f"{FREE_RUNNING_US:.0f} µs = {(exp_time + base) * 1e6:.3f} µs cycle",
    )


def resolve_frame_time(
    explicit: float | None, nframes: int | None = None
) -> tuple[float, str]:
    """Return the per-frame period and a description of where it came from.

    1. explicit ``exp_time`` supplied -> ``exp_time`` + 9 µs overhead
    2. fallback                       -> 9 µs free-running base

    The acquisition exe also writes a ``frame_rate_cnt.txt`` beside the data -
    a 200 MHz tick count for the whole run, latched by the firmware. **It is
    deliberately not read here.** It is written by an exe we do not control, so
    nothing in the analysis path is allowed to depend on it; the period comes
    from the settings the run was started with. See 'resolve_cycle_time' for
    what the counter does contain, and for the measured-vs-nominal gap.

    Parameters
    ----------
    explicit : float | None
        Explicit exposure time in seconds, or None for the free-running base.
    nframes : int | None, optional
        Unused. Kept so existing callers do not break; it was only ever needed
        to turn the counter's tick count into a period. The default is None.

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

    ft = FREE_RUNNING_US * 1e-6
    return ft, f"free-running base ({FREE_RUNNING_US} µs)"


def resolve_live_time(
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
        return resolve_frame_time(exp_time, nframes)
    raise ValueError(
        "firmware_version must be 'short_window' or 'full_window', got "
        f"{firmware_version!r}"
    )
