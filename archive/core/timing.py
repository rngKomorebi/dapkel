"""ARCHIVED from dapkel.core.timing (branch modules-restructure, 2026-10-31).

The cycle-time resolver (unused by any analysis) and the frame-time helper
'resolve_live_time' now inlines. 'resolve_cycle_time' carries the record of
what the exe's frame_rate_cnt.txt counter contains (9.685-9.770 us real
cycle against the nominal 9 us).

Reference only: not packaged, not tested, not maintained (commit 6b49a4b).
"""

from dapkel.core.timing import FIRMWARE_VERSIONS, FREE_RUNNING_US


def resolve_cycle_time(
    firmware_version: str, exp_time_s: float | None = None
) -> tuple[float, str]:
    """Return the length of one acquisition cycle and where it came from.

    This is the wall-clock repetition period of a frame, which is what a
    photon *rate* divides by: ``nframes * n_files * cycle`` is the elapsed
    time of a run.

    Computed from the firmware setting, and **never** from the
    ``frame_rate_cnt.txt`` the acquisition exe writes beside the data: that exe
    is not ours, so no analysis here is allowed to depend on its output. The
    cycle length is a property of the firmware and the requested exposure, and
    ``Kelpie_run.m`` states it directly: ``exp_time_s = 0 -> 9 us``,
    ``1e-6 -> 10 us``, ``2e-6 -> 11 us``.

    For the record, the counter *is* self-consistent where it exists - it is a
    200 MHz tick count over the whole run, latched by the firmware, and across
    every sample folder it fits ``ticks = nframes * N + 138 + window/5ns``
    exactly. What it says is that the real cycle is **9.685-9.770 µs**, i.e.
    685-770 ns per frame longer than the nominal 9 µs used here, and that the
    exposure window sits *inside* the cycle rather than extending it.
    So a rate normalised by the nominal 9 µs runs ~7.8% high. Whether to prefer
    the nominal or a measured cycle is a decision for the experiment, not for
    this function to make silently off a file it happens to find on disk.

    Parameters
    ----------
    firmware_version : str
        ``'short_exposure'`` (the cycle is always 9 µs) or ``'long_exposure'``
        (the cycle is ``exp_time_s`` + 9 µs).
    exp_time_s : float | None, optional
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
        ``'long_exposure'`` is given without ``exp_time_s``.
    """
    if firmware_version not in FIRMWARE_VERSIONS:
        raise ValueError(
            "firmware_version must be one of "
            f"{FIRMWARE_VERSIONS}, got {firmware_version!r}"
        )

    base = FREE_RUNNING_US * 1e-6
    if firmware_version == "short_exposure":
        return base, f"short_exposure: {FREE_RUNNING_US:.0f} µs cycle"
    if exp_time_s is None:
        raise ValueError(
            "firmware_version='long_exposure' needs exp_time_s (the exposure X "
            "requested, in seconds; the firmware adds "
            f"{FREE_RUNNING_US:.0f} µs to it)."
        )
    return (
        exp_time_s + base,
        f"long_exposure: {exp_time_s * 1e6:.3f} µs + "
        f"{FREE_RUNNING_US:.0f} µs = {(exp_time_s + base) * 1e6:.3f} µs cycle",
    )


def resolve_frame_time(
    explicit_s: float | None, nframes: int | None = None
) -> tuple[float, str]:
    """Return the per-frame period and a description of where it came from.

    1. explicit ``exp_time_s`` supplied -> ``exp_time_s`` + 9 µs overhead
    2. fallback                         -> 9 µs free-running base

    The acquisition exe also writes a ``frame_rate_cnt.txt`` beside the data -
    a 200 MHz tick count for the whole run, latched by the firmware. **It is
    deliberately not read here.** It is written by an exe we do not control, so
    nothing in the analysis path is allowed to depend on it; the period comes
    from the settings the run was started with. See 'resolve_cycle_time' for
    what the counter does contain, and for the measured-vs-nominal gap.

    Parameters
    ----------
    explicit_s : float | None
        Explicit exposure time in seconds, or None for the free-running base.
    nframes : int | None, optional
        Unused. Kept so existing callers do not break; it was only ever needed
        to turn the counter's tick count into a period. The default is None.

    Returns
    -------
    tuple[float, str]
        The frame period in seconds and a human-readable source string.
    """
    if explicit_s is not None:
        ft = explicit_s + FREE_RUNNING_US * 1e-6
        return (
            ft,
            f"exp_time_s={explicit_s * 1e6:.1f} µs + "
            f"{FREE_RUNNING_US:.0f} µs overhead",
        )

    ft = FREE_RUNNING_US * 1e-6
    return ft, f"free-running base ({FREE_RUNNING_US} µs)"


#: Board clock period: 200 MHz.
CLK_PERIOD = 5e-9


#: Largest ``abs(delta)`` two timestamps in one frame can produce, in seconds.
#:
#: Both timestamps are codes on the same range, so their difference is bounded by
#: the widest range the frame format can express - 'MAX_TS_CODE' codes, ~586 ns.
#: This is the *physical support* of a delta-t histogram at any clock, which is
#: why 'dapkel.functions.delta_t' histograms over all of it and needs no window
#: parameter: at one code per bin that is ~16 k bins, i.e. ~131 kB.
MAX_DELTA_S = MAX_TS_CODE * CODE_S
