"""Stage 1: resolving how much wall-clock time an acquisition represents.

Counts become rates only once you know how long the sensor was actually
collecting, and that depends on the firmware. These helpers previously lived in
'dcr_analysis' as private functions that 'hitmap_analysis' imported across the
module boundary; they are shared API, so they live here.

Also the single home for the ORT timing scale. **Every time parameter in this
package is in seconds and named with an ``_s`` suffix**; picoseconds appear only
as internal working units, because that is the scale the TDC lives on. Convert
at the boundary with 'to_ps' / 'to_s'.

Two timescales are easy to conflate and must not be:

* ``open_shutter_time_s`` - the photon-sensitive window per frame. An
  acquisition setting, nothing in the data reveals it, so it is asked for. It
  sets the rate normalisation and **nothing else**.
* the **TDC full scale** - the ORT measurement range, i.e. one period of the
  reference clock. The timestamp is a code on this range, so it is what the
  accidental (triangle) background's half-base follows.

They coincide only when the shutter is one clock period, and an 8 µs shutter on
a 10 MHz clock still yields ~100 ns timestamps: the free-running oscillator has
no cycle counter, so timestamps fold into the full scale. **So the full scale is
never taken from the shutter.** It is measured from the data - the largest code
present *is* the range the TDC swept - which is why no analysis entry point in
this package asks for it. See 'full_scale_from_max_code'.

The one place a clock period is genuinely irreducible is building a calibration
LUT ('dapkel.functions.tdc_calibration'): the density test fixes only the
*relative* per-code widths, so an externally known duration has to set the
absolute scale. That parameter is called ``clock_period_s`` and it appears
nowhere else. Everything downstream leans on 'CODE_S' instead, which was itself
measured against known clocks.
"""

from __future__ import annotations

import warnings

__all__ = [
    "CLK_PERIOD",
    "CODE_S",
    "COARSE_BITS",
    "FIRMWARE_VERSIONS",
    "FREE_RUNNING_US",
    "MAX_DELTA_S",
    "MAX_TS_CODE",
    "full_scale_from_max_code",
    "resolve_cycle_time",
    "resolve_frame_time",
    "resolve_live_time",
    "to_ps",
    "to_s",
]

#: Board clock period: 200 MHz.
CLK_PERIOD = 5e-9

#: Base frame period when ``exp_time_s=0`` is sent to the acquisition exe, in
#: µs. The firmware adds this to whatever exposure is requested, so it is both
#: the whole cycle under short-exposure firmware and the overhead under long.
FREE_RUNNING_US = 9.0

#: Width of one ring-oscillator TDC code, in seconds (~71.6 ps).
#:
#: This is set by the ring itself, **not** by the reference clock, so it does
#: not change with the clock frequency - only the number of codes does. Measured
#: on the 2026.08.07 rate-vs-clock set, where the largest code seen scales as
#: 1/f to within 2 %::
#:
#:     clock    full scale   max code   max code / (10 MHz max code)
#:     50 MHz       20 ns         274        0.199   (ideal 0.2)
#:     10 MHz      100 ns        1378        1.000
#:      5 MHz      200 ns        2760        2.003   (ideal 2)
#:      2 MHz      500 ns        6913        5.017   (ideal 5)
#:
#: Only ever a *nominal average*: the ring has strong DNL, which is the whole
#: reason for the per-code LUT in 'dapkel.functions.tdc_calibration'. Use it for
#: sizing code axes and for uncalibrated raw-code data, never in place of a LUT.
CODE_S = 71.58e-12

#: Width of the coarse counter field decoded in 'dapkel.functions.unpack'.
COARSE_BITS = 10

#: Largest TDC code the frame format can express.
#:
#: 'unpack' builds ``ts = (coarse - 1) * 8 + (8 - fine)`` from a ``COARSE_BITS``
#: coarse field, so the code saturates here regardless of clock. At 'CODE_S'
#: that is ~586 ns, which caps the usable reference clock at ~1.7 MHz: below
#: that the coarse counter wraps and the timestamps are meaningless. 2 MHz
#: (500 ns) already uses 85 % of the counter.
MAX_TS_CODE = (2**COARSE_BITS - 2) * 8 + 8

#: Largest ``abs(delta)`` two timestamps in one frame can produce, in seconds.
#:
#: Both timestamps are codes on the same range, so their difference is bounded by
#: the widest range the frame format can express - 'MAX_TS_CODE' codes, ~586 ns.
#: This is the *physical support* of a delta-t histogram at any clock, which is
#: why 'dapkel.functions.delta_t' histograms over all of it and needs no window
#: parameter: at one code per bin that is ~16 k bins, i.e. ~131 kB.
MAX_DELTA_S = MAX_TS_CODE * CODE_S

#: Firmware versions, by how they set the length of one acquisition cycle:
#:
#:     * ``'short_exposure'`` - the cycle is always 9 µs. The photon-sensitive
#:       exposure sits *inside* that cycle and does not change its length, so it
#:       is settable anywhere from ~50 ns up to the full 9 µs. Both extremes are
#:       in use: 100 ns for DCR and crosstalk work, 8 µs for the 2026.08.10
#:       Ne 703 nm HBT run. Never assume a short one - an exposure far longer
#:       than the TDC full scale wraps the timestamps and dilutes any
#:       coincidence peak by ``open_shutter_time_s / full_scale`` (84x for that
#:       run: 8 µs over a 95 ns full scale).
#:     * ``'long_exposure'`` - the caller sets X and the firmware adds 9 µs, so
#:       the cycle is ``X + 9 µs`` and the exposure is essentially the whole of
#:       it.
#:
#: One vocabulary, used by every function that takes a ``firmware_version``:
#: 'resolve_cycle_time', 'resolve_live_time', 'dcr_analysis' and
#: 'hitmap_analysis'. The older ``'short_window'`` / ``'full_window'`` spellings
#: are gone rather than aliased - two names for one setting is how a caller ends
#: up passing the pair that only one of two functions accepts.
FIRMWARE_VERSIONS = ("short_exposure", "long_exposure")


def to_ps(seconds: float) -> float:
    """Convert seconds to picoseconds.

    Parameters
    ----------
    seconds : float
        A time in seconds (the package-wide parameter convention).

    Returns
    -------
    float
        The same time in picoseconds (the internal working unit).
    """
    return float(seconds) * 1e12


def to_s(picoseconds: float) -> float:
    """Convert picoseconds to seconds.

    Parameters
    ----------
    picoseconds : float
        A time in picoseconds (the internal working unit).

    Returns
    -------
    float
        The same time in seconds (the package-wide parameter convention).
    """
    return float(picoseconds) * 1e-12


def full_scale_from_max_code(max_code: int, *, code_s: float = CODE_S) -> float:
    """Return the TDC full scale a run swept, from the largest code it produced.

    The reference clock stops the ring, so the largest code present in a run
    *is* its measurement range - there is no need to be told the clock. This is
    the only resolution path in the package: a stated shutter is not the full
    scale (see the module docstring), and a remembered clock frequency is one
    more thing to get wrong.

    Accuracy is set by 'CODE_S', the nominal code width, which carries the
    ring's few-percent frequency spread; and by sample size, since the largest
    code is an extreme value and runs a little low (~1.4 % at 10 MHz for a
    populated tail). Both are irrelevant to its two uses - seeding a triangle
    half-base that the fit then refines, and bounding a histogram.

    Parameters
    ----------
    max_code : int
        Largest raw TDC code observed, i.e. ``time_series[time_series > 0].max()``
        over the run.
    code_s : float, optional
        Width of one code, in seconds. The default is 'CODE_S'.

    Returns
    -------
    float
        The full scale in seconds.

    Raises
    ------
    ValueError
        Raised when ``max_code`` is not positive, which means no pixel ever
        carried a valid timestamp.
    """
    if max_code <= 0:
        raise ValueError(
            f"max_code must be positive, got {max_code} - no valid timestamp "
            "was found, so the TDC full scale cannot be measured."
        )
    if max_code > MAX_TS_CODE:
        warnings.warn(
            f"The largest code seen ({max_code}) exceeds what the "
            f"{COARSE_BITS}-bit coarse counter can express ({MAX_TS_CODE}), so "
            "the reference clock is too slow and the codes have wrapped. The "
            "timestamps are not trustworthy.",
            UserWarning,
            stacklevel=2,
        )
    return float(max_code) * code_s


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


def resolve_live_time(
    firmware_version: str,
    nframes: int,
    *,
    open_shutter_time_s: float | None = None,
    exp_time_s: float | None = None,
) -> tuple[float | None, str]:
    """Return the live seconds per frame for the rate normalisation.

    The photon rate is ``counts / (nframes * n_files * live_per_frame)``. How
    much of each frame is photon-sensitive depends on the firmware:

    * ``'short_exposure'`` - the current firmware: each ~9 µs frame is mostly
      readout and only the open shutter window (``open_shutter_time_s``) is
      live. That window is settable over more than two decades - ~50 ns to the
      full 9 µs - so it is asked for rather than assumed. Returns ``(None, ...)`` when
      ``open_shutter_time_s`` is not supplied, so the caller can decide whether
      that is an error (DCR) or just "no rate available" (hitmap).

    * ``'long_exposure'`` - the legacy firmware: the whole frame is live, so the
      per-frame time comes from 'resolve_frame_time'.

    Parameters
    ----------
    firmware_version : str
        One of 'FIRMWARE_VERSIONS'.
    nframes : int
        Frames per file (used by the ``'long_exposure'`` resolution).
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (``'short_exposure'``
        only). The default is None.
    exp_time_s : float | None, optional
        Legacy exposure time, in seconds (``'long_exposure'`` only). The default
        is None.

    Returns
    -------
    tuple[float | None, str]
        The live seconds per frame (None if unresolved) and a human-readable
        source string.

    Raises
    ------
    ValueError
        Raised when ``firmware_version`` is not one of 'FIRMWARE_VERSIONS'.
    """
    if firmware_version == "short_exposure":
        if open_shutter_time_s is None:
            return None, "short_exposure (no open_shutter_time_s given)"
        return (
            open_shutter_time_s,
            f"short_exposure: {open_shutter_time_s * 1e9:.1f} ns window",
        )
    if firmware_version == "long_exposure":
        return resolve_frame_time(exp_time_s, nframes)
    raise ValueError(
        f"firmware_version must be one of {FIRMWARE_VERSIONS}, got "
        f"{firmware_version!r}"
    )
