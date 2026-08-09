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

* ``open_shutter_time_s`` - the photon-sensitive window per frame. Sets the rate
  normalisation, and bounds how far apart two detections in one frame can be.
* the **TDC full scale** - the ORT measurement range, i.e. one period of the
  reference clock. The timestamp is a code on this range, so it is what the
  accidental (triangle) background's half-base follows.

They coincide only when the shutter is one clock period. When the shutter is
*longer*, the free-running oscillator has no cycle counter, so timestamps fold
into the full scale: a 500 ns shutter read by a 10 MHz (100 ns) TDC gives a
100 ns triangle, not a 500 ns one. Never assume one from the other -
'resolve_tdc_full_scale_s' takes the full scale from the calibration LUT, which
is self-describing, and only falls back to the shutter when there is no LUT.
"""

from __future__ import annotations

import warnings

__all__ = [
    "CLK_PERIOD",
    "CODE_S",
    "COARSE_BITS",
    "FREE_RUNNING_US",
    "MAX_TS_CODE",
    "codes_for_full_scale",
    "resolve_cycle_time",
    "resolve_frame_time",
    "resolve_live_time",
    "resolve_tdc_full_scale_s",
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


def codes_for_full_scale(
    full_scale_s: float, *, code_s: float = CODE_S, headroom: float = 1.1
) -> int:
    """Size the TDC code axis for a given full scale.

    The ring sweeps ``full_scale_s / code_s`` codes per measurement range; the
    headroom covers per-pixel spread in the ring frequency, and the code axis is
    trimmed to what is actually populated afterwards anyway.

    Parameters
    ----------
    full_scale_s : float
        TDC full scale (one reference-clock period), in seconds.
    code_s : float, optional
        Nominal width of one code, in seconds. The default is 'CODE_S'.
    headroom : float, optional
        Multiplier applied to the nominal code count. The default is 1.1.

    Returns
    -------
    int
        Number of code bins, capped at ``MAX_TS_CODE + 1``.

    Raises
    ------
    ValueError
        Raised when ``full_scale_s`` or ``code_s`` is not positive.

    Warns
    -----
    UserWarning
        Raised when the requested full scale exceeds what the coarse counter
        can express, i.e. the reference clock is too slow and codes will wrap.
    """
    if full_scale_s <= 0:
        raise ValueError(f"full_scale_s must be positive, got {full_scale_s}")
    if code_s <= 0:
        raise ValueError(f"code_s must be positive, got {code_s}")

    nominal = full_scale_s / code_s
    if nominal > MAX_TS_CODE:
        warnings.warn(
            f"A full scale of {full_scale_s * 1e9:.1f} ns needs "
            f"~{nominal:.0f} codes, but the {COARSE_BITS}-bit coarse counter "
            f"saturates at {MAX_TS_CODE} "
            f"(~{MAX_TS_CODE * code_s * 1e9:.0f} ns). The reference clock is "
            f"too slow - below ~{1e-6 / (MAX_TS_CODE * code_s):.2f} MHz the "
            "codes wrap and the timestamps are meaningless.",
            UserWarning,
            stacklevel=2,
        )
    return min(int(nominal * headroom) + 1, MAX_TS_CODE + 1)


def resolve_tdc_full_scale_s(
    *,
    time_lut=None,
    full_scale_s: float | None = None,
    open_shutter_time_s: float | None = None,
) -> tuple[float, str]:
    """Return the TDC full scale to model the accidental background on.

    The timestamp is a code on the TDC's measurement range, so a difference of
    two timestamps is bounded by that range and - phases being uniform within it
    - distributed as a triangle of that half-base. Resolution order, most to
    least trustworthy:

    1. ``time_lut`` - a calibration LUT is self-describing: its largest entry
       *is* the full scale it was built for.
    2. ``full_scale_s`` - stated outright by the caller.
    3. ``open_shutter_time_s`` - a fallback that is only correct when the
       shutter is exactly one clock period. Warns, because a shutter longer than
       the clock period folds (see the module docstring).

    Parameters
    ----------
    time_lut : np.ndarray | None, optional
        A ``(n_codes,)`` or ``(32, 32, n_codes)`` code -> ps LUT. The default
        is None.
    full_scale_s : float | None, optional
        TDC full scale in seconds, stated explicitly. The default is None.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds, used only as a last
        resort. The default is None.

    Returns
    -------
    tuple[float, str]
        The full scale in seconds and a human-readable source string.

    Raises
    ------
    ValueError
        Raised when none of the three inputs is given.

    Warns
    -----
    UserWarning
        Raised when the full scale had to be guessed from the shutter, or when
        a LUT and an explicit ``full_scale_s`` disagree by more than 5 %.
    """
    lut_s = None
    if time_lut is not None:
        lut_s = to_s(float(time_lut.max()))

    if lut_s is not None and full_scale_s is not None:
        if abs(lut_s - full_scale_s) > 0.05 * lut_s:
            warnings.warn(
                f"full_scale_s={full_scale_s * 1e9:.1f} ns disagrees with the "
                f"LUT's own full scale of {lut_s * 1e9:.1f} ns. Using the LUT "
                "- it is the table the timestamps were actually converted "
                "with. Pass the matching calibration instead.",
                UserWarning,
                stacklevel=2,
            )
        return lut_s, f"LUT full scale ({lut_s * 1e9:.2f} ns)"

    if lut_s is not None:
        return lut_s, f"LUT full scale ({lut_s * 1e9:.2f} ns)"
    if full_scale_s is not None:
        return float(full_scale_s), f"stated ({full_scale_s * 1e9:.2f} ns)"
    if open_shutter_time_s is not None:
        warnings.warn(
            "No calibration LUT and no full_scale_s, so the TDC full scale is "
            f"taken from open_shutter_time_s="
            f"{open_shutter_time_s * 1e9:.1f} ns. That is only right when the "
            "shutter is exactly one reference-clock period; a longer shutter "
            "folds the timestamps into the clock period and the accidental "
            "triangle is narrower than this. Pass full_scale_s to be sure.",
            UserWarning,
            stacklevel=2,
        )
        return (
            float(open_shutter_time_s),
            f"assumed = open shutter time ({open_shutter_time_s * 1e9:.1f} ns)",
        )

    raise ValueError(
        "Cannot resolve the TDC full scale: pass a time_lut, or full_scale_s, "
        "or open_shutter_time_s (in seconds)."
    )

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
    50-500 ns exposure window sits *inside* the cycle rather than extending it.
    So a rate normalised by the nominal 9 µs runs ~7.8% high. Whether to prefer
    the nominal or a measured cycle is a decision for the experiment, not for
    this function to make silently off a file it happens to find on disk.

    Parameters
    ----------
    firmware_version : str
        ``'short_exposure'`` (the cycle is always 9 µs) or ``'long_exposure'``
        (the cycle is ``exp_time_s`` + 9 µs). ``'short_window'`` and
        ``'full_window'`` are accepted as aliases of the two.
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

    * ``'short_window'`` - the current firmware: each ~9 µs frame is mostly
      readout and only the open shutter window (``open_shutter_time_s``, e.g.
      ``200e-9`` s, ``< 9 µs``) is live. Returns ``(None, ...)`` when
      ``open_shutter_time_s`` is not supplied, so the caller can decide whether
      that is an error (DCR) or just "no rate available" (hitmap).

    * ``'full_window'`` - the legacy firmware: the whole frame is live, so the
      per-frame time comes from 'resolve_frame_time'.

    Parameters
    ----------
    firmware_version : str
        ``'short_window'`` or ``'full_window'``.
    nframes : int
        Frames per file (used by the ``'full_window'`` resolution).
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds (``'short_window'``
        only). The default is None.
    exp_time_s : float | None, optional
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
        if open_shutter_time_s is None:
            return None, "short_window (no open_shutter_time_s given)"
        return (
            open_shutter_time_s,
            f"short_window: {open_shutter_time_s * 1e9:.1f} ns window",
        )
    if firmware_version == "full_window":
        return resolve_frame_time(exp_time_s, nframes)
    raise ValueError(
        "firmware_version must be 'short_window' or 'full_window', got "
        f"{firmware_version!r}"
    )
