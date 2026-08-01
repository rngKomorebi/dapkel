"""Per-pixel sensor occupancy (hitmaps), in counts and in rate units.

Two modes, matched to what 'unpack' returns:

    * ``mode="timestamp"`` (``ORT``) - counts frames carrying a valid
      first-photon timestamp. At most one firing per macropixel per frame.
    * ``mode="count"`` (``S*C``/``ORC``) - sums the per-pixel photon counts.

The rate normalisation differs between them: ``count`` divides by the live
acquisition window, ``timestamp`` by the wall-clock frame period. Using the
window for occupancy overstates the rate by ~45x. The derivation is in
``docs/guide/hitmap.md``.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import io, plots, reduce, store, timing

__all__ = [
    # stage 1 - compute the hitmap from raw '.bin' files
    "compute_hitmap",
    "compute_hitmap_64",
    # stage 1 - compute and persist ('.npy' + '.meta.json' sidecar)
    "compute_and_save_hitmap",
    "compute_and_save_hitmap_64",
    # stage 2 - plot from an already computed map
    "plot_hitmap",
    # drivers - load a saved hitmap, figures on disk
    "collect_and_plot_hitmap",
]

#: Analysis name: the stage-1 file stem and the results sub-folder.
_KIND = "hitmap"

# SPAD quadrant layout on the full (64, 64) sensor: each S*C tag reads one
# micropixel per 2x2 macropixel, so the four (32, 32) maps interleave onto
# the full grid. The indices run CLOCKWISE, not row-major::
#
#     S0 (0,0)   S1 (0,1)
#     S3 (1,0)   S2 (1,1)
#
# Kept identical to dcr_analysis and crosstalk_analysis so the assembly and
# the cross-talk directions agree.
_SPAD_LAYOUT = {
    "S0C": (0, 0),
    "S1C": (0, 1),
    "S2C": (1, 1),
    "S3C": (1, 0),
}




def _accumulate_hitmap(
    files: list[str],
    mode: str,
    *,
    valid_min: float = 0.0,
    nframes: int | None = None,
    label: str = "",
) -> np.ndarray:
    """Unpack a list of '.bin' files and accumulate the (32, 32) hitmap.

    Parameters
    ----------
    files : list[str]
        Paths to the '.bin' files to unpack and accumulate.
    mode : str
        ``'timestamp'`` for the ORT occupancy map (per pixel, the number of
        frames carrying a valid first-photon timestamp), or ``'count'`` for
        the accumulated photon-count map (S*C / ORC).
    valid_min : float, optional
        Timestamp-validity threshold for ``mode='timestamp'``: a pixel is
        counted in a frame when ``time_series > valid_min``. 'unpack' leaves
        non-fired slots at ``<= 0``. Ignored for ``mode='count'``. The
        default is 0.0.
    nframes : int | None, optional
        Number of frames per file. When None (the default) it is derived
        from each file's size, which is the common case for the uniform
        ORT/S*C acquisitions.
    label : str, optional
        Label used in the progress printout. The default is "".

    Returns
    -------
    np.ndarray
        The (32, 32) hitmap accumulated over every frame of every file.

    Raises
    ------
    ValueError
        Raised when ``mode`` is not 'timestamp' or 'count'.
    """
    if mode not in ("timestamp", "count"):
        raise ValueError(f"mode must be 'timestamp' or 'count', got {mode!r}")

    if mode == "timestamp":
        # Occupancy: frames with a valid first-photon timestamp. Do NOT use
        # photon_counts here — in ORT those bits are the low bits of the
        # coarse timestamp, not a real count (see 'unpack').
        extract = lambda ts, pc: (ts > valid_min).sum(axis=2)  # noqa: E731
    else:  # count
        extract = lambda ts, pc: pc.sum(axis=2)  # noqa: E731

    return reduce.accumulate_frames(
        files,
        extract,
        nframes=nframes,
        need_time_series=mode == "timestamp",
        label=label,
    )


def compute_hitmap(
    folder: str,
    mode: str = "timestamp",
    tag: str = "ORT",
    nframes: int | None = None,
    max_files: int | None = None,
    *,
    valid_min: float = 0.0,
) -> np.ndarray:
    """Unpack the binary data and compute the (32, 32) hitmap.

    Builds a single-readout hitmap from the '.bin' files matching ``tag``.

    Parameters
    ----------
    folder : str
        Path to the folder with the '.bin' data files.
    mode : str, optional
        ``'timestamp'`` for the ORT occupancy hitmap (per pixel, the number
        of frames with a valid first-photon timestamp), or ``'count'`` for
        the accumulated photon-count hitmap (S*C / ORC count programs). The
        default is ``'timestamp'``.
    tag : str, optional
        Filename fragment selecting the files (e.g. ``'ORT'``, ``'ORC'``,
        or a SPAD quadrant tag ``'S0C'``). Use ``''`` to match every
        '.bin' file in the folder. The default is ``'ORT'``.
    nframes : int | None, optional
        Number of frames per file. When None (the default) it is derived
        from each file's size.
    max_files : int | None, optional
        Process at most this many files (after natural sorting). Useful
        for a quick preview over a large batch. The default is None (all
        files).
    valid_min : float, optional
        Timestamp-validity threshold for ``mode='timestamp'`` (see
        '_accumulate_hitmap'). The default is 0.0.

    Returns
    -------
    np.ndarray
        The (32, 32) hitmap.
    """
    files = io.find_bin_files(folder, tag)
    if max_files is not None:
        files = files[:max_files]
    return _accumulate_hitmap(
        files, mode, valid_min=valid_min, nframes=nframes, label=tag or mode
    )


def compute_hitmap_64(
    folder: str,
    nframes: int | None = None,
    max_files: int | None = None,
) -> np.ndarray:
    """Unpack the four S*C tags and assemble the (64, 64) count hitmap.

    Accumulates the four SPAD tags' photon counts (``mode='count'``) and
    interleaves them onto the full sensor. Only the count programs (S*C)
    tile this way; ORT/ORC are (32, 32) and use 'compute_hitmap'.

    Parameters
    ----------
    folder : str
        Path to the folder with the S*C '.bin' data files.
    nframes : int | None, optional
        Number of frames per file. When None (the default) it is derived
        from each file's size.
    max_files : int | None, optional
        Process at most this many files per tag. The default is None.

    Returns
    -------
    np.ndarray
        The (64, 64) accumulated-count hitmap for the full sensor.

    Raises
    ------
    FileNotFoundError
        Raised when no '.bin' files for any of the four SPAD tags are
        found in the folder.
    """
    tag_files = {}
    for tag in _SPAD_LAYOUT:
        files = glob.glob(os.path.join(folder, f"*_{tag}*.bin"))
        if files:
            files = sorted(
                files,
                key=lambda fp: int(
                    "".join(filter(str.isdigit, os.path.basename(fp))) or "0"
                ),
            )
            tag_files[tag] = files[:max_files] if max_files else files
    if not tag_files:
        raise FileNotFoundError(
            "No S0C/S1C/S2C/S3C .bin files found in:\n  " + folder
        )

    quadrants = {
        tag: _accumulate_hitmap(files, "count", nframes=nframes, label=tag)
        for tag, files in tag_files.items()
    }
    return reduce.assemble_64(quadrants, _SPAD_LAYOUT)


def plot_hitmap(
    hitmap: np.ndarray,
    cmap: str | None = None,
    *,
    clabel: str = "hits",
    name: str = "Hitmap",
    show_total: bool = True,
) -> plt.Figure:
    """Render a (32, 32)/(64, 64) map as a 2D image.

    Used both for the raw counts/occupancy map and for the derived rate map
    (see 'collect_and_plot_hitmap'); the caller sets the colorbar label,
    title name and whether a total is meaningful.

    Parameters
    ----------
    hitmap : np.ndarray
        Map to plot, e.g. from 'compute_hitmap'/'compute_hitmap_64', or a
        rate map (counts divided by the active acquisition time).
    cmap : str | None, optional
        Matplotlib colormap name. The default is None — use the active
        style's ``image.cmap``.
    clabel : str, optional
        Colorbar label / units (e.g. ``'counts'``, ``'cps'``). The default
        is ``'hits'``.
    name : str, optional
        Map name used in the title (e.g. ``'Hitmap'``, ``'Photon rate'``).
        The default is ``'Hitmap'``.
    show_total : bool, optional
        Include the summed total in the title. Meaningful for counts, not
        for a rate map. The default is True.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    rows, cols = hitmap.shape
    median, maximum, total = plots.map_stats(hitmap)

    stats = f"median {median:.3g}  max {maximum:.3g}"
    if show_total:
        stats += f"  total {total:.3g}"

    return plots.sensor_map(
        hitmap,
        title=f"{name}  {rows}×{cols}\n{stats}  [{clabel}]",
        clabel=clabel,
        cmap=cmap,
    )






def _acquisition_meta(
    folder: str,
    files: list[str],
    mode: str,
    tag: str,
    nframes: int | None,
    firmware_version: str,
    acq_window: float | None,
    exp_time: float | None,
) -> dict:
    """Build the acquisition metadata saved alongside a hitmap.

    Records what is needed to turn counts into a rate later: the frame count
    and the *live* time per frame, resolved per ``firmware_version`` via
    'dcr_analysis.resolve_live_time' (``'short_window'`` → the user-set
    ``acq_window`` is live; ``'full_window'`` → the whole frame is live).
    The physical rate is ``counts / active_time_s``.

    Parameters
    ----------
    folder : str
        Data folder (used by ``'full_window'`` for ``frame_rate_cnt.txt``).
    files : list[str]
        The '.bin' files that were accumulated.
    mode : str
        ``'timestamp'`` or ``'count'``.
    tag : str
        The tag used to select the files.
    nframes : int | None
        Frames per file, or None to derive from the first file.
    firmware_version : str
        ``'short_window'`` or ``'full_window'``.
    acq_window : float | None
        Photon-sensitive window per frame, in seconds (``'short_window'``).
    exp_time : float | None
        Legacy exposure time, in seconds (``'full_window'``).

    Returns
    -------
    dict
        The metadata dictionary. ``active_time_s`` is None when the live
        time could not be resolved (short_window without ``acq_window``).
    """
    nfr = nframes if nframes is not None else io.frames_in_file(files[0])
    total_frames = int(nfr) * len(files)
    frame_live_s, live_source = timing.resolve_live_time(
        folder,
        firmware_version,
        nfr,
        acq_window=acq_window,
        exp_time=exp_time,
    )
    # Wall-clock frame period (the ~9 µs repetition), independent of the live
    # window. 'timestamp' mode records at most one firing per frame, so its
    # rate ceiling is 1 / frame_period; 'count' mode uses the live window.
    frame_period_s, period_source = timing.resolve_frame_time(folder, exp_time, nfr)
    return {
        "mode": mode,
        "tag": tag,
        "n_files": len(files),
        "nframes": int(nfr),
        "total_frames": total_frames,
        "firmware_version": firmware_version,
        "acq_window_s": acq_window,
        "exp_time_s": exp_time,
        "frame_live_s": frame_live_s,
        "active_time_s": (
            total_frames * frame_live_s if frame_live_s is not None else None
        ),
        "live_time_source": live_source,
        "frame_period_s": frame_period_s,
        "wallclock_time_s": total_frames * frame_period_s,
        "frame_period_source": period_source,
    }


def compute_and_save_hitmap(
    path: str,
    mode: str = "timestamp",
    tag: str = "ORT",
    *,
    valid_min: float = 0.0,
    firmware_version: str = "short_window",
    acq_window: float | None = None,
    exp_time: float | None = None,
    nframes: int | None = None,
    max_files: int | None = None,
) -> str:
    """Unpack the '.bin' files and save the (32, 32) hitmap to '.npy'.

    Unpacks once and saves ``processed/<name>_<tag>_hitmap.npy`` plus a
    ``.meta.json`` sidecar recording the frame count and acquisition window,
    so 'collect_and_plot_hitmap' can later render a rate map without
    re-unpacking.

    Parameters
    ----------
    path : str
        Path to the folder with the '.bin' data files.
    mode : str, optional
        ``'timestamp'`` (ORT occupancy) or ``'count'`` (S*C / ORC photon
        counts). See 'compute_hitmap'. The default is ``'timestamp'``.
    tag : str, optional
        Filename fragment selecting the files. The default is ``'ORT'``.
    valid_min : float, optional
        Timestamp-validity threshold for ``mode='timestamp'``. The default
        is 0.0.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; sets how the live
        time per frame is resolved for the rate map (see
        'dcr_analysis.resolve_live_time').
    acq_window : float | None, optional
        Photon-sensitive window per frame, in seconds (e.g. ``200e-9``),
        used when ``firmware_version='short_window'``. Under that firmware
        each ~9 µs frame is mostly readout and only this window is live, so
        the photon rate is ``counts / (total_frames * acq_window)``. Stored
        in the sidecar for the rate map; None (default) means no rate unless
        supplied later at plot time.
    exp_time : float | None, optional
        Legacy exposure time, in seconds, used when
        ``firmware_version='full_window'``. The default is None.
    nframes : int | None, optional
        Number of frames per file. When None (the default) it is derived
        from each file's size.
    max_files : int | None, optional
        Process at most this many files. The default is None (all files).

    Returns
    -------
    str
        Path to the saved '.npy' hitmap.
    """
    print(
        f"\n> > > Computing hitmap (mode='{mode}', tag='{tag or 'all'}') "
        "and saving the array < < <\n"
    )
    hitmap = compute_hitmap(
        path, mode, tag, nframes, max_files, valid_min=valid_min
    )
    print(
        f"  {mode} hitmap: total {hitmap.sum():.0f} hits  "
        f"median {np.median(hitmap):.0f}  max {hitmap.max():.0f}"
    )

    files = io.find_bin_files(path, tag)
    if max_files is not None:
        files = files[:max_files]
    meta = _acquisition_meta(
        path,
        files,
        mode,
        tag,
        nframes,
        firmware_version,
        acq_window,
        exp_time,
    )
    out_path = store.save_map(
        hitmap, path, kind=_KIND, tag=(tag or mode), meta=meta, quiet=True
    )
    print(f"\n> > > Hitmap saved to {out_path} < < <")
    return out_path


def compute_and_save_hitmap_64(
    path: str,
    *,
    firmware_version: str = "short_window",
    acq_window: float | None = None,
    exp_time: float | None = None,
    nframes: int | None = None,
    max_files: int | None = None,
) -> str:
    """Assemble and save the (64, 64) full-sensor count hitmap to '.npy'.

    Like 'compute_and_save_hitmap' but for the four-quadrant (64, 64)
    accumulated-count map (see 'compute_hitmap_64'); saved as
    ``<name>_64_hitmap.npy`` in ``processed``, with a ``.meta.json``
    sidecar (frame count per SPAD tag; each quadrant is acquired for the
    same duration) for the rate map.

    Parameters
    ----------
    path : str
        Path to the folder with the S*C '.bin' data files.
    firmware_version : str, optional
        ``'short_window'`` (default) or ``'full_window'``; see
        'compute_and_save_hitmap'.
    acq_window : float | None, optional
        Photon-sensitive window per frame, in seconds (e.g. ``200e-9``),
        for ``firmware_version='short_window'``. The default is None.
    exp_time : float | None, optional
        Legacy exposure time, in seconds, for
        ``firmware_version='full_window'``. The default is None.
    nframes : int | None, optional
        Number of frames per file. When None (the default) it is derived
        from each file's size.
    max_files : int | None, optional
        Process at most this many files per tag. The default is None.

    Returns
    -------
    str
        Path to the saved '.npy' hitmap.
    """
    print("\n> > > Computing (64, 64) full-sensor hitmap and saving < < <\n")
    hitmap = compute_hitmap_64(path, nframes, max_files)
    print(
        f"  counts hitmap: total {hitmap.sum():.0f} hits  "
        f"median {np.median(hitmap):.0f}  max {hitmap.max():.0f}"
    )

    # Frame count from one SPAD tag (all four span the same duration).
    tag_files: list[str] = []
    for tag in _SPAD_LAYOUT:
        tag_files = io.find_bin_files(path, f"_{tag}")
        if tag_files:
            break
    if max_files is not None:
        tag_files = tag_files[:max_files]
    meta = _acquisition_meta(
        path,
        tag_files,
        "count",
        "64",
        nframes,
        firmware_version,
        acq_window,
        exp_time,
    )
    out_path = store.save_map(
        hitmap, path, kind=_KIND, tag="64", meta=meta, quiet=True
    )
    print(f"\n> > > Hitmap saved to {out_path} < < <")
    return out_path


def _rate_active_time(
    mode: str,
    meta: dict,
    path: str | None,
    tag: str,
    acq_window: float | None,
) -> tuple[float | None, str]:
    """Return (total active time in seconds, source) for the rate map.

    The normalisation differs by mode (see 'collect_and_plot_hitmap'):

    * ``'count'`` — divide by the live window: ``total_frames * live``, where
      ``live`` is ``acq_window`` (given here or from the sidecar) or, for
      full-window firmware, the resolved per-frame live time.
    * ``'timestamp'`` — divide by the wall-clock frame period:
      ``total_frames * frame_period`` (one firing per frame; ceiling
      ``1 / frame_period``).

    Missing values are recovered from the '.bin' files in ``path`` (sizes for
    the frame count; ``frame_rate_cnt.txt`` / 9 µs for the period) so old
    '.npy' saved before the sidecar existed still get a rate.

    Parameters
    ----------
    mode : str
        ``'timestamp'`` or ``'count'`` (the effective mode from the sidecar).
    meta : dict
        The sidecar metadata (possibly empty).
    path : str | None
        Data folder, used to recover missing values from the '.bin' files.
    tag : str
        Tag selecting the '.bin' files, for the recovery.
    acq_window : float | None
        Live window per frame (s), overriding the sidecar's; ``'count'`` mode.

    Returns
    -------
    tuple[float | None, str]
        The total active time in seconds (None if it cannot be determined)
        and a human-readable source string.
    """
    total_frames = meta.get("total_frames")
    nfr = meta.get("nframes")
    if (not total_frames or not nfr) and path is not None:
        try:
            bins = io.find_bin_files(path, tag)
            nfr = io.frames_in_file(bins[0])
            total_frames = sum(io.frames_in_file(f) for f in bins)
        except (FileNotFoundError, ValueError):
            pass
    if not total_frames:
        return None, "no frame count"

    if mode == "count":
        per_frame = acq_window
        if per_frame is None:
            per_frame = meta.get("acq_window_s") or meta.get("frame_live_s")
        if per_frame is None:
            return None, "no live window"
        return (
            total_frames * per_frame,
            f"count / {per_frame * 1e9:.1f} ns live",
        )

    # timestamp: one firing per frame period (wall clock)
    period = meta.get("frame_period_s")
    if period is None and path is not None and nfr:
        period, _ = timing.resolve_frame_time(path, None, int(nfr))
    if period is None:
        return None, "no frame period"
    return total_frames * period, f"timestamp / {period * 1e6:.3f} µs frame"


def collect_and_plot_hitmap(
    path: str | None = None,
    mode: str = "timestamp",
    tag: str = "ORT",
    *,
    npy_path: str | None = None,
    cmap: str | None = None,
    rate: bool = True,
    acq_window: float | None = None,
) -> np.ndarray:
    """Load a saved hitmap '.npy' and save its hitmap plot(s).

    Reads the saved array - no re-unpacking - and writes the counts hitmap
    to ``results/hitmap``, plus a **second** figure in rate units when the
    active time is known: cps (``count``) or Hz (``timestamp``).

    The two modes normalise differently (live window vs frame period); the
    derivation is in ``docs/guide/hitmap.md``. Timing comes from the
    '.meta.json' sidecar, with ``acq_window`` overriding it if given.

    Locate the '.npy' from ``path`` or directly via ``npy_path``, which
    ignores ``path``.

    Parameters
    ----------
    path : str | None, optional
        Data-folder path used to locate the saved '.npy'. May be None when
        ``npy_path`` is given. Required otherwise.
    mode : str, optional
        Used with ``tag`` to build the default '.npy' name (the suffix is
        ``tag or mode``) and to label the plots. The default is
        ``'timestamp'``.
    tag : str, optional
        Used to build the default '.npy' name. The default is ``'ORT'``.
    npy_path : str | None, optional
        Explicit path to a saved hitmap '.npy' to plot directly; when given,
        ``path`` / ``mode`` / ``tag`` are ignored (mode/units then come from
        the sidecar). The default is None.
    cmap : str | None, optional
        Matplotlib colormap name for the hitmaps. The default is None —
        use the active style's ``image.cmap``.
    rate : bool, optional
        Also save the rate hitmap when the active time is known. The
        default is True.
    acq_window : float | None, optional
        Active acquisition window per frame, in seconds; overrides the
        sidecar value. The default is None (use the sidecar's, if any).

    Returns
    -------
    np.ndarray
        The loaded hitmap (counts/occupancy).

    Raises
    ------
    FileNotFoundError
        Raised when the '.npy' hitmap cannot be found.
    ValueError
        Raised when neither ``path`` nor ``npy_path`` is given.
    """
    if npy_path is not None:
        npy_path = os.path.abspath(npy_path)
        # <root>/processed/<name>.npy -> the figures belong under <root>.
        root = os.path.dirname(os.path.dirname(npy_path))
        base = os.path.basename(npy_path).removesuffix(".npy")
    else:
        if path is None:
            raise ValueError(
                "Provide either path (to locate the saved hitmap) or npy_path."
            )
        root = path
        base = store.map_file_name(
            os.path.basename(os.path.normpath(path)), _KIND, tag or mode
        ).removesuffix(".npy")

    hitmap, meta = store.load_map(
        path, kind=_KIND, tag=(tag or mode), npy_path=npy_path
    )
    mode_eff = meta.get("mode", mode)
    counts_clabel = "photon counts" if mode_eff == "count" else "frames fired"
    print(
        f"  hitmap {hitmap.shape[0]}×{hitmap.shape[1]}: total "
        f"{hitmap.sum():.0f}  median {np.median(hitmap):.0f}  "
        f"max {hitmap.max():.0f}  [{counts_clabel}]"
    )

    # Figures go to 'results/hitmap'; the array stays in 'processed/'.
    results_dir = store.results_dir(root, _KIND)

    # 1) counts / occupancy map
    fig_counts = plot_hitmap(hitmap, cmap=cmap, clabel=counts_clabel)
    store.save_figure(fig_counts, results_dir, f"{base}.png")
    plt.close(fig_counts)

    # 2) rate map. The normalisation depends on the mode:
    #      * 'count'     — photons accumulate within the live window, so the
    #        rate is counts / (total_frames * live_window): counts/s (cps).
    #      * 'timestamp' — at most one firing per frame, so the honest rate is
    #        occupancy / frame_period = counts / (total_frames * frame_period):
    #        firing rate in Hz, ceiling 1 / frame_period (~9 µs → ~111 kHz).
    #    Dividing occupancy by the (short) live window would overstate it by
    #    frame_period / live_window (e.g. 45x for 9 µs / 200 ns).
    if rate:
        active_time, rate_source = _rate_active_time(
            mode_eff, meta, path, tag, acq_window
        )
        if active_time:
            rate_map = hitmap / active_time
            rate_clabel = "cps" if mode_eff == "count" else "Hz"
            print(
                f"  rate ({rate_source}): active time {active_time * 1e3:.3f} "
                f"ms  median {np.median(rate_map):.3g}  "
                f"max {rate_map.max():.3g} {rate_clabel}"
            )
            fig_rate = plot_hitmap(
                rate_map,
                cmap=cmap,
                clabel=rate_clabel,
                name="Photon rate",
                show_total=False,
            )
            store.save_figure(fig_rate, results_dir, f"{base}_rate.png")
            # plt.close(fig_rate)
        else:
            print(
                f"  (no rate map — {rate_source}; pass acq_window=<seconds> "
                "for count mode, or recompute so .meta.json stores the times)"
            )

    return hitmap
