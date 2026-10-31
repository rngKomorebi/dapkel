"""Per-pixel sensor occupancy (hitmaps), in counts and in photon rate.

    process_hitmap   raw '.bin' -> processed/<name>_<tag>_hitmap.npy
    plot_hitmap      that array -> results/hitmap/ the counts map and the rate map

The tag says what a count is ('core.reduce'): ``...T`` programs (``ORT``,
``S0T``) count frames carrying a valid first-photon timestamp - at most one per
pixel per frame - and ``...C`` programs (``ORC``, ``S0C``) sum the photon
counter.

The rate divides by the total *live* (shutter-open) time of the acquisition,
``nframes * n_files * open_shutter_time_s`` - not the full ~9 us frame period.
Only the open shutter is photon-sensitive; normalising by the whole frame would
dilute the rate by the frame/window ratio. See ``docs/guide/hitmap.md``.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import plots, reduce, store, timing

__all__ = ["process_hitmap", "plot_hitmap"]

#: Analysis name: the stage-1 file stem and the results sub-folder.
_KIND = "hitmap"


def process_hitmap(
    path: str,
    nframes: int,
    tag: str = "ORT",
    *,
    firmware_version: str = "short_exposure",
    open_shutter_time_s: float | None = 500e-9,
    exp_time_s: float | None = None,
    max_files: int | None = None,
) -> str:
    """Unpack the '.bin' files, accumulate the hitmap and save it.

    Writes ``processed/<name>_<tag>_hitmap.npy`` (counts, or frames fired for a
    ``...T`` tag) with a ``.meta.json`` sidecar recording the frame count and
    the live time, which is everything 'plot_hitmap' needs for the rate map.

    Parameters
    ----------
    path : str
        Folder with the '.bin' data files.
    nframes : int
        Frames the acquisition recorded per '.bin' file. Sets the rate's
        denominator, ``nframes * n_files * open_shutter_time_s``. Do not infer
        it from the file size - the exe dumps a fixed, larger DDR3 region.
    tag : str, optional
        File tag: ``'ORT'``, ``'ORC'``, ``'S0C'``, ``'S2T'``... The default is
        ``'ORT'``.
    firmware_version : str, optional
        ``'short_exposure'`` (default, only ``open_shutter_time_s`` of each
        frame is live) or ``'long_exposure'`` (the whole ``exp_time_s + 9 µs``
        cycle is); see 'core.timing.resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds; settable from ~50 ns to
        9 µs, so it cannot be guessed. The default is ``500e-9``.
    exp_time_s : float | None, optional
        Exposure requested of ``'long_exposure'`` firmware, in seconds. The
        default is None.
    max_files : int | None, optional
        Use at most this many files. The default is None (all).

    Returns
    -------
    str
        Path to the saved '.npy'.
    """
    live_s, live_source = timing.resolve_live_time(
        firmware_version,
        open_shutter_time_s=open_shutter_time_s,
        exp_time_s=exp_time_s,
    )
    print(f"\n> > > Computing the '{tag}' hitmap < < <\n")
    counts, n_files = reduce.count_map(
        path, tag, nframes=nframes, max_files=max_files
    )
    total_frames = int(nframes) * n_files
    out_path = store.save_map(
        counts,
        path,
        kind=_KIND,
        tag=tag,
        meta={
            "mode": reduce.readout_mode(tag),
            "n_files": n_files,
            "nframes": int(nframes),
            "total_frames": total_frames,
            "firmware_version": firmware_version,
            "open_shutter_time_s": open_shutter_time_s,
            "exp_time_s": exp_time_s,
            "frame_live_s": live_s,
            "live_time_source": live_source,
            "live_time_s": total_frames * live_s,
        },
        quiet=True,
    )
    print(
        f"  total {np.nansum(counts):.0f}  median {np.nanmedian(counts):.0f}  "
        f"max {np.nanmax(counts):.0f}\n\n> > > Hitmap saved to {out_path} < < <"
    )
    return out_path


def plot_hitmap(
    path: str,
    tag: str = "ORT",
    *,
    cmap: str | None = None,
    clip_colorbar_to: float | None = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved hitmap and save its counts and photon-rate maps.

    Writes two figures to ``results/hitmap``: the counts (frames fired) map and
    the photon-rate map, ``hits / (nframes * n_files * open_shutter_time_s)``
    in Hz, both normalisation inputs read from the sidecar. Derivation:
    ``docs/guide/hitmap.md``.

    In timestamp mode only the *first* photon of a frame is timestamped, so the
    rate saturates at one per ``open_shutter_time_s`` - 2 MHz at a 500 ns
    window - and under-reports as it approaches that.

    Parameters
    ----------
    path : str
        Data folder that 'process_hitmap' was run on.
    tag : str, optional
        The tag it was run with. The default is ``'ORT'``.
    cmap : str | None, optional
        Matplotlib colormap name. The default is None - the active style's.
    clip_colorbar_to : float | None, optional
        Percentile the colour scale is clipped at, so a few hot pixels do not
        flatten the rest into one shade. The default is 99. None runs the scale
        to the true maximum - the value quoted in the title either way.

    Returns
    -------
    counts : np.ndarray
        The (32, 32) counts map - frames fired for a ``...T`` tag, photon
        counts for a ``...C`` one.
    rate : np.ndarray
        The (32, 32) photon-rate map, in Hz.

    Raises
    ------
    FileNotFoundError
        Raised when no saved hitmap exists for ``tag``.
    """
    hitmap, meta = store.load_map(path, kind=_KIND, tag=tag)
    rate = hitmap / meta["live_time_s"]
    name = store.map_file_name(
        os.path.basename(os.path.normpath(path)), _KIND, tag
    ).removesuffix(".npy")
    results_dir = store.results_dir(path, _KIND)

    counts_label = "photon counts" if meta["mode"] == "count" else "frames fired"
    for data, clabel, title, suffix in (
        (hitmap, counts_label, "Hitmap", ""),
        (rate, "Hz", "Photon rate", "_rate"),
    ):
        median, maximum, total = plots.map_stats(data)
        stats = f"median {median:.3g}  max {maximum:.3g}"
        if not suffix:
            stats += f"  total {total:.3g}"
        fig = plots.sensor_map(
            data,
            title=f"{title}  {data.shape[0]}×{data.shape[1]}\n{stats}  [{clabel}]",
            clabel=clabel,
            cmap=cmap,
            clip_colorbar_to=clip_colorbar_to,
        )
        store.save_figure(fig, results_dir, f"{name}{suffix}.png")
        plt.close(fig)

    print(
        f"  {meta['total_frames']} frames x {meta['frame_live_s'] * 1e9:.0f} ns "
        f"live = {meta['live_time_s'] * 1e3:.3f} ms;  rate median "
        f"{np.nanmedian(rate):.3g}  max {np.nanmax(rate):.3g} Hz"
    )
    return hitmap, rate
