"""Dark count rate (DCR) maps, per quadrant or for the full sensor.

    process_dcr   raw '.bin' -> processed/<name>_<tag>_dcr.npy (cps)
    plot_dcr      that array -> results/dcr/ heatmap and sorted distribution

DCR is accumulated dark counts divided by the time the sensor was actually
collecting. That live time is NOT the frame period: under the current
``short_exposure`` firmware only the user-set ``open_shutter_time_s`` of each
~9 us frame is photon-sensitive.

What a count is follows from the tag ('core.reduce'): the photon counter for
``...C`` programs, frames carrying a valid timestamp for ``...T`` ones.
In timestamp mode only the *first* dark count of a frame is recorded, so the
rate saturates at one per ``open_shutter_time_s`` (125 kcps at an 8 µs window)
and under-reports below it; ``pileup_correct=True`` inverts that censoring
under Poisson statistics.

See ``docs/guide/dcr.md``.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from dapkel.core import plots, reduce, store, timing

__all__ = ["process_dcr", "plot_dcr"]

#: Analysis name: the stage-1 file stem and results sub-folder.
_KIND = "dcr"


def process_dcr(
    path: str,
    nframes: int,
    tag: str,
    *,
    pileup_correct: bool = False,
    firmware_version: str = "short_exposure",
    open_shutter_time_s: float | None = 500e-9,
    exp_time_s: float | None = None,
    max_files: int | None = None,
) -> str:
    """Unpack the dark '.bin' files, compute the DCR map in cps and save it.

    Parameters
    ----------
    path : str
        Folder with the '.bin' data files.
    nframes : int
        Frames the acquisition recorded per '.bin' file.
    tag : str
        File tag: ``'S0C'``... ``'S3C'``, ``'S0T'``..., ``'ORC'``, ``'ORT'``.
    pileup_correct : bool, optional
        Timestamp tags only. Each frame is a Bernoulli trial - a frame fires
        with ``p = 1 - exp(-r * w)`` because only the first count is recorded -
        so the occupancy is replaced by its maximum-likelihood
        ``-N ln(1 - hits/N)`` before dividing by the live time. A pixel firing
        in every frame is clipped to the half-count level, a finite lower
        bound. The default is False (the raw occupancy rate).
    firmware_version : str, optional
        ``'short_exposure'`` (default) or ``'long_exposure'``; see
        'core.timing.resolve_live_time'.
    open_shutter_time_s : float | None, optional
        Photon-sensitive window per frame, in seconds. The default is
        ``500e-9``.
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
    mode = reduce.readout_mode(tag)
    pileup_correct = pileup_correct and mode == "timestamp"
    print(f"\n> > > Computing the '{tag}' DCR map (mode='{mode}') < < <\n")
    counts, n_files = reduce.count_map(
        path, tag, nframes=nframes, max_files=max_files
    )
    total_frames = int(nframes) * n_files
    if pileup_correct:
        p = np.minimum(counts / total_frames, 1.0 - 0.5 / total_frames)
        counts = -total_frames * np.log1p(-p)
    dcr = counts / (total_frames * live_s)

    # A saturated *pixel* is normal in a DCR map; it is the number attached to
    # it that stops being a rate, so this is a note rather than an error.
    ceiling = 1.0 / live_s
    if mode == "timestamp" and not pileup_correct and np.nanmax(dcr) > 0.05 * ceiling:
        print(
            f"  ! timestamp mode saturates at {ceiling:.3g} cps; the hottest "
            f"pixel is at {np.nanmax(dcr) / ceiling * 100:.0f}% of that and "
            f"{int((dcr > 0.5 * ceiling).sum())} are past 50%, so those read "
            "low. Pass pileup_correct=True for the Poisson-corrected rate."
        )

    out_path = store.save_map(
        dcr,
        path,
        kind=_KIND,
        tag=tag,
        meta={
            "mode": mode,
            "pileup_correct": pileup_correct,
            "n_files": n_files,
            "nframes": int(nframes),
            "firmware_version": firmware_version,
            "open_shutter_time_s": open_shutter_time_s,
            "exp_time_s": exp_time_s,
            "frame_live_s": live_s,
            "live_time_source": live_source,
            "unit": "cps",
        },
        quiet=True,
    )
    print(
        f"  median {np.nanmedian(dcr):.0f}  max {np.nanmax(dcr):.0f} cps\n\n"
        f"> > > DCR map saved to {out_path} < < <"
    )
    return out_path


def plot_dcr(path: str, tag: str, *, cmap: str | None = None) -> np.ndarray:
    """Load a saved DCR map and save its heatmap and sorted distribution.

    Parameters
    ----------
    path : str
        Data folder that 'process_dcr' was run on.
    tag : str
        The tag it was run with.
    cmap : str | None, optional
        Matplotlib colormap name for the heatmap. The default is None - the
        active style's.

    Returns
    -------
    np.ndarray
        The DCR map, in counts per second.

    Raises
    ------
    FileNotFoundError
        Raised when no saved DCR map exists for ``tag``.
    """
    dcr, _ = store.load_map(path, kind=_KIND, tag=tag)
    name = store.map_file_name(
        os.path.basename(os.path.normpath(path)), _KIND, tag
    ).removesuffix(".npy")
    results_dir = store.results_dir(path, _KIND)

    median, maximum, _ = plots.map_stats(dcr)
    fig = plots.sensor_map(
        dcr,
        title=(
            f"DCR map  {dcr.shape[0]}×{dcr.shape[1]} SPADs\n"
            f"median {median:.0f}  max {maximum:.0f}  cps"
        ),
        cmap=cmap,
    )
    store.save_figure(fig, results_dir, f"{name}_heatmap.png")
    plt.close(fig)

    fig = plots.sorted_distribution(
        dcr,
        ylabel="DCR  (cps)",
        title="Sorted DCR distribution",
        logy=True,  # DCR spans decades
        fmt=".0f",
        unit="cps",
    )
    store.save_figure(fig, results_dir, f"{name}_distribution.png")
    plt.close(fig)
    return dcr
