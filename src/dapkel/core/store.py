"""Persistence: the boundary between stage 1 (data) and stage 2 (figures).

The layout under a data folder is::

    <data folder>/
      *.bin            stage 0 - raw, read-only, dapkel never writes here
      processed/       stage 1 - arrays ('.npy') + metadata ('.meta.json')
      results/<kind>/  stage 2 - figures only

One rule follows from it: **stage 2 gets its data by loading a stage-1 artifact,
never by re-unpacking '.bin' files.** Retrying a colormap costs a 'load_map',
not a full re-read of the raw data.

Every array is saved with a ``.meta.json`` sidecar recording how it was
acquired (frame counts, live time, tag), so a saved map can still be turned
into a *rate* months later without the original call arguments.

See 'docs/adding_an_analysis.md' for the contract.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "PROCESSED_DIR",
    "RESULTS_DIR",
    "processed_dir",
    "results_dir",
    "map_file_name",
    "meta_path",
    "read_meta",
    "save_map",
    "load_map",
    "save_figure",
]

#: Sub-folder holding stage-1 artifacts (arrays + metadata).
PROCESSED_DIR = "processed"

#: Sub-folder holding stage-2 artifacts (figures).
RESULTS_DIR = "results"


def processed_dir(folder: str, *, create: bool = True) -> str:
    """Return (and by default create) the ``processed/`` folder of a dataset.

    Parameters
    ----------
    folder : str
        The data folder holding the '.bin' files.
    create : bool, optional
        Create the folder if missing. The default is True.

    Returns
    -------
    str
        Path to ``<folder>/processed``.
    """
    out = os.path.join(folder, PROCESSED_DIR)
    if create:
        os.makedirs(out, exist_ok=True)
    return out


def results_dir(folder: str, kind: str, *, create: bool = True) -> str:
    """Return (and by default create) the figure folder for one analysis.

    Parameters
    ----------
    folder : str
        The data folder holding the '.bin' files.
    kind : str
        Analysis name, used as the sub-folder (e.g. ``'dcr'``, ``'hitmap'``).
    create : bool, optional
        Create the folder if missing. The default is True.

    Returns
    -------
    str
        Path to ``<folder>/results/<kind>``.
    """
    out = os.path.join(folder, RESULTS_DIR, kind)
    if create:
        os.makedirs(out, exist_ok=True)
    return out


def map_file_name(dataset: str, kind: str, tag: str = "") -> str:
    """Build the '.npy' file name for a stage-1 array.

    Parameters
    ----------
    dataset : str
        Name of the dataset, normally the data folder's base name.
    kind : str
        Analysis name (e.g. ``'dcr'``, ``'hitmap'``, ``'crosstalk'``).
    tag : str, optional
        Readout/quadrant tag distinguishing artifacts of the same kind. The
        default is "".

    Returns
    -------
    str
        ``'<dataset>_<tag>_<kind>.npy'``, or ``'<dataset>_<kind>.npy'`` when
        no tag is given.
    """
    parts = [dataset, tag.lower(), kind] if tag else [dataset, kind]
    return "_".join(parts) + ".npy"


def meta_path(npy_path: str) -> str:
    """Return the sidecar '.meta.json' path for a saved '.npy'."""
    return npy_path.removesuffix(".npy") + ".meta.json"


def read_meta(npy_path: str) -> dict:
    """Read a saved array's '.meta.json' sidecar.

    Parameters
    ----------
    npy_path : str
        Path to the saved '.npy'.

    Returns
    -------
    dict
        The metadata, or an empty dict when the sidecar is missing or
        unreadable - callers treat metadata as best-effort.
    """
    path = meta_path(npy_path)
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return {}


def save_map(
    data: np.ndarray,
    folder: str,
    *,
    kind: str,
    tag: str = "",
    meta: dict | None = None,
    file_name: str | None = None,
    quiet: bool = False,
) -> str:
    """Save a stage-1 array into ``processed/`` with a metadata sidecar.

    Parameters
    ----------
    data : np.ndarray
        The array to save.
    folder : str
        The data folder; the array lands in its ``processed/`` sub-folder.
    kind : str
        Analysis name (e.g. ``'dcr'``), used in the file name.
    tag : str, optional
        Readout/quadrant tag. The default is "".
    meta : dict | None, optional
        Acquisition metadata written alongside as '.meta.json'. The default
        is None (an empty sidecar is still written, recording kind and tag).
    file_name : str | None, optional
        Explicit '.npy' file name, overriding the derived one. Used where a
        naming scheme is already established (the TDC LUTs). The default is
        None.
    quiet : bool, optional
        Suppress the "saved to" printout. The default is False.

    Returns
    -------
    str
        Path the array was saved to.
    """
    out_dir = processed_dir(folder)
    dataset = os.path.basename(os.path.normpath(folder))
    name = file_name or map_file_name(dataset, kind, tag)
    out_path = os.path.join(out_dir, name)

    np.save(out_path, data)
    with open(meta_path(out_path), "w") as fh:
        json.dump({"kind": kind, "tag": tag, **(meta or {})}, fh, indent=2)

    if not quiet:
        print(f"\n> > > Saved to {out_path} < < <")
    return out_path


def load_map(
    folder: str | None = None,
    *,
    kind: str = "",
    tag: str = "",
    npy_path: str | None = None,
    file_name: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Load a stage-1 array and its metadata back from ``processed/``.

    Parameters
    ----------
    folder : str | None, optional
        The data folder to look under. May be None when ``npy_path`` is
        given. The default is None.
    kind : str, optional
        Analysis name, used to derive the file name. The default is "".
    tag : str, optional
        Readout/quadrant tag. The default is "".
    npy_path : str | None, optional
        Explicit path to a saved '.npy', bypassing the derivation entirely.
        The default is None.
    file_name : str | None, optional
        Explicit file name within ``processed/``. The default is None.

    Returns
    -------
    tuple[np.ndarray, dict]
        The array and its metadata (an empty dict when no sidecar exists).

    Raises
    ------
    ValueError
        Raised when neither ``npy_path`` nor ``folder`` is given.
    FileNotFoundError
        Raised when the '.npy' does not exist, with the path that was tried.
    """
    if npy_path is None:
        if folder is None:
            raise ValueError("give either 'folder' or 'npy_path'")
        dataset = os.path.basename(os.path.normpath(folder))
        name = file_name or map_file_name(dataset, kind, tag)
        npy_path = os.path.join(folder, PROCESSED_DIR, name)

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(
            f"No saved '{kind or 'array'}' found at:\n  {npy_path}\n"
            f"Run the matching compute_and_save_* first."
        )
    return np.load(npy_path), read_meta(npy_path)


def save_figure(fig: plt.Figure, results_dir: str, file_name: str) -> str:
    """Save a figure into the results folder.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    results_dir : str
        Folder the figure should be saved into. Created if missing.
    file_name : str
        Name of the '.png' file to save the figure as.

    Returns
    -------
    str
        Path the figure was saved to.
    """
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, file_name)
    fig.savefig(out_path)
    print(f"\n> > > Plot is saved as {file_name} in {results_dir} < < <")
    return out_path
