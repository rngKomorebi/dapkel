"""Analysis functions for Kelpie v2 SPAD-camera data.

Each module decodes the raw '.bin' files with 'unpack' and does one kind of
reduction/plot; shared code lives in 'dapkel.core'. Every module declares its
public surface in ``__all__`` - read that list to see what you can call.

Naming convention:

    * ``compute_*``     raw '.bin' files (or an array) in, array out - no I/O
    * ``plot_*``        an already computed array in, a Figure out - no I/O
    * ``collect_and_*`` driver: walks a folder and writes figures to disk
    * ``load_*``        read a saved artifact back

Modules are re-exported here rather than their functions, so call sites stay
qualified (``dcr_analysis.plot_heatmap``): the package has ``plot_heatmap``,
``plot_hitmap`` and ``plot_ct_heatmap``, only unambiguous when the module is
named.

Full documentation: ``docs/index.md``. Adding a measurement:
``docs/adding_an_analysis.md``.
"""

from dapkel.functions import (
    background_subtraction,
    calc_diff,
    crosstalk_analysis,
    data_quality,
    dcr_analysis,
    delta_t,
    hitmap_analysis,
    tdc_calibration,
    unpack,
)

__all__ = [
    # stage 0 - raw data
    "unpack",
    # check the data before analysing it
    "data_quality",
    # stage 1/2 - per-measurement analyses
    "dcr_analysis",
    "crosstalk_analysis",
    "hitmap_analysis",
    "tdc_calibration",
    "calc_diff",
    "background_subtraction",
    "delta_t",
]
