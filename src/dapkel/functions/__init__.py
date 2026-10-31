"""Analysis functions for Kelpie v2 SPAD-camera data.

Every analysis module has exactly two public functions:

    * ``process_<x>(path, nframes, ...)`` - unpack the raw '.bin' files, reduce
      them and **always save** the result under ``<path>/processed/``; returns
      the saved path.
    * ``plot_<x>(path, ...)`` - load that saved result and write the figures to
      ``<path>/results/<x>/``; returns the data it plotted.

So a measurement is always two calls, and replotting never re-reads raw data:

    =====================  ==========================  =======================
    module                 process                     plot
    =====================  ==========================  =======================
    data_quality           process_time_codes          plot_time_codes
    hitmap_analysis        process_hitmap              plot_hitmap
    dcr_analysis           process_dcr                 plot_dcr
    crosstalk_analysis     process_crosstalk           plot_crosstalk
    tdc_calibration        process_tdc_calibration     plot_tdc_calibration
    coincidences           process_coincidences        plot_coincidences
    =====================  ==========================  =======================

'unpack' (stage 0) decodes one '.bin' file and is used by all of them. Shared
code lives in 'dapkel.core'; retired code in the repository's ``archive/``.

Modules are re-exported here rather than their functions, so call sites stay
qualified (``hitmap_analysis.plot_hitmap``).

Full documentation: ``docs/index.md``. Adding a measurement:
``docs/adding_an_analysis.md``.
"""

from dapkel.functions import (
    coincidences,
    crosstalk_analysis,
    data_quality,
    dcr_analysis,
    hitmap_analysis,
    tdc_calibration,
    unpack,
)

__all__ = [
    # stage 0 - raw data
    "unpack",
    # check the data before analysing it
    "data_quality",
    # per-measurement analyses
    "hitmap_analysis",
    "dcr_analysis",
    "crosstalk_analysis",
    "tdc_calibration",
    "coincidences",
]
