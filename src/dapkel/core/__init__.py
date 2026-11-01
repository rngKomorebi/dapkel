"""Shared substrate for every dapkel analysis.

The analyses in 'dapkel.functions' are all the same pipeline with a different
reduction in the middle, so the parts that do not vary live here:

    * io - stage 0: find the raw '.bin' files and size them
      (find_bin_files, BYTES_PER_FRAME).

    * timing - the units (everything in seconds), the TDC code width and the
      photon-sensitive live time a frame represents (resolve_live_time).

    * calibration - the compact 8-tap TDC calibration: load a board's tables
      (board_taps) and turn codes into picoseconds at any clock (codes_to_ps).

    * pairs - which pixel pairs a coincidence analysis is about and what each
      is called (Pixel, as_pixel_list, pair_list, pair_label).

    * reduce - stage 1: fold a set of files into one sensor map (count_map,
      accumulate_frames), the per-pixel TDC-code histogram (code_histogram),
      and the SPAD geometry inside a macropixel (SPAD_LAYOUT).

    * plots - stage 2: the two figure shapes the package draws
      (sensor_map, sorted_distribution) plus NaN-aware map_stats.

    * store - the stage 1 / stage 2 boundary: where artifacts and figures go
      (save_map, load_map, save_figure, PROCESSED_DIR, RESULTS_DIR).

Anything copied into two analysis modules belongs here instead;
'tests/test_api_surface.py' fails if a function name is defined twice. See
'docs/adding_an_analysis.md' for the contract when adding a measurement.
"""

from dapkel.core import calibration, io, pairs, plots, reduce, store, timing

__all__ = ["io", "timing", "calibration", "pairs", "reduce", "plots", "store"]
