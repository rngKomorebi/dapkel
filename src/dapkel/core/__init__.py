"""Shared substrate for every dapkel analysis.

The analyses in 'dapkel.functions' are all the same pipeline with a different
reduction in the middle, so the parts that do not vary live here:

    * io - stage 0: find the raw '.bin' files and size them
      (find_bin_files, frames_in_file, BYTES_PER_FRAME).

    * timing - stage 1: work out the frame period and the photon-sensitive
      live time an acquisition represents (resolve_frame_time,
      resolve_live_time).

    * pairs - stage 1: turn two detector groups into the ordered pixel pairs
      and labels every coincidence artifact is indexed by (pair_list,
      pair_labels, pair_label).

    * reduce - stage 1: fold a set of files into one sensor map
      (accumulate_frames) and interleave four quadrants into the full
      (64, 64) sensor (assemble_64).

    * plots - stage 2: the two figure shapes the package draws
      (sensor_map, sorted_distribution) plus NaN-aware map_stats.

    * store - the stage 1 / stage 2 boundary: where artifacts and figures go
      (save_figure, PROCESSED_DIR, RESULTS_DIR).

Anything copied into two analysis modules belongs here instead;
'tests/test_api_surface.py' fails if a function name is defined twice. See
'docs/adding_an_analysis.md' for the contract when adding a measurement.
"""

from dapkel.core import io, pairs, plots, reduce, store, timing

__all__ = ["io", "timing", "pairs", "reduce", "plots", "store"]
