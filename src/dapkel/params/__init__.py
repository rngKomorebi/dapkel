"""Packaged parameter/calibration data for dapkel.

Holds the compact TDC calibration shipped with the library under
``calibration_data``: one (32, 32, 8) tap-width table per SPAD per board, named
``{daughterboard}_{motherboard}_TDC_COMPACT_SPAD{n}_S{n}.npy`` and built with
'dapkel.functions.tdc_calibration.process_tdc_calibration'. Load them with
'dapkel.core.calibration.board_taps', which resolves this package's data
directory via ``importlib.resources`` so it works whether the library is run
from source or installed via pip.
"""
