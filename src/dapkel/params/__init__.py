"""Packaged parameter/calibration data for dapkel.

Holds the per-pixel TDC density-test lookup tables shipped with the library
under ``calibration_data`` (one ``.npy`` per SPAD per board, named
``{daughterboard}_{motherboard}_TDC_LUT_SPAD{n}_S{n}.npy``). Load them with
'dapkel.functions.tdc_calibration.load_board_lut', which resolves this
package's data directory via ``importlib.resources`` so it works whether the
library is run from source or installed via pip.
"""
