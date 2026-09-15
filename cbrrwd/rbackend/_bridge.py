"""Thin numpy/pandas <-> rpy2 conversion helpers."""

from __future__ import annotations

from . import require_rpy2


def numpy_to_rpy2(p_in):
    ro, numpy2ri, _, localconverter, _ = require_rpy2()
    with localconverter(ro.default_converter + numpy2ri.converter):
        return ro.conversion.get_conversion().py2rpy(p_in)


def rpy2_to_numpy(r_in):
    ro, numpy2ri, _, localconverter, _ = require_rpy2()
    with localconverter(ro.default_converter + numpy2ri.converter):
        return ro.conversion.get_conversion().rpy2py(r_in)


def pandas_to_rpy2(p_in):
    ro, _, pandas2ri, localconverter, _ = require_rpy2()
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.get_conversion().py2rpy(p_in)


def rpy2_to_pandas(r_in):
    ro, _, pandas2ri, localconverter, _ = require_rpy2()
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.get_conversion().rpy2py(r_in)
