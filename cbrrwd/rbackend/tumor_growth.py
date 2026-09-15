"""Tumor growth/decay model fitting via the R ``tumgr`` package
(Wilkerson et al., Lancet Oncology, 2017)."""

from __future__ import annotations

import pandas as pd

from . import r_package, require_rpy2
from ._bridge import pandas_to_rpy2, rpy2_to_pandas


def gdrate(df_in: pd.DataFrame, pval: float = 0.05, plot: bool = False) -> pd.DataFrame:
    """Fit tumor growth/decay models (Wilkerson et al., Lancet Oncology, 2017)
    and return the results table.

    Needs the ``r`` extra plus the R package ``tumgr``.
    """
    ro, *_ = require_rpy2()
    tumgr = r_package("tumgr")

    df_out = tumgr.gdrate(pandas_to_rpy2(df_in), pval, ro.vectors.BoolVector([plot])).rx2("results")

    return rpy2_to_pandas(df_out)
