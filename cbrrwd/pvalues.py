"""p-value helpers: significance codes, multiple-testing correction, and a
column-wise Fisher's exact test with BH-FDR."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

__all__ = ["cut_p", "fdr", "fisher_test"]


def cut_p(p) -> str:
    """Convert a p-value to a compact significance code.

    ``< 0.001`` -> ``'***'``, ``< 0.01`` -> ``'**'``, ``< 0.05`` -> ``'*'``,
    ``< 0.1`` -> the value formatted as ``'0.07'``, otherwise ``'ns'``.
    """
    p = float(p)
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return f"{p:.2f}"
    else:
        return "ns"


def fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (NaN-free input expected)."""
    return multipletests(pvals, method="fdr_bh")[1]


def fisher_test(df: pd.DataFrame, ref) -> pd.DataFrame:
    """Fisher's exact test of ``ref`` against every column of ``df``, with BH-FDR.

    Each column of ``df`` is cross-tabulated against ``ref``; columns whose
    contingency table has fewer than 2 categories on either axis are dropped.
    2x2 tables use :func:`scipy.stats.fisher_exact`; larger (r x c) tables
    need the optional ``r`` extra (see :func:`cbrrwd.rbackend.contingency.fisher_exact_rc`).

    Returns
    -------
    pd.DataFrame
        Indexed by ``df`` column name, columns ``pval``, ``fdr``, ``odds_ratio``
        (``odds_ratio`` is NaN for tables larger than 2x2), sorted by ``pval``.
    """

    out = []

    for k, v in df.items():

        tab = pd.crosstab(ref, v)

        if not all(i >= 2 for i in tab.shape):
            orr_out, pval = np.nan, np.nan
        elif tab.shape == (2, 2):
            orr_out, pval = fisher_exact(tab.values)
        else:
            from .rbackend.contingency import fisher_exact_rc
            orr_out, pval = fisher_exact_rc(tab.values)

        out.append((pval, orr_out))

    df_out = pd.DataFrame(out, index=df.columns, columns=["pval", "odds_ratio"])
    df_out = df_out.dropna(subset=["pval"])

    df_out.insert(1, "fdr", fdr(df_out["pval"]))

    return df_out.sort_values("pval")
