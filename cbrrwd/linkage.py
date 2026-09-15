"""Linking a reference timepoint per patient to the closest matching record
in a second (e.g. lab-value) table -- a common real-world-data task when two
tables are keyed by patient but not aligned to the same timestamps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["find_closest"]


def find_closest(d: pd.DataFrame,
                  ref: pd.Series,
                  code: str,
                  code_var: str = "CODE",
                  patient_var: str = "PATISAPNR",
                  date_var: str = "LABEINDAT",
                  target_var: str = "ERGEBNISF") -> pd.DataFrame:
    """For each ``(patient, timestamp)`` entry in ``ref``, find the closest
    ``d`` row for the same patient and lab ``code`` by date.

    Parameters
    ----------
    d : pd.DataFrame
        Long-format lab/record table with patient id, code, date and value columns.
    ref : pd.Series
        MultiIndex-ed (``patient``, ``timestamp``) series of reference values to match against.
    code : str
        The ``code_var`` value in ``d`` to restrict candidate rows to.
    code_var, patient_var, date_var, target_var : str
        Column names in ``d``.

    Returns
    -------
    pd.DataFrame
        One row per ``ref`` entry: ``(patient, timestamp, ref_value, matched_value, days_diff)``.
        ``matched_value``/``days_diff`` are NaN when the patient has no matching record.
    """

    res = []

    for pa, vs in ref.groupby(level=0):

        dq = d.query(f"{patient_var}=={pa} & {code_var}=='{code}'")

        for (pa, ts), org_val in vs.items():

            if len(dq) > 0:

                tdiff = (dq[date_var] - ts).abs()
                idxmin = tdiff.idxmin()

                new_value = dq.loc[idxmin, target_var]
                time_diff = tdiff.loc[idxmin].days

                res.append((pa, ts, org_val, new_value, time_diff))

            else:

                res.append((pa, ts, org_val, np.nan, np.nan))

    return pd.DataFrame(res)
