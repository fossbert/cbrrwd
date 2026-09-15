import numpy as np
import pandas as pd

from cbrrwd.linkage import find_closest


def _ref():
    return pd.Series(
        [10.0, 12.0, 5.0],
        index=pd.MultiIndex.from_tuples([
            (1, pd.Timestamp("2024-01-10")),
            (1, pd.Timestamp("2024-02-01")),
            (2, pd.Timestamp("2024-01-15")),
        ]),
    )


def _labs():
    return pd.DataFrame({
        "PATISAPNR": [1, 1, 1],
        "CODE": ["CA", "CA", "CA"],
        "LABEINDAT": pd.to_datetime(["2024-01-08", "2024-01-20", "2024-03-01"]),
        "ERGEBNISF": [9.5, 11.0, 20.0],
    })


def test_find_closest_matches_nearest_by_date():
    res = find_closest(_labs(), _ref(), code="CA")
    res.columns = ["patient", "ts", "ref_val", "matched_val", "days_diff"]

    p1 = res[res["patient"] == 1].set_index("ts")
    assert p1.loc[pd.Timestamp("2024-01-10"), "matched_val"] == 9.5
    assert p1.loc[pd.Timestamp("2024-01-10"), "days_diff"] == 2
    assert p1.loc[pd.Timestamp("2024-02-01"), "matched_val"] == 11.0


def test_find_closest_patient_without_matching_records_is_nan_not_stale():
    # Regression test: a patient with no rows in `d` must get NaN for their
    # *own* reference entry, not leftover (ts, value) from a previous patient.
    res = find_closest(_labs(), _ref(), code="CA")
    res.columns = ["patient", "ts", "ref_val", "matched_val", "days_diff"]

    p2 = res[res["patient"] == 2]
    assert len(p2) == 1
    assert p2.iloc[0]["ts"] == pd.Timestamp("2024-01-15")
    assert p2.iloc[0]["ref_val"] == 5.0
    assert np.isnan(p2.iloc[0]["matched_val"])
    assert np.isnan(p2.iloc[0]["days_diff"])
