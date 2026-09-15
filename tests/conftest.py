import pandas as pd
import pytest

from cbrrwd import Medication, Regime


@pytest.fixture
def flot_regime():
    fu = Medication("FU", [1], 14, planned_applications=4)
    ox = Medication("OX", [1], 14, planned_applications=4)
    doc = Medication("DOC", [1], 14, planned_applications=4)
    return Regime("FLOT", fu, ox, doc)


@pytest.fixture
def plf_regime():
    # weekly-ish schedule with an optional medication, mirrors PLF/OLF
    fu = Medication("FU", [1, 8, 15, 22, 29, 36], 49, planned_applications=12)
    cis = Medication("CIS", [1, 15, 29], 49, planned_applications=6)
    ox = Medication("OX", [1, 15, 29], 49, planned_applications=6, required=False)
    return Regime("PLF", fu, cis, ox)


@pytest.fixture
def chemo_row():
    return pd.Series({
        "Erste_Gabe_Datum": "2024-01-01",
        "Letzte_Gabe_Datum": "2024-03-01",
        "Therapieprotokoll_Name": "FLOT",
    })
