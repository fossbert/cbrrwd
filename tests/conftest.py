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
def sequence_like_regime():
    # schematic: Gem/nab-paclitaxel backbone from day 1, FOLFOX (FU+OX) added
    # only from day 29 of a 42-day cycle -- mirrors the SEQUENCE regimen
    gem = Medication("GEM", [1, 8, 15], 42)
    fu = Medication("FU", [29], 42, required=False)
    ox = Medication("OX", [29], 42, required=False)
    return Regime("SEQUENCE", gem, fu, ox)


@pytest.fixture
def chemo_row():
    return pd.Series({
        "Erste_Gabe_Datum": "2024-01-01",
        "Letzte_Gabe_Datum": "2024-03-01",
        "Therapieprotokoll_Name": "FLOT",
    })


@pytest.fixture
def regime_dict(flot_regime, sequence_like_regime):
    return {"FLOT": flot_regime, "SEQUENCE": sequence_like_regime}


@pytest.fixture
def patient_row_valid():
    return pd.Series({
        "Erste_Gabe_Datum": "2024-01-01",
        "Letzte_Gabe_Datum": "2024-02-12",  # 42 days
        "Therapieprotokoll_Name": "FLOT",
        "Applizierte_Medikamente_Detail": "FU-4x100-2x80_OX-4x100-2x80_DOC-2x100-2x80",
    })


@pytest.fixture
def patient_row_bad_dates():
    return pd.Series({
        "Erste_Gabe_Datum": "2024-03-01",
        "Letzte_Gabe_Datum": "2024-01-01",  # before Erste_Gabe -> invalid
        "Therapieprotokoll_Name": "FLOT",
        "Applizierte_Medikamente_Detail": "FU-4x100_OX-4x100_DOC-2x100",
    })


@pytest.fixture
def patient_row_unknown_protocol():
    return pd.Series({
        "Erste_Gabe_Datum": "2024-01-01",
        "Letzte_Gabe_Datum": "2024-02-12",
        "Therapieprotokoll_Name": "UNKNOWN",
        "Applizierte_Medikamente_Detail": "FU-4x100",
    })


@pytest.fixture
def patient_row_invalid_application_string():
    return pd.Series({
        "Erste_Gabe_Datum": "2024-01-01",
        "Letzte_Gabe_Datum": "2024-02-12",
        "Therapieprotokoll_Name": "FLOT",
        "Applizierte_Medikamente_Detail": "FU-4x100",  # missing required OX, DOC
    })
