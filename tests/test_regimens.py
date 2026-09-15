import numpy as np
import pandas as pd
import pytest

from cbrrwd.regimens import (
    Medication,
    calc_total_days_on_therapy,
    calculate_applications,
    calculate_rdi,
    determine_treatment_days,
    med_info,
    parse_application_string,
    unpack_regime,
    validate_chemo_protocol,
    validate_meds,
)


def test_calc_total_days_on_therapy(chemo_row):
    assert calc_total_days_on_therapy(chemo_row) == 60


def test_calc_total_days_on_therapy_rejects_reversed_dates():
    row = pd.Series({"Erste_Gabe_Datum": "2024-03-01", "Letzte_Gabe_Datum": "2024-01-01"})
    with pytest.raises(ValueError, match="liegt vor erster Gabe"):
        calc_total_days_on_therapy(row)


def test_validate_chemo_protocol(chemo_row, flot_regime):
    assert validate_chemo_protocol(chemo_row, {"FLOT": flot_regime}) == "FLOT"


def test_validate_chemo_protocol_unknown(chemo_row):
    with pytest.raises(ValueError, match="not found"):
        validate_chemo_protocol(chemo_row, {"PLF": object()})


def test_determine_treatment_days_across_cycles():
    days = determine_treatment_days(np.array([1]), 14, 60)
    assert list(days) == [1, 15, 29, 43, 57]


def test_determine_treatment_days_empty_when_shorter_than_first_day():
    days = determine_treatment_days(np.array([8, 15]), 14, 5)
    assert len(days) == 0


def test_calculate_applications_regular():
    n_applications, last_day = calculate_applications([1], 14, 60)
    assert (n_applications, last_day) == (5, 56)


def test_calculate_applications_exact_cycle_multiple():
    # 14 days on therapy with a q14 single-day schedule: the boundary case
    # where the last cycle's application must still be counted.
    n_applications, last_day = calculate_applications([1], 14, 14)
    assert (n_applications, last_day) == (2, 14)


def test_validate_meds_required_and_optional(plf_regime):
    required, optional = plf_regime.get_required_optional()
    assert required == ("FU", "CIS")
    assert optional == ("OX",)

    # OX (optional) present
    assert validate_meds("FU-6x100_CIS-6x100_OX-6x100", required, optional)
    # OX (optional) absent -- still valid
    assert validate_meds("FU-6x100_CIS-6x100", required, optional)
    # required CIS missing -- invalid
    assert not validate_meds("FU-6x100", required, optional)
    # unlisted medication -- invalid
    assert not validate_meds("FU-6x100_CIS-6x100_XYZ-1x100", required, optional)


def test_parse_application_string_flot(flot_regime, chemo_row):
    dot = calc_total_days_on_therapy(chemo_row)
    applied = "FU-4x100-2x80_OX-4x100-2x80_DOC-2x100-2x80"

    res = parse_application_string(applied, dot, flot_regime)

    assert list(res.index) == ["FU", "OX", "DOC"]
    assert res.loc["FU", "applications"] == 6
    assert res.loc["FU", "avg_dose"] == pytest.approx((4 * 100 + 2 * 80) / 6)
    assert res.loc["DOC", "applications"] == 4
    assert (res["time_on_treatment_real"] == dot).all()


def test_parse_application_string_missing_optional_medication_gets_zero(plf_regime):
    res = parse_application_string("FU-6x100_CIS-6x100", time_on_treatment_real=90, regime=plf_regime)
    assert res.loc["OX", "applications"] == 0
    assert res.loc["OX", "avg_dose"] == 0


def test_parse_application_string_rejects_invalid_regime(flot_regime):
    with pytest.raises(ValueError, match="Could not validate"):
        parse_application_string("FU-4x100", time_on_treatment_real=60, regime=flot_regime)


def test_med_info_rejects_unknown_key():
    fu = Medication("FU", [1], 14, planned_applications=4)
    with pytest.raises(ValueError, match="Info key"):
        med_info(fu, "bogus")


def test_unpack_regime(flot_regime):
    assert unpack_regime(flot_regime, "cycle_len") == {"FU": 14, "OX": 14, "DOC": 14}


def test_calculate_rdi_on_plan():
    rdi = calculate_rdi(
        avg_dose=100, applications=4, time_on_treatment_real=43,
        time_on_treatment_asper_applications=43, applications_planned=4,
        avg_dose_planned=100, time_on_treatment_planned=43,
    )
    assert rdi == 100.0


def test_calculate_rdi_reduced_dose():
    rdi = calculate_rdi(
        avg_dose=90, applications=4, time_on_treatment_real=45,
        time_on_treatment_asper_applications=43, applications_planned=4,
        avg_dose_planned=100, time_on_treatment_planned=43,
    )
    assert rdi == 86.0


def test_calculate_rdi_therapy_canceled():
    rdi = calculate_rdi(
        avg_dose=100, applications=2, time_on_treatment_real=15,
        time_on_treatment_asper_applications=15, applications_planned=4,
        avg_dose_planned=100, time_on_treatment_planned=43,
    )
    assert rdi == 50.0


def test_calculate_rdi_longer_than_planned_caps_at_full_intensity():
    rdi = calculate_rdi(
        avg_dose=100, applications=6, time_on_treatment_real=71,
        time_on_treatment_asper_applications=71, applications_planned=4,
        avg_dose_planned=100, time_on_treatment_planned=43,
    )
    assert rdi == 100.0
