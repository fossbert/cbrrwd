import numpy as np
import pandas as pd
import pytest

from cbrrwd.regimens import (
    Medication,
    applications_to_treatment_days,
    calc_total_days_on_therapy,
    calculate_applications,
    calculate_rdi,
    calculate_rdi_theoretical,
    determine_treatment_days,
    med_info,
    parse_application_string,
    parse_patient_regimen,
    theoretical_applications_table,
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


def test_calculate_applications_zero_before_late_medication_start():
    # FOLFOX added from day 29 of a 42-day cycle: therapy stopped at day 20,
    # well before FOLFOX was ever reached -- used to raise IndexError
    assert calculate_applications([29], 42, 20) == (0, None)
    # elapsed=28 means today IS calendar day 29 (day 1 of therapy == elapsed
    # 0, see determine_treatment_days) -> FOLFOX's own start day has been
    # reached, 1 application
    assert calculate_applications([29], 42, 28) == (1, 28)
    assert calculate_applications([29], 42, 29) == (1, 28)


def test_calculate_applications_zero_applications_multi_day_schedule():
    # Erste_Gabe == Letzte_Gabe (0 days elapsed) means we're on day 1 of
    # therapy -- if day 1 is itself a treatment day, that first application
    # is credited; a schedule that doesn't start on day 1 still gets 0 and
    # must not crash (this used to raise IndexError)
    assert calculate_applications([1, 8, 15], 21, 0) == (1, 0)
    assert calculate_applications([8, 15, 22], 28, 0) == (0, None)


def test_calculate_applications_zero_applications_single_day_not_day_one():
    # even the existing single-day special case only rescues day-1 schedules
    # -- used to raise IndexError
    assert calculate_applications([15], 30, 0) == (0, None)


def test_calculate_applications_full_compliance_multi_day_schedule():
    # regression test for a real off-by-one: a fully compliant patient on
    # [1, 8, 15]/q28 (3 cycles = 9 applications) has calc_total_days_on_therapy
    # == 70 (Erste_Gabe to Letzte_Gabe, day 1 == elapsed 0); this must count
    # all 9 applications, not 8 -- see determine_treatment_days
    assert calculate_applications([1, 8, 15], 28, 70) == (9, 70)


@pytest.mark.parametrize("treatment_days, cycle_len_days", [
    ([1], 14),                      # single-day, e.g. FLOT's FU/OX/DOC
    ([1, 8], 21),                   # Gem/Cis
    ([1, 8, 15], 28),               # Gem/nab-Paclitaxel (GNP)
    ([1, 8, 15, 22, 29, 36], 49),   # PLF -- FU
    ([1, 15, 29], 49),              # PLF -- CIS
    ([29], 42),                     # medication added later in the cycle
], ids=["FLOT", "GEMCIS", "GNP", "PLF-FU", "PLF-CIS", "late-start"])
def test_applications_to_treatment_days_roundtrips_with_calculate_applications(treatment_days, cycle_len_days):
    # applications_to_treatment_days (n -> elapsed day of the n-th dose) and
    # calculate_applications (elapsed day -> applications so far) must be
    # exact inverses of each other across regimes -- both walk the same
    # elapsed-days/day-in-therapy convention (see determine_treatment_days).
    # A mismatch here means a patient's real applications_theoretical count
    # would be wrong at exactly the boundary that matters most: full or
    # near-full compliance.
    for n in range(1, 30):
        elapsed = applications_to_treatment_days(treatment_days, cycle_len_days, n)

        applications_at, last_day_at = calculate_applications(treatment_days, cycle_len_days, elapsed)
        assert (applications_at, last_day_at) == (n, elapsed)

        if elapsed > 0:
            applications_before, _ = calculate_applications(treatment_days, cycle_len_days, elapsed - 1)
            assert applications_before < n


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


def test_parse_application_string_nan_applied(flot_regime):
    res = parse_application_string(np.nan, time_on_treatment_real=60, regime=flot_regime)

    assert list(res.index) == ["FU", "OX", "DOC"]
    assert res["applications"].isna().all()
    assert res["avg_dose"].isna().all()
    assert res["time_on_treatment_asper_applications"].isna().all()
    assert (res["time_on_treatment_real"] == 60).all()


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


def test_calculate_rdi_theoretical_full_compliance():
    # every application that was theoretically possible in the real elapsed
    # time was actually given, at full dose -> RDI == 100
    rdi = calculate_rdi_theoretical(
        avg_dose=100, applications=4,
        avg_dose_theoretical=100, applications_theoretical=4,
    )
    assert rdi == 100.0


def test_calculate_rdi_theoretical_fewer_applications_than_theoretical():
    # only half of what was theoretically achievable in the real elapsed
    # time was given -- whether that's due to genuine delay or to
    # structurally dropped within-cycle days doesn't matter here, both
    # collapse to the same applications ratio
    rdi = calculate_rdi_theoretical(
        avg_dose=100, applications=4,
        avg_dose_theoretical=100, applications_theoretical=8,
    )
    assert rdi == 50.0


def test_calculate_rdi_theoretical_reduced_dose():
    rdi = calculate_rdi_theoretical(
        avg_dose=90, applications=4,
        avg_dose_theoretical=100, applications_theoretical=4,
    )
    assert rdi == 90.0


def test_calculate_rdi_theoretical_single_application():
    rdi = calculate_rdi_theoretical(
        avg_dose=100, applications=1,
        avg_dose_theoretical=100, applications_theoretical=1,
    )
    assert rdi == 100.0


def test_calculate_rdi_theoretical_no_applications():
    rdi = calculate_rdi_theoretical(
        avg_dose=0, applications=0,
        avg_dose_theoretical=100, applications_theoretical=9,
    )
    assert rdi == 0.0


def test_calculate_rdi_theoretical_no_theoretical_applications_possible():
    # elapsed time too short for even one theoretical application -- must
    # not raise ZeroDivisionError
    rdi = calculate_rdi_theoretical(
        avg_dose=100, applications=1,
        avg_dose_theoretical=100, applications_theoretical=0,
    )
    assert rdi == 0.0


def test_theoretical_applications_table(flot_regime):
    res = theoretical_applications_table(flot_regime, 60)

    assert list(res.index) == ["FU", "OX", "DOC"]
    assert (res["applications_theoretical"] == 5).all()
    assert (res["avg_dose_theoretical"] == 100).all()
    assert (res["last_treatment_day_theoretical"] == 56).all()


def test_theoretical_applications_table_zero_before_late_start(sequence_like_regime):
    # therapy stopped at day 20, well before FOLFOX (day 29) could be reached
    res = theoretical_applications_table(sequence_like_regime, 20)

    assert res.loc["GEM", "applications_theoretical"] == 3
    assert res.loc["FU", "applications_theoretical"] == 0
    # None becomes NaN once pandas builds the column (mixed with int last-day
    # values for other rows)
    assert pd.isna(res.loc["FU", "last_treatment_day_theoretical"])


def test_parse_patient_regimen_success(patient_row_valid, regime_dict):
    result, error = parse_patient_regimen(patient_row_valid, regime_dict)

    assert error is None
    assert list(result.index) == ["FU", "OX", "DOC"]
    assert "applications" in result.columns
    assert "applications_theoretical" in result.columns
    assert result.loc["FU", "applications"] == 6


def test_parse_patient_regimen_flags_applications_exceeding_theoretical(patient_row_valid, regime_dict):
    # patient_row_valid records 6 FU/OX applications in 42 days on a q14
    # schedule, where only 4 fit protocol timing (see calc above) -- likely
    # a data error (wrong dates/cycle count), so this must be flagged
    result, error = parse_patient_regimen(patient_row_valid, regime_dict)

    assert error is None
    assert result.loc["FU", "applications_exceed_theoretical"]
    assert result.loc["OX", "applications_exceed_theoretical"]
    # DOC matches its theoretical count exactly -- nothing to flag
    assert not result.loc["DOC", "applications_exceed_theoretical"]


def test_parse_patient_regimen_bad_dates_returns_row_as_error(patient_row_bad_dates, regime_dict):
    result, error = parse_patient_regimen(patient_row_bad_dates, regime_dict)

    assert result is None
    assert error is patient_row_bad_dates


def test_parse_patient_regimen_unknown_protocol_returns_row_as_error(patient_row_unknown_protocol, regime_dict):
    result, error = parse_patient_regimen(patient_row_unknown_protocol, regime_dict)

    assert result is None
    assert error is patient_row_unknown_protocol


def test_parse_patient_regimen_invalid_application_string_returns_row_as_error(
    patient_row_invalid_application_string, regime_dict
):
    result, error = parse_patient_regimen(patient_row_invalid_application_string, regime_dict)

    assert result is None
    assert error is patient_row_invalid_application_string
