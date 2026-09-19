"""Systemic-therapy regimens: model a protocol, parse what was actually
applied from a free-text record, and derive the real dose intensity (RDI).

A :class:`Regime` is a named bundle of :class:`Medication` objects, each with
its own within-cycle treatment-day pattern and (optionally) a planned number
of applications. :func:`parse_application_string` takes a compact record of
what was actually given -- e.g. ``"FU-4x100-2x80_OX-4x100-2x80_DOC-2x100-2x80"``
-- and returns per-medication counts and average relative dose, alongside the
theoretical (per elapsed days) and planned (per protocol) equivalents needed
to compute :func:`calculate_rdi`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "Medication",
    "Regime",
    "validate_meds",
    "parse_application_string",
    "calculate_applications",
    "applications_to_treatment_days",
    "determine_treatment_days",
    "calc_total_days_on_therapy",
    "validate_chemo_protocol",
    "med_info",
    "unpack_regime",
    "calculate_rdi",
    "calculate_rdi_theoretical",
    "theoretical_applications_table",
    "parse_patient_regimen",
]


@dataclass(frozen=True)
class Medication:
    """One drug within a :class:`Regime`: its schedule and (optionally) target dose count.

    Parameters
    ----------
    name : str
        Medication name/code, matched case-insensitively against application strings.
    treatment_days : list
        Days within one cycle on which the drug is given, e.g. ``[1]`` or ``[1, 8, 15]``.
    cycle_len : int
        Length of one cycle in days.
    planned_applications : int, optional
        Target number of applications per protocol.
    required : bool
        Whether this medication must appear for the applied string to validate
        against the regime (see :func:`validate_meds`).
    """

    name: str
    treatment_days: list
    cycle_len: int
    planned_applications: int = None
    required: bool = True


@dataclass(init=False)
class Regime:
    """A named protocol: an ordered bundle of :class:`Medication` objects."""

    name: str
    meds: tuple[Medication, ...] = field(default_factory=tuple)

    def __init__(self, name: str, *args: Medication):
        self.name = name
        self.meds = args

    def get_required_optional(self):
        required = tuple(m.name for m in self.meds if m.required)
        optional = tuple(m.name for m in self.meds if not m.required)
        return required, optional


def validate_meds(application_string: str, required: Iterable[str], optional: Iterable[str]) -> bool:
    """
    Validate whether applied medications match a protocol with required and optional drugs.

    The function parses an application string of the form:
        "FU-4x100-2x80_OX-4x100-2x80_DOC-2x100-2x80"
    by extracting the medication names (substring before the first `-`)
    and normalizing them to uppercase.

    Validation rules:
        - All required medications must be present with the exact expected frequency.
        - Optional medications may be present or absent.
        - Any medication not listed as required or optional causes validation to fail.

    Notes
    -----
    Matching is case-insensitive. Medication frequency (duplicates) is
    respected. The order of medications in the input string does not matter.
    """

    meds_applied = [s.split('-')[0].upper() for s in application_string.split("_")]

    applied_counter = Counter(meds_applied)
    required_counter = Counter(m.upper() for m in required)
    optional_set = {m.upper() for m in optional}

    for med, count in required_counter.items():
        if applied_counter[med] != count:
            return False

    remaining = applied_counter - required_counter

    if any(m not in optional_set for m in remaining):
        return False

    return True


def parse_application_string(application_string: str, time_on_treatment_real: int, regime: Regime) -> pd.DataFrame:
    """
    Parse a medication application string into a DataFrame while ensuring that
    all medications in the regime (including optional ones) are represented.

    Optional medications not present in the input string are included with
    zero applications.

    If `application_string` is NaN (no record of what was applied), a
    DataFrame with the same shape and columns is returned, but with NaN in
    place of every derived value instead of 0. This keeps it usable by the
    same downstream code (e.g. :func:`calculate_rdi`) while distinguishing
    "no data available" from a genuine zero (e.g. an optional medication that
    was not given).

    ``time_on_treatment_real`` in the result is the regimen-wide real span
    (Erste_Gabe to Letzte_Gabe), shared as-is by every medication in the
    regime -- including one that only enters partway through a cycle (e.g.
    FOLFOX added from day 29 of a 42-day Gem/nab-paclitaxel cycle, as in the
    SEQUENCE regimen) or that stops early while another component continues.
    Neither case is corrected for here: a medication with a staggered start
    or end is charged with the full regimen-wide real time, which
    understates its RDI. Combos with genuinely staggered components are
    better modeled as separate regime lines with their own
    Erste_Gabe/Letzte_Gabe, each parsed with its own
    :func:`parse_application_string` call -- see :func:`parse_patient_regimen`.

    Returns
    -------
    pd.DataFrame
        Indexed by medication name, with columns ``applications``,
        ``avg_dose``, ``time_on_treatment_real`` and
        ``time_on_treatment_asper_applications``.
    """

    if pd.isna(application_string):
        med_list = [
            {
                "medication": med_obj.name,
                "applications": np.nan,
                "avg_dose": np.nan,
                "time_on_treatment_real": time_on_treatment_real,
                "time_on_treatment_asper_applications": np.nan,
            }
            for med_obj in regime.meds
        ]
        return pd.DataFrame(med_list).set_index("medication")

    required, optional = regime.get_required_optional()

    if not validate_meds(application_string, required, optional):
        raise ValueError(
            f"Could not validate all medications that are part of {getattr(regime, 'name')}"
        )

    treatment_days_dict = unpack_regime(regime, "treatment_days")
    cycle_len_days_dict = unpack_regime(regime, "cycle_len")

    parsed = {}

    for medinfo in application_string.split("_"):
        med, *doses = medinfo.upper().split("-")

        n_applications = 0
        doses_percent = 0

        for dose in doses:
            n_times, dose_percent = dose.split("X")
            n_times = int(n_times)

            n_applications += n_times
            doses_percent += int(dose_percent) * n_times

        avg = doses_percent / n_applications if n_applications else 0

        parsed[med] = (n_applications, avg)

    med_list = []

    for med_obj in regime.meds:  # preserves original regime order

        med = med_obj.name

        n_applications, avg_dose = parsed.get(med, (0, 0))

        med_treatment_days = treatment_days_dict.get(med)

        time_on_treatment_asper_applications = applications_to_treatment_days(
            med_treatment_days,
            cycle_len_days_dict.get(med),
            n_applications
        )

        med_list.append({
            "medication": med,
            "applications": n_applications,
            "avg_dose": avg_dose,
            "time_on_treatment_real": time_on_treatment_real,
            "time_on_treatment_asper_applications": time_on_treatment_asper_applications,
        })

    return pd.DataFrame(med_list).set_index("medication")


def calculate_applications(treatment_days: list, cycle_len_days: int, total_days_on_therapy: int):
    """
    Calculate the number of treatment applications and the final treatment day.

    This helper converts a per-cycle treatment schedule into actual application
    counts over the observed total days on therapy (see
    :func:`determine_treatment_days` for the elapsed-day/day-number convention
    this relies on).

    Parameters
    ----------
    treatment_days : list
        Scheduled treatment days within a cycle, e.g. [1] or [1, 8].
    cycle_len_days : int
        Length of one treatment cycle in days, e.g. 14.
    total_days_on_therapy : int
        Total observed days the patient remained on therapy.

    Returns
    -------
    tuple
        ``(applications_count, last_treatment_day)``. ``last_treatment_day``
        is ``None`` if ``total_days_on_therapy`` falls before this
        medication's own first possible treatment day (0 applications) --
        e.g. therapy stopped before a medication added only later in the
        cycle (day 29 of a 42-day cycle, say) was ever reached.
    """

    treatment_days = np.array(treatment_days)

    treatment_days_array_pruned = determine_treatment_days(treatment_days, cycle_len_days, total_days_on_therapy)

    if len(treatment_days_array_pruned) == 0:
        return 0, None

    # treatment_days_array_pruned holds 1-indexed day-in-therapy numbers (see
    # determine_treatment_days); convert to the elapsed-days scale used
    # everywhere else (day 1 -> elapsed 0).
    last_treatment_day = treatment_days_array_pruned[-1] - 1

    return len(treatment_days_array_pruned), last_treatment_day


def applications_to_treatment_days(treatment_days: list[int], cycle_len_days: int, n_applications: int) -> int:
    """
    Compute the calendar day of the n-th treatment administration.

    This helper converts a treatment schedule defined within a single cycle into
    the absolute day index for the requested application number. It assumes that
    treatment repeats every cycle_len_days.

    Returns an *elapsed*-days value (0-indexed: the first possible treatment
    day is 0 elapsed days), the same convention :func:`calc_total_days_on_therapy`
    and :func:`determine_treatment_days` use -- the inverse of this function.
    ``n_applications=1`` on a schedule starting on day 1 therefore returns 0,
    not 1.

    Parameters
    ----------
    treatment_days : list[int]
        Days within a single cycle when treatment is given.
    cycle_len_days : int
        Duration of one treatment cycle in days.
    n_applications : int
        1-based index of the application to locate.

    Returns
    -------
    int
        The day number corresponding to the n-th treatment application.
    """

    n_per_cycle = len(treatment_days)
    cycle_idx = (n_applications - 1) // n_per_cycle
    day_idx = (n_applications - 1) % n_per_cycle

    number_of_days = treatment_days[day_idx] + cycle_idx * cycle_len_days - 1

    if number_of_days < 0:  # in case there are 0 applications
        number_of_days = 0

    return number_of_days


def determine_treatment_days(treatment_days: np.ndarray, cycle_len_days: int, total_days_on_therapy: int) -> np.ndarray:
    """
    Return every treatment day across repeated cycles up to total_days_on_therapy (inclusive).

    ``treatment_days`` (and the values this function returns) are 1-indexed
    calendar-day-in-therapy numbers -- day 1 is the first possible treatment
    day. ``total_days_on_therapy``, by contrast, is *elapsed* days since the
    first dose (0-indexed: the first dose day itself is 0 elapsed days,
    matching :func:`calc_total_days_on_therapy` and the day/elapsed
    convention :func:`applications_to_treatment_days` uses in the opposite
    direction). Calendar day ``d`` is reached once elapsed time has caught up
    to it, i.e. once ``total_days_on_therapy >= d - 1`` -- equivalently,
    ``d <= total_days_on_therapy + 1``, which is what the comparisons below
    use throughout.

    ``treatment_days`` is the pattern within one cycle (e.g. [1, 8, 15] for a
    weekly schedule on a 21-day cycle). The pattern is shifted by
    ``k * cycle_len_days`` for k = 0, 1, 2, ... and concatenated, then trimmed
    at ``total_days_on_therapy`` (inclusive, on the elapsed-day scale above).
    """
    total_days = total_days_on_therapy + 1  # elapsed days -> 1-indexed day-in-therapy scale

    if total_days < treatment_days.min():
        return treatment_days[:0]  # empty, preserves dtype

    # How many cycle shifts can still produce a day <= total_days?
    # Smallest day in cycle k is treatment_days.min() + k * cycle_len_days,
    # so k_max = floor((total_days - min) / cycle_len_days).
    k_max = (total_days - treatment_days.min()) // cycle_len_days
    offsets = np.arange(k_max + 1) * cycle_len_days

    all_days = (treatment_days + offsets[:, None]).ravel()

    return all_days[all_days <= total_days]


def calc_total_days_on_therapy(row: pd.Series, last_date: str = "Letzte_Gabe_Datum", first_date: str = "Erste_Gabe_Datum"):
    """Days between first and last recorded application (raises if last precedes first)."""

    last_date = pd.to_datetime(getattr(row, last_date))
    first_date = pd.to_datetime(getattr(row, first_date))

    days_on_treatment = (last_date - first_date).days

    if days_on_treatment < 0:
        raise ValueError(f"Letzte Gabe {last_date} liegt vor erster Gabe {first_date}")

    return days_on_treatment


def validate_chemo_protocol(row: pd.Series, regime_dict: dict, attribute_key: str = "Therapieprotokoll_Name"):
    """Look up ``row``'s protocol name in ``regime_dict``, raising if unknown."""

    protocol_name = getattr(row, attribute_key)

    if protocol_name not in regime_dict:
        raise ValueError(f"Protocol name {protocol_name} not found in protocol info dictionary")

    return protocol_name


def med_info(medication: Medication, info_key: str):
    """Return one field (``name``/``treatment_days``/``cycle_len``/``planned_applications``) of ``medication``."""

    if info_key not in ["name", "treatment_days", "cycle_len", "planned_applications"]:
        raise ValueError(
            f"Info key {info_key} not valid. Must be one of 'medication', 'treatment_days', "
            "'cycle_len' or 'planned applications'"
        )

    return getattr(medication, info_key)


def unpack_regime(regime: Regime, info_key: str):
    """Return ``{medication_name: info_key value}`` for every medication in ``regime``."""

    return {getattr(med, "name"): med_info(med, info_key) for med in regime.meds}


def calculate_rdi(avg_dose: float,
                   applications: int,
                   time_on_treatment_real: int,
                   time_on_treatment_asper_applications: int,
                   applications_planned: int,
                   avg_dose_planned: float,
                   time_on_treatment_planned: int) -> float:
    
    """Berechnet die reale Dosisintensität (RDI) unter Berücksichtigung von Abbruch, Verzögerung
    und dynamischer Protokoll-Erweiterung bei ungewöhnlich langer Therapiedauer.
    """
    # 1. Fallunterscheidung: Liegt eine ungewöhnlich lange Therapie vor oder wurde abgebrochen?
    longer_than_anticipated = applications > applications_planned
    therapy_canceled = applications < applications_planned

    # 2. Dynamische Anpassung der PLAN-WERTE bei Übertherapie: Wir berechnen, wie
    # lange der Plan für diese höhere Anzahl an Applikationen laut medizinischem
    # Protokoll (ganzzahlig!) gelaufen wäre -- das sind die neuen, skalierten
    # Plan-Erwartungswerte.
    if longer_than_anticipated:
        effective_applications_planned = applications
        effective_time_planned = time_on_treatment_asper_applications
    else:
        # Standardfall: Regulär oder abgebrochen (Plan bleibt wie initial definiert).
        effective_applications_planned = applications_planned
        effective_time_planned = time_on_treatment_planned

    # 3. Geplante Dosisintensität (DI_planned wandert mit, wenn länger als geplante Therapie vorliegt)
    dose_intensity_planned = (avg_dose_planned * effective_applications_planned) / effective_time_planned

    # 4. Zeitkorrektur-Logik für den realen Nenner
    if therapy_canceled:
        relevant_time = time_on_treatment_planned
    else:
        # Bei regulärer Zielerreichung ODER Übertherapie gilt: Hat der Patient
        # länger gebraucht als der (skalierte) Plan vorsieht?
        relevant_time = max(time_on_treatment_real, effective_time_planned)

    # 5. Tatsächliche Dosisintensität und finale RDI
    dose_intensity_real = (avg_dose * applications) / relevant_time

    return round((dose_intensity_real / dose_intensity_planned) * 100, 2)


def calculate_rdi_theoretical(avg_dose: float,
                               applications: int,
                               avg_dose_theoretical: float,
                               applications_theoretical: int) -> float:

    """RDI-Nullmodell fuer Therapien ohne fixe Zyklenzahl (z.B. palliative Regime).

    Anders als :func:`calculate_rdi` gibt es hier kein extern definiertes
    Plan-Ende (keine geplante Zyklenzahl), gegen das ab- oder ueberschritten
    werden koennte. Stattdessen wird die real verabreichte kumulative Dosis
    (``avg_dose * applications``) gegen die kumulative Dosis verglichen, die
    bei Volldosis und protokollgerechter Taktung in derselben real
    verstrichenen Zeit theoretisch moeglich gewesen waere
    (``avg_dose_theoretical * applications_theoretical``, siehe
    :func:`theoretical_applications_table`). Die RDI haengt damit rein am
    Verhaeltnis erhaltener zu theoretisch moeglichen Gaben -- anders als eine
    reine Zeitverhaeltnis-Rechnung wird kein zusaetzliches "Tempo"-Defizit
    unterstellt, wenn lediglich einzelne Gaben innerhalb eines ansonsten
    unveraenderten Zyklus ausgelassen wurden (z.B. Tag 8 bei Gem/nab-Paclitaxel
    q28[1,8,15]: 6 statt 9 Gaben bei gleichem Start-/Enddatum ergeben 6/9,
    nicht weniger).

    Ein echter zeitlicher Verzug bleibt trotzdem erfasst: je laenger die reale
    Therapiedauer bei gleicher Gabenzahl, desto groesser
    ``applications_theoretical`` (mehr Gaben waeren in der Zeit moeglich
    gewesen) und desto niedriger die resultierende RDI.

    Bekannte Grenze: Fuer Mehrtages-Schemata kann
    :func:`calculate_applications`/:func:`determine_treatment_days` an der
    Zyklusgrenze ein ``applications_theoretical`` liefern, das um 1 zu niedrig
    ist -- bei voller Compliance kann die RDI dadurch knapp ueber 100%
    liegen. Bewusst nicht korrigiert (siehe Tests), da Kombinationen mit
    Medikamenten, die erst im Verlauf des Zyklus einsteigen bzw. frueher
    enden (z.B. SEQUENCE), ohnehin separat pro Regime-Zeile geparst werden --
    siehe :func:`parse_patient_regimen`.
    """

    # 1. Ohne reale oder theoretische Applikationen ist keine Dosisintensitaet
    # definiert -- und applications_theoretical waere 0, was Punkt 2 als
    # Nenner nicht vertraegt.
    if applications == 0 or applications_theoretical == 0:
        return 0.0

    # 2. Reale vs. theoretisch moegliche kumulative Dosis in derselben Zeit.
    dose_real = avg_dose * applications
    dose_theoretical = avg_dose_theoretical * applications_theoretical

    return round((dose_real / dose_theoretical) * 100, 2)


def theoretical_applications_table(regime: Regime, total_days_on_therapy: int) -> pd.DataFrame:
    """
    Per medication: the applications theoretically achievable at full dose
    and on-protocol timing, given ``total_days_on_therapy``.

    A thin wrapper around :func:`calculate_applications` over every
    medication in ``regime``, returned as a DataFrame indexed by medication
    name (mirrors :func:`parse_application_string`'s indexing) so it can be
    joined onto its output, e.g. as the theoretical/null-model side of a
    real-vs-theoretical comparison.

    Returns
    -------
    pd.DataFrame
        Columns ``applications_theoretical``, ``avg_dose_theoretical``
        (always 100 -- full dose is the definition of "theoretical"), and
        ``last_treatment_day_theoretical`` (``NaN`` where 0 applications
        were theoretically possible, e.g. therapy stopped before a
        medication added only later in the cycle was ever reached --
        :func:`calculate_applications` returns ``None`` for this case, which
        pandas coerces to ``NaN`` once the column is built).
    """

    rows = []

    for med in regime.meds:
        n_theoretical, last_day = calculate_applications(
            med.treatment_days, med.cycle_len, total_days_on_therapy
        )
        rows.append({
            "medication": med.name,
            "applications_theoretical": n_theoretical,
            "avg_dose_theoretical": 100,
            "last_treatment_day_theoretical": last_day,
        })

    return pd.DataFrame(rows).set_index("medication")


def parse_patient_regimen(row: pd.Series, regime_dict: dict) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """
    Build the per-medication real-vs-theoretical table for one patient row.

    Composes :func:`calc_total_days_on_therapy`, :func:`validate_chemo_protocol`,
    :func:`parse_application_string` and :func:`theoretical_applications_table`
    into the single call site meant to be used inside a per-patient loop.

    Returns
    -------
    tuple
        ``(result, error)`` -- exactly one of the two is not ``None``. On any
        of the three known failure modes (invalid Erste/Letzte_Gabe dates,
        an unknown ``Therapieprotokoll_Name``, or an application string that
        does not validate against the resolved regime) this returns
        immediately with ``row`` as the error, rather than falling through to
        later steps that would otherwise run with a stale value left over
        from a previous call.
    """

    try:
        dot = calc_total_days_on_therapy(row)
    except ValueError:
        return None, row

    try:
        prtcol = validate_chemo_protocol(row, regime_dict)
    except ValueError:
        return None, row

    regime = regime_dict[prtcol]
    applied = getattr(row, "Applizierte_Medikamente_Detail")

    try:
        res_applied = parse_application_string(applied, dot, regime)
    except ValueError:
        return None, row

    res_theoretical = theoretical_applications_table(regime, dot)

    return res_applied.join(res_theoretical), None
