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
    "ChemoDetail",
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


@dataclass
class ChemoDetail:
    """Per-medication result row: applications, average dose, time on treatment."""

    medication: str = None
    applications: int = None
    avg_dose: int = None
    time_on_treatment_real: int = None
    time_on_treatment_asper_applications: int = None


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

    Returns
    -------
    pd.DataFrame
        Indexed by medication name.
    """

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

        time_on_treatment_asper_applications = applications_to_treatment_days(
            treatment_days_dict.get(med),
            cycle_len_days_dict.get(med),
            n_applications
        )

        med_list.append(
            ChemoDetail(
                med,
                n_applications,
                avg_dose,
                time_on_treatment_real,
                time_on_treatment_asper_applications
            )
        )

    return pd.DataFrame(med_list).set_index("medication")


def calculate_applications(treatment_days: list, cycle_len_days: int, total_days_on_therapy: int):
    """
    Calculate the number of treatment applications and the final treatment day.

    This helper converts a per-cycle treatment schedule into actual application
    counts over the observed total days on therapy. It adjusts single-day
    treatment protocols when the total therapy duration is an exact multiple of
    the cycle length so that the last cycle application is still counted.

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
        ``(applications_count, last_treatment_day)``.
    """

    treatment_days = np.array(treatment_days)

    if len(treatment_days) == 1 and (total_days_on_therapy % cycle_len_days == 0):

        # For protocols that repeat every 14 days, for example, we need to add
        # 1 to count the last application.
        total_days_on_therapy += 1

    treatment_days_array_pruned = determine_treatment_days(treatment_days, cycle_len_days, total_days_on_therapy)

    last_treatment_day = treatment_days_array_pruned[-1]

    if last_treatment_day > 1:
        last_treatment_day -= 1

    return len(treatment_days_array_pruned), last_treatment_day


def applications_to_treatment_days(treatment_days: list[int], cycle_len_days: int, n_applications: int) -> int:
    """
    Compute the calendar day of the n-th treatment administration.

    This helper converts a treatment schedule defined within a single cycle into
    the absolute day index for the requested application number. It assumes that
    treatment repeats every cycle_len_days.

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
    elif number_of_days == 0:  # in case there is 1 application, the function ends up at 0, correct this case
        number_of_days = 1

    return number_of_days


def determine_treatment_days(treatment_days: np.ndarray, cycle_len_days: int, total_days_on_therapy: int) -> np.ndarray:
    """
    Return every treatment day across repeated cycles up to total_days_on_therapy (inclusive).

    ``treatment_days`` is the pattern within one cycle (e.g. [1, 8, 15] for a
    weekly schedule on a 21-day cycle). The pattern is shifted by
    ``k * cycle_len_days`` for k = 0, 1, 2, ... and concatenated, then trimmed
    at ``total_days_on_therapy``.
    """
    if total_days_on_therapy < treatment_days.min():
        return treatment_days[:0]  # empty, preserves dtype

    # How many cycle shifts can still produce a day <= total_days_on_therapy?
    # Smallest day in cycle k is treatment_days.min() + k * cycle_len_days,
    # so k_max = floor((total - min) / cycle_len_days).
    k_max = (total_days_on_therapy - treatment_days.min()) // cycle_len_days
    offsets = np.arange(k_max + 1) * cycle_len_days

    all_days = (treatment_days + offsets[:, None]).ravel()

    return all_days[all_days <= total_days_on_therapy]


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
