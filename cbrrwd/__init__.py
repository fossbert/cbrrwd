"""cbrrwd -- real-world-data analysis helpers, extracted from a working analysis script.

Focus is systemic (chemo/immuno-)therapy: model a protocol as a
:class:`~cbrrwd.regimens.Regime` of :class:`~cbrrwd.regimens.Medication`
objects, parse what was actually applied from a free-text record, and derive
the real dose intensity (RDI) against the plan. A handful of general
real-world-data helpers (patient/date linkage, significance codes, Fisher's
exact test) round this out.

Submodules: :mod:`cbrrwd.regimens`, :mod:`cbrrwd.linkage`, :mod:`cbrrwd.pvalues`,
:mod:`cbrrwd.plotting` (needs the ``plotting`` extra), and :mod:`cbrrwd.rbackend`
(needs the ``r`` extra).
"""

from __future__ import annotations

from .linkage import find_closest
from .pvalues import cut_p, fdr, fisher_test
from .regimens import (
    Medication,
    Regime,
    SuspectApplicationCountWarning,
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # regimens
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
    "SuspectApplicationCountWarning",
    # linkage
    "find_closest",
    # pvalues
    "cut_p",
    "fdr",
    "fisher_test",
]
