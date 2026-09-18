# cbrrwd

Real-world-data analysis helpers for systemic (chemo/immuno-)therapy: model a
protocol, parse what a patient actually received from a free-text application
record, and derive the real dose intensity (RDI) against it -- for both a
fixed-cycle-count protocol (e.g. neoadjuvant FLOT) and an open-ended one
(e.g. palliative therapy continued until progression or toxicity). A
handful of general real-world-data helpers (patient/date linkage,
significance codes, Fisher's exact test) round this out.

## Install

```bash
pip install -e .                 # core: numpy, pandas, scipy, statsmodels
pip install -e '.[r]'            # fisher_test on r x c tables, gdrate (needs R + tumgr)
pip install -e '.[all,test]'
```

## Layout

| Module | Contents |
| --- | --- |
| `cbrrwd.regimens` | `Medication`, `Regime`, `validate_meds`, `parse_application_string`, `parse_patient_regimen`, `calculate_applications`, `theoretical_applications_table`, `applications_to_treatment_days`, `real_time_since_medication_start`, `determine_treatment_days`, `calc_total_days_on_therapy`, `validate_chemo_protocol`, `med_info`, `unpack_regime`, `calculate_rdi`, `calculate_rdi_theoretical` |
| `cbrrwd.linkage` | `find_closest` (nearest record by date, per patient) |
| `cbrrwd.pvalues` | `cut_p`, `fdr`, `fisher_test` |
| `cbrrwd.rbackend.contingency` | `fisher_exact_rc` (r x c fallback for `fisher_test`) |
| `cbrrwd.rbackend.tumor_growth` | `gdrate` (Wilkerson et al. 2017 tumor growth/decay model, needs CRAN package `tumgr`) |

The most-used helpers are re-exported at the top level (`import cbrrwd as rwd`).
The rest of this README walks through the dose-intensity workflow, which is
the core of the package.

## Modeling a protocol: `Regime` and `Medication`

A `Regime` is a named, ordered bundle of `Medication` objects. Each
`Medication` carries its own within-cycle schedule:

```python
import cbrrwd as rwd

FU  = rwd.Medication("FU",  treatment_days=[1], cycle_len=14, planned_applications=4)
OX  = rwd.Medication("OX",  treatment_days=[1], cycle_len=14, planned_applications=4)
DOC = rwd.Medication("DOC", treatment_days=[1], cycle_len=14, planned_applications=4)
flot = rwd.Regime("FLOT", FU, OX, DOC)
```

- `treatment_days` -- days *within one cycle* the drug is given, e.g. `[1]`
  or `[1, 8, 15]`. It does not have to start on day 1: a drug added only
  later in a combination (e.g. FOLFOX from day 29 of a 42-day cycle, see
  below) is `treatment_days=[29]`.
- `cycle_len` -- length of one cycle in days.
- `planned_applications` -- target number of applications per protocol, if
  there is one. Leave it `None` for a drug given until progression/toxicity
  rather than for a fixed number of cycles (see `calculate_rdi_theoretical`
  below).
- `required` -- whether the drug must appear in an application string for it
  to validate against this regime (default `True`; set `False` for an
  optional component).

## Parsing what was actually given: `parse_application_string`

The real-world record of what a patient received is a compact string, one
segment per medication: `"FU-4x100-2x80_OX-4x100-2x80_DOC-2x100-2x80"` means
FU was given 4 times at 100% dose, then 2 times at 80%; similarly for OX and
DOC.

```python
dot = rwd.calc_total_days_on_therapy(row)  # from Erste_Gabe_Datum / Letzte_Gabe_Datum
protocol = rwd.validate_chemo_protocol(row, regime_dict)
regime = regime_dict[protocol]

applied = rwd.parse_application_string(row.Applizierte_Medikamente_Detail, dot, regime)
```

`applied` is indexed by medication name, with one row each:

```
            applications  avg_dose  time_on_treatment_real  time_on_treatment_real_since_start  time_on_treatment_asper_applications
medication
FU                     4     100.0                       42                                  42                                     42
OX                     4     100.0                       42                                  42                                     42
DOC                    4     100.0                       42                                  42                                     42
```

- `applications` / `avg_dose` -- how many doses were given, and their average
  relative dose (%).
- `time_on_treatment_real` -- the regimen-wide real span (Erste_Gabe to
  Letzte_Gabe), the same for every medication in the row.
- `time_on_treatment_real_since_start` -- that span re-anchored to *this*
  medication's own first possible treatment day (see "Medications that start
  mid-cycle" below). Identical to `time_on_treatment_real` for a medication
  starting on day 1 of the cycle, which is the common case.
- `time_on_treatment_asper_applications` -- the calendar day the observed
  number of applications implies under protocol timing (`applications_to_treatment_days`).

An optional medication (`required=False`) absent from the string gets `0`
applications, not a missing row. If `application_string` itself is missing
(`NaN`), every derived column comes back `NaN` rather than `0` -- "no
record" and "given zero times" are kept distinguishable.

## Dose intensity, fixed cycle count: `calculate_rdi`

Use this when the protocol specifies a number of cycles up front -- e.g.
neoadjuvant FLOT for gastric cancer, always 4 cycles. `calculate_rdi`
compares the real dose intensity actually achieved against the planned one,
handling both early discontinuation and courses that ran *longer* than
planned (the plan is scaled to the observed application count before
comparing, so a clinician-extended course isn't penalized for simply having
more applications).

```python
appl_planned = rwd.applications_to_treatment_days(
    FU.treatment_days, FU.cycle_len, FU.planned_applications
)  # protocol day of the 4th (last planned) FU application -> 42

rdi = rwd.calculate_rdi(
    avg_dose=applied.loc["FU", "avg_dose"],
    applications=applied.loc["FU", "applications"],
    time_on_treatment_real=applied.loc["FU", "time_on_treatment_real"],
    time_on_treatment_asper_applications=applied.loc["FU", "time_on_treatment_asper_applications"],
    applications_planned=FU.planned_applications,
    avg_dose_planned=100,
    time_on_treatment_planned=appl_planned,
)
```

Four scenarios against the same 4-cycle FLOT plan (`applications_planned=4`,
`time_on_treatment_planned=42`):

| Scenario | applications | avg_dose | real time | RDI (FU) |
| --- | --- | --- | --- | --- |
| Exactly on plan | 4 | 100 | 42 | **100.0** |
| Dose reduced to 90% | 4 | 90 | 42 | **90.0** |
| Stopped after 2 of 4 cycles | 2 | 100 | 14 | **50.0** |
| 6 cycles instead of 4, on time | 6 | 100 | 70 | **100.0** (capped) |

## Dose intensity, open-ended therapy: `calculate_rdi_theoretical`

Palliative therapy is different: there is no pre-specified number of cycles
to compare against -- treatment simply continues until it stops working or
becomes intolerable. `calculate_rdi_theoretical` uses a null model instead of
a plan: it compares the real dose intensity against the dose intensity that
the *same number of applications* would have achieved at full dose (100%) and
strictly on-protocol timing. `applications` and
`time_on_treatment_asper_applications` take over the role `calculate_rdi`'s
plan arguments play, scaled to what was actually given rather than to a
fixed protocol.

```python
GEM = rwd.Medication("GEM", treatment_days=[1, 8, 15], cycle_len=28)
gem_regime = rwd.Regime("GEM", GEM)

applied = rwd.parse_application_string("GEM-9x100", time_on_treatment_real=70, regime=gem_regime)
row = applied.loc["GEM"]

rdi = rwd.calculate_rdi_theoretical(
    row.avg_dose, row.applications,
    row.time_on_treatment_real_since_start, row.time_on_treatment_asper_applications,
)
```

Four scenarios for weekly-x3/q28 gemcitabine (day 1, 8, 15 of a 28-day cycle,
i.e. 9 applications per protocol in 70 days):

| Scenario | applications | avg_dose | real time | RDI |
| --- | --- | --- | --- | --- |
| 9 applications, exactly on the day1/8/15 q28 schedule | 9 | 100 | 70d | **100.0** |
| Same 9 applications, but given every 2 weeks instead | 9 | 100 | 112d | **62.5** |
| Stopped after 2 applications, both on time | 2 | 100 | 7d | **100.0** |
| Dose reduced to 80%, applications spread over double the protocol time | 4 | 80 | 56d | **40.0** |

Note on the "every 2 weeks" row: a back-of-envelope rate comparison
(1 dose / 14 days vs. 3 doses / 28 days) suggests a 33% reduction (66.7%),
not the 37.5% (62.5%) the function returns. The difference is a real
boundary effect, not a bug: `time_on_treatment_asper_applications` measures
the *span* from the first to the last of the 9 applications, not a sustained
steady-state rate, so at a small application count the "missing" interval
after the very last dose matters proportionally more. The two converge as
the application count grows.

## Medications that start mid-cycle: `real_time_since_medication_start`

Occasionally one component of a combination starts later than day 1 -- e.g.
FOLFOX (5-FU + oxaliplatin) added only from day 29 of a 42-day
Gem/nab-paclitaxel cycle, as in the SEQUENCE regimen for pancreatic ductal
adenocarcinoma:

```python
GEM = rwd.Medication("GEM", [1, 8, 15], 42)
FU  = rwd.Medication("FU",  [29],       42, required=False)
OX  = rwd.Medication("OX",  [29],       42, required=False)
sequence = rwd.Regime("SEQUENCE", GEM, FU, OX)

applied = rwd.parse_application_string("GEM-6x100_FU-2x100_OX-2x100", time_on_treatment_real=83, regime=sequence)
```

```
            applications  avg_dose  time_on_treatment_real  time_on_treatment_real_since_start  time_on_treatment_asper_applications
medication
GEM                    6     100.0                       83                                   83                                     56
FU                     2     100.0                       83                                   55                                     70
OX                     2     100.0                       83                                   55                                     70
```

FOLFOX could not possibly start before day 29, so 28 of the 83 real days are
not attributable to it -- `time_on_treatment_real_since_start` correctly
shows 55, not 83. Feeding the raw `time_on_treatment_real` into
`calculate_rdi_theoretical` instead would understate FOLFOX's RDI, charging
it for time it was never eligible to use:

```python
rwd.calculate_rdi_theoretical(100, 2, 55, 70)   # correct: 100.0
rwd.calculate_rdi_theoretical(100, 2, 83, 70)   # wrong:    84.34 -- penalized for 28 days it never had
```

Always pass `time_on_treatment_real_since_start` (not `time_on_treatment_real`)
into `calculate_rdi`/`calculate_rdi_theoretical`; for a medication starting
on day 1 of the cycle the two columns are identical, so this is safe to do
unconditionally.

**Known limitation.** This re-anchoring only corrects for a *late start*,
because a medication's own first possible day is a structural fact of the
protocol, independent of what actually happened. There is no equivalent
correction for a medication that *stops early* while another component of
the same combination continues -- e.g. 2 cycles of Gem/nab-paclitaxel +
FOLFOX per SEQUENCE, then a 3rd cycle of FOLFOX alone as bridging. *When*
Gem/nab-paclitaxel's own last dose was given is a clinical fact, not
derivable from the schedule, and it is not captured anywhere in
`application_string` or the single, shared `time_on_treatment_real` for the
whole regimen. The component that stopped early is still charged with the
full regimen-wide real time, understating its RDI. Fixing this properly
needs medication-specific first/last-application dates, or the therapy
course split into separate segments per regimen/protocol change -- each
parsed with its own `parse_application_string` call and its own
`time_on_treatment_real` -- rather than one combined call across a regimen
switch.

## What was theoretically achievable: `calculate_applications` / `theoretical_applications_table`

Separately from RDI, you can ask: given `total_days_on_therapy`, how many
applications of a medication would protocol timing have allowed, at most?

```python
n_theoretical, last_day = rwd.calculate_applications(
    treatment_days=[1, 8, 15], cycle_len_days=42, total_days_on_therapy=20
)
# (3, 14)
```

`last_day` is `None` (shown as `NaN` once placed in a DataFrame) when 0
applications were theoretically possible -- e.g. therapy stopped before a
medication added only later in the cycle was ever reached:

```python
rwd.calculate_applications([29], 42, 20)   # (0, None) -- day 29 never reached
```

`theoretical_applications_table` runs this over every medication in a
`Regime` at once, indexed the same way as `parse_application_string`'s
output so the two can be `.join()`ed:

```python
rwd.theoretical_applications_table(sequence, total_days_on_therapy=20)
```

```
            applications_theoretical  avg_dose_theoretical  last_treatment_day_theoretical
medication
GEM                                 3                    100                             14.0
FU                                  0                    100                              NaN
OX                                  0                    100                              NaN
```

## Processing many patients: `parse_patient_regimen`

`parse_patient_regimen` composes `calc_total_days_on_therapy`,
`validate_chemo_protocol`, `parse_application_string` and
`theoretical_applications_table` into the single call site meant for a
per-patient loop. It returns `(result, error)` -- exactly one is not
`None` -- and returns immediately on the first failure, so a bad row never
falls through to a later step using a stale value from a previous patient:

```python
res = {}
errs = {}

for pid, row in v1.set_index("case_id").iterrows():
    result, error = rwd.parse_patient_regimen(row, regime_dict)
    if error is not None:
        errs[pid] = error
    else:
        res[pid] = result
```

`result` is `parse_application_string`'s output joined with
`theoretical_applications_table`'s -- the real-vs-theoretical table for
every medication in that patient's regime in one DataFrame. `errs` collects
the original row for every patient that failed validation (bad Erste/Letzte
Gabe dates, an unknown `Therapieprotokoll_Name`, or an application string
that doesn't validate against the resolved regime), for review.

## Other modules

- `rwd.find_closest(d, ref, code)` -- for each `(patient, timestamp)` in
  `ref`, finds the closest same-patient/same-code row in a long-format
  lab/record table `d` by date.
- `rwd.cut_p(p)` -- p-value to significance code (`***`/`**`/`*`/`ns`).
- `rwd.fdr(pvals)` -- Benjamini-Hochberg FDR correction.
- `rwd.fisher_test(df, ref)` -- Fisher's exact test of `ref` against every
  column of `df`, with BH-FDR; 2x2 tables use SciPy, r x c tables need the
  `r` extra (`pip install cbrrwd[r]`, plus base R).
