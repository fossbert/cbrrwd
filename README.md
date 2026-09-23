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
| `cbrrwd.regimens` | `Medication`, `Regime`, `validate_meds`, `parse_application_string`, `parse_patient_regimen`, `calculate_applications`, `theoretical_applications_table`, `planned_applications_table`, `applications_to_treatment_days`, `determine_treatment_days`, `calc_total_days_on_therapy`, `validate_chemo_protocol`, `med_info`, `unpack_regime`, `calculate_rdi`, `calculate_rdi_theoretical`, `calculate_rdi_combined`, `SuspectApplicationCountWarning` |
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

Getting from a raw patient row to that string's regime takes two small
lookups first:

- `calc_total_days_on_therapy(row)` -- days between `Erste_Gabe_Datum` (first
  dose) and `Letzte_Gabe_Datum` (last dose); column names are overridable
  keyword args if your table names them differently. Raises `ValueError` if
  the last dose precedes the first (a data error, not a valid course).
- `validate_chemo_protocol(row, regime_dict)` -- looks `row.Therapieprotokoll_Name`
  up in `regime_dict` and returns it unchanged; raises `ValueError` if that
  protocol name isn't a key in `regime_dict` (typo, or a protocol not
  modeled yet).

```python
dot = rwd.calc_total_days_on_therapy(row)
protocol = rwd.validate_chemo_protocol(row, regime_dict)
regime = regime_dict[protocol]

applied = rwd.parse_application_string(row.Applizierte_Medikamente_Detail, dot, regime)
```

`parse_application_string` itself first calls `validate_meds` on the string
against `regime`: every *required* medication must appear with exactly the
frequency (segment count) the regime expects, an *optional* one may appear
or not, and any medication that's neither required nor optional fails
validation -- matching is case-insensitive and segment order doesn't matter.
A mismatch raises `ValueError`. For each medication present it then sums the
`NxPercent%` segments into a total application count and a dose-weighted
average (`avg_dose`), and calls `applications_to_treatment_days` (below) to
translate that count into the protocol day it would fall on.

`applied` is indexed by medication name, with one row each:

```
            applications   avg_dose  time_on_treatment_real  time_on_treatment_asper_applications
medication
FU                     6  93.333333                      42                                    70
OX                     6  93.333333                      42                                    70
DOC                    4  90.000000                      42                                    42
```

- `applications` / `avg_dose` -- how many doses were given, and their average
  relative dose (%).
- `time_on_treatment_real` -- the regimen-wide real span (Erste_Gabe to
  Letzte_Gabe), the same for every medication in the row. For a medication
  that starts partway through the regimen (see below) this is still the
  whole-regimen span, not re-anchored to that medication's own start.
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

`calculate_rdi` takes a medication's row and the column names to read its
inputs from -- the defaults already match what `parse_patient_regimen`
produces (see below), so once you're inside a row from there you're "in
cbrrwd's own world" and normally pass nothing but the row itself:

```python
appl_planned = rwd.applications_to_treatment_days(
    FU.treatment_days, FU.cycle_len, FU.planned_applications
)  # protocol day of the 4th (last planned) FU application -> 42
# (this is exactly what planned_applications_table computes per medication --
# see "What the protocol calls for" below for the version that does it for you)

row = applied.loc["FU"].copy()
row["applications_planned"] = FU.planned_applications
row["avg_dose_planned"] = 100
row["time_on_treatment_planned"] = appl_planned

rdi = rwd.calculate_rdi(row)
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
a plan: it compares the real cumulative dose given against the cumulative
dose that full-dose, on-protocol timing would have achieved *in the same
real elapsed time* (`applications_theoretical`, from
`theoretical_applications_table` below) -- i.e. it's a pure ratio of doses
given vs. doses theoretically possible, with no separate time-based penalty.

```python
GEM = rwd.Medication("GEM", treatment_days=[1, 8, 15], cycle_len=28)
gem_regime = rwd.Regime("GEM", GEM)

applied = rwd.parse_application_string("GEM-9x100", time_on_treatment_real=70, regime=gem_regime)
theoretical = rwd.theoretical_applications_table(gem_regime, total_days_on_therapy=70)
row = applied.join(theoretical).loc["GEM"]

rdi = rwd.calculate_rdi_theoretical(row)
```

Scenarios for weekly-x3/q28 gemcitabine (day 1, 8, 15 of a 28-day cycle):

| Scenario | applications | avg_dose | real time | applications_theoretical | RDI |
| --- | --- | --- | --- | --- | --- |
| 9 applications, exactly on the day1/8/15 q28 schedule | 9 | 100 | 70d | 9 | **100.0** |
| Same 9 applications, but given every 2 weeks instead | 9 | 100 | 112d | 13 | **69.23** |
| Stopped after 2 applications, both on time | 2 | 100 | 7d | 2 | **100.0** |
| Day 8 dropped 3x (6 of 9 given), same start/end as full course | 6 | 100 | 70d | 9 | **66.67** |
| Dose reduced to 80%, applications spread over double the protocol time | 4 | 80 | 56d | 7 | **45.71** |

When `applications > applications_theoretical`, `applications_theoretical` is
raised to `applications` before the ratio is computed, capping the result at
`avg_dose` -- without this, a real course just 1 day faster than protocol
pace can already tip `applications_theoretical` down by 1 at a cycle
boundary and send RDI to 125% for what is, for all practical purposes, a
fully compliant course. The cap doesn't hide the underlying gap, though:
`parse_patient_regimen` still flags every such row via
`applications_exceed_theoretical` and raises a `SuspectApplicationCountWarning`
naming the case id and the size of the gap in both applications and days
(see below) -- a 1-2 day gap is normal rounding noise, a double-digit one
(as with the FOLFIRINOX case above) is almost always bad Erste_Gabe_Datum/
Letzte_Gabe_Datum or a wrong cycle count.

## Medications that start mid-cycle

A component of a combination can start later than day 1 -- e.g. FOLFOX
(5-FU + oxaliplatin) added only from day 29 of a 42-day Gem/nab-paclitaxel
cycle, as in the SEQUENCE regimen for pancreatic ductal adenocarcinoma
(`treatment_days=[29]`, see `Regime`/`Medication` above). `parse_application_string`
does not re-anchor `time_on_treatment_real` to such a medication's own start,
so it gets charged with the full regimen-wide real time, which understates
its RDI. The robust fix is to not model this as one combined regime at all:
split the patient's course into separate segments at the point the protocol
changed (e.g. `GNP_SEQUENCE` and `FOLFOX_SEQUENCE` as two rows with their own
Erste_Gabe/Letzte_Gabe), each parsed with its own `parse_patient_regimen`
call -- see the note on `case_id` uniqueness below.

## What was theoretically achievable: `calculate_applications` / `theoretical_applications_table`

Separately from RDI, you can ask: given `total_days_on_therapy`, how many
applications of a medication would protocol timing have allowed, at most?
`calculate_applications` answers this via `determine_treatment_days`, which
does the actual day-bookkeeping: it takes the within-cycle pattern (e.g.
`[1, 8, 15]`), repeats it every `cycle_len_days` for as many cycles as fit,
and trims the result at `total_days_on_therapy`. `calculate_applications`
then just counts how many of those days occurred and reports the last one:

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

The other direction -- from an application *count* to the protocol day it
falls on -- is `applications_to_treatment_days` (used above by
`parse_application_string` and by the fixed-cycle-plan example below): given
`n_applications`, it walks the same within-cycle/`cycle_len_days` pattern
forward and returns that day, e.g. the 4th FU application on a day-1/q14
schedule falls on protocol day 42.

`theoretical_applications_table` runs `calculate_applications` over every
medication in a `Regime` at once (pulling each medication's `treatment_days`/
`cycle_len` via the small `unpack_regime`/`med_info` accessors rather than
looping over `regime.meds` by hand), indexed the same way as
`parse_application_string`'s output so the two can be `.join()`ed:

```python
sequence = rwd.Regime("SEQUENCE",
    rwd.Medication("GEM", [1, 8, 15], 42),
    rwd.Medication("FU",  [29],       42, required=False),
    rwd.Medication("OX",  [29],       42, required=False),
)
rwd.theoretical_applications_table(sequence, total_days_on_therapy=20)
```

```
            applications_theoretical  avg_dose_theoretical  last_treatment_day_theoretical
medication
GEM                                 3                    100                             14.0
FU                                  0                    100                              NaN
OX                                  0                    100                              NaN
```

## What the protocol calls for: `planned_applications_table`

The plan-side counterpart to `theoretical_applications_table`, needed by
`calculate_rdi`: for every medication with `planned_applications` set, it
calls `applications_to_treatment_days` (above) once to get
`time_on_treatment_planned` -- the protocol day of the *last planned*
application -- and reports `applications_planned` and `avg_dose_planned`
(always 100) alongside it. Unlike the theoretical table, this doesn't depend
on any observed patient at all, only on the protocol itself. A medication
with `planned_applications=None` (dosed until progression/toxicity, not for
a fixed cycle count) gets `NaN` in all three columns rather than `0`, so it
stays distinguishable from "0 planned" downstream:

```python
rwd.planned_applications_table(flot)
```

```
            applications_planned  avg_dose_planned  time_on_treatment_planned
medication
FU                              4               100                         42
OX                              4               100                         42
DOC                             4               100                         42
```

## Processing many patients: `parse_patient_regimen`

`parse_patient_regimen` composes `calc_total_days_on_therapy`,
`validate_chemo_protocol`, `parse_application_string`,
`theoretical_applications_table` and `planned_applications_table` into the
single call site meant for a per-patient loop. It returns `(result, error)`
-- exactly one is not `None` -- and returns immediately on the first
failure, so a bad row never falls through to a later step using a stale
value from a previous patient:

```python
res = {}
errs = {}

for idx, row in v1.iterrows():
    pid = row["ID"]
    result, error = rwd.parse_patient_regimen(row, regime_dict, case_id="ID")
    if error is not None:
        errs[idx] = row
    else:
        res[idx] = result.assign(case_id=pid)
```

A raw input table's own index is rarely meaningful (`iterrows()` typically
walks a default `RangeIndex`, and the real patient id lives in an ordinary
column, `"ID"` here) -- `case_id="ID"` tells `parse_patient_regimen` to read
that column for the id it puts in the warning below, instead of falling back
to `row.name`. If your table *does* have the patient id as its index (e.g.
after `df.set_index("case_id")`), just omit `case_id` and it uses `row.name`
directly. Either way, `res`/`errs` above are your own dicts -- key them by
whatever you find convenient (`idx`, `pid`, ...); `parse_patient_regimen`
doesn't care, `case_id` only affects the warning message.

`result` is `parse_application_string`'s output joined with
`theoretical_applications_table`'s and `planned_applications_table`'s -- one
row per medication with everything either RDI function needs already on it:
the real columns (`applications`, `avg_dose`, ...), the theoretical-null-model
columns (`applications_theoretical`, `avg_dose_theoretical`, ...) and the
fixed-cycle-plan columns (`applications_planned`, `avg_dose_planned`,
`time_on_treatment_planned` -- `NaN` for a medication with no
`planned_applications`, i.e. dosed open-ended). This is what makes a mixed
regime (some medications on a fixed cycle count, others dosed until
progression/toxicity) a single parse instead of two: `calculate_rdi` and
`calculate_rdi_theoretical` each just read the columns they need off the same
row, and `calculate_rdi_combined` picks the right one automatically per
medication --

```python
result["rdi"] = result.apply(rwd.calculate_rdi_combined, axis=1)
```

`result` also carries an
`applications_exceed_theoretical` column (`applications > applications_theoretical`)
flagging rows worth checking before trusting their RDI -- usually bad input
data, not a real dose-dense course (see `calculate_rdi_theoretical` above).
Every flagged row also raises a `SuspectApplicationCountWarning`, naming the
`case_id` value resolved as above:

```
SuspectApplicationCountWarning: case_id='PATIENT-007' medication=FU: 5 Gaben
dokumentiert, aber nur 4 waeren im beobachteten Zeitraum protokollgerecht
getaktet moeglich gewesen (reale Zeit 54d vs. 56d fuer diese Gabenzahl,
Differenz -2d) -- Erste_Gabe_Datum/Letzte_Gabe_Datum und Zyklenzahl pruefen.
```

`errs` collects the original row for every patient that failed validation
(bad Erste/Letzte Gabe dates, an unknown `Therapieprotokoll_Name`, or an
application string that doesn't validate against the resolved regime), for
review.

The patient id -- whether it's the DataFrame index or an `"ID"` column --
must be unique per row you feed in: a patient whose regime changed mid-course
(see above) needs one row per segment, each with its own
Erste_Gabe/Letzte_Gabe -- otherwise the later row silently overwrites the
earlier one in `res`.

## Other modules

- `rwd.find_closest(d, ref, code)` -- for each `(patient, timestamp)` in
  `ref`, finds the closest same-patient/same-code row in a long-format
  lab/record table `d` by date.
- `rwd.cut_p(p)` -- p-value to significance code (`***`/`**`/`*`/`ns`).
- `rwd.fdr(pvals)` -- Benjamini-Hochberg FDR correction.
- `rwd.fisher_test(df, ref)` -- Fisher's exact test of `ref` against every
  column of `df`, with BH-FDR; 2x2 tables use SciPy, r x c tables need the
  `r` extra (`pip install cbrrwd[r]`, plus base R).
