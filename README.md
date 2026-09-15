# cbrrwd

Real-world-data analysis helpers, extracted from a working analysis script.
Focus is systemic (chemo/immuno-)therapy: model a protocol, parse what a
patient actually received from a free-text application record, and derive
the real dose intensity (RDI) against plan. A handful of general real-world
-data helpers (patient/date linkage, significance codes, Fisher's exact test)
round this out.

## Install

```bash
pip install -e .                 # core: numpy, pandas, scipy, statsmodels
pip install -e '.[plotting]'     # sqrt_ax, label_heatmap (needs matplotlib)
pip install -e '.[r]'            # fisher_test on r x c tables, gdrate (needs R + tumgr)
pip install -e '.[all,test]'
```

## Layout

| Module | Contents |
| --- | --- |
| `cbrrwd.regimens` | `Medication`, `Regime`, `ChemoDetail`, `validate_meds`, `parse_application_string`, `calculate_applications`, `applications_to_treatment_days`, `determine_treatment_days`, `calc_total_days_on_therapy`, `validate_chemo_protocol`, `med_info`, `unpack_regime`, `calculate_rdi` |
| `cbrrwd.linkage` | `find_closest` (nearest record by date, per patient) |
| `cbrrwd.pvalues` | `cut_p`, `fdr`, `fisher_test` |
| `cbrrwd.plotting` | `sqrt_ax`, `label_heatmap` (needs the `plotting` extra) |
| `cbrrwd.rbackend.contingency` | `fisher_exact_rc` (r x c fallback for `fisher_test`) |
| `cbrrwd.rbackend.tumor_growth` | `gdrate` (Wilkerson et al. 2017 tumor growth/decay model, needs CRAN package `tumgr`) |

The most-used helpers are re-exported at the top level:

```python
import cbrrwd as rwd

fu = rwd.Medication("FU", treatment_days=[1], cycle_len=14, planned_applications=4)
ox = rwd.Medication("OX", treatment_days=[1], cycle_len=14, planned_applications=4)
doc = rwd.Medication("DOC", treatment_days=[1], cycle_len=14, planned_applications=4)
flot = rwd.Regime("FLOT", fu, ox, doc)

applied = rwd.parse_application_string("FU-4x100-2x80_OX-4x100-2x80_DOC-2x100-2x80",
                                        time_on_treatment_real=45, regime=flot)
```

## Dose intensity from an applied-medication string

A protocol is a `Regime` of `Medication` objects, each with its own
within-cycle schedule (`treatment_days`, `cycle_len`) and planned number of
applications:

```python
import cbrrwd as rwd

FU  = rwd.Medication("FU",  [1], 14, planned_applications=4)
OX  = rwd.Medication("OX",  [1], 14, planned_applications=4)
DOC = rwd.Medication("DOC", [1], 14, planned_applications=4)
flot = rwd.Regime("FLOT", FU, OX, DOC)

regime_dict = {"FLOT": flot}
```

For each patient row, compute days on therapy, validate the protocol name,
and parse what was actually applied (`"FU-4x100-2x80_OX-4x100-2x80_DOC-2x100-2x80"`
means: FU given 4x at 100% dose then 2x at 80%, etc.):

```python
dot = rwd.calc_total_days_on_therapy(row)  # from Erste_Gabe_Datum / Letzte_Gabe_Datum
protocol = rwd.validate_chemo_protocol(row, regime_dict)
regime = regime_dict[protocol]

applied = rwd.parse_application_string(row.Applizierte_Medikamente_Detail, dot, regime)
```

`applied` has one row per medication with `applications`, `avg_dose`,
`time_on_treatment_real` and `time_on_treatment_asper_applications` (the day
the observed number of applications implies under the protocol schedule).
Compare against the fully-planned course per medication:

```python
for med in regime.meds:
    appl_planned = rwd.applications_to_treatment_days(med.treatment_days, med.cycle_len,
                                                        med.planned_applications)
```

Then the real dose intensity, per medication:

```python
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

`calculate_rdi` handles both truncated therapy (fewer applications than
planned) and courses that ran longer than planned (the plan is scaled up to
the observed application count before comparing).

## Fisher's exact test with FDR

```python
res = rwd.fisher_test(df, ref)   # df: one column per feature, ref: grouping variable
```

2x2 tables use SciPy; r x c tables need the `r` extra (`pip install cbrrwd[r]`,
plus base R -- no extra CRAN package).

## Migrating from `utils.py`

* `import utils as ut` -> `import cbrrwd as rwd`. Unchanged calls: `Medication`,
  `Regime`, `ChemoDetail`, `validate_meds`, `parse_application_string`,
  `calculate_applications`, `applications_to_treatment_days`,
  `determine_treatment_days`, `calc_total_days_on_therapy`,
  `validate_chemo_protocol`, `med_info`, `unpack_regime`, `calculate_rdi`,
  `cut_p`, `find_closest`, `sqrt_ax` (needs `plotting` extra), `label_heatmap`
  (needs `plotting` extra), `gdrate` (`cbrrwd.rbackend.tumor_growth.gdrate`,
  needs `r` extra).
* `fisher_test` now uses SciPy for 2x2 tables (no R needed) and only falls
  back to R for r x c tables.
* `find_closest`: fixed a bug where patients with no matching record in `d`
  used the stale `(ts, org_val)` from a previous patient's last reference
  entry instead of iterating their own reference entries.
* Dropped (superseded): `rpy2fisher` (use `fisher_test` on a single column, or
  `cbrrwd.rbackend.contingency.fisher_exact_rc` directly), the standalone
  `numpy_to_rpy2`/`pandas_to_rpy2`/etc. converters (now internal to
  `cbrrwd.rbackend`).
