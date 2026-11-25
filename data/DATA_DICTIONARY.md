# Data dictionary

## `mapping_features.csv` (output of `src/carto_mapping_features.py`; one row per patient)

| Column | Type | Definition |
|---|---|---|
| `patient_folder` | string | De-identified patient/study folder key used to join to the clinical table |
| `n_points` | int | Number of valid mapped points parsed for the patient |
| `earliest_LAT` | float (ms) | Most negative local activation time; LAT = Map_Annotation − Reference_Annotation |
| `LAT_median` | float (ms) | Median LAT across the patient's points |
| `bip_at_earliest` | float (mV) | Bipolar voltage at the earliest-activation point |
| `uni_at_earliest` | float (mV) | **Unipolar voltage at the earliest-activation point (feature of interest)** |
| `bip_min` | float (mV) | Minimum bipolar voltage across points |
| `bip_median` | float (mV) | Median bipolar voltage |
| `uni_median` | float (mV) | Median unipolar voltage |

Invalid/sentinel voltages (|value| ≥ 9000) are discarded before aggregation.

## Acquisition and annotation (CARTO 3 v7)

- **Catheter:** a single catheter type throughout — open-irrigated 3.5 mm contact-force–sensing
  catheter with surround-flow irrigation (ThermoCool SmartTouch SF, Biosense Webster).
- **Contact force:** points were acquired only at adequate catheter–tissue contact
  (**contact force > 10 g**). The exported studies (CARTO 3 v7.5) contain no point-level contact-force
  files; VisiTag contact-force logs were absent or empty, and the radiofrequency logs (`RF_*.txt`)
  record power, duration, impedance and temperature but not contact force. Contact force is therefore
  not available as a column here or as a model covariate.
- **Unipolar reference:** Wilson's central terminal (0.5–100 Hz); bipolar from the distal pair (30–500 Hz).
- **Local activation annotation:** maximum negative dV/dt of the unipolar electrogram.
- **Reference annotation:** set automatically by the CARTO 3 v7 Advanced Reference Annotation algorithm on
  surface lead V5 in all patients. Relative LAT is an intra-beat offset; only its ranking (earliest site) is used.

## Clinical / outcome spreadsheet — NOT shared (identifiable)

Columns are located by case-insensitive substring (after Turkish→ASCII normalization), so exact
headers may vary:

| Needed field | Matched by | Definition |
|---|---|---|
| Patient name | `isim` / `name` | Joined to `patient_folder` after normalization |
| Recurrence | `recurrence` | Outcome; 1 = recurrence (<80% burden reduction @3-month Holter, same PVC morphology), 0 = success |
| Coupling interval | `coupling interval` | Covariate for the second post hoc sensitivity model |
| Localisation | `localisation` | Arrhythmia origin (RVOT, LVOT, LCC/RCC/NCC, AMC, LV summit, intramural, papillary, fascicular, …), assigned from 12-lead ECG and confirmed by mapping |
| Atrial fibrillation | `fibrillation` | Covariate for the post hoc sensitivity models |
| Number of PVC morphologies | `morpholog` | Covariate for the post hoc sensitivity models |
| LV ejection fraction | `ejection` | Baseline characteristic |
| PVC burden (24-h Holter) | `burden` | Baseline characteristic |

**Origin grouping.** Four mutually exclusive categories: RVOT; other outflow-tract
{LVOT, LCC, RCC, NCC, AMC, LV SUMMIT}; presumed intramural (all of outflow-tract origin in this
cohort); non-outflow-tract (everything else). Presumed-intramural was recorded when biventricular
mapping and/or ablation at two adjacent chambers was required — a procedural presumption, looser
than activation-based criteria.

## `corrections.json` — OPTIONAL, NOT shared (git-ignored)

Dataset-specific cleaning kept out of the code so that no patient identifiers appear in this
repository. See `corrections.example.json`. Shape:

```json
{
  "voltage_fixes": { "<normalized patient_folder>": { "uni_at_earliest": 0.0, "bip_at_earliest": 0.0 } },
  "name_aliases":  { "<normalized spelling>": "<normalized canonical>" }
}
```

In the study, two earliest-site voltages were re-measured after an acquisition artifact, and one
spelling variant was mapped to its canonical form; duplicate study folders are collapsed
automatically by normalized key. None of these require editing the analysis code.

## CARTO point-export XML (raw input to `carto_mapping_features.py`) — NOT shared

Each `*_Point_Export.xml` provides `<Annotations Reference_Annotation Map_Annotation/>`,
`<Voltages Unipolar Bipolar/>`, `<Impedances/>` and `<ECG/>`. It does **not** contain a
contact-force element.
