# Data dictionary

## `mapping_features.csv` (output of `src/carto_mapping_features.py`; one row per patient)

| Column | Type | Definition |
|---|---|---|
| `patient_folder` | string | De-identified patient/study folder key used to join to the clinical table |
| `n_points` | int | Number of valid mapped points parsed for the patient |
| `earliest_LAT` | float (ms) | Most negative local activation time; LAT = Map_Annotation − Reference_Annotation |
| `LAT_median` | float (ms) | Median LAT across the patient's points |
| `bip_at_earliest` | float (mV) | Bipolar voltage at the earliest-activation point |
| `uni_at_earliest` | float (mV) | **Unipolar voltage at the earliest-activation point (primary predictor)** |
| `bip_min` | float (mV) | Minimum bipolar voltage across points |
| `bip_median` | float (mV) | Median bipolar voltage |
| `uni_median` | float (mV) | Median unipolar voltage |

Invalid/sentinel voltages (|value| ≥ 9000) are discarded before aggregation.

## Clinical / outcome spreadsheet (`Kopya VES1 son.xlsx`) — NOT shared (identifiable)

The scripts locate columns by case-insensitive substring, so exact headers may vary:

| Needed field | Matched by | Definition |
|---|---|---|
| Patient name | `İSİM` / `ISIM` | Joined to `patient_folder` after Turkish→ASCII normalisation |
| Recurrence | `Recurrence` | Outcome; 1 = recurrence (<80% burden reduction @3-month Holter), 0 = success |
| Localisation | `Localisation` | Arrhythmia origin (RVOT, LVOT, LCC/RCC/NCC, AMC, LV summit, intramural, papillary, fascicular, …) |
| ABC-VT score | `ABC-VT` | Clinical risk score (benchmark / adjustment covariate) |

**Origin grouping.** Outflow-tract set = {RVOT, LVOT, LCC, RCC, NCC, AMC, LV SUMMIT,
INTRAMURAL}; everything else is coded `nonOT = 1` (non-outflow-tract).

## CARTO point-export XML (raw input to `carto_mapping_features.py`) — NOT shared

Each `*_Point_Export.xml` provides:
`<Annotations Reference_Annotation Map_Annotation/>` (LAT = Map − Reference) and
`<Voltages Unipolar Bipolar/>`. Points are grouped by the top-level folder under the export
root (= one patient/study).
