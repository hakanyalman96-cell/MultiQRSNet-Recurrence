# Unipolar voltage and PVC-ablation recurrence

Code accompanying:

**Low unipolar voltage at the earliest activation site is associated with recurrence after catheter
ablation of premature ventricular complexes: an exploratory electroanatomic study.**
Yalman H, Soysal AU, Çimci M, İkitimur B, Yalın K. *Journal of Interventional Cardiac
Electrophysiology (under review).*

The exact code that produced every number, table and figure in the revised manuscript is frozen in the
tagged release **`jice-r1-unipolar`**. The repository name reflects the broader research project; this
release and the files listed under *Current pipeline* are the ones relevant to the paper.

No raw electroanatomic exports or patient-identifiable data are included.

---

## What the study does

For patients undergoing point-by-point CARTO-guided PVC ablation (157 patients, 31 recurrences), we
take the **earliest activation site** (most negative relative LAT) and the **unipolar voltage** at that
site, and examine its association with 3-month recurrence (<80% PVC-burden reduction, same PVC
morphology) independently of arrhythmia origin (RVOT, other outflow tract, presumed intramural,
non-outflow tract). Analyses are exploratory and hypothesis-generating.

## Current pipeline (used for the paper)

| File | Produces |
|---|---|
| `src/carto_mapping_features.py` | `mapping_features.csv` from raw CARTO point-export XML |
| `firth.py` | Firth penalized-likelihood logistic regression |
| `analyze_recurrence.py` | `results.json`, `table1.json`: AUROC, quartile trend, three origin-parameterized Firth models, post hoc sensitivity models (atrial fibrillation + PVC-morphology count; + coupling interval), 100 × 5-fold cross-validation with bootstrap 95% CIs, the 5.4 mV cut-off and multi-threshold operating characteristics, nested cross-validation, per-stratum AUROC, Table 1 |
| `feature_screen.py` | Origin-adjusted multiplicity screen of the seven mapping features (Online Resource 3) |
| `make_figures.py`, `make_flowchart.py` | Figures 1–4 |
| `corrections.example.json` | Template for optional, git-ignored dataset-specific cleaning (keeps identifiers out of the code) |

See `ANALYSIS.md` for details and `data/DATA_DICTIONARY.md` for the input schema and CARTO
acquisition/annotation settings.

## Reproduce

```bash
pip install -r requirements.txt
python src/carto_mapping_features.py /path/to/CARTO_export_root      # -> mapping_features.csv
# optional: cp corrections.example.json corrections.json  (and edit)
python analyze_recurrence.py mapping_features.csv clinical.xlsx      # -> results.json, table1.json
python feature_screen.py     mapping_features.csv clinical.xlsx      # -> feature_screen.json
python make_figures.py       mapping_features.csv clinical.xlsx      # -> Figure_2-4
python make_flowchart.py                                             # -> Figure_1
```

`data/mapping_features_EXAMPLE.csv` is a synthetic file that shows the feature layout.

## Analysis history (transparency)

Earlier exploratory scripts are kept for transparency but were **not** used for the reported results:
`recurrence_cv.py`, `prepare_dataset.py`, `Run_Recurrence_CV.ipynb`, `build_recurrence_docx.py`,
`src/recurrence_mapping_fullcohort.py`, `src/recurrence_unipolar.py` and
`src/recurrence_unipolar_robust.py`. An earlier version compared mapping features with the ABC-VT
clinical score; that comparison was dropped, and the final analysis benchmarks against arrhythmia
origin. Those legacy scripts additionally need `pandas` and `statsmodels`.

## Data availability

Raw CARTO exports and the patient-level clinical spreadsheet are not shared publicly because they may
contain identifiable information and are governed by institutional data-governance policy. De-identified
data underlying the findings are available from the corresponding author on reasonable request,
subject to institutional approval and a data-transfer agreement.

## License

MIT — see `LICENSE`. The license covers the code only.
