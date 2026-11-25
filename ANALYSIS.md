# Recurrence / unipolar-voltage analysis

Reproduces the analysis in *"Low unipolar voltage at the earliest activation site is associated
with recurrence after catheter ablation of premature ventricular complexes: an exploratory
electroanatomic study."*

The identifiable inputs (the clinical spreadsheet and the raw CARTO exports) are **not** shared.
This directory provides the de-identified code and the feature schema so the pipeline is fully
inspectable; `data/mapping_features_EXAMPLE.csv` shows the expected feature layout.

## Files

| File | Purpose |
|---|---|
| `firth.py` | Firth penalized-likelihood logistic regression (Jeffreys prior), robust step-halving |
| `analyze_recurrence.py` | Main analysis → `results.json`, `table1.json` (AUROC, quartile trend, three origin-parameterized Firth models, AF + PVC-morphology sensitivity model, leakage-free 5-fold cross-validation, Youden cut-off and multi-threshold operating characteristics, per-stratum AUROC) |
| `feature_screen.py` | Origin-adjusted multiplicity screen of the seven candidate mapping features (Bonferroni/Holm/BH) → `feature_screen.json` |
| `src/carto_mapping_features.py` | Parses raw CARTO `*_Point_Export.xml` into `mapping_features.csv` |
| `corrections.example.json` | Template for optional, git-ignored dataset-specific cleaning (no identifiers in code) |

See `data/DATA_DICTIONARY.md` for the feature schema, CARTO acquisition/annotation details
(SmartTouch SF catheter, >10 g contact-force acquisition, Wilson-central-terminal unipolar
reference, max −dV/dt unipolar LAT, contact force not retained in the export), and the
`corrections.json` convention.

## Reproduce

```bash
pip install -r requirements.txt            # numpy, scipy, scikit-learn, openpyxl
# (optional) cp corrections.example.json corrections.json  and edit for your dataset
python analyze_recurrence.py  mapping_features.csv  clinical.xlsx
python feature_screen.py      mapping_features.csv  clinical.xlsx
```

`mapping_features.csv` is produced by `python src/carto_mapping_features.py /path/to/CARTO_export_root`.

All analyses are exploratory and hypothesis-generating; see the manuscript for interpretation and limitations.
