# MultiQRSNet-Recurrence

Code and de-identified derived data for:

**Low unipolar voltage at the earliest activation site independently predicts recurrence after catheter ablation of premature ventricular complexes.**
Yalman H, et al. *(under review).*

This repository lets others reproduce the analysis from de-identified, mapping-derived
features. It does **not** contain raw electroanatomic exports or patient-identifiable data.

---

## What the study does

For patients undergoing point-by-point CARTO-guided PVC ablation, we extract, for each
patient, the **earliest activation site** (the mapped point with the most negative local
activation time, LAT = Map annotation − Reference annotation) and the **unipolar/bipolar
voltage** at that site. We then test whether unipolar voltage at the earliest site
predicts 3-month recurrence (<80% PVC-burden reduction) beyond the ABC-VT clinical risk
score and beyond the arrhythmia origin. Analyses are exploratory / hypothesis-generating,
single-centre, and require external validation.

## Repository structure

```
MultiQRSNet-Recurrence/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── carto_mapping_features.py        # parse CARTO point-export XML -> mapping_features.csv
│   ├── recurrence_mapping_fullcohort.py # 7-feature Bonferroni screen vs ABC-VT
│   ├── recurrence_unipolar.py           # characterise the unipolar signal (AUROC, independence, CV)
│   └── recurrence_unipolar_robust.py    # Firth OR (per mV), Youden cut-off, subgroup consistency
├── data/
│   ├── DATA_DICTIONARY.md               # column definitions for every input/output
│   └── mapping_features_EXAMPLE.csv     # SYNTHETIC example (not real patients) to run the code
├── reporting/
│   └── TRIPOD_AI_checklist.md           # TRIPOD+AI reporting checklist
└── build_recurrence_docx.py            # (optional) render manuscript/supplements to .docx
```

## Reproduce the analysis

```bash
pip install -r requirements.txt

# 1) From raw CARTO exports (NOT shared here) -> per-patient mapping features
python src/carto_mapping_features.py /path/to/CARTO_export_root
#    -> writes mapping_features.csv

# 2) Screen all 7 mapping features against ABC-VT (Bonferroni)
python src/recurrence_mapping_fullcohort.py
#    inputs: Kopya VES1 son.xlsx (clinical + outcome) and mapping_features.csv

# 3) Characterise the unipolar signal
python src/recurrence_unipolar.py

# 4) Robustness: Firth OR per mV, Youden cut-off, subgroup direction
pip install firthlogist -q
python src/recurrence_unipolar_robust.py
```

Steps 2–4 need the clinical/outcome spreadsheet (`Kopya VES1 son.xlsx`) with the columns
listed in `data/DATA_DICTIONARY.md`. To smoke-test the pipeline without any real data,
point the scripts at `data/mapping_features_EXAMPLE.csv` (synthetic).

## Data availability

De-identified mapping-derived features (the per-patient `mapping_features.csv` schema) and
all analysis code are provided in this repository. **Raw CARTO electroanatomic exports and
the patient-level clinical spreadsheet are not shared publicly** because they may contain
identifiable information and are governed by the institutional data-sharing policy.
De-identified data underlying the findings are available from the corresponding author
(hakan.yalman.96@gmail.com) on reasonable request, subject to institutional review-board /
data-governance approval and a data-transfer agreement.

## Requirements

Python ≥ 3.9. See `requirements.txt` (numpy, pandas, scikit-learn, scipy, statsmodels,
matplotlib, openpyxl, firthlogist).

## Citation

If you use this code, please cite the paper (details to be added on acceptance) and this
repository.

## License

MIT — see `LICENSE`. The license covers the code only; clinical data are not licensed for
redistribution.
