# Recurrence classification (cured vs. not-cured) — `recurrence_cv.py`

Patient-level 5-fold cross-validated model for predicting recurrence after
catheter ablation of idiopathic PVCs, using the `cured` / `notcured` labels
already present in the dataset folder structure.

This script is intended to **replace** the fixed train/test-folder protocol in
`data_aug_mdi.py` for the recurrence task.

## Quick start — no installation required

Open `Run_Recurrence_CV.ipynb` in Google Colab and run the cells in order.
It mounts Google Drive (where the dataset lives), locates `train-test-folder`,
verifies patient grouping, and runs the full cross-validation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hakanyalman96-cell/MultiQRSNet-Recurrence/blob/main/Run_Recurrence_CV.ipynb)

> Replace `USERNAME` in the badge URL above with your GitHub username after
> pushing this repository.

To run locally instead:

```bash
git clone https://github.com/hakanyalman96-cell/MultiQRSNet-Recurrence.git
cd MultiQRSNet-Recurrence
pip install -r requirements.txt
python recurrence_cv.py --data /path/to/train-test-folder --sanity-check
python recurrence_cv.py --data /path/to/train-test-folder --epochs 60
```

## Why this differs from `data_aug_mdi.py`

| | `data_aug_mdi.py` | `recurrence_cv.py` |
|---|---|---|
| Task | LVOT vs RVOT (localization) | cured vs recurrence |
| Splitting | fixed `train_*` / `test_*` folders | **patient-level `StratifiedGroupKFold`, 5 folds** |
| Leakage control | none in code — depends on how folders were filled | grouping by patient ID + hard assertion per fold |
| Model selection | checkpoint per epoch named by **test** accuracy, best chosen afterwards | inner **validation** split; test fold never used for selection |
| Band-pass filter | computed but **not applied** (`return data` — filtered return commented out) | applied |
| Normalization | none | per-lead robust (median / IQR) |
| MDI | one scalar broadcast as a constant 13th channel | **12-dim vector** fused at the classifier head (`--no-mdi` to ablate) |
| Class imbalance | not addressed | class-weighted loss (default), optional label smoothing / mixup |
| Metrics | accuracy only | accuracy, precision, recall, F1, **AUC, PR-AUC**, with 95% CIs |
| Unit of analysis | recording | recording **and patient** (mean probability per patient) |
| Head | `flatten → Linear(4736, 1024)` (~5M params) | global average pooling → small head (~10⁵ params) |

The constant-channel MDI is worth noting: a channel that holds the same value at
every timestep cannot contribute information, which is the likely explanation
for its near-zero SHAP attribution — not that the network "learned to ignore a
redundant input".

## Requirements

```
python >= 3.9
torch, numpy, pandas, scipy, scikit-learn >= 0.24   # StratifiedGroupKFold
```

## Step 1 — verify patient grouping first (do not skip)

Everything depends on `patient_id_from_path()` correctly recovering which
recordings belong to the same patient. Run:

```bash
python recurrence_cv.py --data /path/to/train-test-folder --sanity-check
```

Inspect the printed report:

* `recordings/patient` should be > 1 if you have multiple recordings per patient.
  If **every recording became its own patient**, grouping is doing nothing and
  the leakage protection is inactive — fix `--patient-id` before going further.
* Check the `example paths -> patient id` lines look right.
* The fold composition table is what goes into Supplementary Table S2.

If the default heuristic is wrong, choose a mode explicitly:

```bash
--patient-id parent                          # patient = parent folder name
--patient-id stem                            # patient = filename up to first _ or -
--patient-id regex --patient-pattern '(P\d+)'   # custom
```

The script aborts if any patient ID maps to both a cured and a not-cured
recording, since that indicates mis-parsing or inconsistent folders.

## Step 2 — train

```bash
python recurrence_cv.py --data /path/to/train-test-folder --out results_recurrence --epochs 60
```

Useful flags:

```
--folds 5              number of CV folds
--no-mdi               ablate the MDI feature (run this to justify keeping it)
--no-class-weight      disable inverse-frequency class weighting
--mixup 0.4            enable mixup (0 = off)
--label-smoothing 0.05
--no-tune-threshold    keep the decision threshold at 0.5
--sites LVOT RVOT      restrict to specific origins
--no-filter --no-norm --no-augment    ablations
```

## Outputs

| file | contents |
|---|---|
| `fold_composition.csv` | patients / recordings / events per fold → **Supplementary Table S2** |
| `fold_metrics.csv` | per-fold accuracy, precision, recall, F1, AUC, PR-AUC → **Supplementary Table S3** |
| `oof_predictions.csv` | out-of-fold probability for every recording |
| `summary.json` | pooled metrics, per-fold mean ± 95% CI, patient-level bootstrap CIs |

## Reporting guidance

* Report the **patient-level pooled out-of-fold** metrics as primary — the
  patient, not the recording, is the independent unit.
* Always report **PR-AUC alongside its no-skill baseline** (the recurrence
  prevalence, ~0.21). An accuracy of ~0.79 is achievable by predicting "cured"
  for everyone, so accuracy alone is not evidence of a useful model.
* Report the 95% CIs. With ~33 events they will be wide, and that is the honest
  result.
* If the ablation (`--no-mdi`) does not degrade performance, say so rather than
  claiming the MDI contributes.

## Reproducibility

`--seed` controls the fold assignment, weight initialisation, and augmentation.
Fold assignment depends on the seed, so report the seed used. Note that results
on a dataset this small will vary meaningfully across seeds; running several
seeds and reporting the spread is more honest than reporting a single run.
