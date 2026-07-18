"""
recurrence_cv.py
================
Cured vs. Recurrence ("notcured") classification from 12-lead surface ECG,
with patient-level stratified 5-fold cross-validation.

This script replaces the fixed train/test-folder protocol used in
`data_aug_mdi.py`. The differences are deliberate and each one addresses a
specific methodological problem:

  1. PATIENT-LEVEL SPLITTING. Multiple ECG recordings exist per patient.
     Splitting by recording lets the same patient appear in both train and
     test, which inflates performance ("identity leakage"). Here folds are
     assigned to unique PATIENTS (stratified by outcome) and then mapped back
     to their recordings, so every recording of a patient stays in one fold and
     the event rate is preserved in each fold.
  2. MODEL SELECTION ON A VALIDATION SPLIT, never on the test fold. The old
     script saved a checkpoint per epoch named by *test* accuracy and the best
     was picked afterwards - that is selection on the test set and biases the
     reported number upward.
  3. THE BAND-PASS FILTER IS ACTUALLY APPLIED. In the old script
     `process_text_file_2` computed `filtered_data` but returned raw `data`
     (the filtered return was commented out).
  4. MDI AS A REAL 12-DIM FEATURE VECTOR fed to the classifier head, instead of
     a single scalar broadcast across all 2500 timesteps as a constant channel.
     A constant channel carries no information, which is why its SHAP
     attribution was ~0. Use --no-mdi to ablate it.
  5. CLASS IMBALANCE handled explicitly (class-weighted loss by default;
     optional label smoothing / mixup), and reported with imbalance-aware
     metrics (PR-AUC) alongside accuracy.
  6. LENGTH-PRESERVING AUGMENTATION. The old time-stretch returned an array of
     a different length, which cannot be assigned back into a fixed-size row.
  7. PATIENT-LEVEL METRICS in addition to recording-level metrics: a patient's
     probability is the mean over their recordings. This is the clinically
     meaningful unit and the one that should be reported as primary.

Outputs (written to --out):
  fold_metrics.csv     per-fold metrics            -> Supplementary Table S3
  fold_composition.csv per-fold patient/event counts -> Supplementary Table S2
  oof_predictions.csv  out-of-fold predictions (recording and patient level)
  summary.json         pooled metrics with 95% CIs

Usage:
  python recurrence_cv.py --data /path/to/train-test-folder --sanity-check
  python recurrence_cv.py --data /path/to/train-test-folder --epochs 60
"""

import argparse
import json
import os
import random
import re
import warnings
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as tdata
from scipy.signal import butter, filtfilt
from scipy.stats import t as student_t
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
LEAD_NAMES = ["V1(22)", "V2(23)", "V3(24)", "V4(25)", "V5(26)", "V6(27)",
              "I(110)", "II(111)", "III(112)", "aVL(171)", "aVR(172)", "aVF(173)"]
N_LEADS = 12
N_SAMPLES = 2500
FS = 1000
SITES = ["LVOT", "RVOT", "RLVOT"]
# Folder name -> label. 0 = cured, 1 = recurrence.
LABEL_FOLDERS = {"train_cured": 0, "test_cured": 0,
                 "train_notcured": 1, "test_notcured": 1}
SPLIT_FOLDERS = set(LABEL_FOLDERS) | {"test_sakura"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Patient identification  -- THE MOST IMPORTANT PART TO VERIFY
# --------------------------------------------------------------------------
def patient_id_from_path(path: str, mode: str = "auto", pattern: str = None) -> str:
    """Derive a patient identifier from an ECG file path.

    The leakage guarantee is only as good as this function. Run with
    --sanity-check and inspect the printed report before trusting any result:
    if every recording maps to a unique ID, grouping is doing nothing.

    modes:
      auto   - use the parent directory name when it is a per-patient folder,
               otherwise the filename prefix before the first '_' or '-'.
      parent - always use the parent directory name.
      stem   - always use the filename prefix before the first '_' or '-'.
      regex  - use --patient-pattern, first capture group.
    """
    p = Path(path)
    if mode == "regex":
        if not pattern:
            raise ValueError("--patient-pattern is required with --patient-id regex")
        m = re.search(pattern, p.name)
        if not m:
            raise ValueError(f"pattern {pattern!r} did not match {p.name!r}")
        return m.group(1)
    if mode == "parent":
        return p.parent.name
    if mode == "stem":
        return re.split(r"[_\-]", p.stem)[0]
    # auto
    if p.parent.name not in SPLIT_FOLDERS:
        return p.parent.name
    return re.split(r"[_\-]", p.stem)[0]


def build_index(data_root: str, id_mode: str, id_pattern: str,
                sites=None) -> pd.DataFrame:
    """Scan the dataset folders and build a recording-level table."""
    sites = sites or SITES
    rows = []
    for site in sites:
        for folder, label in LABEL_FOLDERS.items():
            d = os.path.join(data_root, site, folder)
            if not os.path.isdir(d):
                continue
            for f in sorted(glob(os.path.join(d, "**", "*.txt"), recursive=True)):
                rows.append({
                    "path": f,
                    "site": site,
                    "source_folder": folder,
                    "label": label,
                    "patient": f"{site}:{patient_id_from_path(f, id_mode, id_pattern)}",
                })
    if not rows:
        raise SystemExit(f"No .txt recordings found under {data_root}. Check --data.")
    df = pd.DataFrame(rows)

    # A patient must not carry conflicting labels.
    bad = df.groupby("patient")["label"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise SystemExit(
            "These patient IDs map to BOTH cured and recurrence recordings, which "
            "means the ID parsing is wrong or the folders are inconsistent:\n"
            + "\n".join(f"  {p}" for p in bad.index[:20])
            + "\nFix --patient-id / --patient-pattern before proceeding."
        )
    return df


def build_index_from_manifest(manifest: str) -> pd.DataFrame:
    """Load a manifest.csv produced by prepare_dataset.py.

    Required columns: path, patient, label. Optional: site, approach.
    Relative paths are resolved against the manifest's own directory.
    """
    df = pd.read_csv(manifest)
    need = {"path", "patient", "label"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{manifest}: missing required column(s) {sorted(missing)}")
    root = os.path.dirname(os.path.abspath(manifest))
    df["path"] = [p if os.path.isabs(p) else os.path.join(root, p) for p in df["path"]]
    gone = [p for p in df["path"] if not os.path.exists(p)]
    if gone:
        raise SystemExit(f"{len(gone)} file(s) in the manifest do not exist, e.g.\n  {gone[0]}")
    df["label"] = df["label"].astype(int)
    if "site" not in df.columns:
        df["site"] = "NA"
    df["source_folder"] = "manifest"

    bad = df.groupby("patient")["label"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise SystemExit("These patients carry conflicting labels in the manifest:\n"
                         + "\n".join(f"  {p}" for p in bad.index[:20]))
    return df


def report_index(df: pd.DataFrame):
    n_rec, n_pat = len(df), df["patient"].nunique()
    per = df.groupby("patient").size()
    pat = df.drop_duplicates("patient")
    print("\n=== DATASET ===")
    print(f"recordings          : {n_rec}")
    print(f"patients            : {n_pat}")
    print(f"recordings/patient  : mean {per.mean():.2f}  median {per.median():.0f}  "
          f"min {per.min()}  max {per.max()}")
    print(f"patients cured      : {(pat.label == 0).sum()}")
    print(f"patients recurrence : {(pat.label == 1).sum()} "
          f"({100 * (pat.label == 1).mean():.1f}%)")
    print("\nby site (patients):")
    print(pat.groupby(["site", "label"]).size().unstack(fill_value=0)
          .rename(columns={0: "cured", 1: "recurrence"}))
    if per.max() == 1:
        print("\n!! Every recording became its own patient. Patient-level grouping "
              "is having no effect - check --patient-id.")
    print("\nexample paths -> patient id:")
    for _, r in df.head(5).iterrows():
        print(f"  {r['path']}\n    -> {r['patient']}")


# --------------------------------------------------------------------------
# Signal loading and preprocessing
# --------------------------------------------------------------------------
def butter_bandpass(x, low=0.5, high=120.0, fs=FS, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def read_ecg(path: str) -> np.ndarray:
    """Read a CARTO export .txt into a (12, 2500) array.

    Assumes line index 2 (the third line) is the column header row and that
    lead columns are named as in LEAD_NAMES.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    if len(lines) < 4:
        raise SystemExit(f"{path}: fewer than 4 lines; not a CARTO export?")
    headers = lines[2].split()
    missing = [n for n in LEAD_NAMES if n not in headers]
    if missing:
        raise SystemExit(
            f"\nCould not find these lead columns in the header row of\n  {path}\n"
            f"missing: {missing}\n"
            f"header row (line 3) actually contains:\n  {headers[:40]}\n\n"
            "The parser expects the third line to be the header and leads named "
            "like 'V1(22)', 'I(110)'. If your export differs, edit LEAD_NAMES "
            "(and the header line index) at the top of this file."
        )
    idx = [headers.index(name) for name in LEAD_NAMES]
    out = np.zeros((N_LEADS, N_SAMPLES), dtype=np.float64)
    n = 0
    for line in lines[3:]:
        items = line.split()
        if not items:
            continue
        if n >= N_SAMPLES:
            break
        for r, c in enumerate(idx):
            out[r, n] = float(items[c])
        n += 1
    if n < N_SAMPLES:  # pad short records by edge repetition
        out[:, n:] = out[:, n - 1:n] if n > 0 else 0.0
    return out


def preprocess(x: np.ndarray, do_filter=True, do_norm=True) -> np.ndarray:
    if do_filter:
        x = np.stack([butter_bandpass(x[i]) for i in range(x.shape[0])])
    if do_norm:  # robust per-lead scaling: median / IQR
        med = np.median(x, axis=1, keepdims=True)
        q75, q25 = np.percentile(x, [75, 25], axis=1, keepdims=True)
        x = (x - med) / (q75 - q25 + 1e-6)
    return np.ascontiguousarray(x, dtype=np.float32)


def compute_mdi(x: np.ndarray) -> np.ndarray:
    """Maximum Deflection Index per lead -> 12-dim vector in [0, 1].

    Note this returns a VECTOR (one value per lead). The original script
    collapsed it to one scalar and broadcast it as a constant 13th channel,
    which carries no per-lead information.
    """
    mid = x.shape[1] // 2
    w = x[:, mid - 100: mid + 100]
    w = w - np.median(w, axis=1, keepdims=True)
    return (np.argmax(np.abs(w), axis=1) / w.shape[1]).astype(np.float32)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class ECGDataset(tdata.Dataset):
    def __init__(self, table: pd.DataFrame, augment=False, cache=None,
                 do_filter=True, do_norm=True):
        self.t = table.reset_index(drop=True)
        self.augment = augment
        self.cache = cache if cache is not None else {}
        self.do_filter, self.do_norm = do_filter, do_norm

    def __len__(self):
        return len(self.t)

    def _load(self, path):
        if path not in self.cache:
            sig = preprocess(read_ecg(path), self.do_filter, self.do_norm)
            self.cache[path] = (sig, compute_mdi(sig))
        return self.cache[path]

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        """Length-preserving ECG augmentation (training folds only)."""
        x = x.copy()
        n = x.shape[1]
        if random.random() < 0.5:  # additive noise
            x += np.random.normal(0, 0.01, x.shape).astype(np.float32)
        if random.random() < 0.5:  # per-lead amplitude scaling
            x *= np.random.uniform(0.9, 1.1, (x.shape[0], 1)).astype(np.float32)
        if random.random() < 0.5:  # circular time shift
            x = np.roll(x, random.randint(-50, 50), axis=1)
        if random.random() < 0.2:  # polarity inversion
            x = -x
        if random.random() < 0.3:  # baseline wander
            f = np.random.uniform(0.05, 0.5)
            tt = np.arange(n) / FS
            x += (0.05 * np.sin(2 * np.pi * f * tt)).astype(np.float32)
        if random.random() < 0.5:  # time warp, resampled back to n
            factor = random.uniform(0.8, 1.2)
            src = np.linspace(0, n - 1, int(round(n * factor)))
            tmp = np.stack([np.interp(src, np.arange(n), x[i]) for i in range(x.shape[0])])
            dst = np.linspace(0, tmp.shape[1] - 1, n)
            x = np.stack([np.interp(dst, np.arange(tmp.shape[1]), tmp[i])
                          for i in range(tmp.shape[0])]).astype(np.float32)
        return x

    def __getitem__(self, i):
        r = self.t.iloc[i]
        sig, mdi = self._load(r["path"])
        if self.augment:
            sig = self._augment(sig)
            mdi = compute_mdi(sig)
        return (torch.from_numpy(np.ascontiguousarray(sig)),
                torch.from_numpy(mdi),
                torch.tensor(int(r["label"])),
                i)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class RecurrenceNet(nn.Module):
    """Compact 1D-CNN with global average pooling and an MDI-fused head.

    Deliberately small. With ~33 recurrence patients, the original
    flatten->Linear(4736,1024) head (several million parameters) is far past
    what the data can support.
    """

    def __init__(self, in_ch=N_LEADS, width=32, mdi_dim=N_LEADS,
                 use_mdi=True, dropout=0.5, n_classes=2):
        super().__init__()
        self.use_mdi = use_mdi

        def block(ci, co):
            return nn.Sequential(
                nn.Conv1d(ci, co, 7, padding=3), nn.BatchNorm1d(co), nn.ReLU(),
                nn.Conv1d(co, co, 5, padding=2), nn.BatchNorm1d(co), nn.ReLU(),
                nn.MaxPool1d(2))

        self.features = nn.Sequential(
            block(in_ch, width), block(width, width * 2), block(width * 2, width * 4))
        self.pool = nn.AdaptiveAvgPool1d(1)
        head_in = width * 4 + (mdi_dim if use_mdi else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes))

    def forward(self, x, mdi=None):
        z = self.pool(self.features(x)).flatten(1)
        if self.use_mdi:
            z = torch.cat([z, mdi], dim=1)
        return self.head(z)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def patient_level_folds(df: pd.DataFrame, n_splits: int, seed: int):
    """Assign folds to PATIENTS (not recordings), stratified by outcome.

    StratifiedGroupKFold stratifies at the sample level while grouping, so when
    patients contribute unequal numbers of recordings the event rate per fold
    can drift far from the cohort rate. Assigning folds to unique patients and
    then mapping back to their recordings gives exact patient-level
    stratification and makes leakage impossible by construction.
    """
    pat = df.drop_duplicates("patient")[["patient", "label"]].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pat["fold"] = -1
    for k, (_, te) in enumerate(skf.split(pat, pat["label"])):
        pat.loc[te, "fold"] = k
    fold_of = dict(zip(pat["patient"], pat["fold"]))
    f = df["patient"].map(fold_of).values
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]


def compute_metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
    }
    # AUC is undefined if the fold happens to contain a single class
    out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    out["pr_auc"] = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return out


def mean_ci(vals, conf=0.95):
    v = np.asarray([x for x in vals if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    m = v.mean()
    if len(v) < 2:
        return m, float("nan"), float("nan")
    sem = v.std(ddof=1) / np.sqrt(len(v))
    h = sem * student_t.ppf(0.5 + conf / 2, len(v) - 1)
    return m, m - h, m + h


def bootstrap_ci(y, p, fn, n_boot=2000, conf=0.95, seed=0):
    """Bootstrap CI over independent units (patients)."""
    rng = np.random.default_rng(seed)
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(fn(y[idx], p[idx]))
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Train / evaluate one fold
# --------------------------------------------------------------------------
def run_epoch(model, loader, device, criterion, optimizer=None, mixup=0.0):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, probs, ys, idxs = 0.0, [], [], []
    for sig, mdi, y, i in loader:
        sig, mdi, y = sig.to(device), mdi.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            if train and mixup > 0 and random.random() < 0.5:
                lam = np.random.beta(mixup, mixup)
                perm = torch.randperm(sig.size(0), device=device)
                sig = lam * sig + (1 - lam) * sig[perm]
                out = model(sig, mdi)
                loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[perm])
            else:
                out = model(sig, mdi)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        tot_loss += loss.item() * sig.size(0)
        probs.append(torch.softmax(out.detach(), 1)[:, 1].cpu().numpy())
        ys.append(y.cpu().numpy())
        idxs.append(i.numpy())
    return (tot_loss / len(loader.dataset), np.concatenate(probs),
            np.concatenate(ys), np.concatenate(idxs))


def train_fold(tr_tab, va_tab, te_tab, args, device, cache, fold):
    tr = ECGDataset(tr_tab, augment=not args.no_augment, cache=cache,
                    do_filter=not args.no_filter, do_norm=not args.no_norm)
    va = ECGDataset(va_tab, augment=False, cache=cache,
                    do_filter=not args.no_filter, do_norm=not args.no_norm)
    te = ECGDataset(te_tab, augment=False, cache=cache,
                    do_filter=not args.no_filter, do_norm=not args.no_norm)

    dl = lambda ds, sh: tdata.DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                         num_workers=args.workers, drop_last=False)
    tr_dl, va_dl, te_dl = dl(tr, True), dl(va, False), dl(te, False)

    model = RecurrenceNet(width=args.width, use_mdi=not args.no_mdi,
                          dropout=args.dropout).to(device)

    # class weighting: inverse frequency, normalised to mean 1
    if args.class_weight:
        counts = np.bincount(tr_tab["label"].values, minlength=2).astype(float)
        w = counts.sum() / (2.0 * np.maximum(counts, 1))
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score, best_state, best_epoch, patience = -np.inf, None, -1, 0
    for ep in range(args.epochs):
        run_epoch(model, tr_dl, device, criterion, optimizer, mixup=args.mixup)
        _, vp, vy, _ = run_epoch(model, va_dl, device, criterion)
        # select on PR-AUC: the imbalance-aware criterion. NEVER the test fold.
        score = (average_precision_score(vy, vp) if len(np.unique(vy)) > 1
                 else -np.inf)
        if score > best_score:
            best_score, best_epoch, patience = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if args.patience and patience >= args.patience:
                break
        sched.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    # tune the decision threshold on VALIDATION, then apply to test unchanged
    _, vp, vy, _ = run_epoch(model, va_dl, device, criterion)
    thr = 0.5
    if args.tune_threshold and len(np.unique(vy)) > 1:
        cand = np.unique(np.round(vp, 3))
        f1s = [f1_score(vy, (vp >= c).astype(int), zero_division=0) for c in cand]
        thr = float(cand[int(np.argmax(f1s))])

    _, tp, ty, ti = run_epoch(model, te_dl, device, criterion)
    print(f"  fold {fold}: best epoch {best_epoch + 1} (val PR-AUC {best_score:.3f}), "
          f"threshold {thr:.3f}")
    return tp, ty, ti, thr, best_epoch + 1


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="path to a train-test-folder style directory")
    src.add_argument("--manifest", help="path to manifest.csv from prepare_dataset.py")
    ap.add_argument("--out", default="results_recurrence")
    ap.add_argument("--sites", nargs="*", default=SITES)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--mixup", type=float, default=0.0, help="beta alpha; 0 disables")
    ap.add_argument("--patience", type=int, default=15, help="0 disables early stop")
    ap.add_argument("--class-weight", action="store_true", default=True)
    ap.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    ap.add_argument("--tune-threshold", action="store_true", default=True)
    ap.add_argument("--no-tune-threshold", dest="tune_threshold", action="store_false")
    ap.add_argument("--no-mdi", action="store_true", help="ablate the MDI feature")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--patient-id", default="auto",
                    choices=["auto", "parent", "stem", "regex"])
    ap.add_argument("--patient-pattern", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanity-check", action="store_true",
                    help="print dataset/fold composition and exit without training")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.manifest:
        df = build_index_from_manifest(args.manifest)
    else:
        df = build_index(args.data, args.patient_id, args.patient_pattern, args.sites)
    report_index(df)

    # ---- patient-level stratified folds -----------------------------------
    folds = patient_level_folds(df, args.folds, args.seed)

    comp = []
    for k, (tr_i, te_i) in enumerate(folds, 1):
        te = df.iloc[te_i].drop_duplicates("patient")
        tr = df.iloc[tr_i].drop_duplicates("patient")
        comp.append({
            "fold": k,
            "test_patients": len(te),
            "test_recordings": len(te_i),
            "test_cured": int((te.label == 0).sum()),
            "test_recurrence": int((te.label == 1).sum()),
            "test_recurrence_pct": round(100 * (te.label == 1).mean(), 1),
            "train_patients": len(tr),
            "train_recordings": len(tr_i),
        })
    comp_df = pd.DataFrame(comp)
    print("\n=== FOLD COMPOSITION (Supplementary Table S2) ===")
    print(comp_df.to_string(index=False))

    # hard check: no patient in both sides of any fold
    for k, (tr_i, te_i) in enumerate(folds, 1):
        overlap = set(df.iloc[tr_i]["patient"]) & set(df.iloc[te_i]["patient"])
        assert not overlap, f"LEAKAGE in fold {k}: {overlap}"
    print("\nleakage check passed: no patient appears in train and test of the same fold")

    comp_df.to_csv(os.path.join(args.out, "fold_composition.csv"), index=False)
    if args.sanity_check:
        print("\n--sanity-check set; stopping before training.")
        return

    # ---- cross-validation --------------------------------------------------
    cache, rows, oof = {}, [], np.full(len(df), np.nan)
    print(f"\n=== TRAINING ({args.folds}-fold, device={device}) ===")
    for k, (tr_i, te_i) in enumerate(folds, 1):
        tr_all = df.iloc[tr_i]
        # inner validation split, also patient-grouped
        inner = patient_level_folds(tr_all.reset_index(drop=True), 5, args.seed)
        i_tr, i_va = inner[0]
        p, y, idx, thr, ep = train_fold(tr_all.iloc[i_tr], tr_all.iloc[i_va],
                                        df.iloc[te_i], args, device, cache, k)
        oof[te_i[idx]] = p
        m = compute_metrics(y, p, thr)
        m.update(fold=k, threshold=thr, best_epoch=ep)
        rows.append(m)

    fold_df = pd.DataFrame(rows)[
        ["fold", "n", "n_pos", "best_epoch", "threshold", "accuracy",
         "precision", "recall", "f1", "auc", "pr_auc"]]
    fold_df.to_csv(os.path.join(args.out, "fold_metrics.csv"), index=False)

    # ---- pooled out-of-fold, recording and patient level --------------------
    df = df.assign(prob=oof)
    df.to_csv(os.path.join(args.out, "oof_predictions.csv"), index=False)
    pat_oof = df.groupby(["patient", "label"], as_index=False)["prob"].mean()

    thr_mean = float(fold_df["threshold"].mean())
    rec_m = compute_metrics(df["label"].values, df["prob"].values, thr_mean)
    pat_m = compute_metrics(pat_oof["label"].values, pat_oof["prob"].values, thr_mean)

    py, pp = pat_oof["label"].values, pat_oof["prob"].values
    auc_lo, auc_hi = bootstrap_ci(py, pp, roc_auc_score, seed=args.seed)
    pr_lo, pr_hi = bootstrap_ci(py, pp, average_precision_score, seed=args.seed)

    summary = {
        "config": vars(args),
        "n_recordings": int(len(df)),
        "n_patients": int(pat_oof.shape[0]),
        "n_recurrence_patients": int(py.sum()),
        "per_fold_mean_95ci": {
            m: dict(zip(("mean", "lo", "hi"), mean_ci(fold_df[m])))
            for m in ["accuracy", "precision", "recall", "f1", "auc", "pr_auc"]},
        "pooled_oof_recording_level": rec_m,
        "pooled_oof_patient_level": pat_m,
        "patient_level_bootstrap_ci": {
            "auc": [auc_lo, auc_hi], "pr_auc": [pr_lo, pr_hi]},
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- report ------------------------------------------------------------
    print("\n=== PER-FOLD METRICS (Supplementary Table S3) ===")
    print(fold_df.round(4).to_string(index=False))
    print("\n=== POOLED: mean across folds (95% CI) ===")
    for m in ["accuracy", "precision", "recall", "f1", "auc", "pr_auc"]:
        mu, lo, hi = mean_ci(fold_df[m])
        print(f"  {m:<10} {mu:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
    print("\n=== POOLED OUT-OF-FOLD, PATIENT LEVEL (report this as primary) ===")
    for k2, v in pat_m.items():
        print(f"  {k2:<10} {v if isinstance(v, int) else round(v, 4)}")
    print(f"  AUC    95% CI (bootstrap): {auc_lo:.4f}-{auc_hi:.4f}")
    print(f"  PR-AUC 95% CI (bootstrap): {pr_lo:.4f}-{pr_hi:.4f}")
    print(f"\nPrevalence (no-skill PR-AUC baseline): {py.mean():.4f}")
    print(f"\nArtifacts written to {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
