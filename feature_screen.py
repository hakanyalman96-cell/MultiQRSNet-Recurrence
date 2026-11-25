"""
Origin-adjusted multiplicity screen of the seven candidate mapping features
(reproduces Online Resource 3). De-identified; see analyze_recurrence.py for the
input/corrections convention.

Usage: python feature_screen.py mapping_features.csv clinical.xlsx
"""
import sys, csv, re, json, unicodedata
import numpy as np
from sklearn.metrics import roc_auc_score
from firth import firth_logit

MAP_CSV = sys.argv[1] if len(sys.argv) > 1 else "mapping_features.csv"
CLIN    = sys.argv[2] if len(sys.argv) > 2 else "clinical.xlsx"

def norm(s):
    if s is None: return ""
    s = str(s).strip()
    tr = {"ç":"c","Ç":"c","ğ":"g","Ğ":"g","ı":"i","İ":"i","ö":"o","Ö":"o","ş":"s","Ş":"s","ü":"u","Ü":"u"}
    s = "".join(tr.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()

try: CFG = json.load(open("corrections.json"))
except FileNotFoundError: CFG = {}
VFIX = {norm(k): v for k, v in CFG.get("voltage_fixes", {}).items()}
ALIAS = {norm(k): norm(v) for k, v in CFG.get("name_aliases", {}).items()}

with open(MAP_CSV) as f:
    rows = [r for r in csv.DictReader(f) if r["patient_folder"].strip()]
for r in rows:
    k = norm(r["patient_folder"])
    if k in VFIX:
        for col, val in VFIX[k].items(): r[col] = str(val)
mp = {}
for r in rows: mp.setdefault(norm(r["patient_folder"]), r)

def load_clinical(path):
    if path.lower().endswith((".xlsx", ".xlsm")):
        import openpyxl
        ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
        data = list(ws.iter_rows(values_only=True)); hdr = list(data[0])
        return hdr, [dict(zip(hdr, r)) for r in data[1:] if r[0]]
    with open(path) as f:
        rd = csv.DictReader(f); return rd.fieldnames, list(rd)
hdr, clin = load_clinical(CLIN)
def col(sub):
    for h in hdr:
        if norm(sub) in norm(h): return h
    raise KeyError(sub)
NAME = col("isim") if any("isim" in norm(h) for h in hdr) else col("name")
REC, LOC = col("recurrence"), col("localisation")
def grp(L):
    L = str(L).strip().upper()
    if L == "RVOT": return "RVOT"
    if L in ("LVOT","LCC","RCC","NCC","AMC","LV SUMMIT"): return "otherOT"
    if L == "INTRAMURAL": return "intramural"
    return "nonOT"
feats = {"uni_at_earliest":"Unipolar voltage at earliest site (primary)","bip_at_earliest":"Bipolar voltage at earliest site",
         "bip_median":"Bipolar voltage, median","uni_median":"Unipolar voltage, median","bip_min":"Bipolar voltage, minimum",
         "earliest_LAT":"Earliest relative LAT","LAT_median":"Relative LAT, median"}
D = []
for c in clin:
    k = norm(c[NAME]); k = ALIAS.get(k, k)
    if k not in mp: continue
    m = mp[k]; d = dict(y=1 if str(c[REC]).strip() in ("1","1.0") else 0, g=grp(c[LOC]))
    for fk in feats: d[fk] = float(m[fk])
    D.append(d)
n = len(D); y = np.array([d["y"] for d in D])
oOT = np.array([d["g"]=="otherOT" for d in D], float); IM = np.array([d["g"]=="intramural" for d in D], float); nOT = np.array([d["g"]=="nonOT" for d in D], float)
def wins(x, lo=1, hi=99): a, b = np.percentile(x, [lo, hi]); return np.clip(x, a, b)
res = []
for fk, lbl in feats.items():
    x = wins(np.array([d[fk] for d in D])); xs = (x-x.mean())/x.std()
    b, se, p = firth_logit(np.column_stack([oOT, IM, nOT, xs]), y); res.append([lbl, b[-1], p[-1]])
res.sort(key=lambda r: r[2]); pv = [r[2] for r in res]; m = len(pv)
order = np.argsort(pv); holm = [None]*m
for rk, i in enumerate(order): holm[i] = min(1, (m-rk)*pv[i])
ob = np.argsort(pv); bh = [min(1, pv[ob[k]]*m/(k+1)) for k in range(m)]
for k in range(m-2, -1, -1): bh[k] = min(bh[k], bh[k+1])
bho = [None]*m
for k, i in enumerate(ob): bho[i] = bh[k]
out = [dict(feature=r[0], coef=r[1], p=r[2], bonferroni=min(1, r[2]*m), holm=holm[i], bh=bho[i]) for i, r in enumerate(res)]
json.dump(out, open("feature_screen.json", "w"), indent=1)
print(f"n={n} events={int(y.sum())}")
for r in out:
    print(f"  {r['feature']:42s} p={r['p']:.4f} Bonf={r['bonferroni']:.3f} Holm={r['holm']:.3f} BH={r['bh']:.3f}")
print("wrote feature_screen.json")
