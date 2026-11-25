"""
Reproduce the unipolar-voltage / PVC-ablation-recurrence analysis.

De-identified: this script contains no patient identifiers. Dataset-specific
value corrections and spelling aliases (if any) are read from an OPTIONAL,
git-ignored `corrections.json` (see corrections.example.json); without it the
raw inputs are used as-is. Duplicate study folders are collapsed by normalized
key automatically.

Inputs (NOT shared in this repository; identifiable):
  mapping_features.csv  -- output of src/carto_mapping_features.py (one row/patient)
  clinical table (.xlsx/.csv) -- outcome + localisation + covariates; columns are
                               matched by case-insensitive substring (see DATA_DICTIONARY.md)

Outputs:
  results.json, table1.json
Usage:
  python analyze_recurrence.py mapping_features.csv clinical.xlsx
"""
import sys, csv, re, json, unicodedata
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from firth import firth_logit, or_ci

MAP_CSV = sys.argv[1] if len(sys.argv) > 1 else "mapping_features.csv"
CLIN    = sys.argv[2] if len(sys.argv) > 2 else "clinical.xlsx"
rng = np.random.default_rng(20260115)

def norm(s):
    if s is None: return ""
    s = str(s).strip()
    tr = {"ç":"c","Ç":"c","ğ":"g","Ğ":"g","ı":"i","İ":"i","ö":"o","Ö":"o","ş":"s","Ş":"s","ü":"u","Ü":"u"}
    s = "".join(tr.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()

# optional, git-ignored dataset-specific corrections
try:
    CFG = json.load(open("corrections.json"))
except FileNotFoundError:
    CFG = {}
VFIX = {norm(k): v for k, v in CFG.get("voltage_fixes", {}).items()}
ALIAS = {norm(k): norm(v) for k, v in CFG.get("name_aliases", {}).items()}

# ---- mapping features (apply optional value fixes, then de-duplicate) ----
with open(MAP_CSV) as f:
    rows = [r for r in csv.DictReader(f) if r["patient_folder"].strip()]
for r in rows:
    k = norm(r["patient_folder"])
    if k in VFIX:
        for col, val in VFIX[k].items():
            r[col] = str(val)
mp = {}
for r in rows:
    mp.setdefault(norm(r["patient_folder"]), r)   # collapse duplicate study folders

# ---- clinical table ----
def load_clinical(path):
    if path.lower().endswith((".xlsx", ".xlsm")):
        import openpyxl
        ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
        data = list(ws.iter_rows(values_only=True))
        hdr = list(data[0]); return hdr, [dict(zip(hdr, r)) for r in data[1:] if r[0]]
    with open(path) as f:
        rd = csv.DictReader(f); return rd.fieldnames, list(rd)
hdr, clin = load_clinical(CLIN)
def col(sub, required=True):
    for h in hdr:
        if norm(sub) in norm(h): return h
    if required: raise KeyError(f"column matching {sub!r} not found")
    return None
NAME = col("isim", required=False) or col("name")
REC, LOC = col("recurrence"), col("localisation")
AF = col("fibrillation", required=False)
EF = col("ejection", required=False)
BURD = col("burden with 24", required=False) or col("burden", required=False)
NMORPH = col("morpholog", required=False)

def fnum(v):
    try: return float(v)
    except: return None
def grp(L):
    L = str(L).strip().upper()
    if L == "RVOT": return "RVOT"
    if L in ("LVOT","LCC","RCC","NCC","AMC","LV SUMMIT"): return "otherOT"
    if L == "INTRAMURAL": return "intramural"
    return "nonOT"

D = []
for crow in clin:
    k = norm(crow[NAME]); k = ALIAS.get(k, k)
    if k not in mp: continue
    m = mp[k]
    D.append(dict(
        y = 1 if str(crow[REC]).strip() in ("1", "1.0") else 0,
        uni = float(m["uni_at_earliest"]), bip = float(m["bip_at_earliest"]),
        latE = float(m["earliest_LAT"]), g = grp(crow[LOC]),
        af = fnum(crow[AF]) if AF else None,
        ef = fnum(crow[EF]) if EF else None,
        burden = fnum(crow[BURD]) if BURD else None,
        nmorph = fnum(crow[NMORPH]) if NMORPH else None))
n = len(D); events = sum(d["y"] for d in D)
y = np.array([d["y"] for d in D]); uni = np.array([d["uni"] for d in D])
print(f"n={n} events={events} ({100*events/n:.1f}%)")

RVOT = np.array([d["g"]=="RVOT" for d in D], float); otherOT = np.array([d["g"]=="otherOT" for d in D], float)
intram = np.array([d["g"]=="intramural" for d in D], float); nonOT = np.array([d["g"]=="nonOT" for d in D], float)

def boot_ci(score, yy, nb=5000):
    a = []; idx = np.arange(len(yy))
    for _ in range(nb):
        s = rng.choice(idx, len(idx), True)
        if len(np.unique(yy[s])) < 2: continue
        a.append(roc_auc_score(yy[s], score[s]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

sc = -uni; A = roc_auc_score(y, sc); Alo, Ahi = boot_ci(sc, y)
uni_rec = [d["uni"] for d in D if d["y"]==1]; uni_nor = [d["uni"] for d in D if d["y"]==0]
mr = np.percentile(uni_rec, [50,25,75]); mn = np.percentile(uni_nor, [50,25,75])
mw = stats.mannwhitneyu(uni_rec, uni_nor, alternative="two-sided").pvalue
qe = np.quantile(uni, [0,.25,.5,.75,1.]); qi = np.clip(np.digitize(uni, qe[1:-1]), 0, 3)
quart = [[int((qi==q).sum()), int(y[qi==q].sum()), float(y[qi==q].mean())] for q in range(4)]
def ca_trend(yy, gi, ng):
    N = len(yy); Rt = np.array([yy[gi==g].sum() for g in range(ng)]); Nt = np.array([(gi==g).sum() for g in range(ng)])
    sb = (Nt*np.arange(ng)).sum()/N; T = (Rt*np.arange(ng)).sum() - yy.sum()*sb; pb = yy.sum()/N
    return float(2*stats.norm.sf(abs(T/np.sqrt(pb*(1-pb)*((Nt*(np.arange(ng)-sb)**2).sum())))))
trend_p = ca_trend(y, qi, 4)

def report(cols, names):
    X = np.column_stack(cols); b, se, p = firth_logit(X, y); out = {}
    for i, nm in enumerate(names):
        idx = i+1
        if nm == "uni":
            o = or_ci(b, se, idx, -1, 1); o5 = or_ci(b, se, idx, -1, 5)
            out["uni"] = [o[0],o[1],o[2],o[3]]; out["uni5"] = [o5[0],o5[1],o5[2],o5[3]]
        else:
            o = or_ci(b, se, idx, 1, 1); out[nm] = [o[0],o[1],o[2],o[3]]
    return out
uF = report([uni], ["uni"])
M1 = report([uni, nonOT], ["uni","nonOT"])
M2 = report([uni, intram, nonOT], ["uni","intramural","nonOT"])
M3 = report([uni, otherOT, intram, nonOT], ["uni","otherOT","intramural","nonOT"])
af = np.array([d["af"] if d["af"] is not None else 0 for d in D], float)
nm_ = np.array([d["nmorph"] if d["nmorph"] is not None else np.nan for d in D])
nm_f = np.where(np.isnan(nm_), np.nanmedian(nm_), nm_)
MS = report([uni, otherOT, intram, nonOT, af, (nm_f-nm_f.mean())/nm_f.std()],
            ["uni","otherOT","intramural","nonOT","AF","nmorph"])

def cv_rep(featfun, reps=50):
    aucs = []
    for rr in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=rr); oof = np.zeros(n); X = featfun()
        for tr, te in skf.split(X, y):
            mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd==0] = 1
            clf = LogisticRegression(max_iter=1000).fit((X[tr]-mu)/sd, y[tr]); oof[te] = clf.predict_proba((X[te]-mu)/sd)[:,1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), (float(np.percentile(aucs,2.5)), float(np.percentile(aucs,97.5)))
of = lambda: np.column_stack([otherOT, intram, nonOT]); ouf = lambda: np.column_stack([otherOT, intram, nonOT, uni])
cv_o, _ = cv_rep(of); cv_ou, ci_ou = cv_rep(ouf)
Xo = of(); clf = LogisticRegression(max_iter=1000).fit((Xo-Xo.mean(0))/(Xo.std(0)+1e-9), y)
origin_is = roc_auc_score(y, clf.predict_proba((Xo-Xo.mean(0))/(Xo.std(0)+1e-9))[:,1])

fpr, tpr, thr = roc_curve(y, sc); k = np.argmax(tpr-fpr); cut = -thr[k]
def opchar(c):
    low = uni <= c
    tp=int((low&(y==1)).sum()); fp=int((low&(y==0)).sum()); fn=int((~low&(y==1)).sum()); tn=int((~low&(y==0)).sum())
    return dict(cut=c, tp=tp, fp=fp, fn=fn, tn=tn,
        sens=tp/(tp+fn) if tp+fn else 0, spec=tn/(tn+fp) if tn+fp else 0,
        ppv=tp/(tp+fp) if tp+fp else 0, npv=tn/(tn+fn) if tn+fn else 0,
        rate_le=tp/(tp+fp) if tp+fp else 0, rate_gt=fn/(fn+tn) if fn+tn else 0)
ins = opchar(cut)
cutoff_table = [opchar(t) for t in sorted(set([3,4,5,round(cut,1),6,7,8]))]
skf = StratifiedKFold(5, shuffle=True, random_state=7); pred = np.zeros(n, bool)
for tr, te in skf.split(uni, y):
    f2, t2, th2 = roc_curve(y[tr], -uni[tr]); kk = np.argmax(t2-f2); pred[te] = uni[te] <= -th2[kk]
ntp=int((pred&(y==1)).sum()); nfp=int((pred&(y==0)).sum()); nfn=int((~pred&(y==1)).sum()); ntn=int((~pred&(y==0)).sum())
nested = dict(sens=ntp/(ntp+nfn), spec=ntn/(ntn+nfp), npv=ntn/(ntn+nfn))
bins=[0,2,4,6,8,10,1e9]; blab=["0-2","2-4","4-6","6-8","8-10",">10"]; rec_bins=[]
for i in range(len(bins)-1):
    sel=(uni>bins[i])&(uni<=bins[i+1]); nn=int(sel.sum()); kk=int(y[sel].sum())
    rec_bins.append(dict(label=blab[i], n=nn, rec=kk, rate=(kk/nn if nn else None)))
strat = {}
for g in ["RVOT","otherOT","intramural","nonOT"]:
    sel = np.array([d["g"]==g for d in D])
    strat[g] = float(roc_auc_score(y[sel], -uni[sel])) if len(np.unique(y[sel]))==2 else None
og = {g:[int(np.array([d["g"]==g for d in D]).sum()), int(y[np.array([d["g"]==g for d in D])].sum())] for g in ["RVOT","otherOT","intramural","nonOT"]}

json.dump(dict(n=n, events=events,
  uni=dict(AUROC=A, CI=[Alo,Ahi], mw_p=float(mw), med_rec=list(mr), med_nor=list(mn),
           mean_rec=float(np.mean(uni_rec)), mean_nor=float(np.mean(uni_nor))),
  trend_p=trend_p, quartiles=quart,
  uni_firth_uni=dict(per1=uF["uni"][0], per1_CI=[uF["uni"][1],uF["uni"][2]], p=uF["uni"][3],
                     per5=uF["uni5"][0], per5_CI=[uF["uni5"][1],uF["uni5"][2]]),
  firth=dict(M1_nonOT=M1, M2_intramural=M2, M3_4cat=M3), sensitivity=MS,
  origin_alone_AUROC=[origin_is], CV=dict(origin=cv_o, origin_plus_uni=cv_ou, CI=list(ci_ou)),
  cut=float(cut), insample={k_:ins[k_] for k_ in ["sens","spec","ppv","npv","tp","fp","fn","tn","rate_le","rate_gt"]},
  cutoff_table=cutoff_table, nestedCV=nested, rec_bins=rec_bins, per_stratum=strat, origin_groups=og),
  open("results.json","w"), indent=1)
print("wrote results.json")

def cont(label, key, dec=0):
    rec=[d[key] for d in D if d["y"]==1 and d.get(key) is not None]; nor=[d[key] for d in D if d["y"]==0 and d.get(key) is not None]; allv=[d[key] for d in D if d.get(key) is not None]
    if not (rec and nor): return None
    p = stats.mannwhitneyu(rec, nor, alternative="two-sided").pvalue
    fmt = lambda v: f"{np.percentile(v,50):.{dec}f} [{np.percentile(v,25):.{dec}f}-{np.percentile(v,75):.{dec}f}]"
    return dict(var=label, rec=fmt(rec), nor=fmt(nor), overall=fmt(allv), p=("<0.001" if p<0.001 else f"{p:.3f}"))
def cat(label, key, val=1):
    rec=[d[key] for d in D if d["y"]==1 and d.get(key) is not None]; nor=[d[key] for d in D if d["y"]==0 and d.get(key) is not None]
    if not (rec and nor): return None
    cnt = lambda v: f"{int(sum(1 for x in v if x==val))} ({100*sum(1 for x in v if x==val)/len(v):.0f}%)"
    a=sum(1 for x in rec if x==val); b=len(rec)-a; cc=sum(1 for x in nor if x==val); dd=len(nor)-cc
    p = stats.fisher_exact([[a,b],[cc,dd]])[1]
    return dict(var=label, rec=cnt(rec), nor=cnt([d[key] for d in D if d["y"]==0 and d.get(key) is not None]), overall=cnt([d[key] for d in D if d.get(key) is not None]), p=("<0.001" if p<0.001 else f"{p:.3f}"))
t1_rows = [r for r in [cat("Atrial fibrillation","af"), cont("LV ejection fraction (%)","ef",0),
           cont("PVC burden, 24-h Holter (%)","burden",0), cont("Number of PVC morphologies","nmorph",0),
           cont("Earliest-site unipolar voltage (mV)","uni",1), cont("Earliest-site bipolar voltage (mV)","bip",1),
           cont("Earliest-site relative LAT (ms)","latE",0)] if r]
json.dump(dict(n=n, n_rec=events, n_nor=n-events, rows=t1_rows), open("table1.json","w"), indent=1)
print("wrote table1.json")
