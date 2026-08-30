"""
FULL-COHORT recurrence analysis (ALL VPC ablation origins, not only outflow-tract):
outflow-tract PLUS papillary-muscle, fascicular, annular, para-Hisian, etc. This
broadens the population and increases the event count/power, at the cost of mixing
heterogeneous origins (interpret as a general 'VPC-ablation recurrence' question).

Tests each electroanatomic-mapping feature's incremental value over the ABC-VT score
with patient-level 5-fold CV and a Bonferroni correction (alpha = 0.05/7). Transparent:
all features reported. Outcome = recurrence (<80% burden reduction @3mo). Exploratory.

Run:  python3 recurrence_mapping_fullcohort.py
Inputs : Kopya VES1 son.xlsx and mapping_features.csv
Outputs: recurrence_mapping_fullcohort.json, Figure_Recurrence_fullcohort.png
"""
import json, re, unicodedata, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

XLSX="Kopya VES1 son.xlsx"; MAP="mapping_features.csv"; SEED=42; N_SPLITS=5
FEATURES=["earliest_LAT","LAT_median","bip_at_earliest","uni_at_earliest","bip_min","bip_median","uni_median"]
ALPHA=0.05/len(FEATURES)
def norm(s):
    s=str(s).strip().lower().replace("ı","i").replace("İ","i")
    s=unicodedata.normalize("NFKD",s); s="".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]","",re.sub(r"\s+"," ",s)).strip()
def find(df,h):
    for x in h:
        for c in df.columns:
            if x.lower() in str(c).lower(): return c
def dauc(y,pa,pb,nb=5000,seed=SEED):
    r=np.random.default_rng(seed); n=len(y); d=[]
    for _ in range(nb):
        i=r.integers(0,n,n)
        if len(set(y[i]))<2: continue
        try: d.append(roc_auc_score(y[i],pb[i])-roc_auc_score(y[i],pa[i]))
        except: pass
    d=np.array(d); return float(np.nanpercentile(d,2.5)),float(np.nanpercentile(d,97.5)),float(2*min((d<=0).mean(),(d>=0).mean()))
def cvpred(A,X,y):
    skf=StratifiedKFold(N_SPLITS,shuffle=True,random_state=SEED); base=np.zeros(len(y)); comb=np.zeros(len(y))
    for tr,te in skf.split(A,y):
        sA=StandardScaler().fit(A[tr]); At,Ae=sA.transform(A[tr]),sA.transform(A[te])
        L=lambda a,b: LogisticRegression(penalty="l2",C=0.5,class_weight="balanced",max_iter=2000).fit(a,y[tr]).predict_proba(b)[:,1]
        base[te]=L(At,Ae)
        if X is not None:
            sX=StandardScaler().fit(X[tr]); Xt,Xe=sX.transform(X[tr]),sX.transform(X[te]); comb[te]=L(np.hstack([At,Xt]),np.hstack([Ae,Xe]))
    return base,comb

def main():
    df=pd.read_excel(XLSX)
    rec=find(df,["Recurrence"]); loc=find(df,["Localisation"]); name=find(df,["İSİM","ISIM"]); abc=find(df,["ABC-VT"])
    d=df.copy()   # NO outflow-tract filter -> all origins
    d["key"]=d[name].map(norm); d["y"]=pd.to_numeric(d[rec],errors="coerce"); d["abc"]=pd.to_numeric(d[abc],errors="coerce")
    d=d.dropna(subset=["y"])
    # show origin mix
    g=d[loc].astype(str).str.strip()
    print("recurrence by origin (top):")
    tmp=d.assign(loc=g);
    for o,sub in tmp.groupby("loc"):
        yy=sub["y"];
        if len(yy)>=3: print(f"   {o[:26]:26s} {int((yy==1).sum())}/{len(yy)}")
    m=pd.read_csv(MAP); m["key"]=m["patient_folder"].map(norm)
    for c in FEATURES: m[c]=pd.to_numeric(m[c],errors="coerce")
    d=d.merge(m[["key"]+FEATURES],on="key",how="left")
    d=d[d[FEATURES].notna().any(axis=1)].reset_index(drop=True)
    y=d["y"].astype(int).values
    A=d["abc"].fillna(d["abc"].median()).values.reshape(-1,1)
    base,_=cvpred(A,None,y); base_auc=roc_auc_score(y,base)
    print(f"\nFULL COHORT: n={len(y)}  recurrence events={int(y.sum())} ({100*y.mean():.1f}%)")
    print(f"ABC-VT baseline AUROC={base_auc:.3f} | Bonferroni threshold p<{ALPHA:.4f}\n")
    print(f"{'feature':16s} {'comb AUROC':11s} {'dAUROC':8s} {'95% CI':18s} {'p':7s} sig")
    res={"n":int(len(y)),"events":int(y.sum()),"ABC_VT_AUROC":round(base_auc,3),"alpha":round(ALPHA,4),"features":{}}
    best=None
    for c in FEATURES:
        X=d[c].fillna(d[c].median()).values.reshape(-1,1)
        _,comb=cvpred(A,X,y); ca=roc_auc_score(y,comb); da=ca-base_auc; lo,hi,p=dauc(y,base,comb)
        sig="YES" if p<ALPHA else "no"
        res["features"][c]=dict(combined_AUROC=round(ca,3),dAUROC=round(da,3),CI=[round(lo,3),round(hi,3)],p=round(p,4),survives_bonferroni=(p<ALPHA))
        print(f"{c:16s} {ca:5.3f}       {da:+.3f}   ({lo:+.3f},{hi:+.3f})    {p:.3f}  {sig}")
        if best is None or ca>best[1]: best=(c,ca,comb)
    print(f"\nBest: {best[0]} combined AUROC {best[1]:.3f}")
    plt.figure(figsize=(6,6))
    fpr,tpr,_=roc_curve(y,base); plt.plot(fpr,tpr,lw=2,color="#7f8c8d",label=f"ABC-VT: {base_auc:.2f}")
    fpr,tpr,_=roc_curve(y,best[2]); plt.plot(fpr,tpr,lw=2,color="#1f8a4c",label=f"ABC-VT + {best[0]}: {best[1]:.2f}")
    plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("1 - specificity"); plt.ylabel("Sensitivity")
    plt.title(f"Full-cohort recurrence: best mapping feature over ABC-VT ({int(y.sum())}/{len(y)})")
    plt.legend(loc="lower right",fontsize=9); plt.tight_layout(); plt.savefig("Figure_Recurrence_fullcohort.png",dpi=250)
    json.dump(res,open("recurrence_mapping_fullcohort.json","w"),indent=2)
    print("\nsaved recurrence_mapping_fullcohort.json and Figure_Recurrence_fullcohort.png")
    print("Broadened population (mixed origins) -> interpret as general VPC-ablation recurrence; still exploratory.")

if __name__=="__main__": main()
