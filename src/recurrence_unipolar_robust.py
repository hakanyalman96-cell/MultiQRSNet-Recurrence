"""
Robustness / clinical-usability add-ons for the unipolar-voltage recurrence finding
(full cohort, all VPC origins). Complements recurrence_unipolar.py.

Adds the numbers a reviewer will ask for and that make the effect interpretable:
  1. Interpretable effect size — OR per 1 mV and per 5 mV LOWER unipolar voltage,
     from a Firth penalized logistic regression (handles the near-separation that made
     the per-SD OR collapse to ~0.02 with a uninterpretable CI in the plain model).
  2. Optimal clinical cut-off (Youden J) with sensitivity/specificity/PPV/NPV and the
     recurrence rate above vs below the cut-off (with 2x2 and Fisher p).
  3. Adjusted effect — Firth logistic recurrence ~ ABC-VT + nonOT + unipolar(mV),
     confirming independence with a stable (non-collapsing) OR.
  4. Subgroup direction check — is the "lower voltage -> recurrence" direction present in
     BOTH the outflow-tract and the non-outflow-tract strata (i.e. not driven by one origin)?

Exploratory / single-centre / needs external validation.

Run:  pip install firthlogist -q ; python3 recurrence_unipolar_robust.py
Inputs : Kopya VES1 son.xlsx and mapping_features.csv
Outputs: recurrence_unipolar_robust.json
"""
import json, re, unicodedata, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

XLSX="Kopya VES1 son.xlsx"; MAP="mapping_features.csv"; FEAT="uni_at_earliest"
OT={"RVOT","LVOT","LCC","RCC","NCC","AMC","LV SUMMIT","INTRAMURAL"}

def norm(s):
    s=str(s).strip().lower().replace("ı","i").replace("İ","i")
    s=unicodedata.normalize("NFKD",s); s="".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]","",re.sub(r"\s+"," ",s)).strip()
def find(df,h):
    for x in h:
        for c in df.columns:
            if x.lower() in str(c).lower(): return c

def firth_or(y, X, colnames):
    """Firth penalized logistic; returns dict col->(OR, lo, hi, p). Falls back to
    statsmodels Logit if firthlogist is unavailable."""
    try:
        from firthlogist import FirthLogisticRegression
        m=FirthLogisticRegression(max_iter=1000)
        m.fit(np.asarray(X,float), np.asarray(y,int))
        b=m.coef_; se=m.bse_; p=m.pvals_
        return {c:(float(np.exp(b[i])),float(np.exp(b[i]-1.96*se[i])),
                   float(np.exp(b[i]+1.96*se[i])),float(p[i])) for i,c in enumerate(colnames)}
    except Exception as e:
        import statsmodels.api as sm
        r=sm.Logit(np.asarray(y,int), sm.add_constant(np.asarray(X,float))).fit(disp=0)
        b=r.params[1:]; se=r.bse[1:]; p=r.pvalues[1:]
        return {c:(float(np.exp(b[i])),float(np.exp(b[i]-1.96*se[i])),
                   float(np.exp(b[i]+1.96*se[i])),float(p[i])) for i,c in enumerate(colnames)}

def main():
    df=pd.read_excel(XLSX)
    rec=find(df,["Recurrence"]); loc=find(df,["Localisation"]); name=find(df,["İSİM","ISIM"]); abc=find(df,["ABC-VT"])
    d=df.copy(); d["key"]=d[name].map(norm)
    d["y"]=pd.to_numeric(d[rec],errors="coerce"); d["abc"]=pd.to_numeric(d[abc],errors="coerce")
    g=d[loc].astype(str).str.strip().str.upper(); d["nonOT"]=(~g.isin(OT)).astype(int)
    d=d.dropna(subset=["y"])
    m=pd.read_csv(MAP); m["key"]=m["patient_folder"].map(norm); m[FEAT]=pd.to_numeric(m[FEAT],errors="coerce")
    d=d.merge(m[["key",FEAT]],on="key",how="left").dropna(subset=[FEAT]).reset_index(drop=True)
    y=d["y"].astype(int).values; v=d[FEAT].values
    res={"n":int(len(y)),"events":int(y.sum())}
    print(f"n={len(y)}  events={int(y.sum())} ({100*y.mean():.1f}%)\n")

    # 1 + 3: interpretable + adjusted effect (Firth), coding LOWER voltage as the risk direction
    neg=-v  # so a positive coefficient == higher risk with lower voltage
    uni_only=firth_or(y, neg.reshape(-1,1), ["neg_uni_mV"])
    or1=uni_only["neg_uni_mV"]
    or5=(or1[0]**5, or1[1]**5, or1[2]**5, or1[3])
    print("1) INTERPRETABLE EFFECT (Firth, univariable):")
    print(f"   OR per 1 mV LOWER unipolar voltage : {or1[0]:.2f} ({or1[1]:.2f}-{or1[2]:.2f}), p={or1[3]:.3f}")
    print(f"   OR per 5 mV LOWER unipolar voltage : {or5[0]:.2f} ({or5[1]:.2f}-{or5[2]:.2f})")
    res["firth_OR_per_1mV_lower"]=dict(OR=round(or1[0],3),CI=[round(or1[1],3),round(or1[2],3)],p=round(or1[3],4))
    res["firth_OR_per_5mV_lower"]=dict(OR=round(or5[0],3),CI=[round(or5[1],3),round(or5[2],3)])

    Xadj=np.column_stack([d["abc"].fillna(d["abc"].median()).values, d["nonOT"].values, neg])
    adj=firth_or(y, Xadj, ["ABC_VT","nonOT","neg_uni_mV"])
    print("\n3) ADJUSTED (Firth: recurrence ~ ABC-VT + nonOT + unipolar mV):")
    for c in ["ABC_VT","nonOT","neg_uni_mV"]:
        o=adj[c]; lbl="per 1 mV lower" if c=="neg_uni_mV" else ""
        print(f"   {c:11s} OR {o[0]:.2f} ({o[1]:.2f}-{o[2]:.2f}), p={o[3]:.3f}  {lbl}")
    res["adjusted_firth"]={c:dict(OR=round(adj[c][0],3),CI=[round(adj[c][1],3),round(adj[c][2],3)],p=round(adj[c][3],4)) for c in adj}

    # 2: Youden cut-off + operating characteristics
    fpr,tpr,thr=roc_curve(y,neg); j=np.argmax(tpr-fpr); cut=-thr[j]  # back to mV
    low=v<=cut
    tp=int((low&(y==1)).sum()); fp=int((low&(y==0)).sum()); fn=int((~low&(y==1)).sum()); tn=int((~low&(y==0)).sum())
    sens=tp/(tp+fn) if tp+fn else np.nan; spec=tn/(tn+fp) if tn+fp else np.nan
    ppv=tp/(tp+fp) if tp+fp else np.nan; npv=tn/(tn+fn) if tn+fn else np.nan
    _,fisher_p=stats.fisher_exact([[tp,fp],[fn,tn]])
    rate_low=tp/(tp+fp) if tp+fp else np.nan; rate_high=fn/(fn+tn) if fn+tn else np.nan
    print(f"\n2) OPTIMAL CUT-OFF (Youden): unipolar voltage <= {cut:.2f} mV flags high risk")
    print(f"   sens {sens:.2f} | spec {spec:.2f} | PPV {ppv:.2f} | NPV {npv:.2f}")
    print(f"   recurrence rate  <= {cut:.2f} mV: {rate_low:.0%} ({tp}/{tp+fp})   vs  > {cut:.2f} mV: {rate_high:.0%} ({fn}/{fn+tn})   Fisher p={fisher_p:.4f}")
    res["cutoff_mV"]=round(float(cut),2)
    res["operating"]=dict(sens=round(sens,3),spec=round(spec,3),PPV=round(ppv,3),NPV=round(npv,3),
                          rate_low=round(float(rate_low),3),rate_high=round(float(rate_high),3),fisher_p=round(float(fisher_p),4),
                          tp=tp,fp=fp,fn=fn,tn=tn)

    # 4: subgroup direction consistency
    print("\n4) DIRECTION CONSISTENCY BY STRATUM (AUROC 'lower voltage -> recurrence'):")
    res["subgroups"]={}
    for lbl,mask in [("outflow-tract", d["nonOT"].values==0), ("non-outflow-tract", d["nonOT"].values==1)]:
        ys=y[mask]; vs=v[mask]
        if len(ys)>=10 and 0<ys.sum()<len(ys):
            au=roc_auc_score(ys,-vs)
            mr=vs[ys==1].mean() if (ys==1).any() else np.nan; mn=vs[ys==0].mean() if (ys==0).any() else np.nan
            print(f"   {lbl:18s} n={len(ys)} ev={int(ys.sum())}  AUROC {au:.3f}  (recur {mr:.1f} vs no-recur {mn:.1f} mV)")
            res["subgroups"][lbl]=dict(n=int(len(ys)),events=int(ys.sum()),AUROC=round(float(au),3),
                                       mean_recur=round(float(mr),2),mean_norecur=round(float(mn),2))
        else:
            print(f"   {lbl:18s} n={len(ys)} ev={int(ys.sum())}  (too few to estimate)")
            res["subgroups"][lbl]=dict(n=int(len(ys)),events=int(ys.sum()))

    json.dump(res,open("recurrence_unipolar_robust.json","w"),indent=2)
    print("\nsaved recurrence_unipolar_robust.json")

if __name__=="__main__": main()
