"""
Characterize the standalone / independent effect of UNIPOLAR VOLTAGE at the earliest
activation site (uni_at_earliest) on VPC-ablation recurrence (full cohort).

Reports:
  1. direction & effect size — mean uni voltage in recurrence vs no-recurrence,
     Mann-Whitney p, univariate AUROC, and odds ratio per 1-SD change.
  2. INDEPENDENCE — multivariable logistic recurrence ~ ABC-VT + nonOT + uni_voltage
     (is uni voltage still significant after adjusting for the clinical score AND for
     non-outflow-tract origin? i.e., is it more than an origin proxy?).
  3. discrimination — patient-level 5-fold CV AUROC: ABC-VT vs ABC-VT + uni voltage,
     with bootstrap 95% CI and calibration.
  4. figure — voltage distribution by outcome + ROC.

Exploratory / hypothesis-generating; single centre; needs external validation.

Run:  python3 recurrence_unipolar.py
Inputs : Kopya VES1 son.xlsx and mapping_features.csv
Outputs: recurrence_unipolar.json, Figure_Recurrence_unipolar.png
"""
import json, re, unicodedata, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.api as sm

XLSX="Kopya VES1 son.xlsx"; MAP="mapping_features.csv"; SEED=42; N_SPLITS=5; FEAT="uni_at_earliest"
def norm(s):
    s=str(s).strip().lower().replace("ı","i").replace("İ","i")
    s=unicodedata.normalize("NFKD",s); s="".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]","",re.sub(r"\s+"," ",s)).strip()
def find(df,h):
    for x in h:
        for c in df.columns:
            if x.lower() in str(c).lower(): return c
def ci(y,p,fn,nb=5000,seed=SEED):
    r=np.random.default_rng(seed); n=len(y); o=[]
    for _ in range(nb):
        i=r.integers(0,n,n)
        if len(set(y[i]))<2: continue
        try: o.append(fn(y[i],p[i]))
        except: pass
    return (np.nanpercentile(o,2.5),np.nanpercentile(o,97.5)) if o else (np.nan,np.nan)

def main():
    df=pd.read_excel(XLSX)
    rec=find(df,["Recurrence"]); loc=find(df,["Localisation"]); name=find(df,["İSİM","ISIM"]); abc=find(df,["ABC-VT"])
    d=df.copy(); d["key"]=d[name].map(norm); d["y"]=pd.to_numeric(d[rec],errors="coerce"); d["abc"]=pd.to_numeric(d[abc],errors="coerce")
    g=d[loc].astype(str).str.strip().str.upper(); OT={"RVOT","LVOT","LCC","RCC","NCC","AMC","LV SUMMIT","INTRAMURAL"}
    d["nonOT"]=(~g.isin(OT)).astype(int)
    d=d.dropna(subset=["y"])
    m=pd.read_csv(MAP); m["key"]=m["patient_folder"].map(norm); m[FEAT]=pd.to_numeric(m[FEAT],errors="coerce")
    d=d.merge(m[["key",FEAT]],on="key",how="left").dropna(subset=[FEAT]).reset_index(drop=True)
    y=d["y"].astype(int).values; v=d[FEAT].values
    print(f"n={len(y)}  events={int(y.sum())} ({100*y.mean():.1f}%)")

    # 1. direction & effect size
    v1=v[y==1]; v0=v[y==0]
    U,pmw=stats.mannwhitneyu(v1,v0)
    au=roc_auc_score(y,-v); au=au  # negative if lower voltage -> recurrence
    lo,hi=ci(y,-v,roc_auc_score)
    z=(v-v.mean())/v.std()
    orsd=sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    b=orsd.params[1]; se=orsd.bse[1]
    print(f"\n1) DIRECTION / EFFECT SIZE ({FEAT}):")
    print(f"   recurrence mean {v1.mean():.3f} mV vs no-recurrence {v0.mean():.3f} mV  (Mann-Whitney p={pmw:.3f})")
    print(f"   univariate AUROC (lower voltage -> recurrence): {au:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    print(f"   OR per +1 SD unipolar voltage: {np.exp(b):.2f} (95% CI {np.exp(b-1.96*se):.2f}-{np.exp(b+1.96*se):.2f}), p={orsd.pvalues[1]:.3f}")

    # 2. independence from ABC-VT and non-OT origin
    X=d[["abc"]].copy(); X["abc"]=X["abc"].fillna(X["abc"].median())
    X["nonOT"]=d["nonOT"].values; X["uni_z"]=z
    Xc=sm.add_constant(X)
    mv=sm.Logit(y,Xc).fit(disp=0)
    print("\n2) INDEPENDENCE (multivariable logistic: recurrence ~ ABC-VT + nonOT + unipolar voltage)")
    for term in ["abc","nonOT","uni_z"]:
        b=mv.params[term]; se=mv.bse[term]
        print(f"   {term:8s} OR {np.exp(b):.2f} (95% CI {np.exp(b-1.96*se):.2f}-{np.exp(b+1.96*se):.2f}), p={mv.pvalues[term]:.3f}")
    res={"n":int(len(y)),"events":int(y.sum()),
         "univariate":{"AUROC":round(au,3),"CI":[round(lo,3),round(hi,3)],"mannwhitney_p":round(float(pmw),4),
                       "mean_recur":round(float(v1.mean()),3),"mean_norecur":round(float(v0.mean()),3)},
         "adjusted_uni_OR_perSD":round(float(np.exp(mv.params['uni_z'])),3),
         "adjusted_uni_p":round(float(mv.pvalues['uni_z']),4)}

    # 3. CV discrimination ABC-VT vs ABC-VT+uni
    A=d[["abc"]].fillna(d["abc"].median()).values; V=v.reshape(-1,1)
    skf=StratifiedKFold(N_SPLITS,shuffle=True,random_state=SEED); base=np.zeros(len(y)); comb=np.zeros(len(y))
    for tr,te in skf.split(A,y):
        sA=StandardScaler().fit(A[tr]); sV=StandardScaler().fit(V[tr])
        L=lambda a,b: LogisticRegression(penalty="l2",C=0.5,class_weight="balanced",max_iter=2000).fit(a,y[tr]).predict_proba(b)[:,1]
        base[te]=L(sA.transform(A[tr]),sA.transform(A[te]))
        comb[te]=L(np.hstack([sA.transform(A[tr]),sV.transform(V[tr])]),np.hstack([sA.transform(A[te]),sV.transform(V[te])]))
    ab=roc_auc_score(y,base); cb=roc_auc_score(y,comb); lo2,hi2=ci(y,comb,roc_auc_score)
    print(f"\n3) DISCRIMINATION (patient-level 5-fold CV):")
    print(f"   ABC-VT alone AUROC {ab:.3f}")
    print(f"   ABC-VT + unipolar voltage AUROC {cb:.3f} (95% CI {lo2:.3f}-{hi2:.3f})  Brier {brier_score_loss(y,comb):.3f}")
    res["CV"]={"ABC_VT":round(ab,3),"ABC_VT_plus_uni":round(cb,3),"CI":[round(lo2,3),round(hi2,3)]}

    fig,ax=plt.subplots(1,2,figsize=(11,5))
    ax[0].boxplot([v0,v1],labels=["no recurrence","recurrence"],showmeans=True)
    ax[0].set_ylabel(f"{FEAT} (mV)"); ax[0].set_title(f"Unipolar voltage by outcome (p={pmw:.3f})")
    fpr,tpr,_=roc_curve(y,base); ax[1].plot(fpr,tpr,color="#7f8c8d",lw=2,label=f"ABC-VT: {ab:.2f}")
    fpr,tpr,_=roc_curve(y,comb); ax[1].plot(fpr,tpr,color="#1f8a4c",lw=2,label=f"ABC-VT + unipolar: {cb:.2f}")
    ax[1].plot([0,1],[0,1],"k--",lw=1); ax[1].set_xlabel("1-spec"); ax[1].set_ylabel("sens"); ax[1].legend(loc="lower right"); ax[1].set_title("Recurrence discrimination")
    plt.tight_layout(); plt.savefig("Figure_Recurrence_unipolar.png",dpi=250)
    json.dump(res,open("recurrence_unipolar.json","w"),indent=2)
    print("\nsaved recurrence_unipolar.json and Figure_Recurrence_unipolar.png")

if __name__=="__main__": main()
