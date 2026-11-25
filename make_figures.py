"""Figures 2-4 for the JICE revision (reads results.json from analyze_recurrence.py).
Usage: python make_figures.py mapping_features.csv clinical.xlsx"""
import csv,re,json,unicodedata
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
import openpyxl
plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
BLUE="#0072B2";VERM="#D55E00";GREY="#7F7F7F";INK="#1a1a1a";GREEN="#4C8C4C"
R=json.load(open('results.json'))
def norm(s):
    if s is None:return ''
    s=str(s).strip();tr={'ç':'c','Ç':'c','ğ':'g','Ğ':'g','ı':'i','İ':'i','ö':'o','Ö':'o','ş':'s','Ş':'s','ü':'u','Ü':'u'}
    s=''.join(tr.get(ch,ch) for ch in s);s=unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode()
    return re.sub(r'\s+',' ',s).strip().lower()
import sys
MAP_CSV = sys.argv[1] if len(sys.argv) > 1 else "mapping_features.csv"
CLIN    = sys.argv[2] if len(sys.argv) > 2 else "clinical.xlsx"
try: CFG=json.load(open('corrections.json'))
except FileNotFoundError: CFG={}
VFIX={norm(k):v for k,v in CFG.get('voltage_fixes',{}).items()}
ALIAS={norm(k):norm(v) for k,v in CFG.get('name_aliases',{}).items()}
with open(MAP_CSV) as f: rows=[r for r in csv.DictReader(f) if r['patient_folder'].strip()]
for r in rows:
    for colname,val in VFIX.get(norm(r['patient_folder']),{}).items(): r[colname]=str(val)
mp={}
for r in rows: mp.setdefault(norm(r['patient_folder']),r)
ws=openpyxl.load_workbook(CLIN,data_only=True).worksheets[0]
data=list(ws.iter_rows(values_only=True));hdr=list(data[0]);clin=[dict(zip(hdr,r)) for r in data[1:] if r[0]]
def col(sub):
    for h in hdr:
        if norm(sub) in norm(h): return h
    raise KeyError(sub)
NAME=col('isim') if any('isim' in norm(h) for h in hdr) else col('name');REC=col('Recurrence');LOC=col('Localisation')
def grp(L):
    L=str(L).strip().upper()
    if L=='RVOT':return 'RVOT'
    if L in('LVOT','LCC','RCC','NCC','AMC','LV SUMMIT'):return 'otherOT'
    if L=='INTRAMURAL':return 'intramural'
    return 'nonOT'
uni=[];y=[];og=[]
for c in clin:
    k=norm(c[NAME]);k=ALIAS.get(k,k)
    if k not in mp:continue
    uni.append(float(mp[k]['uni_at_earliest']));y.append(1 if str(c[REC]).strip() in('1','1.0') else 0);og.append(grp(c[LOC]))
uni=np.array(uni);y=np.array(y);og=np.array(og);n=len(y);cut=R['cut']
oOT=(og=='otherOT').astype(float);IM=(og=='intramural').astype(float);nOT=(og=='nonOT').astype(float)

# ---- FIG 2 ----
fig,ax=plt.subplots(1,2,figsize=(10,4.3))
gN=uni[y==0];gR=uni[y==1]
bp=ax[0].boxplot([gN,gR],positions=[1,2],widths=0.55,patch_artist=True,showmeans=True,
 meanprops=dict(marker='D',markerfacecolor='white',markeredgecolor=INK,markersize=6),
 medianprops=dict(color=INK,lw=1.6),whiskerprops=dict(color=GREY),capprops=dict(color=GREY))
for p,c in zip(bp['boxes'],["#EAF3EA","#FBE9DE"]):p.set_facecolor(c);p.set_edgecolor(INK)
for i,(g,c) in enumerate(zip([gN,gR],[BLUE,VERM])):
    x=np.random.default_rng(1).normal(i+1,0.055,len(g));ax[0].plot(x,g,'o',ms=3,color=c,alpha=0.5,zorder=3)
ax[0].axhline(cut,ls='--',color=GREY,lw=1);ax[0].text(2.47,cut+0.4,f"{cut:.1f} mV",ha='right',va='bottom',fontsize=8,color=GREY)
ax[0].set_xticks([1,2]);ax[0].set_xticklabels([f"No recurrence\n(n={int((y==0).sum())})",f"Recurrence\n(n={int((y==1).sum())})"])
ax[0].set_ylabel("Unipolar voltage at earliest site (mV)");ax[0].set_ylim(0,28);ax[0].set_title("A",loc='left',fontweight='bold')
ax[0].text(0.5,0.97,f"Mann–Whitney P = {R['uni']['mw_p']:.3f}",transform=ax[0].transAxes,ha='center',va='top',fontsize=8.5)
def oof_reps(feat,reps=100):
    out=[]
    for rr in range(reps):
        skf=StratifiedKFold(5,shuffle=True,random_state=rr);pr=np.zeros(n)
        for tr,te in skf.split(feat,y):
            mu=feat[tr].mean(0);sd=feat[tr].std(0);sd[sd==0]=1
            clf=LogisticRegression(max_iter=1000).fit((feat[tr]-mu)/sd,y[tr]);pr[te]=clf.predict_proba((feat[te]-mu)/sd)[:,1]
        out.append(pr)
    return out
Po=oof_reps(np.column_stack([oOT,IM,nOT]));Pou=oof_reps(np.column_stack([oOT,IM,nOT,uni]))
ao=np.array([roc_auc_score(y,p) for p in Po]);aou=np.array([roc_auc_score(y,p) for p in Pou])
rstar=int(np.argmin(np.abs(ao-ao.mean())+np.abs(aou-aou.mean())))   # repetition closest to the mean
print("Fig2B representative repetition",rstar,round(ao[rstar],3),round(aou[rstar],3))
for p,lab,c in [(Po[rstar],f"Origin alone (mean AUC {R['CV']['origin']:.2f})",GREY),(Pou[rstar],f"Origin + unipolar (mean AUC {R['CV']['origin_plus_uni']:.2f})",BLUE)]:
    fpr,tpr,_=roc_curve(y,p);ax[1].plot(fpr,tpr,color=c,lw=2,label=lab)
ax[1].plot([0,1],[0,1],ls=':',color=GREY,lw=1);ax[1].set_xlabel("1 − specificity");ax[1].set_ylabel("Sensitivity")
ax[1].set_title("B",loc='left',fontweight='bold');ax[1].legend(loc='lower right',fontsize=8,frameon=False);ax[1].set_aspect('equal')
fig.tight_layout();fig.savefig("Figure_2.png",dpi=400,bbox_inches='tight');fig.savefig("Figure_2.pdf",bbox_inches='tight');print("Fig2 saved")

# ---- FIG 3 ----
fig,ax=plt.subplots(1,2,figsize=(10,4.3))
qs=R['quartiles'];rates=[q[2] for q in qs];ns=[q[0] for q in qs];ks=[q[1] for q in qs]
lo=[stats.beta.ppf(.025,k+.5,nn-k+.5) for nn,k in zip(ns,ks)];hi=[stats.beta.ppf(.975,k+.5,nn-k+.5) for nn,k in zip(ns,ks)]
xp=np.arange(4)
ax[0].bar(xp,[r*100 for r in rates],color=["#B23B3B","#D98C5F","#9FB89F","#4C8C4C"],edgecolor=INK,width=.68)
ax[0].errorbar(xp,[r*100 for r in rates],yerr=[[(r-l)*100 for r,l in zip(rates,lo)],[(h-r)*100 for r,h in zip(rates,hi)]],fmt='none',ecolor=INK,capsize=4,lw=1)
ax[0].set_xticks(xp);ax[0].set_xticklabels([f"Q1\n(lowest)\nn={ns[0]}",f"Q2\nn={ns[1]}",f"Q3\nn={ns[2]}",f"Q4\n(highest)\nn={ns[3]}"],fontsize=8.5)
ax[0].set_ylabel("3-month recurrence (%)");ax[0].set_title("A",loc='left',fontweight='bold');ax[0].set_ylim(0,55)
ax[0].text(0.5,0.97,f"Cochran–Armitage trend P = {R['trend_p']:.3f}",transform=ax[0].transAxes,ha='center',va='top',fontsize=8.5)
fpr,tpr,_=roc_curve(y,-uni);ax[1].plot(fpr,tpr,color=VERM,lw=2,label=f"Unipolar voltage (AUC {R['uni']['AUROC']:.2f})")
Fo=np.column_stack([oOT,IM,nOT]);clf=LogisticRegression(max_iter=1000).fit((Fo-Fo.mean(0))/(Fo.std(0)+1e-9),y)
fpr2,tpr2,_=roc_curve(y,clf.predict_proba((Fo-Fo.mean(0))/(Fo.std(0)+1e-9))[:,1]);ax[1].plot(fpr2,tpr2,color=GREY,lw=2,label=f"Origin alone (AUC {R['origin_alone_AUROC'][0]:.2f})")
pred=uni<=cut;se=(pred&(y==1)).sum()/(y==1).sum();sp=(~pred&(y==0)).sum()/(y==0).sum()
ax[1].plot(1-sp,se,'o',ms=8,color=INK,zorder=5);ax[1].annotate(f"≤{cut:.1f} mV",(1-sp,se),textcoords="offset points",xytext=(8,-10),fontsize=8)
ax[1].plot([0,1],[0,1],ls=':',color=GREY,lw=1);ax[1].set_xlabel("1 − specificity");ax[1].set_ylabel("Sensitivity")
ax[1].set_title("B",loc='left',fontweight='bold');ax[1].legend(loc='lower right',fontsize=8,frameon=False);ax[1].set_aspect('equal')
fig.tight_layout();fig.savefig("Figure_3.png",dpi=400,bbox_inches='tight');fig.savefig("Figure_3.pdf",bbox_inches='tight');print("Fig3 saved")

# ---- FIG 4: observed recurrence vs voltage + operating characteristics across thresholds ----
fig,ax=plt.subplots(1,2,figsize=(10,4.3))
rb=[b for b in R['rec_bins'] if b['n']>0]
xb=np.arange(len(rb));rates=[b['rate'] for b in rb];ns=[b['n'] for b in rb];ks=[b['rec'] for b in rb]
lo=[stats.beta.ppf(.025,k+.5,nn-k+.5) for nn,k in zip(ns,ks)];hi=[stats.beta.ppf(.975,k+.5,nn-k+.5) for nn,k in zip(ns,ks)]
ax[0].bar(xb,[r*100 for r in rates],color=BLUE,alpha=.75,edgecolor=INK,width=.72)
ax[0].errorbar(xb,[r*100 for r in rates],yerr=[[(r-l)*100 for r,l in zip(rates,lo)],[(h-r)*100 for r,h in zip(rates,hi)]],fmt='none',ecolor=INK,capsize=3,lw=1)
ax[0].set_xticks(xb);ax[0].set_xticklabels([f"{b['label']}\nn={b['n']}" for b in rb],fontsize=8)
ax[0].set_xlabel("Unipolar voltage at earliest site (mV)");ax[0].set_ylabel("Observed 3-month recurrence (%)")
ax[0].set_title("A",loc='left',fontweight='bold');ax[0].set_ylim(0,55)
ct=R['cutoff_table']
cx=[r['cut'] for r in ct]
for key,lab,c,mk in [('sens','Sensitivity',VERM,'o'),('spec','Specificity',BLUE,'s'),('ppv','PPV',GREEN,'^'),('npv','NPV',"#8856a7",'D')]:
    ax[1].plot(cx,[r[key] for r in ct],marker=mk,color=c,lw=1.6,ms=5,label=lab)
ax[1].axvline(R['cut'],ls='--',color=GREY,lw=1);ax[1].text(R['cut']+0.1,0.02,f"Youden ≤{R['cut']:.1f}",fontsize=7.5,color=GREY)
ax[1].set_xlabel("Unipolar-voltage cut-off (≤ mV)");ax[1].set_ylabel("Operating characteristic")
ax[1].set_title("B",loc='left',fontweight='bold');ax[1].set_ylim(0,1);ax[1].legend(fontsize=7.5,frameon=False,ncol=2,loc='upper center')
fig.tight_layout();fig.savefig("Figure_4.png",dpi=400,bbox_inches='tight');fig.savefig("Figure_4.pdf",bbox_inches='tight');print("Fig4 saved")
