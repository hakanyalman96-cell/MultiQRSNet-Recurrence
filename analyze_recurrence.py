"""Reproduces every statistic in the JICE revision of "Low unipolar voltage at the earliest
activation site is associated with recurrence after catheter ablation of premature ventricular
complexes" (Firth models, post hoc sensitivity models, repeated cross-validation with bootstrap CIs,
single 5.4 mV cut-off operating characteristics, Table 1).

De-identified: no patient identifiers appear in this code. Dataset-specific value corrections and
spelling aliases are read from an OPTIONAL, git-ignored corrections.json (see corrections.example.json).
Inputs are not shared (identifiable): mapping_features.csv and the clinical table.
Usage: python analyze_recurrence.py mapping_features.csv clinical.xlsx  ->  results.json, table1.json
"""
import sys, csv, re, json, unicodedata
import numpy as np
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import openpyxl
from firth import firth_logit, or_ci
rng = np.random.default_rng(20260115)

def norm(s):
    if s is None: return ''
    s=str(s).strip();tr={'ç':'c','Ç':'c','ğ':'g','Ğ':'g','ı':'i','İ':'i','ö':'o','Ö':'o','ş':'s','Ş':'s','ü':'u','Ü':'u'}
    s=''.join(tr.get(ch,ch) for ch in s);s=unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode()
    return re.sub(r'\s+',' ',s).strip().lower()
MAP_CSV = sys.argv[1] if len(sys.argv) > 1 else "mapping_features.csv"
CLIN    = sys.argv[2] if len(sys.argv) > 2 else "clinical.xlsx"
# Optional, git-ignored dataset-specific cleaning (no identifiers in code): see corrections.example.json
try: CFG=json.load(open('corrections.json'))
except FileNotFoundError: CFG={}
VFIX={norm(k):v for k,v in CFG.get('voltage_fixes',{}).items()}
ALIAS={norm(k):norm(v) for k,v in CFG.get('name_aliases',{}).items()}

# mapping features (apply optional value fixes, then collapse duplicate study folders)
with open(MAP_CSV) as f: rows=[r for r in csv.DictReader(f) if r['patient_folder'].strip()]
for r in rows:
    k=norm(r['patient_folder'])
    for colname,val in VFIX.get(k,{}).items(): r[colname]=str(val)
mp={}
for r in rows: mp.setdefault(norm(r['patient_folder']), r)

def load_clinical(path):
    if path.lower().endswith(('.xlsx','.xlsm')):
        ws=openpyxl.load_workbook(path,data_only=True).worksheets[0]
        data=list(ws.iter_rows(values_only=True)); hdr=list(data[0]); return hdr,[dict(zip(hdr,r)) for r in data[1:] if r[0]]
    with open(path) as f:
        rd=csv.DictReader(f); return rd.fieldnames,list(rd)
hdr,clin=load_clinical(CLIN)
def col(sub):
    for h in hdr:
        if norm(sub) in norm(h): return h
    raise KeyError(sub)
NAME=col('isim') if any('isim' in norm(h) for h in hdr) else col('name')
REC=col('Recurrence');LOC=col('Localisation');AGE=col('yas');SEX=col('cinsiyet')
HT=col('hypertension');DM=col('Diabetes');CAD=col('Coronary Artery');AF=col('Atrial fibrillation')
EF=col('Preprocedural Left Ventricular Ejection');BURD=col('PVC Burden with 24');QRS=col('PVC QRS duration')
NMORPH=col('Numbers of PVC morphology');COUP=col('PVC Coupling interval')
def fnum(v):
    try: return float(v)
    except: return None
def grp(L):
    L=str(L).strip().upper()
    if L=='RVOT': return 'RVOT'
    if L in ('LVOT','LCC','RCC','NCC','AMC','LV SUMMIT'): return 'otherOT'
    if L=='INTRAMURAL': return 'intramural'
    return 'nonOT'
D=[]
for c in clin:
    k=norm(c[NAME]); k=ALIAS.get(k,k)
    if k not in mp: continue
    m=mp[k]
    D.append(dict(y=1 if str(c[REC]).strip() in('1','1.0') else 0, uni=float(m['uni_at_earliest']),
        bip=float(m['bip_at_earliest']), latE=float(m['earliest_LAT']), g=grp(c[LOC]),
        age=fnum(c[AGE]),sex=fnum(c[SEX]),ht=fnum(c[HT]),dm=fnum(c[DM]),cad=fnum(c[CAD]),af=fnum(c[AF]),
        ef=fnum(c[EF]),burden=fnum(c[BURD]),qrs=fnum(c[QRS]),nmorph=fnum(c[NMORPH]),coup=fnum(c[COUP])))
n=len(D); events=sum(d['y'] for d in D)
y=np.array([d['y'] for d in D]); uni=np.array([d['uni'] for d in D])
print(f"n={n} events={events} ({100*events/n:.1f}%)")
RVOT=np.array([d['g']=='RVOT' for d in D],float);otherOT=np.array([d['g']=='otherOT' for d in D],float)
intram=np.array([d['g']=='intramural' for d in D],float);nonOT=np.array([d['g']=='nonOT' for d in D],float)

def boot_ci(score,yy,nb=5000):
    a=[];idx=np.arange(len(yy))
    for _ in range(nb):
        s=rng.choice(idx,len(idx),True)
        if len(np.unique(yy[s]))<2: continue
        a.append(roc_auc_score(yy[s],score[s]))
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
sc=-uni; A=roc_auc_score(y,sc); Alo,Ahi=boot_ci(sc,y)
uni_rec=[d['uni'] for d in D if d['y']==1]; uni_nor=[d['uni'] for d in D if d['y']==0]
mr=np.percentile(uni_rec,[50,25,75]); mn=np.percentile(uni_nor,[50,25,75])
mw=stats.mannwhitneyu(uni_rec,uni_nor,alternative='two-sided').pvalue
# quartiles
qe=np.quantile(uni,[0,.25,.5,.75,1.0]); qi=np.clip(np.digitize(uni,qe[1:-1]),0,3)
quart=[[int((qi==q).sum()),int(y[qi==q].sum()),float(y[qi==q].mean())] for q in range(4)]
def ca_trend(yy,gi,ng):
    N=len(yy);Rt=np.array([yy[gi==g].sum() for g in range(ng)]);Nt=np.array([(gi==g).sum() for g in range(ng)])
    sb=(Nt*np.arange(ng)).sum()/N;T=(Rt*np.arange(ng)).sum()-yy.sum()*sb;pb=yy.sum()/N
    return float(2*stats.norm.sf(abs(T/np.sqrt(pb*(1-pb)*((Nt*(np.arange(ng)-sb)**2).sum())))))
trend_p=ca_trend(y,qi,4)

def report(cols,names):
    X=np.column_stack(cols);b,se,p=firth_logit(X,y);out={}
    for i,nm in enumerate(names):
        idx=i+1
        if nm=='uni':
            o=or_ci(b,se,idx,-1,1);o5=or_ci(b,se,idx,-1,5)
            out['uni']=[o[0],o[1],o[2],o[3]];out['uni5']=[o5[0],o5[1],o5[2],o5[3]]
        else:
            o=or_ci(b,se,idx,1,1);out[nm]=[o[0],o[1],o[2],o[3]]
    return out
uF=report([uni],['uni'])
M1=report([uni,nonOT],['uni','nonOT'])
M2=report([uni,intram,nonOT],['uni','intramural','nonOT'])
M3=report([uni,otherOT,intram,nonOT],['uni','otherOT','intramural','nonOT'])

# NEW: sensitivity model adjusting for AF + number of PVC morphologies (reviewer request)
af=np.array([d['af'] if d['af'] is not None else 0 for d in D],float)
nm_=np.array([d['nmorph'] if d['nmorph'] is not None else np.nan for d in D])
nm_f=np.where(np.isnan(nm_),np.nanmedian(nm_),nm_)
Msens=report([uni,otherOT,intram,nonOT,af,(nm_f-nm_f.mean())/nm_f.std()],['uni','otherOT','intramural','nonOT','AF','nmorph'])
cp_=np.array([d['coup'] if d['coup'] is not None else np.nan for d in D])
cp_f=np.where(np.isnan(cp_),np.nanmedian(cp_),cp_)
n_cp_missing=int(np.isnan(cp_).sum()); n_nm_missing=int(np.isnan(nm_).sum())
Msens2=report([uni,otherOT,intram,nonOT,af,(nm_f-nm_f.mean())/nm_f.std(),(cp_f-cp_f.mean())/cp_f.std()],['uni','otherOT','intramural','nonOT','AF','nmorph','coup'])
# coupling interval by origin group (median) and Kruskal-Wallis
cp_by_origin={g:float(np.nanmedian(cp_[np.array([d['g']==g for d in D])])) for g in ['RVOT','otherOT','intramural','nonOT']}
cp_kw_p=float(stats.kruskal(*[cp_[np.array([d['g']==g for d in D])&~np.isnan(cp_)] for g in ['RVOT','otherOT','intramural','nonOT']]).pvalue)
print("S2 (+coupling) uni OR:",[round(x,2) for x in Msens2['uni'][:3]],"p=%.4f"%Msens2['uni'][3],"| coup OR/SD",[round(x,2) for x in Msens2['coup'][:3]],"p=%.3f"%Msens2['coup'][3],"| AF p=%.3f"%Msens2['AF'][3],"| missing coup",n_cp_missing,"nmorph",n_nm_missing)
print("coupling by origin",cp_by_origin,"KW p=%.4f"%cp_kw_p)
print("Sensitivity (origin+uni+AF+morph) uni OR:",[round(x,2) for x in Msens['uni'][:3]],"p=%.4f"%Msens['uni'][3],
      "| AF OR",[round(x,2) for x in Msens['AF'][:3]],"p=%.3f"%Msens['AF'][3],
      "| morph OR",[round(x,2) for x in Msens['nmorph'][:3]],"p=%.3f"%Msens['nmorph'][3])

# CV AUC
REPS=100
def cv_rep(featfun,reps=REPS):
    aucs=[];oofs=[]
    for rr in range(reps):
        skf=StratifiedKFold(5,shuffle=True,random_state=rr);oof=np.zeros(n);X=featfun()
        for tr,te in skf.split(X,y):
            mu=X[tr].mean(0);sd=X[tr].std(0);sd[sd==0]=1
            clf=LogisticRegression(max_iter=1000).fit((X[tr]-mu)/sd,y[tr]);oof[te]=clf.predict_proba((X[te]-mu)/sd)[:,1]
        aucs.append(roc_auc_score(y,oof));oofs.append(oof)
    return np.array(aucs),oofs
of=lambda:np.column_stack([otherOT,intram,nonOT]);ouf=lambda:np.column_stack([otherOT,intram,nonOT,uni])
a_o,O_o=cv_rep(of);a_ou,O_ou=cv_rep(ouf);dlt=a_ou-a_o
cv_o=float(a_o.mean());cv_ou=float(a_ou.mean())
ci_o=[float(np.percentile(a_o,2.5)),float(np.percentile(a_o,97.5))];ci_ou=[float(np.percentile(a_ou,2.5)),float(np.percentile(a_ou,97.5))]
# sampling uncertainty: paired patient-level bootstrap (1000) of each repetition's out-of-fold predictions;
# report the median 2.5/97.5 bounds across the 100 repetitions
rb=np.random.default_rng(11);idx=np.arange(n);L={'o':[],'ou':[],'d':[]};U={'o':[],'ou':[],'d':[]}
for r in range(REPS):
    bo=[];bu=[];bd=[]
    for _ in range(1000):
        ss=rb.choice(idx,n,True)
        if len(np.unique(y[ss]))<2: continue
        x1=roc_auc_score(y[ss],O_o[r][ss]);x2=roc_auc_score(y[ss],O_ou[r][ss]);bo.append(x1);bu.append(x2);bd.append(x2-x1)
    for k,v in (('o',bo),('ou',bu),('d',bd)): L[k].append(np.percentile(v,2.5));U[k].append(np.percentile(v,97.5))
boot=dict(origin=[float(np.median(L['o'])),float(np.median(U['o']))],origin_plus_uni=[float(np.median(L['ou'])),float(np.median(U['ou']))],
          delta=float(dlt.mean()),delta_CI=[float(np.median(L['d'])),float(np.median(U['d']))])
print(f"CV {REPS}x5: origin {cv_o:.3f} [rep {ci_o[0]:.3f}-{ci_o[1]:.3f}] | +uni {cv_ou:.3f} [rep {ci_ou[0]:.3f}-{ci_ou[1]:.3f}] | delta mean {dlt.mean():.3f} [rep {np.percentile(dlt,2.5):.3f}-{np.percentile(dlt,97.5):.3f}] pos {np.mean(dlt>0):.2f}")
print("bootstrap 95% CI (median across reps):",{k:(round(v,3) if isinstance(v,float) else [round(x,3) for x in v]) for k,v in boot.items()})
Xo=of();clf=LogisticRegression(max_iter=1000).fit((Xo-Xo.mean(0))/(Xo.std(0)+1e-9),y)
origin_is=roc_auc_score(y,clf.predict_proba((Xo-Xo.mean(0))/(Xo.std(0)+1e-9))[:,1])

# Youden cutoff + operating chars
fpr,tpr,thr=roc_curve(y,sc);k=np.argmax(tpr-fpr);cut_exact=float(-thr[k]);cut=round(cut_exact,1)
def opchar(c):
    low=uni<=c;tp=int((low&(y==1)).sum());fp=int((low&(y==0)).sum());fn=int((~low&(y==1)).sum());tn=int((~low&(y==0)).sum())
    return dict(cut=c,tp=tp,fp=fp,fn=fn,tn=tn,
        sens=tp/(tp+fn) if tp+fn else 0,spec=tn/(tn+fp) if tn+fp else 0,
        ppv=tp/(tp+fp) if tp+fp else 0,npv=tn/(tn+fn) if tn+fn else 0,
        rate_le=tp/(tp+fp) if tp+fp else 0,rate_gt=fn/(fn+tn) if fn+tn else 0)
ins=opchar(cut)
# NEW: operating characteristics at several thresholds (reviewer request)
thresholds=[3,4,5,cut,6,7,8]
cutoff_table=[opchar(t) for t in sorted(set(thresholds))]
for r in cutoff_table:
    print(f" cut<= {r['cut']:>4}: sens {r['sens']:.2f} spec {r['spec']:.2f} ppv {r['ppv']:.2f} npv {r['npv']:.2f} rate<= {100*r['rate_le']:.0f}% rate> {100*r['rate_gt']:.0f}%")
# nested CV
skf=StratifiedKFold(5,shuffle=True,random_state=7);pred=np.zeros(n,bool)
for tr,te in skf.split(uni,y):
    f2,t2,th2=roc_curve(y[tr],-uni[tr]);kk=np.argmax(t2-f2);pred[te]=uni[te]<=-th2[kk]
ntp=int((pred&(y==1)).sum());nfp=int((pred&(y==0)).sum());nfn=int((~pred&(y==1)).sum());ntn=int((~pred&(y==0)).sum())
nested=dict(sens=ntp/(ntp+nfn),spec=ntn/(ntn+nfp),npv=ntn/(ntn+nfn))

# NEW: observed recurrence vs unipolar voltage, fixed 2-mV bins (for the plot)
bins=[0,2,4,6,8,10,100];blab=['0–2','2–4','4–6','6–8','8–10','>10']
rec_bins=[]
for i in range(len(bins)-1):
    sel=(uni>bins[i])&(uni<=bins[i+1]);nn=int(sel.sum());kk=int(y[sel].sum())
    rec_bins.append(dict(label=blab[i],n=nn,rec=kk,rate=(kk/nn if nn else None)))

# per-stratum
strat={}
for g in ['RVOT','otherOT','intramural','nonOT']:
    sel=np.array([d['g']==g for d in D])
    strat[g]=float(roc_auc_score(y[sel],-uni[sel])) if len(np.unique(y[sel]))==2 else None
og={g:[int(np.array([d['g']==g for d in D]).sum()),int(y[np.array([d['g']==g for d in D])].sum())] for g in ['RVOT','otherOT','intramural','nonOT']}

res=dict(n=n,events=events,
 uni=dict(AUROC=A,CI=[Alo,Ahi],mw_p=float(mw),med_rec=list(mr),med_nor=list(mn),
          mean_rec=float(np.mean(uni_rec)),mean_nor=float(np.mean(uni_nor))),
 trend_p=trend_p,quartiles=quart,
 uni_firth_uni=dict(per1=uF['uni'][0],per1_CI=[uF['uni'][1],uF['uni'][2]],p=uF['uni'][3],per5=uF['uni5'][0],per5_CI=[uF['uni5'][1],uF['uni5'][2]]),
 firth=dict(M1_nonOT=M1,M2_intramural=M2,M3_4cat=M3),sensitivity=Msens,
 origin_alone_AUROC=[origin_is],CV=dict(origin=cv_o,origin_plus_uni=cv_ou,CI=list(ci_ou),CI_origin=ci_o,reps=REPS,delta_mean=float(dlt.mean()),delta_range=[float(np.percentile(dlt,2.5)),float(np.percentile(dlt,97.5))],delta_min=float(dlt.min()),frac_delta_pos=float(np.mean(dlt>0)),boot=boot),sensitivity2=Msens2,coupling_by_origin=cp_by_origin,coupling_kw_p=cp_kw_p,missing=dict(coup=n_cp_missing,nmorph=n_nm_missing),
 cut=float(cut),cut_exact=cut_exact,insample=dict(**{k:ins[k] for k in ['sens','spec','ppv','npv','tp','fp','fn','tn','rate_le','rate_gt']}),
 cutoff_table=cutoff_table,nestedCV=nested,rec_bins=rec_bins,per_stratum=strat,origin_groups=og)
json.dump(res,open('results.json','w'),indent=1)
print("WROTE results.json")

# Table 1
def cont(label,key,dec=0):
    rec=[d[key] for d in D if d['y']==1 and d[key] is not None];nor=[d[key] for d in D if d['y']==0 and d[key] is not None];allv=[d[key] for d in D if d[key] is not None]
    p=stats.mannwhitneyu(rec,nor,alternative='two-sided').pvalue if rec and nor else 1
    fmt=lambda v:f"{np.percentile(v,50):.{dec}f} [{np.percentile(v,25):.{dec}f}–{np.percentile(v,75):.{dec}f}]"
    return dict(var=label,rec=fmt(rec),nor=fmt(nor),overall=fmt(allv),p=("<0.001" if p<0.001 else f"{p:.3f}"))
def cat(label,key,val=1):
    rec=[d[key] for d in D if d['y']==1 and d[key] is not None];nor=[d[key] for d in D if d['y']==0 and d[key] is not None];allv=[d[key] for d in D if d[key] is not None]
    cnt=lambda v:f"{int(sum(1 for x in v if x==val))} ({100*sum(1 for x in v if x==val)/len(v):.0f}%)" if v else "—"
    a=sum(1 for x in rec if x==val);b=len(rec)-a;cc=sum(1 for x in nor if x==val);dd=len(nor)-cc
    p=stats.fisher_exact([[a,b],[cc,dd]])[1]
    return dict(var=label,rec=cnt(rec),nor=cnt(nor),overall=cnt(allv),p=("<0.001" if p<0.001 else f"{p:.3f}"))
# origin distribution (4 categories), overall chi-square P
labg={'RVOT':'RVOT','otherOT':'Other outflow-tract','intramural':'Presumed intramural†','nonOT':'Non-outflow-tract'}
tab=np.array([[sum(1 for d in D if d['g']==g and d['y']==yy) for yy in (0,1)] for g in labg])
orig_p=stats.chi2_contingency(tab)[1]
orig_rows=[dict(var='Arrhythmia origin',rec='',nor='',overall='',p=("<0.001" if orig_p<0.001 else f"{orig_p:.3f}"))]
for i,(g,lbl) in enumerate(labg.items()):
    nr=int(tab[i,1]);nn=int(tab[i,0])
    orig_rows.append(dict(var='   '+lbl,rec=f"{nr} ({100*nr/events:.0f}%)",nor=f"{nn} ({100*nn/(n-events):.0f}%)",overall=f"{nr+nn} ({100*(nr+nn)/n:.0f}%)",p=''))
print('origin chi2 P=%.4f'%orig_p)
t1=dict(n=n,n_rec=events,n_nor=n-events,origin_p=float(orig_p),rows=orig_rows+[
 cont('Age (years)','age',0),cat('Female sex','sex',1),cat('Hypertension','ht'),cat('Diabetes mellitus','dm'),
 cat('Coronary artery disease','cad'),cat('Atrial fibrillation','af'),cont('LV ejection fraction (%)','ef',0),
 cont('PVC burden, 24-h Holter (%)','burden',0),cont('PVC QRS duration (ms)','qrs',0),
 cont('Number of PVC morphologies','nmorph',0),cont('PVC coupling interval (ms)','coup',0),
 cont('Earliest-site unipolar voltage (mV)','uni',1),cont('Earliest-site bipolar voltage (mV)','bip',1),
 cont('Earliest-site relative LAT (ms)*','latE',0)])
json.dump(t1,open('table1.json','w'),indent=1);print("WROTE table1.json")
