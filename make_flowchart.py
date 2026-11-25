"""Figure 1 — study flow, four mutually-exclusive origin categories as parallel branches."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R=json.load(open('results.json')); OG=R['origin_groups']
INK="#1a1a1a"; BLUE="#0072B2"; VERM="#D55E00"; GREY="#7F7F7F"; LBLUE="#DCE9F5"; LGREY="#EFEFEF"
plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
fig,ax=plt.subplots(figsize=(9.6,9.2)); ax.set_xlim(0,100); ax.set_ylim(0,116); ax.axis("off")
def box(x,y,w,h,lines,fc="white",ec=INK,fs=9.0,bold0=True,lw=1.1,align="center"):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.5,rounding_size=1.3",linewidth=lw,edgecolor=ec,facecolor=fc,zorder=2))
    ha={"center":"center","left":"left"}[align]; tx=x if align=="center" else x-w/2+2
    nn=len(lines); step=h/(nn+0.4); y0=y+h/2-step*0.85
    for i,ln in enumerate(lines):
        ax.text(tx,y0-i*step,ln,ha=ha,va="center",fontsize=fs if not(i==0 and bold0) else fs+0.3,fontweight="bold" if (i==0 and bold0) else "normal",color=INK,zorder=3)
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,lw=1.1,color=INK,zorder=1,shrinkA=0,shrinkB=0))
box(30,110,56,9,["Consecutive acutely successful PVC ablations","July 2020 – July 2025  (n = 159)"],fs=8.4)
arr(30,105.5,30,100.5)
box(80,104,34,9,["Excluded (n = 2)","• Missing / uninterpretable","   electroanatomic map export"],fc=LGREY,fs=8.0,align="left")
ax.add_patch(FancyArrowPatch((30,103),(63,104),arrowstyle="-|>",mutation_scale=11,lw=1.1,color=INK))
box(30,94,46,8.5,["Analysis cohort (n = 157)"],fc=LBLUE)
ax.text(56,93,"Arrhythmia origin\n(12-lead ECG, confirmed by mapping)",ha="left",va="center",fontsize=8.2,style="italic",color=GREY)
# four parallel origin branches
xs=[14,38,62,86]
labs=[("RVOT",OG['RVOT']),("Other outflow-tract",OG['otherOT']),("Presumed intramural*",OG['intramural']),("Non-outflow-tract",OG['nonOT'])]
for x,(nm,(nn,rec)) in zip(xs,labs):
    arr(30,89.7,x,82.5)
    box(x,76,22,13,[nm,f"n = {nn}",f"Recurrence {rec}",f"Success {nn-rec}"],fc="white",fs=8.0)
    arr(x,69.5,x,63)
box(50,57,74,9,[f"3-month Holter outcome:  {R['n']-R['events']} success ({100*(R['n']-R['events'])/R['n']:.1f}%)  vs  {R['events']} recurrence ({100*R['events']/R['n']:.1f}%)","Recurrence = <80% PVC-burden reduction, same PVC morphology confirmed"],fc=LBLUE,fs=8.4)
arr(50,52.5,50,46.5)
box(50,38,86,15,["Primary analysis — unipolar voltage at the earliest activation site",
 "Association with recurrence independent of origin (Firth regression, three origin models)",
 "· incremental value over origin: leakage-free 5-fold cross-validation repeated 100 times",
 "· post hoc sensitivity models: + atrial fibrillation and PVC-morphology count; + coupling interval"],fc="white",ec=BLUE,fs=8.0,lw=1.4)
ax.text(2,22,"* All presumed-intramural foci were of outflow-tract origin (required biventricular mapping and/or ablation at two adjacent chambers).\n"
        "  Points acquired only at >10 g contact force with a SmartTouch SF catheter.  RVOT, right-ventricular outflow tract.",
        ha="left",va="top",fontsize=7.2,color="#555")
fig.savefig("Figure_1.png",dpi=400,bbox_inches="tight");fig.savefig("Figure_1.pdf",bbox_inches="tight");print("Fig1 saved")
