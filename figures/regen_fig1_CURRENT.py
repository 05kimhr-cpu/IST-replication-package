import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from pathlib import Path
OUT=Path("_jin_edit/figures")
plt.rcParams.update({"font.family":"serif","font.size":10,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.05})
fig,ax=plt.subplots(figsize=(7.6,4.6)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
def box(cx,cy,w,h,text,fs=8.2,lw=1.2):
    ax.add_patch(mpatches.FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.12",linewidth=lw,edgecolor="black",facecolor="white",zorder=3))
    ax.text(cx,cy,text,ha="center",va="center",fontsize=fs,zorder=4,multialignment="center")
def arrow(x0,y0,x1,y1):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle="->",color="black",lw=1.1),zorder=2)
for x,label in [(1.35,"Data"),(4.6,"Evaluation stage"),(8.35,"Construct")]:
    ax.text(x,5.7,label,ha="center",va="center",fontsize=9,fontweight="bold",bbox=dict(boxstyle="round,pad=0.2",fc="0.88",ec="black",lw=0.7))
# Data (left)
box(1.35,4.5,2.2,0.82,"MCMD corpus\n2,907 controlled pairs\n8 languages")
box(1.35,3.0,2.2,0.82,"1,600 diffs, 3x7B LLMs\n4,800 generations\n8 languages")
box(1.35,1.5,2.2,0.82,"80 stratified pairs\n2 blind human raters")
# Stages (center) -- shortened, no hatch
box(4.6,4.5,2.7,0.82,"RQ1 . Controlled perturbation\n2,907 pairs\n5 metrics + NLI probe")
box(4.6,3.0,2.7,0.82,"RQ2 . Premise-swap diagnostic\ngold vs diff premise\nBART-MNLI + DeBERTa")
box(4.6,1.5,2.7,0.82,"RQ3 . Human triangulation\ndiff-description accuracy\n2 blind raters")
box(4.6,0.32,2.7,0.5,"RQ4 . Robustness checks")
# Constructs (right)
box(8.35,4.5,2.3,0.82,"Metric blindness\nto meaning flips")
box(8.35,3.0,2.3,0.82,"Reference proximity\nnot = diff grounding")
box(8.35,1.5,2.3,0.82,"Instrument-dependent;\nhuman vs NLI diverge")
# Arrows Data->Stage
arrow(2.45,4.5,3.25,4.5); arrow(2.45,3.0,3.25,3.0); arrow(2.45,1.5,3.25,1.5)
# Stage->Construct
arrow(5.95,4.5,7.2,4.5); arrow(5.95,3.0,7.2,3.0); arrow(5.95,1.5,7.2,1.5)
# Stage3 -> RQ4
arrow(4.6,1.09,4.6,0.57)
fig.savefig(OUT/"fig1_evidence_funnel.pdf"); fig.savefig(OUT/"fig1_evidence_funnel.png"); plt.close(fig)
print("fig1 regenerated (clean, no hatch, shortened)")
