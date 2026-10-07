"""Generate the five consolidated main-paper figures from committed result JSON.

Outputs:
  paper/figures/Fig1_causal_transfer.{svg,pdf}
  paper/figures/Fig2_regime_map.{svg,pdf}
  paper/figures/Fig3_periodicity_remapping.{svg,pdf}
  paper/figures/Fig4_theta_replication.{svg,pdf}
  paper/figures/Fig5_geolife.{svg,pdf}

Run from repository root:
    python -m src.eval.make_submission_figures
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/"results"
OUT=ROOT/"paper"/"figures"
OUT.mkdir(parents=True,exist_ok=True)

def load(name):
    return json.loads((RES/name).read_text())

def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f"{name}.svg",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)

def panel(ax,label):
    ax.text(-0.12,1.10,label,transform=ax.transAxes,fontweight="bold",fontsize=12,va="top")

# ------------------------------------------------------------------
# FIGURE 1 — causal language transfer
# ------------------------------------------------------------------
tor=load("torus_llm.json")
tim=load("elapsed_time_llm.json")
dr=load("deadreckoning_llm_agg.json")
mf=load("multiframe_llm_agg.json")

fig=plt.figure(figsize=(12,7))
gs=fig.add_gridspec(2,2)

ax=fig.add_subplot(gs[0,0]); panel(ax,"A")
ax.axis("off")
boxes=[
    (0.02,0.42,"Self-motion\ntrajectory"),
    (0.28,0.42,"Cognitive-map\nstate"),
    (0.56,0.42,"Gated spatial→text\nfusion"),
    (0.82,0.42,"Frozen LLM\n+ LoRA"),
]
for x,y,t in boxes:
    ax.text(x,y,t,transform=ax.transAxes,ha="center",va="center",
            bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="black"))
for a,b in zip(boxes[:-1],boxes[1:]):
    ax.annotate("",xy=(b[0]-0.08,b[1]),xytext=(a[0]+0.08,a[1]),
                xycoords=ax.transAxes,textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->"))
ax.text(.5,.13,"Moves never enter the prompt",ha="center",transform=ax.transAxes,fontstyle="italic")

ax=fig.add_subplot(gs[0,1]); panel(ax,"B")
lens=["8","16","24"]; x=np.arange(3); w=.35
on=[tor["results_by_len"][k]["on_mean"] for k in lens]
off=[tor["results_by_len"][k]["off_mean"] for k in lens]
onci=[tor["results_by_len"][k]["on_ci95"] for k in lens]
offci=[tor["results_by_len"][k]["off_ci95"] for k in lens]
ax.bar(x-w/2,on,w,yerr=onci,capsize=3,label="Cortex ON")
ax.bar(x+w/2,off,w,yerr=offci,capsize=3,label="OFF")
ax.set_xticks(x,lens); ax.set_xlabel("Path length"); ax.set_ylabel("Exact accuracy")
ax.set_ylim(0,1.05); ax.set_title("Leakage-proof torus readout"); ax.legend(frameon=False)

ax=fig.add_subplot(gs[1,0]); panel(ax,"C")
names=["Elapsed time\nexact","WHERE","FACING","LANDMARK"]
on=[tim["exact"]["on_mean"],dr["where"]["on"][0],dr["facing"]["on"][0],mf["landmark"]["on"][0]]
off=[tim["exact"]["off_mean"],dr["where"]["off"][0],dr["facing"]["off"][0],mf["landmark"]["off"][0]]
x=np.arange(len(names)); w=.35
ax.bar(x-w/2,on,w,label="ON"); ax.bar(x+w/2,off,w,label="OFF")
ax.set_xticks(x,names); ax.set_ylim(0,1); ax.set_ylabel("Accuracy")
ax.set_title("Causal readouts through latent state")

ax=fig.add_subplot(gs[1,1]); panel(ax,"D")
labels=["WHERE\n(grid)","FACING\n(HD)","WHERE\n(grid)","LANDMARK\n(OVC)"]
intact=[dr["where"]["on"][0],dr["facing"]["on"][0],mf["where"]["on"][0],mf["landmark"]["on"][0]]
own=[dr["where"]["no_grid_lesion"],dr["facing"]["no_hd_lesion"],mf["where"]["no_grid_lesion"],mf["landmark"]["no_ovc_lesion"]]
other=[dr["where"]["no_hd_lesion"],dr["facing"]["no_grid_lesion"],mf["where"]["no_ovc_lesion"],mf["landmark"]["no_grid_lesion"]]
x=np.arange(len(labels)); w=.26
ax.bar(x-w,intact,w,label="Intact")
ax.bar(x,own,w,label="Own-organ lesion")
ax.bar(x+w,other,w,label="Other-organ lesion")
ax.set_xticks(x,labels); ax.set_ylim(0,.65); ax.set_ylabel("Accuracy")
ax.set_title("Organ-specific double dissociation"); ax.legend(frameon=False,fontsize=8)
save(fig,"Fig1_causal_transfer")

# ------------------------------------------------------------------
# FIGURE 2 — regime map / honest characterization
# ------------------------------------------------------------------
sig=load("significance.json")["comparisons"]
phase=load("phase_diagram.json")
fig=plt.figure(figsize=(12,7)); gs=fig.add_gridspec(2,2)

ax=fig.add_subplot(gs[0,0]); panel(ax,"A")
key="extrapolation distance@T24: grid vs NoPE+sum xf (NULL)"
d=sig[key]
ax.bar(["Grid","NoPE+sum"],[d["mean_A"],d["mean_B"]])
ax.set_ylim(.8,1.0); ax.set_ylabel("Distance accuracy @ T24")
ax.set_title(f"Certified Euclidean tie\np={d['p_perm']:.2f}, d={d['cohen_d']:.2f}")

ax=fig.add_subplot(gs[0,1]); panel(ax,"B")
k="extrapolation distance@T24: grid vs place"; d=sig[k]
ax.bar(["Grid","Place"],[d["mean_A"],d["mean_B"]])
ax.set_ylim(.6,1.0); ax.set_ylabel("Distance accuracy @ T24")
ax.set_title("Bounded place code loses range")

ax=fig.add_subplot(gs[1,:]); panel(ax,"C")
codes=phase["codes"]; regimes=[r["regime"] for r in phase["matrix"]]
mat=np.array([[np.nan if r["cells"][c]["value"] is None else r["cells"][c]["value"] for c in codes]
              for r in phase["matrix"]])
im=ax.imshow(mat,aspect="auto",vmin=0,vmax=1,cmap="viridis")
ax.set_xticks(np.arange(len(codes)),codes)
ax.set_yticks(np.arange(len(regimes)),regimes)
for i,r in enumerate(phase["matrix"]):
    for j,c in enumerate(codes):
        cell=r["cells"][c]
        txt="—" if cell["value"] is None else f"{cell['value']:.2f}\n{cell['outcome']}"
        ax.text(j,i,txt,ha="center",va="center",fontsize=8,
                color="white" if (cell["value"] or 0) < .5 else "black")
ax.set_title("Predictive regime map: wins, ties, and failures")
fig.colorbar(im,ax=ax,label="Task metric")
save(fig,"Fig2_regime_map")

# ------------------------------------------------------------------
# FIGURE 3 — periodicity and remapping
# ------------------------------------------------------------------
torus=load("torus.json")
nec=load("code_necessity.json")
mm=load("multimap_task.json")
fig=plt.figure(figsize=(13.5,5.2)); gs=fig.add_gridspec(1,3)

ax=fig.add_subplot(gs[0,0]); panel(ax,"A")
T=[8,16,32,64]
for name in ["grid (periodic)","NoPE+sum Transformer","place (Euclidean)"]:
    vals=[torus["results"][name][str(t)]["within_45deg"]["mean"] for t in T]
    ci=[torus["results"][name][str(t)]["within_45deg"]["ci95"] for t in T]
    ax.errorbar(T,vals,yerr=ci,marker="o",label=name)
ax.set_xscale("log",base=2); ax.set_xticks(T,T)
ax.set_ylim(0,1.05); ax.set_xlabel("Path length / wraps"); ax.set_ylabel("Within 45°")
ax.set_title("Periodicity is load-bearing on a torus"); ax.legend(frameon=False,fontsize=7)

ax=fig.add_subplot(gs[0,1]); panel(ax,"B")
Ms=[1,2,4,8,16]
for name in ["grid + remap","place + remap","grid, NO remap","additive (raw 2-D)"]:
    vals=[nec["multimap"][name][str(m)]["mean"] for m in Ms]
    ci=[nec["multimap"][name][str(m)]["ci95"] for m in Ms]
    ax.errorbar(Ms,vals,yerr=ci,marker="o",label=name)
ax.set_xscale("log",base=2); ax.set_xticks(Ms,Ms)
ax.set_ylim(0,1.05); ax.set_xlabel("Number of contexts"); ax.set_ylabel("One-shot recall")
ax.set_title("Remapping prevents context collisions")
ax.legend(frameon=False,fontsize=7)

ax=fig.add_subplot(gs[0,2]); panel(ax,"C")
Ms=[1,2,4,8,16,32]
for name in ["grid + remap","grid, no remap","additive (raw 2-D)"]:
    vals=[mm["results"][name][str(m)]["mean"] for m in Ms]
    ax.plot(Ms,vals,marker="o",label=name)
ax.set_xscale("log",base=2); ax.set_xticks(Ms,Ms)
ax.set_ylim(0,1.05); ax.set_xlabel("Number of labelled contexts"); ax.set_ylabel("Accuracy")
ax.set_title("A context label substitutes for remapping")
ax.legend(frameon=False,fontsize=7)
save(fig,"Fig3_periodicity_remapping")

# ------------------------------------------------------------------
# FIGURE 4 — theta sweep + cross-backbone replication
# ------------------------------------------------------------------
q=load("theta_sweep_llm_agg.json")
s=load("sweep_llm_smollm2_v2.json")
fig=plt.figure(figsize=(12,5.5)); gs=fig.add_gridspec(1,3)

ax=fig.add_subplot(gs[0,0]); panel(ax,"A")
vals=[q["on"][0],
      q["conditions"]["off_text_only"]["acc"][0],
      q["conditions"]["no_sweep"]["acc"][0],
      q["conditions"]["shuffled"]["acc"][0]]
cis=[q["on"][1],
     q["conditions"]["off_text_only"]["acc"][1],
     q["conditions"]["no_sweep"]["acc"][1],
     q["conditions"]["shuffled"]["acc"][1]]
ax.bar(["ON","OFF","NO-SWEEP","WRONG\nHEADING"],vals,yerr=cis,capsize=3)
ax.axhline(.5,ls="--",lw=1)
ax.set_ylim(0,1); ax.set_ylabel("Accuracy"); ax.set_title("Qwen2.5-1.5B")

ax=fig.add_subplot(gs[0,1]); panel(ax,"B")
keys=["on","off","no_sweep","shuffle"]
means=[np.mean([r[k] for r in s["per_seed"]]) for k in keys]
ax.bar(["ON","OFF","NO-SWEEP","WRONG\nHEADING"],means)
ax.axhline(.5,ls="--",lw=1)
ax.set_ylim(0,1); ax.set_ylabel("Accuracy"); ax.set_title("SmolLM2-1.7B")

ax=fig.add_subplot(gs[0,2]); panel(ax,"C")
diff=[r["on"]-r["no_sweep"] for r in s["per_seed"]]
ax.axhline(0,lw=1)
ax.scatter(np.arange(8),diff)
for i,dv in enumerate(diff):
    ax.plot([i,i],[0,dv],lw=1)
ax.set_xticks(np.arange(8),[str(i) for i in range(8)])
ax.set_xlabel("Seed"); ax.set_ylabel("ON − NO-SWEEP")
ax.set_title("SmolLM2 paired effect\np=0.0156")
save(fig,"Fig4_theta_replication")

# ------------------------------------------------------------------
# FIGURE 5 — GeoLife external validation
# ------------------------------------------------------------------
g=load("geolife_external_v2.json")
fig=plt.figure(figsize=(12,5.5)); gs=fig.add_gridspec(1,3)

ax=fig.add_subplot(gs[0,0]); panel(ax,"A")
ax.axis("off")
stages=[
    (.50,.84,"GeoLife GPS trajectories"),
    (.50,.65,"East/north displacement windows"),
    (.50,.46,"User-disjoint 70/15/15 split"),
    (.50,.27,"Fixed grid population → readout"),
]
for x,y,t in stages:
    ax.text(x,y,t,ha="center",va="center",transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35",fc="white",ec="black"),fontsize=8.5)
for (_,ya,_),(_,yb,_) in zip(stages[:-1],stages[1:]):
    ax.annotate("",xy=(.50,yb+.07),xytext=(.50,ya-.07),xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->"))
ax.text(.5,.06,"Scale + distance bins fixed from TRAIN users only",ha="center",
        transform=ax.transAxes,fontstyle="italic",fontsize=8)

ax=fig.add_subplot(gs[0,1]); panel(ax,"B")
tasks=["bearing","distance"]; names=["Bearing","Distance"]
grid=[g["tasks"][t]["grid_mean"] for t in tasks]
raw=[g["tasks"][t]["raw_mlp_mean"] for t in tasks]
off=[g["tasks"][t]["off"] for t in tasks]
x=np.arange(2); w=.24
ax.bar(x-w,grid,w,label="GRID"); ax.bar(x,raw,w,label="RAW-MLP"); ax.bar(x+w,off,w,label="OFF")
ax.set_xticks(x,names); ax.set_ylim(0,1.05); ax.set_ylabel("Accuracy")
ax.set_title("External validation on unseen users"); ax.legend(frameon=False,fontsize=8)

ax=fig.add_subplot(gs[0,2]); panel(ax,"C")
effects=[g["tasks"][t]["user_paired_grid_minus_off_mean"] for t in tasks]
cis=[g["tasks"][t]["user_paired_grid_minus_off_ci95_bootstrap"] for t in tasks]
lo=[e-c[0] for e,c in zip(effects,cis)]; hi=[c[1]-e for e,c in zip(effects,cis)]
ax.errorbar(effects,np.arange(2),xerr=np.array([lo,hi]),fmt="o",capsize=4)
ax.axvline(0,lw=1)
ax.set_yticks(np.arange(2),names); ax.set_xlabel("GRID − OFF accuracy")
ax.set_title("Held-out user effects (n=28)")
save(fig,"Fig5_geolife")

print("Wrote:")
for p in sorted(OUT.glob("Fig*.svg")):
    print(p.relative_to(ROOT))
