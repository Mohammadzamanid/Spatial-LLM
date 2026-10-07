"""External validation on Microsoft GeoLife GPS trajectories.

Pre-registered Gate-A experiment for Spatial-LLM.

Scientific question
-------------------
Does the same fixed, biologically constrained grid population preserve endpoint
spatial information on REAL human trajectories from users never seen during
readout training?

The experiment is intentionally representation-level first. It does not train
an LLM and it does not modify the cortex. GeoLife users are split BEFORE any
normalization; the spatial scale and distance-bin thresholds are estimated from
TRAIN users only.

Conditions
----------
GRID: fixed 6-module hexagonal grid population after integrating real EN steps.
RAW:  exact additive endpoint displacement (dx,dy), passed to the same-size MLP
      readout; this is the non-neural integration baseline.
OFF:  majority-class predictor estimated on train users.
ORACLE: label recomputed directly from endpoint displacement (task ceiling).

Primary confirmatory test: 8 readout seeds, GRID vs OFF on 8-way bearing,
paired exact sign-flip test over seeds. Distance (6 train-quantile bins) is
secondary. RAW is a calibration baseline, not a straw man we expect GRID to beat.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from src.models.neuro.trajectory_cortex import _HexGridModules

TARGET_MEDIAN_STEP = 0.50  # matches original synthetic speed scale (~0.2..0.8)
LENGTHS = (8,16,24)
N_CLASSES_BEARING = 8
N_CLASSES_DISTANCE = 6


def load_jsonl(path):
    out=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def cap_records(records, n, seed=0):
    if n is None or len(records)<=n:
        return records
    rng=random.Random(seed)
    idx=list(range(len(records))); rng.shuffle(idx)
    return [records[i] for i in idx[:n]]


def train_scale(train_records):
    steps=[]
    for r in train_records:
        for x,y in r["moves_xy_m"]:
            d=math.hypot(x,y)
            if d>0: steps.append(d)
    med=float(np.median(np.asarray(steps,dtype=np.float64)))
    if not np.isfinite(med) or med<=0:
        raise RuntimeError("Invalid training median GPS step")
    return med/TARGET_MEDIAN_STEP, med


def distance_edges(train_records):
    d=np.asarray([r["distance_m"] for r in train_records],dtype=np.float64)
    qs=np.quantile(d,np.linspace(0,1,N_CLASSES_DISTANCE+1)[1:-1])
    # guard against repeated quantiles
    for i in range(1,len(qs)):
        if qs[i] <= qs[i-1]:
            qs[i]=np.nextafter(qs[i-1],np.inf)
    return qs


def labels(records, edges):
    bearing=np.asarray([int(r["bearing_sector_8"]) for r in records],dtype=np.int64)
    dist=np.asarray([np.searchsorted(edges,float(r["distance_m"]),side="right")
                     for r in records],dtype=np.int64)
    return bearing,dist


@torch.no_grad()
def features(records, scale_m_per_unit, device, batch=512):
    grid=_HexGridModules(embed_dim=64,n_modules=6,base_spacing=1.6).to(device).eval()
    grid_feats=[]; raw_feats=[]; lens=[]
    for i in range(0,len(records),batch):
        rr=records[i:i+batch]
        T=max(len(r["moves_xy_m"]) for r in rr)
        if any(len(r["moves_xy_m"])!=T for r in rr):
            # records are normally grouped mixed-length; pad by zero motion
            pass
        v=torch.zeros(len(rr),T,3,dtype=torch.float32,device=device)
        raw=torch.zeros(len(rr),2,dtype=torch.float32,device=device)
        for j,r in enumerate(rr):
            m=torch.tensor(r["moves_xy_m"],dtype=torch.float32,device=device)/scale_m_per_unit
            v[j,:len(m),:2]=m
            raw[j]=m.sum(0)
        _,cells=grid(v,return_cells=True)
        grid_feats.append(cells.cpu())
        raw_feats.append(raw.cpu())
        lens.extend([len(r["moves_xy_m"]) for r in rr])
    return torch.cat(grid_feats),torch.cat(raw_feats),np.asarray(lens,dtype=np.int64)


class Readout(nn.Module):
    def __init__(self,din,ncls):
        super().__init__()
        self.net=nn.Sequential(
            nn.LayerNorm(din),
            nn.Linear(din,64),nn.GELU(),nn.Dropout(0.10),
            nn.Linear(64,ncls)
        )
    def forward(self,x): return self.net(x)


def fit_eval(xtr,ytr,xv,yv,xte,yte,ncls,seed,device,epochs=80,bs=256):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m=Readout(xtr.shape[1],ncls).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
    lossf=nn.CrossEntropyLoss()
    xtr=xtr.to(device); ytr=torch.as_tensor(ytr,device=device)
    xv=xv.to(device); yv=torch.as_tensor(yv,device=device)
    best=None; best_acc=-1; bad=0
    for ep in range(epochs):
        m.train()
        perm=torch.randperm(len(xtr),device=device)
        for i in range(0,len(xtr),bs):
            ix=perm[i:i+bs]
            opt.zero_grad(set_to_none=True)
            loss=lossf(m(xtr[ix]),ytr[ix]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc=(m(xv).argmax(1)==yv).float().mean().item()
        if acc>best_acc+1e-5:
            best_acc=acc; bad=0
            best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else:
            bad+=1
            if bad>=10: break
    m.load_state_dict(best); m.eval()
    with torch.no_grad():
        pred=m(xte.to(device)).argmax(1).cpu().numpy()
    return float((pred==yte).mean()),pred


def majority_acc(ytr,yte):
    vals,cnt=np.unique(ytr,return_counts=True); maj=vals[cnt.argmax()]
    return float((yte==maj).mean())


def signflip_exact(diffs):
    diffs=np.asarray(diffs,dtype=float); obs=abs(diffs.mean()); n=len(diffs)
    count=0
    for mask in range(1<<n):
        signs=np.asarray([1 if (mask>>i)&1 else -1 for i in range(n)])
        count += abs((diffs*signs).mean()) >= obs-1e-12
    return count/(1<<n)


def mean_ci(xs):
    x=np.asarray(xs,float); m=float(x.mean())
    ci=1.96*float(x.std(ddof=1))/math.sqrt(len(x)) if len(x)>1 else 0.0
    return m,ci


def run(a):
    device="cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    root=Path(a.data_dir)
    train=cap_records(load_jsonl(root/"train.jsonl"),a.max_train,1)
    val=cap_records(load_jsonl(root/"val.jsonl"),a.max_val,2)
    test=cap_records(load_jsonl(root/"test.jsonl"),a.max_test,3)
    print(f"device={device} records train/val/test={len(train)}/{len(val)}/{len(test)}",flush=True)

    scale,med=train_scale(train)
    edges=distance_edges(train)
    print(f"TRAIN-only median GPS step={med:.3f} m -> {TARGET_MEDIAN_STEP:.2f} model units",flush=True)
    print("TRAIN-only distance-bin edges (m):",np.round(edges,2).tolist(),flush=True)

    xb_tr,xr_tr,Ltr=features(train,scale,device,a.feature_batch)
    xb_v,xr_v,Lv=features(val,scale,device,a.feature_batch)
    xb_te,xr_te,Lte=features(test,scale,device,a.feature_batch)
    yb_tr,yd_tr=labels(train,edges); yb_v,yd_v=labels(val,edges); yb_te,yd_te=labels(test,edges)

    result={
      "protocol":"geolife_external_v1","dataset":"Microsoft GeoLife GPS Trajectories",
      "user_disjoint":True,"train_only_scale_m_per_unit":scale,"train_median_step_m":med,
      "target_median_step_model_units":TARGET_MEDIAN_STEP,
      "distance_edges_m":[float(x) for x in edges],
      "n":{"train":len(train),"val":len(val),"test":len(test)},
      "seeds":list(range(a.seeds)),"tasks":{}
    }

    for task,ncls,ytr,yv,yte in [
        ("bearing",N_CLASSES_BEARING,yb_tr,yb_v,yb_te),
        ("distance",N_CLASSES_DISTANCE,yd_tr,yd_v,yd_te)
    ]:
        off=majority_acc(ytr,yte)
        rows=[]
        print(f"\n{task.upper()} | OFF majority={off:.3f}",flush=True)
        for seed in range(a.seeds):
            ga,_=fit_eval(xb_tr,ytr,xb_v,yv,xb_te,yte,ncls,seed,device)
            ra,_=fit_eval(xr_tr,ytr,xr_v,yv,xr_te,yte,ncls,1000+seed,device)
            rows.append({"seed":seed,"grid":ga,"raw":ra,"off":off})
            print(f" seed {seed}: GRID {ga:.3f} | RAW {ra:.3f} | OFF {off:.3f}",flush=True)
        grid=[r["grid"] for r in rows]; raw=[r["raw"] for r in rows]; offv=[off]*len(rows)
        mg,cg=mean_ci(grid); mr,cr=mean_ci(raw)
        pg=signflip_exact(np.asarray(grid)-off)
        pr=signflip_exact(np.asarray(grid)-np.asarray(raw))
        bylen={}
        # Final trained-seed accuracy by length is intentionally omitted here because the
        # readout predictions are seed-specific; pooled seed inference is the confirmatory unit.
        result["tasks"][task]={
          "per_seed":rows,
          "grid_mean":mg,"grid_ci95":cg,"raw_mean":mr,"raw_ci95":cr,"off":off,
          "grid_minus_off":mg-off,"p_grid_vs_off_exact_signflip":pg,
          "grid_minus_raw":mg-mr,"p_grid_vs_raw_exact_signflip":pr,
        }
        print(f" SUMMARY GRID {mg:.3f}±{cg:.3f} | RAW {mr:.3f}±{cr:.3f} | OFF {off:.3f}",flush=True)
        print(f" GRID-OFF {mg-off:+.3f}, exact p={pg:.4f}; GRID-RAW {mg-mr:+.3f}, p={pr:.4f}",flush=True)

    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print("\nwrote",out,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",default="data/geolife_benchmark")
    ap.add_argument("--out",default="results/geolife_external_v1.json")
    ap.add_argument("--seeds",type=int,default=8)
    ap.add_argument("--max_train",type=int,default=24000)
    ap.add_argument("--max_val",type=int,default=6000)
    ap.add_argument("--max_test",type=int,default=12000)
    ap.add_argument("--feature_batch",type=int,default=512)
    ap.add_argument("--cpu",action="store_true")
    run(ap.parse_args())
