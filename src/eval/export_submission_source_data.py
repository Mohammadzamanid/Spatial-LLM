"""Export compact source-data CSVs for the five main figures from committed JSON."""
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/"results"; OUT=ROOT/"paper"/"source_data"; OUT.mkdir(parents=True,exist_ok=True)
def load(n): return json.loads((RES/n).read_text())
def write(name,rows,fields):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

# Fig 1
tor=load("torus_llm.json"); rows=[]
for T,d in tor["results_by_len"].items():
    rows += [{"panel":"1B","task":"torus","x":T,"condition":"ON","mean":d["on_mean"],"ci95":d["on_ci95"],"n":6,"p":d["paired_p"]},
             {"panel":"1B","task":"torus","x":T,"condition":"OFF","mean":d["off_mean"],"ci95":d["off_ci95"],"n":6,"p":d["paired_p"]}]
write("Fig1_source.csv",rows,["panel","task","x","condition","mean","ci95","n","p"])

# Fig 2
sig=load("significance.json")["comparisons"]; rows=[]
for k,d in sig.items():
    rows.append({"comparison":k,"mean_A":d["mean_A"],"mean_B":d["mean_B"],"mean_diff":d["mean_diff"],
                 "ci95_low":d["ci95"][0],"ci95_high":d["ci95"][1],"p":d["p_perm"],"cohen_d":d["cohen_d"],"n":d["n"]})
write("Fig2_source.csv",rows,["comparison","mean_A","mean_B","mean_diff","ci95_low","ci95_high","p","cohen_d","n"])

# Fig 3
tor=load("torus.json"); nec=load("code_necessity.json"); mm=load("multimap_task.json"); rows=[]
for cond,vals in tor["results"].items():
    for T,d in vals.items():
        rows.append({"panel":"3A","condition":cond,"x":T,"mean":d["within_45deg"]["mean"],"ci95":d["within_45deg"]["ci95"]})
for cond,vals in nec["multimap"].items():
    for M,d in vals.items():
        rows.append({"panel":"3B","condition":cond,"x":M,"mean":d["mean"],"ci95":d["ci95"]})
for cond,vals in mm["results"].items():
    for M,d in vals.items():
        rows.append({"panel":"3C","condition":cond,"x":M,"mean":d["mean"],"ci95":d["ci95"]})
write("Fig3_source.csv",rows,["panel","condition","x","mean","ci95"])

# Fig 4
q=load("theta_sweep_llm_agg.json"); s=load("sweep_llm_smollm2_v2.json"); rows=[]
rows.append({"model":"Qwen2.5-1.5B","seed":"aggregate","condition":"ON","accuracy":q["on"][0]})
for k,d in q["conditions"].items():
    rows.append({"model":"Qwen2.5-1.5B","seed":"aggregate","condition":k,"accuracy":d["acc"][0]})
for r in s["per_seed"]:
    for k in ("on","off","no_sweep","shuffle"):
        rows.append({"model":"SmolLM2-1.7B","seed":r["seed"],"condition":k,"accuracy":r[k]})
write("Fig4_source.csv",rows,["model","seed","condition","accuracy"])

# Fig 5
g=load("geolife_external_v2.json"); rows=[]
for task,d in g["tasks"].items():
    for r in d["per_seed"]:
        rows += [{"task":task,"unit":"readout_seed","id":r["seed"],"condition":"GRID","accuracy":r["grid"]},
                 {"task":task,"unit":"readout_seed","id":r["seed"],"condition":"RAW-MLP","accuracy":r["raw"]},
                 {"task":task,"unit":"readout_seed","id":r["seed"],"condition":"OFF","accuracy":r["off"]}]
    for u in d["heldout_users"]:
        rows += [{"task":task,"unit":"heldout_user","id":u["user_id"],"condition":"GRID","accuracy":u["grid"]},
                 {"task":task,"unit":"heldout_user","id":u["user_id"],"condition":"RAW-MLP","accuracy":u["raw"]},
                 {"task":task,"unit":"heldout_user","id":u["user_id"],"condition":"OFF","accuracy":u["off"]}]
write("Fig5_source.csv",rows,["task","unit","id","condition","accuracy"])

print("Wrote source data to",OUT)
