"""Plot GeoLife external-validation summary from committed JSON."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
data=json.loads((ROOT/"results/geolife_external_v2.json").read_text())

tasks=["bearing","distance"]
labels=["Bearing (8-way)","Distance (6-bin)"]
grid=[data["tasks"][t]["grid_mean"] for t in tasks]
raw=[data["tasks"][t]["raw_mlp_mean"] for t in tasks]
off=[data["tasks"][t]["off"] for t in tasks]

x=np.arange(len(tasks)); w=0.24
fig,ax=plt.subplots(figsize=(7.2,4.6))
ax.bar(x-w,grid,w,label="GRID")
ax.bar(x,raw,w,label="RAW-MLP")
ax.bar(x+w,off,w,label="OFF")
ax.set_xticks(x,labels)
ax.set_ylim(0,1.05)
ax.set_ylabel("Accuracy")
ax.legend(frameon=False)
ax.set_title("GeoLife external validation — unseen users")
fig.tight_layout()
out=ROOT/"results/geolife_external_v2.svg"
fig.savefig(out,bbox_inches="tight")
print(out)
