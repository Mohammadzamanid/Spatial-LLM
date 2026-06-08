"""Aggregate results/<label>_seed*.json into one paste-back block + mean/std.

    python scripts/collect_results.py coord2d
"""
import glob
import json
import statistics as st
import sys

label = sys.argv[1] if len(sys.argv) > 1 else "coord2d"
files = sorted(glob.glob(f"results/{label}_seed*.json"))
runs = [json.load(open(f)) for f in files]
accs = [r["balanced_accuracy"] for r in runs if r.get("balanced_accuracy") is not None]

payload = {
    "experiment": label,
    "task": "elevation",
    "dataset": "cities15000",
    "source": "external",
    "reproduced_in_repo": True,
    "summary": {
        "mean": round(st.mean(accs), 4) if accs else None,
        "std": round(st.pstdev(accs), 4) if len(accs) > 1 else 0.0,
        "values": [round(a, 4) for a in accs],
    },
    "runs": runs,
}
print(f"\nfound {len(runs)} run(s) for label '{label}'")
print("===PASTE-THIS-BACK===")
print(json.dumps(payload, indent=2))
print("===END===")
