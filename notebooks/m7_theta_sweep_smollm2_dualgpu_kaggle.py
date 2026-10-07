# M7 SmolLM2 V2 — dual-GPU Kaggle launcher (Python 3.13 safe)
# IMPORTANT: no editable install. The worker injects the repo root into sys.path,
# and this launcher also exports PYTHONPATH for subprocesses.

import os, subprocess, time, json, math
from pathlib import Path
import torch

ROOT = Path("/kaggle/working")
REPO = ROOT / "Spatial-LLM"

assert torch.cuda.is_available(), "Enable GPU."
assert torch.cuda.device_count() >= 2, f"Need T4 x2; found {torch.cuda.device_count()} GPU(s)."
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

if REPO.exists():
    subprocess.run(["rm", "-rf", str(REPO)], check=True)

subprocess.run([
    "git", "clone", "--branch", "submission-ready-2026-10", "--single-branch",
    "https://github.com/Mohammadzamanid/Spatial-LLM.git", str(REPO)
], check=True)

os.chdir(REPO)
subprocess.run(["git", "log", "-1", "--oneline"], check=True)

subprocess.run(
    'pip -q install -U "transformers>=4.40" peft accelerate huggingface_hub',
    shell=True, check=True
)
subprocess.run("pip -q uninstall -y torchao", shell=True)

base_env = os.environ.copy()
base_env["HF_HUB_DISABLE_XET"] = "1"
base_env["PYTHONPATH"] = str(REPO) + os.pathsep + base_env.get("PYTHONPATH", "")

# Fail fast before downloading/model training.
check = subprocess.run(
    ["python", "-c",
     "from src.models.fusion import MultiScaleSpatialFusion; "
     "from src.models.neuro.theta_sweep import ThetaSweepSampler; "
     "print('SRC IMPORT CHECK: OK')"],
    cwd=REPO, env=base_env, capture_output=True, text=True
)
print(check.stdout)
if check.stderr:
    print(check.stderr)
assert check.returncode == 0, "Import check failed."

from huggingface_hub import snapshot_download
snapshot_download("HuggingFaceTB/SmolLM2-1.7B-Instruct")
print("SmolLM2 cached.")

OUTDIR = REPO / "results_sweep_llm_smollm2_v2"
OUTDIR.mkdir(exist_ok=True)

env0 = base_env.copy(); env0["CUDA_VISIBLE_DEVICES"] = "0"
env1 = base_env.copy(); env1["CUDA_VISIBLE_DEVICES"] = "1"

worker = str(REPO / "notebooks/m7_theta_sweep_smollm2_worker.py")
cmd0 = ["python", "-u", worker, "--seeds", "0", "2", "4", "6"]
cmd1 = ["python", "-u", worker, "--seeds", "1", "3", "5", "7"]

log0_path = ROOT / "smollm_gpu0.log"
log1_path = ROOT / "smollm_gpu1.log"
log0 = open(log0_path, "w", buffering=1)
log1 = open(log1_path, "w", buffering=1)

print("Launching GPU0 seeds 0,2,4,6 and GPU1 seeds 1,3,5,7")
p0 = subprocess.Popen(cmd0, cwd=REPO, env=env0, stdout=log0, stderr=subprocess.STDOUT)
p1 = subprocess.Popen(cmd1, cwd=REPO, env=env1, stdout=log1, stderr=subprocess.STDOUT)

last = {0:"",1:""}
while p0.poll() is None or p1.poll() is None:
    time.sleep(30)
    for gpu, path in [(0,log0_path),(1,log1_path)]:
        try:
            lines = path.read_text(errors="ignore").splitlines()
            tail = "\n".join(lines[-4:])
            if tail != last[gpu]:
                print(f"\n--- GPU {gpu} ---\n{tail}")
                last[gpu] = tail
        except Exception:
            pass

log0.close(); log1.close()
print("return codes:", p0.returncode, p1.returncode)

if p0.returncode != 0:
    print("\nGPU0 LOG\n", log0_path.read_text(errors="ignore")[-10000:])
if p1.returncode != 0:
    print("\nGPU1 LOG\n", log1_path.read_text(errors="ignore")[-10000:])
assert p0.returncode == 0 and p1.returncode == 0, "Worker failed."

results = []
for seed in range(8):
    f = OUTDIR / f"seed{seed}.json"
    assert f.exists(), f"Missing {f}"
    results.append(json.loads(f.read_text()))

def ci95(xs):
    n=len(xs); m=sum(xs)/n
    sd=math.sqrt(sum((x-m)**2 for x in xs)/(n-1))
    return m,1.96*sd/math.sqrt(n)

def exact_signflip_p(diffs):
    obs=abs(sum(diffs)/len(diffs)); total=2**len(diffs); count=0
    for mask in range(total):
        vals=[d*(1 if (mask>>i)&1 else -1) for i,d in enumerate(diffs)]
        count += abs(sum(vals)/len(vals)) >= obs-1e-12
    return count/total

on=[r["on"] for r in results]
print("\n"+"="*76)
print("SMOLLM2 V2 — SECOND-BACKBONE REPLICATION")
print("="*76)
for r in results:
    print(f"seed {r['seed']}: ON {r['on']:.1%} | OFF {r['off']:.1%} | "
          f"NO-SWEEP {r['no_sweep']:.1%} | SHUFFLE {r['shuffle']:.1%}")

summary={}
for key,label in [("off","OFF"),("no_sweep","NO-SWEEP"),("shuffle","WRONG-HEADING")]:
    alt=[r[key] for r in results]
    diffs=[a-b for a,b in zip(on,alt)]
    mo,cio=ci95(on); mc,cic=ci95(alt)
    p=exact_signflip_p(diffs)
    summary[key]={
        "on_mean":mo,"on_ci95":cio,"control_mean":mc,"control_ci95":cic,
        "delta":sum(diffs)/len(diffs),"exact_signflip_p":p,
        "wins":sum(d>0 for d in diffs),"losses":sum(d<0 for d in diffs),
        "ties":sum(d==0 for d in diffs)
    }
    print(f"\nON {mo:.1%} ± {cio:.1%} vs {label} {mc:.1%} ± {cic:.1%}")
    print(summary[key])

final={
    "model":"HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "protocol":"theta_sweep_second_backbone_v2",
    "training_steps":1600,"n_seeds":8,
    "scoring":"forced-choice logits for 0 vs 1",
    "per_seed":results,"comparisons":summary
}
final_path=REPO/"results_sweep_llm_smollm2_v2_aggregate.json"
final_path.write_text(json.dumps(final,indent=2))
print("\nSaved:", final_path)
print("\nPRIMARY: ON vs NO-SWEEP")
print(json.dumps(summary["no_sweep"],indent=2))
