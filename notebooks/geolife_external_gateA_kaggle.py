# GeoLife Gate-A external validation — Kaggle launcher
# One T4 is sufficient. This downloads the official Microsoft GeoLife archive,
# builds a user-disjoint benchmark, and runs the preregistered representation test.

import os, subprocess, zipfile
from pathlib import Path

ROOT=Path("/kaggle/working")
REPO=ROOT/"Spatial-LLM"
ZIP=ROOT/"Geolife_Trajectories_1.3.zip"
EXTRACT=ROOT/"geolife"

# fresh publication branch
if REPO.exists():
    subprocess.run(["rm","-rf",str(REPO)],check=True)
subprocess.run([
    "git","clone","--branch","submission-ready-2026-10","--single-branch",
    "https://github.com/Mohammadzamanid/Spatial-LLM.git",str(REPO)
],check=True)
os.chdir(REPO)
subprocess.run(["git","log","-1","--oneline"],check=True)

env=os.environ.copy()
env["PYTHONPATH"]=str(REPO)+os.pathsep+env.get("PYTHONPATH","")

# fail-fast import check
subprocess.run([
    "python","-c",
    "from src.data.geolife_trajectory import build_benchmark; "
    "from src.eval.geolife_external import run; "
    "print('GEOLIFE IMPORT CHECK: OK')"
],cwd=REPO,env=env,check=True)

# official Microsoft Download Center binary
url=("https://download.microsoft.com/download/f/4/8/"
     "f4894aa5-fdbc-481e-9285-d5f8c4c4f039/"
     "Geolife%20Trajectories%201.3.zip")

if not ZIP.exists():
    print("Downloading official Microsoft GeoLife archive...")
    subprocess.run(["wget","-q","--show-progress","-O",str(ZIP),url],check=True)
else:
    print("Using cached archive:",ZIP)

print("archive MB:",round(ZIP.stat().st_size/1024**2,1))
assert ZIP.stat().st_size > 250*1024**2, "GeoLife download looks incomplete."

if EXTRACT.exists():
    subprocess.run(["rm","-rf",str(EXTRACT)],check=True)
EXTRACT.mkdir(parents=True)
print("Extracting...")
with zipfile.ZipFile(ZIP) as z:
    z.extractall(EXTRACT)

data_dirs=[p for p in EXTRACT.rglob("Data") if p.is_dir()]
assert data_dirs, "Could not find GeoLife Data directory after extraction."
DATA=data_dirs[0]
print("GeoLife Data:",DATA)

# Build non-overlapping trajectory windows. Limit per source trajectory to keep
# the benchmark broad across users/files instead of dominated by long traces.
BENCH=REPO/"data/geolife_benchmark"
subprocess.run([
    "python","-m","src.data.geolife_trajectory",
    "--root",str(DATA),
    "--out",str(BENCH),
    "--lengths","8","16","24",
    "--max_windows_per_file","4"
],cwd=REPO,env=env,check=True)

print((BENCH/"metadata.json").read_text())

# External validation. One GPU accelerates feature extraction/readouts; this is
# much smaller than the LLM experiments.
subprocess.run([
    "python","-u","-m","src.eval.geolife_external",
    "--data_dir",str(BENCH),
    "--out","results/geolife_external_v1.json",
    "--seeds","8",
    "--max_train","24000",
    "--max_val","6000",
    "--max_test","12000"
],cwd=REPO,env=env,check=True)

print("\nFINAL RESULT JSON")
print("="*80)
print((REPO/"results/geolife_external_v1.json").read_text())
