"""
modal_app.py — run a Spatial-LLM multi-seed train+eval sweep on a Modal T4, in the
BACKGROUND (fire-and-forget). Modal is NOT a notebook: you launch this file with the
modal CLI and the GPU work happens on Modal's servers, so you can close everything.

    pip install modal
    modal setup                        # one-time browser auth (or `modal token set ...`)
    modal run --detach modal_app.py    # background; survives closing your laptop/tab

Results are written to a Modal Volume ('spatial-llm-results') AND printed to the logs.
After it finishes, fetch them:

    modal volume get spatial-llm-results coord2d_ALL.json .
    # or watch logs live:  modal app logs spatial-llm-sweep

Change the arm by editing main() at the bottom (config / label / coords_in_text).
"""
import modal

app = modal.App("spatial-llm-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.2.2",
        "transformers==4.40.2", "peft==0.10.0", "accelerate==0.28.0",
        "datasets", "bitsandbytes",
        "geopandas", "shapely", "mercantile", "pyproj",
        "pyyaml", "pillow", "numpy", "pandas",
    )
)

results_vol = modal.Volume.from_name("spatial-llm-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("spatial-llm-hf-cache", create_if_missing=True)


@app.function(gpu="T4", image=image, timeout=3 * 60 * 60,
              volumes={"/results": results_vol, "/cache": hf_cache})
def sweep(config: str, label: str, seeds: list[int], coords_in_text: bool):
    import json
    import os
    import subprocess

    # fresh clone each run so we always pick up the latest main
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/Mohammadzamanid/Spatial-LLM.git", "/root/repo"],
        check=True,
    )
    os.chdir("/root/repo")
    env = {
        **os.environ,
        "HF_HUB_DISABLE_XET": "1",
        "HF_HOME": "/cache/huggingface",      # cache the base model across runs
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }

    data_cmd = ["python", "-m", "src.data.real_datasets", "--dataset", "cities15000",
                "--task", "elevation", "--n_train", "8000", "--n_val", "1000"]
    if not coords_in_text:
        data_cmd.append("--no-coords-in-text")
    subprocess.run(data_cmd, check=True, env=env)

    runs = []
    for s in seeds:
        out = f"outputs/{label}_seed{s}"
        print(f"\n===== TRAIN {label} seed={s} -> {out} =====", flush=True)
        subprocess.run(["python", "-u", "-m", "src.training.trainer",
                        "--config", config, "--seed", str(s), "--output_dir", out],
                       check=True, env=env)
        rj = f"/results/{label}_seed{s}.json"
        subprocess.run(["python", "-m", "src.eval.accuracy", "--config", config,
                        "--checkpoint", out, "--val", "data/processed/val.jsonl",
                        "--dump-gates", "--seed", str(s),
                        "--label", f"{label}_seed{s}", "--results-json", rj],
                       check=True, env=env)
        runs.append(json.load(open(rj)))
        results_vol.commit()
        hf_cache.commit()

    payload = {"experiment": label, "task": "elevation",
               "coords_in_text": coords_in_text, "seeds": list(seeds), "runs": runs}
    with open(f"/results/{label}_ALL.json", "w") as f:
        json.dump(payload, f, indent=2)
    results_vol.commit()
    print("\n===PASTE-THIS-BACK===")
    print(json.dumps(payload, indent=2))
    print("===END===")
    return payload


@app.local_entrypoint()
def main():
    # Edit to run a different arm (e.g. configs/coord_3d_noleak.yaml, coords_in_text=False).
    sweep.remote(
        config="configs/coord_2d_noleak.yaml",   # lat/lon only -> must LEARN elevation
        label="coord2d",
        seeds=[42, 43, 44],
        coords_in_text=False,
    )
