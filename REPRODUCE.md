# Reproducibility

The flagship submission is built from committed JSON result artifacts and reproducible CPU/GPU scripts.
The main paper is intentionally narrower than the full repository; extended neuroscience-inspired analyses
are preserved in `paper/SUPPLEMENTARY_RESULTS.md`.

## Environment

Core CPU experiments:
- Python 3.11 or 3.12
- dependencies from `requirements.txt` / `pyproject.toml`
- CI verifies both Python versions

Language experiments additionally require:
- `transformers>=4.40`
- `peft`
- `accelerate`
- NVIDIA T4-class GPU or better

## Main-paper evidence map

| Main figure / claim | Regeneration | Committed artifact |
|---|---|---|
| Fig. 1 — causal language transfer | GPU notebooks listed below | `results/torus_llm.json`, `elapsed_time_llm.json`, `deadreckoning_llm_agg.json`, `multiframe_llm_agg.json` |
| Fig. 2 — regime map and certified Euclidean null | `python -m src.eval.significance --n_fast 20 --n_slow 8`; `python -m src.eval.phase_diagram` | `results/significance.json`, `phase_diagram.json` |
| Fig. 3 — periodicity/remapping mechanisms | `python -m src.eval.torus --seeds 8`; `python -m src.eval.multimap_task --seeds 5` | `results/torus.json`, `multimap_task.json` |
| Fig. 4 — theta-sweep + second backbone | Qwen + SmolLM2 GPU notebooks below | `results/theta_sweep_llm_agg.json`, `sweep_llm_smollm2_v2.json` |
| Fig. 5 — GeoLife external validation | `python -m src.eval.geolife_external_v2 --data_dir data/geolife_benchmark --seeds 8` | `results/geolife_external_v2.json` |

## Consolidated main figures

After the result JSONs are present:

```bash
python -m src.eval.make_submission_figures
```

This writes vector outputs under:

```
paper/figures/Fig1_causal_transfer.{svg,pdf}
paper/figures/Fig2_regime_map.{svg,pdf}
paper/figures/Fig3_periodicity_remapping.{svg,pdf}
paper/figures/Fig4_theta_replication.{svg,pdf}
paper/figures/Fig5_geolife.{svg,pdf}
```

## GPU language experiments

| Result | Script / notebook |
|---|---|
| Qwen torus causal ON/OFF | `src/training/train_trajectory.py --task torus --constrained_velocity` / project Kaggle notebook |
| Qwen elapsed-time readout | `notebooks/m3_temporal_full_kaggle.py` |
| Qwen dead-reckoning organ lesions | `notebooks/m5_deadreckoning_llm_kaggle.py` |
| Qwen multi-reference-frame lesions | `notebooks/m6_multiframe_llm_kaggle.py` |
| Qwen theta-sweep ablation | `notebooks/m7_theta_sweep_llm_kaggle.py` |
| SmolLM2 theta-sweep replication | `notebooks/m7_theta_sweep_smollm2_worker.py` + dual-GPU launcher |
| GeoLife external validation | `notebooks/geolife_external_gateA_v2_kaggle.py` |

For GPU experiments, the committed result JSON is the paper artifact. The notebooks record model,
training, evaluation, and seed settings; all seeds, including non-convergent/null seeds, remain reported.

## GeoLife external validation

The repository does **not** redistribute Microsoft's GeoLife archive. Build the benchmark from the official
GeoLife Trajectories 1.3 dataset:

```bash
python -m src.data.geolife_trajectory \
  --root /path/to/Geolife/Data \
  --out data/geolife_benchmark \
  --lengths 8 16 24 \
  --max_windows_per_file 4

python -m src.eval.geolife_external_v2 \
  --data_dir data/geolife_benchmark \
  --out results/geolife_external_v2.json \
  --seeds 8 \
  --max_train 24000 \
  --max_val 6000 \
  --max_test 12000
```

The user split, normalization, and distance thresholds are derived exactly as documented in
`paper/SUBMISSION_GATE.md`.

## Extended / supplementary experiments

The full repository contains many additional mechanistic and neuroscience-inspired evaluations
(successor maps, time cells, replay, semantic warping, 3-D/local-order grids, content binding,
basal-ganglia action selection, behaving agents, and others). These are retained for transparency and
follow-up work but are not required for the central five-figure submission. See:

- `paper/SUPPLEMENTARY_RESULTS.md`
- `results/*.json`
- `src/eval/*.py`

## Submission archive

See `paper/RELEASE_CHECKLIST.md`.

The exact submission commit should be tagged and released on GitHub, then archived through Zenodo.
After Zenodo mints the DOI, update `CITATION.cff`, README, and the manuscript code-availability statement
without changing the scientific result artifacts.
