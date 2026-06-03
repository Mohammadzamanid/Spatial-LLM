# Results

Committed experiment results live here. `outputs/` is **gitignored** — checkpoints
and per-run `eval_results.json` die with the ephemeral Kaggle container — so the
*reportable* numbers are copied here to version them alongside the code that
produced them. (This is why the Step-1 0.974/0.717 result previously lived only
in chat: it was never written anywhere tracked.)

## Convention
- One JSON per experiment, named `<experiment>.json`.
- `python -m src.eval.accuracy --results-json results/<name>.json` writes a
  structured result (overall + balanced accuracy, per-class recall, fusion gates,
  seed) — emit it on Kaggle, paste the block back, commit it here.
- Multi-seed runs store one entry per seed so error bars are reproducible.
- Always record provenance: `source` (kaggle/local), `reproduced_in_repo`, seeds.

## Files
- `step1_coord_2d_vs_3d.json` — Step 1: 3D (lat/lon/elevation) vs 2D coords on the
  elevation task. The baseline "trusted root".
- `per_module_gating.json` — shared-gate vs per-module-gate on coord_3d, with the
  per-module gate read-out. Populated after running
  `notebooks/per_module_gating_kaggle.ipynb`.
