# Source data map

The main figures are generated entirely from committed JSON artifacts. No manual data extraction is required.

| Figure | Panel | Source artifact |
|---|---|---|
| Fig. 1 | b | `results/torus_llm.json` |
| Fig. 1 | c | `results/elapsed_time_llm.json`, `results/deadreckoning_llm_agg.json`, `results/multiframe_llm_agg.json` |
| Fig. 1 | d | `results/deadreckoning_llm_agg.json`, `results/multiframe_llm_agg.json` |
| Fig. 2 | a | `results/significance.json` |
| Fig. 2 | b | `results/significance.json`, `results/extrapolation.json` |
| Fig. 2 | c | `results/phase_diagram.json` |
| Fig. 3 | a | `results/torus.json` |
| Fig. 3 | b | `results/code_necessity.json` |
| Fig. 3 | c | `results/multimap_task.json` |
| Fig. 4 | a | `results/theta_sweep_llm_agg.json` |
| Fig. 4 | b,c | `results/sweep_llm_smollm2_v2.json` |
| Fig. 5 | b,c | `results/geolife_external_v2.json` |

Regenerate all five main figures with:

```bash
python -m src.eval.make_submission_figures
```

The plotting code is `src/eval/make_submission_figures.py`.
