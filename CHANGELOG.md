# Changelog

## [Unreleased]

### Added
- **`coords_in_text` flag (no-coordinate-leak mode)** — omit lat/lon from the prompt so location reaches the model ONLY through the spatial channel (coord/elevation embedder + grid cells). Without it the LLM just reads coordinates off the text and the fusion gates never open (confirmed: gates stayed ~0, balanced acc ~0.99 from text alone). New configs `coord_3d_noleak.yaml` + `coord_3d_permod_noleak.yaml`; data via `real_datasets.py --no-coords-in-text`.
- **Committed experiment results** under `results/` — `per_module_gating.json` records the leaky-run finding (per-module ≈ shared because the spatial pathway was unused).
- **Per-module fusion gating** ("synchronization") — each spatial module (coordinate/elevation, grid cells, place-cell memory, tile) gets its own zero-init tanh gate in `SpatialFusionLayer`, attended and gated independently, so the model learns to weight grid cells vs elevation vs place memory per task and the trained gates read out which module each task relied on. Toggle via `model.per_module_gates`; preserves the zero-init identity that keeps generation coherent, and the shared-gate default leaves existing checkpoints loading unchanged.
- **`configs/coord_3d_permod.yaml`** — treatment-arm config for A/B-ing per-module vs shared gating (identical to `coord_3d.yaml` except `per_module_gates: true`).
- **Reproducibility seed knob** — `training.seed` seeds python/numpy/torch and the HuggingFace data sampler up front, enabling honest multi-seed error bars across runs.
- **Fusion tests** — zero-init identity (shared + per-module), per-module gate count, independent per-module routing, and no-tile / no-`group_sizes` fallbacks.

### Changed
- `SpatialLLM._encode_spatial` now also returns per-module token spans (`group_sizes`), threaded through `MultiScaleSpatialFusion` for per-module gating.

## [0.1.0] — 2025-05-25

### Added
- **GridCellEncoder** — entorhinal cortex-inspired hexagonal multi-scale coordinate encoder (6 modules, learnable rotations)
- **CoordinateEmbedder** — Fourier feature lat/lon encoding with optional token output
- **HippocampalMemory** — place cell sparse population code + episodic spatial buffer with k-WTA sparsity
- **SpatialTileEncoder** — ViT-based map/satellite tile encoder with LLM-space projection
- **SpatialPredictiveCoding** — hierarchical neocortical prediction error (auxiliary training signal)
- **SpatialNeuromodulator** — dopamine/ACh/NE-inspired context-conditioned gain modulation
- **AdaptiveGain** — norepinephrine-style uncertainty-driven contrast control
- **PredictionErrorGate** — dopamine-style novelty gating for spatial representations
- **MultiScaleSpatialFusion** — cross-attention fusion of spatial tokens into LLM hidden states
- **SpatialLLM** — full model integrating all above with LoRA-adapted Mistral-7B
- **SpatialQADataset** — GeoPandas-backed JSONL dataset with tile support
- **SpatialTokenizer** — spatial prompt injection + teacher-forcing label masking
- **TileFetcher** — OSM/ESRI satellite tile downloader with retry logic
- **HaversineLoss + SpatialLMLoss** — geo-aware differentiable training objectives
- **Evaluation metrics** — Haversine error, BBox IoU, within-N-km accuracy, exact match
- **Benchmark runner** — full eval pipeline with coordinate parsing
- **Synthetic data generator** — 20-city anchor QA generation (`src/data/synthetic.py`)
- **FastAPI inference server** — `/predict`, `/predict/batch`, `/health`, `/model/info`
- **CheckpointManager** — best-metric checkpoint tracking
- **GitHub Actions CI** — automated test + lint on Python 3.11/3.12
- **Docker** — Dockerfile + docker-compose for GPU deployment
- **38 passing unit + integration tests**
- **MODEL_CARD.md** — full architecture and usage documentation
