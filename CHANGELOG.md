# Changelog

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
