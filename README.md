# 🌍 Spatial-LLM

A multimodal language model that fuses **geospatial coordinates**, **satellite/map tile imagery**, and **vector geodata** with a fine-tuned LLM backbone for spatial reasoning tasks.

## Architecture

```
Map Tile (image) ──→ ViT Encoder ────┐
                                      ├──→ Cross-Attention Fusion ──→ LLM (LoRA) ──→ Spatial Answer
Coordinates (lat/lon) ──→ Fourier ───┘
                         Embedder
```

## Project Structure

```
Spatial-LLM/
├── src/
│   ├── data/           # Loaders, tokenizers, tile fetchers
│   ├── models/         # Encoder, embedder, fusion, LLM wrapper
│   ├── training/       # Trainer, loss, callbacks
│   └── eval/           # Spatial metrics, benchmarks
├── data/               # raw / processed / tiles
├── configs/            # YAML hyperparameter configs
├── tests/              # Unit tests
└── notebooks/          # Exploration notebooks
```

## Quickstart

```bash
pip install -r requirements.txt

# Train
python -m src.training.trainer --config configs/train_config.yaml

# Evaluate
python -m src.eval.benchmark --config configs/train_config.yaml
```

## Key Features
- **Fourier coordinate embeddings** — lat/lon as continuous features, not raw text
- **ViT-based tile encoder** — visual grounding from satellite/map imagery
- **Cross-attention fusion** — spatial tokens injected into LLM layers
- **LoRA fine-tuning** — efficient adaptation of Mistral-7B or LLaMA-3
- **Haversine-aware evaluation** — geographically meaningful accuracy metrics

## Datasets
| Dataset | Use |
|---|---|
| OSM QA | Spatial entity QA pairs |
| SpatialBench | Spatial reasoning benchmarks |
| BigEarthNet | Satellite imagery + labels |
| GeoQA | Geographic question answering |
| WHU-RS19 | Remote sensing scene classification |
