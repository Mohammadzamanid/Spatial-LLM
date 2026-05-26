# 🌍 Spatial-LLM

[![CI](https://github.com/Mohammadzamanid/Spatial-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohammadzamanid/Spatial-LLM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-117%20passing-brightgreen)](https://github.com/Mohammadzamanid/Spatial-LLM/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A neuroscience-inspired multimodal language model that fuses **grid cell coordinate encoding**, **hippocampal spatial memory**, **predictive coding**, and **neuromodulation** with a LoRA-fine-tuned LLM backbone for geographic reasoning tasks.

---

## Architecture

```
Coordinates (lat/lon)
  ├─→ GridCellEncoder    (entorhinal cortex — 6-scale hexagonal lattice)
  └─→ CoordinateEmbedder (Fourier features)
                                         ↘
Map Tile (224×224)                        Cross-Attention   →  LLM (LoRA)  →  Answer
  └─→ ViT SpatialTileEncoder            ↗        ↑
                                         Neuromodulation
Episodic Buffer                         (Dopamine / NE / ACh gating)
  └─→ HippocampalMemory  (place cells + k-WTA sparsity)

Auxiliary training signal:
  SpatialPredictiveCoding  (neocortical prediction error — self-supervised)
```

### Neuroscience Components

| Component | Brain Region | Mechanism |
|---|---|---|
| `GridCellEncoder` | Medial entorhinal cortex | 6-module hexagonal lattice, learnable rotations, multi-scale (0.01°–24°) |
| `CoordinateEmbedder` | Spatial place encoding | Fourier feature mapping, 64 frequency bands |
| `HippocampalMemory` | Hippocampus CA1/CA3 | Place cell population code, k-WTA sparsity (k=50/512), episodic buffer |
| `SpatialPredictiveCoding` | Neocortex | Hierarchical prediction error, 3-level architecture (Rao & Ballard 1999) |
| `SpatialNeuromodulator` | Dopamine / ACh | Context-conditioned gain + bias modulation |
| `AdaptiveGain` | Norepinephrine (LC-NE) | Uncertainty-driven contrast amplification |
| `PredictionErrorGate` | Dopamine (VTA) | Novelty-gated spatial signal routing |
| `SpatialTileEncoder` | Visual cortex V1–V4 | ViT-base-patch16-224 → LLM projection |
| `MultiScaleSpatialFusion` | Cortico-hippocampal | Cross-attention, 2-layer, 8 heads |


### Complete Neuroscience Stack (single neuron → network)

Spatial-LLM implements **20 neuroscience-grounded modules** across every level of brain organization. This is the differentiator: most "spatial" models stop at coordinate embeddings; this one models the actual computational primitives the mammalian navigation system uses.

| Level | Module | Biological Basis | Reference |
|---|---|---|---|
| **Single neuron** | `LIFNeuron` | Leaky integrate-and-fire membrane dynamics + surrogate-gradient spikes | Gerstner & Kistler 2002 |
| | `AdaptiveLIFNeuron` | Spike-frequency adaptation (cortical pyramidal cells) | Neftci et al. 2019 |
| | `DendriticNeuron` | Multi-compartment dendrites, NMDA-style supralinear branches | Gidon et al., *Science* 2020 |
| **Synapse** | `HebbianLayer` | "Fire together, wire together" with Oja normalization | Oja 1982 |
| | `STDPLayer` | Spike-timing-dependent plasticity (LTP/LTD) | Bi & Poo 1998 |
| | `ShortTermPlasticity` | Facilitation + depression | Tsodyks & Markram 1997 |
| **Microcircuit** | `DivisiveNormalization` | Canonical cortical gain control | Carandini & Heeger 2012 |
| | `LateralInhibition` | Surround suppression / competition | — |
| | `EIBalanceLayer` | Excitatory/inhibitory balance, Dale's law (80/20) | — |
| | `CorticalColumn` | Canonical L4→L2/3→L5/6 microcircuit | Douglas & Martin 2004 |
| **Spatial cells** | `GridAttractorNetwork` | Toroidal continuous attractor → hexagonal grids | Burak & Fiete 2009 |
| | `HeadDirectionCells` | Ring attractor heading code (von Mises tuning) | Taube et al. 1990 |
| | `BoundaryVectorCells` | Fire at preferred distance/angle from boundaries | Lever et al. 2009 |
| | `SpeedCells` | Velocity signal driving path integration | Kropff et al., *Nature* 2015 |
| **Oscillations** | `ThetaOscillator` | 4–8 Hz theta rhythm gating | Buzsáki 2002 |
| | `PhasePrecession` | Position encoded as theta phase | O'Keefe & Recce 1993 |
| | `ThetaGammaCoupling` | 7±2 item working-memory buffer (nested gamma) | Lisman & Idiart 1995 |
| | `SharpWaveRipple` | Offline replay for memory consolidation | Buzsáki 2015 |
| **Plus prior** | `PlaceCellMemory`, `PredictiveCoding`, `Neuromodulation` | hippocampus / neocortex / dopamine-ACh-NE | Rao & Ballard 1999 |

These are unified in `BrainSpatialCortex` (≈2.8M params) — coordinates flow through grid attractors, head-direction & speed cells, theta phase coding, and boundary cells, integrated by dendritic neurons and a canonical cortical column before fusing into the LLM.


---

## Benchmarks

> **Status:** The full LLM (Mistral-7B) is not yet fine-tuned (needs GPU). But the **spatial encoding stack has been trained and measured** on controlled tasks below. These are **real measured numbers** from `experiments_v2.py`, reproducible on CPU in minutes. No projections in this section.

### Measured: spatial encoder comparison (CPU, identical budget)

All encoders use a 64-dim output + identical linear head, same data, same training steps.

**Task A — Coordinate denoising/regression** (input = true location + 2° noise; reconstruct true location). Lower Haversine error is better.

| Encoder | Mean error (km) | Median error (km) |
|---|---|---|
| **Raw MLP (baseline)** | **226.5** | **212.9** |
| BrainSpatialCortex (full stack) | 231.5 | 216.9 |
| Grid cells | 297.0 | 269.8 |
| Fourier | 350.7 | 321.6 |

**Task B — Fine-grained spatial classification** (100 classes, 1°×1° grid cells). Higher accuracy is better; chance = 1%.

| Encoder | Test accuracy |
|---|---|
| **Fourier** | **99.7%** |
| Grid cells | 95.7% |
| BrainSpatialCortex (full stack) | 64.3% |
| Raw MLP (baseline) | 37.0% |

### What these results actually tell us (honest reading)

1. **Spatial inductive biases help enormously on fine discrimination.** On Task B, Fourier and grid-cell encoders hit 95–99% while a plain MLP manages only 37%. Periodic spatial codes carve up space far better than raw coordinates — this is the core thesis, and it holds.
2. **But "more brain" is not automatically better.** The full `BrainSpatialCortex` (attractors + dendrites + oscillations + cells) *underperforms its own simpler components* on both tasks. The extra machinery adds parameters and optimization difficulty without payoff on these simple tasks. Complexity must earn its place.
3. **Task structure decides the winner.** On smooth denoising (Task A) a linear MLP wins because the target is essentially a smoothed input; on fine discrimination (Task B) periodic codes dominate. There is no universally best encoder.
4. **A real bug was found and fixed by measurement.** The grid-cell encoder initially scored 8.6% (near chance) due to frequency aliasing — its scales (0.01°) were wrong for global coordinates. After the fix (1°–32° wavelengths) it reached 95.7%. This is *why* you measure instead of trusting that "brain-inspired = good".

### Literature baselines (for context, measured by their authors)

| Task | Model | Result | Source |
|---|---|---|---|
| Image geolocation | GeoCLIP | 19.4 km median | Vivanco et al., NeurIPS 2023 |
| Image geolocation | PlaNet | 523 km median | Weyand et al., CVPR 2016 |
| Geographic QA | GPT-4o | 71.2% EM | OpenAI 2024 |

*The LLM-integrated numbers will be added here only after real fine-tuning — never projected.*


---

## Quickstart

```bash
git clone https://github.com/Mohammadzamanid/Spatial-LLM.git
cd Spatial-LLM
pip install -e ".[dev]"

# Generate training data
python -m src.data.synthetic --n_train 5000 --n_val 500 --output_dir data/processed/

# Run all 91 tests
pytest tests/ -v

# Train
python -m src.training.trainer --config configs/train_config.yaml

# Inference
python -m src.inference \
  --config configs/train_config.yaml \
  --checkpoint outputs/best \
  --lat 35.6895 --lon 139.6917 \
  --question "What type of urban area is this?"

# API server
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
cd docker
docker compose up --build
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "What city is at these coordinates?", "lat": 35.6895, "lon": 139.6917}'
```

---

## Project Structure

```
Spatial-LLM/
├── src/
│   ├── models/
│   │   ├── grid_cell_encoder.py    ← entorhinal cortex (hexagonal multi-scale)
│   │   ├── coord_embedder.py       ← Fourier coordinate embedding
│   │   ├── place_cell_memory.py    ← hippocampal place cells + episodic buffer
│   │   ├── spatial_encoder.py      ← ViT tile encoder
│   │   ├── predictive_coding.py    ← neocortical prediction error
│   │   ├── neuromodulation.py      ← dopamine / NE / ACh gating
│   │   ├── fusion.py               ← cross-attention spatial→LLM fusion
│   │   └── llm_wrapper.py          ← full model (LoRA + all components)
│   ├── data/
│   │   ├── loader.py               ← GeoPandas JSONL dataset
│   │   ├── tokenizer.py            ← spatial prompt injection
│   │   ├── tile_fetcher.py         ← OSM/ESRI tile downloader
│   │   └── synthetic.py            ← 200+ city anchor QA generator
│   ├── training/
│   │   ├── trainer.py              ← HuggingFace Trainer entry point
│   │   └── loss.py                 ← Haversine loss + combined spatial LM loss
│   ├── eval/
│   │   ├── metrics.py              ← Haversine, BBox IoU, within-N-km
│   │   └── benchmark.py            ← full eval pipeline
│   ├── api/
│   │   └── server.py               ← FastAPI server (predict / batch / health)
│   ├── utils/
│   │   ├── checkpoint.py           ← best-metric checkpoint manager
│   │   └── logging_config.py       ← structured logging
│   └── inference.py                ← production inference wrapper
├── tests/                          ← 91 passing tests (unit + integration + real data)
├── configs/train_config.yaml       ← all hyperparameters
├── docker/                         ← Dockerfile + docker-compose
├── notebooks/explore_data.ipynb
├── MODEL_CARD.md
└── CHANGELOG.md
```

---

## Recommended Datasets

| Dataset | Size | Use |
|---|---|---|
| [GeoQA](https://github.com/panyw5/GeoQA) | 4,998 QA pairs | Primary fine-tuning target |
| [OSM QA](https://osmlab.github.io/osm-qa-tiles/) | ~100k entries | Spatial entity QA |
| [BigEarthNet](https://bigearth.net/) | 590,326 tiles | Satellite imagery + labels |
| [WHU-RS19](http://captain.whu.edu.cn/repository.html) | 1,005 images | Remote sensing scenes |
| [SpatialBench](https://huggingface.co/datasets/allenai/SpatialBench) | varies | Spatial reasoning evaluation |

---

## Citation

```bibtex
@software{spatial_llm_2025,
  author  = {Mohammadzamanid},
  title   = {Spatial-LLM: Neuroscience-Inspired Spatial Language Model},
  year    = {2025},
  url     = {https://github.com/Mohammadzamanid/Spatial-LLM}
}
```
