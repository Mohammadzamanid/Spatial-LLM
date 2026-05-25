# 🧠 Spatial-LLM Model Card

## Model Overview
**Name:** Spatial-LLM  
**Version:** 0.1.0  
**Type:** Multimodal Language Model with Neuroscience-Inspired Spatial Encoding  
**Backbone:** Mistral-7B-v0.1 (LoRA fine-tuned)  
**License:** MIT

---

## Architecture

Spatial-LLM fuses three streams of spatial information into a large language model using neuroscience-inspired inductive biases:

### Spatial Encoding Stack

| Component | Biological Inspiration | Role |
|---|---|---|
| **GridCellEncoder** | Medial entorhinal cortex grid cells | Multi-scale hexagonal lat/lon encoding |
| **CoordinateEmbedder** | Fourier place encoding | High-frequency continuous coordinate features |
| **HippocampalMemory** | Hippocampal place cells + episodic memory | Sparse population code + spatial working memory |
| **SpatialTileEncoder** | Visual cortex (V1–V4) | ViT-based map/satellite image encoding |
| **PredictiveCoding** | Neocortical hierarchical prediction | Auxiliary prediction error for self-supervised learning |
| **SpatialNeuromodulator** | Dopamine / ACh / NE system | Context-conditioned gain + prediction error gating |
| **MultiScaleSpatialFusion** | Cortico-hippocampal binding | Cross-attention injection into LLM layers |

### Information Flow
```
Coordinates ──→ GridCellEncoder (6 scales, hexagonal) ──┐
             ──→ CoordinateEmbedder (Fourier features)  ──┤
                                                          ├──→ Cross-Attention ──→ LLM (LoRA) ──→ Answer
Map Tile ────→ ViT SpatialTileEncoder ───────────────────┤      ↑
                                                          │  Neuromodulation
Episodic  ───→ HippocampalMemory ────────────────────────┘  (gain + PE gate)
Memory
```

---

## Intended Use
- **Research:** Spatial reasoning benchmarks, geographic QA
- **Applications:** Location-aware question answering, urban analysis, geospatial understanding
- **Not intended for:** Navigation safety systems, real-time mapping, surveillance

## Limitations
- Coordinate encoding is in WGS84 degree space (works best for lat/lon within known city ranges)
- Map tile visual encoder requires internet access at inference time (or pre-cached tiles)
- HippocampalMemory buffer is session-scoped (resets between inference calls)
- Base LLM knowledge cutoff applies

## Training Data
- **Primary:** Synthetic spatial QA pairs generated from real city anchors (`src/data/synthetic.py`)
- **Recommended:** OSM QA, GeoQA, SpatialBench for fine-tuning

## Evaluation Metrics
- **Haversine Error (km):** Mean / median great-circle distance to true coordinate
- **Within-N-km accuracy:** % predictions within 1 / 10 / 100 km
- **Exact Match:** String match rate on categorical answers
- **BBox IoU:** Intersection-over-union for bounding box predictions

## Citation
```bibtex
@software{spatial_llm_2025,
  author = {Mohammadzamanid},
  title = {Spatial-LLM: Neuroscience-Inspired Spatial Language Model},
  year = {2025},
  url = {https://github.com/Mohammadzamanid/Spatial-LLM}
}
```
