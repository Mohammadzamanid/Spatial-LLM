"""
src/api/server.py
FastAPI production inference server for Spatial-LLM.

Endpoints:
  GET  /health          — liveness check
  POST /predict         — spatial QA inference
  POST /predict/batch   — batch inference
  GET  /model/info      — model metadata

Usage:
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from ..utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ── Request / Response schemas ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2048)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    use_tile: bool = Field(True, description="Fetch and use map tile for this location")
    max_new_tokens: int = Field(128, ge=1, le=512)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class PredictResponse(BaseModel):
    answer: str
    lat: float
    lon: float
    latency_ms: float
    model_version: str = "spatial-llm-0.1.0"


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., max_length=16)


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


# ── App lifecycle ──────────────────────────────────────────────────────────────

_inference_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _inference_engine
    import yaml, os
    from ..inference import SpatialLLMInference

    config_path = os.environ.get("CONFIG_PATH", "configs/train_config.yaml")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "outputs/best")

    try:
        logger.info(f"Loading model from {checkpoint_path}...")
        _inference_engine = SpatialLLMInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Allow server to start without model — health check will report it
        _inference_engine = None

    yield

    logger.info("Shutting down — releasing model")
    if _inference_engine is not None:
        del _inference_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Spatial-LLM API",
    description="Neuroscience-inspired spatial language model inference",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        model_loaded=_inference_engine is not None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if _inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        answer = _inference_engine.predict(
            question=req.question,
            lat=req.lat,
            lon=req.lon,
            use_tile=req.use_tile,
            max_new_tokens=req.max_new_tokens,
        )
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(answer=answer, lat=req.lat, lon=req.lon, latency_ms=latency_ms)


@app.post("/predict/batch", response_model=list[PredictResponse])
async def predict_batch(req: BatchPredictRequest):
    if _inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for r in req.requests:
        t0 = time.perf_counter()
        try:
            answer = _inference_engine.predict(
                question=r.question, lat=r.lat, lon=r.lon,
                use_tile=r.use_tile, max_new_tokens=r.max_new_tokens,
            )
        except Exception as e:
            answer = f"ERROR: {e}"
        latency_ms = (time.perf_counter() - t0) * 1000
        results.append(PredictResponse(answer=answer, lat=r.lat, lon=r.lon, latency_ms=latency_ms))
    return results


@app.get("/model/info")
async def model_info():
    return {
        "name": "Spatial-LLM",
        "version": "0.1.0",
        "components": [
            "GridCellEncoder (entorhinal cortex)",
            "HippocampalMemory (place cells)",
            "PredictiveCoding (neocortex)",
            "SpatialNeuromodulator (dopamine/ACh/NE)",
            "ViT SpatialTileEncoder",
            "LoRA fine-tuned LLM",
        ],
        "model_loaded": _inference_engine is not None,
    }
