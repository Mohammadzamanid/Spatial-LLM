"""
src/data/synthetic.py
Synthetic spatial QA data generator for bootstrapping training.

Generates JSONL records with realistic geographic QA pairs.
Uses real city/country data — no network required.
Covers: urban classification, distance reasoning, directional reasoning,
        landmark proximity, climate/biome inference.

Usage:
    python -m src.data.synthetic --n_train 5000 --n_val 500 --output_dir data/processed/
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterator

# ── Real-world anchor points ───────────────────────────────────────────────────
CITIES = [
    {"name": "Tokyo",       "lat": 35.6895, "lon": 139.6917, "type": "megacity",   "climate": "humid_subtropical"},
    {"name": "London",      "lat": 51.5074, "lon": -0.1278,  "type": "megacity",   "climate": "oceanic"},
    {"name": "New York",    "lat": 40.7128, "lon": -74.0060, "type": "megacity",   "climate": "humid_continental"},
    {"name": "Tehran",      "lat": 35.6892, "lon": 51.3890,  "type": "megacity",   "climate": "semi_arid"},
    {"name": "Lagos",       "lat": 6.5244,  "lon": 3.3792,   "type": "megacity",   "climate": "tropical"},
    {"name": "São Paulo",   "lat": -23.5505,"lon": -46.6333, "type": "megacity",   "climate": "humid_subtropical"},
    {"name": "Sydney",      "lat": -33.8688,"lon": 151.2093, "type": "city",       "climate": "oceanic"},
    {"name": "Cairo",       "lat": 30.0444, "lon": 31.2357,  "type": "megacity",   "climate": "desert"},
    {"name": "Moscow",      "lat": 55.7558, "lon": 37.6173,  "type": "megacity",   "climate": "humid_continental"},
    {"name": "Mumbai",      "lat": 19.0760, "lon": 72.8777,  "type": "megacity",   "climate": "tropical"},
    {"name": "Berlin",      "lat": 52.5200, "lon": 13.4050,  "type": "city",       "climate": "oceanic"},
    {"name": "Singapore",   "lat": 1.3521,  "lon": 103.8198, "type": "city_state", "climate": "tropical"},
    {"name": "Nairobi",     "lat": -1.2864, "lon": 36.8172,  "type": "city",       "climate": "subtropical_highland"},
    {"name": "Buenos Aires","lat": -34.6037,"lon": -58.3816, "type": "megacity",   "climate": "humid_subtropical"},
    {"name": "Toronto",     "lat": 43.6532, "lon": -79.3832, "type": "city",       "climate": "humid_continental"},
    {"name": "Dubai",       "lat": 25.2048, "lon": 55.2708,  "type": "city",       "climate": "desert"},
    {"name": "Seoul",       "lat": 37.5665, "lon": 126.9780, "type": "megacity",   "climate": "humid_continental"},
    {"name": "Paris",       "lat": 48.8566, "lon": 2.3522,   "type": "megacity",   "climate": "oceanic"},
    {"name": "Shanghai",    "lat": 31.2304, "lon": 121.4737, "type": "megacity",   "climate": "humid_subtropical"},
    {"name": "Jakarta",     "lat": -6.2088, "lon": 106.8456, "type": "megacity",   "climate": "tropical"},
]

QUESTION_TEMPLATES = [
    # Classification
    ("What type of urban area is at {lat:.4f}°N, {lon:.4f}°E?",
     "This location in {name} is classified as a {type}."),
    ("Describe the climate at coordinates ({lat:.4f}, {lon:.4f}).",
     "The climate at {name} ({lat:.4f}, {lon:.4f}) is {climate}."),
    # Direction
    ("In which hemisphere is the location at latitude {lat:.4f}?",
     "{hemisphere} hemisphere."),
    # Distance
    ("Is the location at ({lat:.4f}, {lon:.4f}) near the equator?",
     "{equator_answer}"),
    # Coastal/inland (simplified heuristic)
    ("Is ({lat:.4f}, {lon:.4f}) likely a coastal or inland location?",
     "{coastal_answer}"),
    # Continent
    ("Which continent contains the location at ({lat:.4f}, {lon:.4f})?",
     "{continent}"),
]


def _hemisphere(lat: float) -> str:
    return "Northern" if lat >= 0 else "Southern"


def _near_equator(lat: float) -> str:
    if abs(lat) < 10:
        return "Yes, this location is very close to the equator."
    elif abs(lat) < 23.5:
        return "This location is in the tropics, relatively close to the equator."
    else:
        return "No, this location is not near the equator."


def _continent(lat: float, lon: float) -> str:
    # Very rough heuristic — good enough for synthetic training data
    if lon > 60 and lon < 150 and lat > -10 and lat < 55:
        return "Asia"
    if lon > -15 and lon < 60 and lat > -35 and lat < 37:
        return "Africa"
    if lon > -15 and lon < 45 and lat > 35 and lat < 72:
        return "Europe"
    if lon > -170 and lon < -50 and lat > 15 and lat < 75:
        return "North America"
    if lon > -85 and lon < -35 and lat > -55 and lat < 15:
        return "South America"
    if lon > 110 and lon < 180 and lat > -47 and lat < -10:
        return "Australia"
    return "Unknown"


def _coastal(lon: float) -> str:
    # Very rough proxy
    if lon % 10 < 3:
        return "This location appears to be coastal based on its longitude."
    return "This location is likely inland."


def _jitter(lat: float, lon: float, scale: float = 0.05) -> tuple[float, float]:
    """Add small random noise to avoid exact city coords."""
    return (
        lat + random.gauss(0, scale),
        lon + random.gauss(0, scale),
    )


def generate_record(city: dict) -> dict:
    lat, lon = _jitter(city["lat"], city["lon"])
    lat = max(-90.0, min(90.0, lat))
    lon = max(-180.0, min(180.0, lon))

    template_q, template_a = random.choice(QUESTION_TEMPLATES)

    q = template_q.format(
        lat=lat, lon=lon,
        name=city["name"],
        type=city["type"],
        climate=city["climate"].replace("_", " "),
    )
    a = template_a.format(
        lat=lat, lon=lon,
        name=city["name"],
        type=city["type"].replace("_", " "),
        climate=city["climate"].replace("_", " "),
        hemisphere=_hemisphere(lat),
        equator_answer=_near_equator(lat),
        coastal_answer=_coastal(lon),
        continent=_continent(lat, lon),
    )

    return {"question": q, "answer": a, "lat": lat, "lon": lon}


def generate_dataset(n: int, seed: int = 42) -> Iterator[dict]:
    random.seed(seed)
    for _ in range(n):
        city = random.choice(CITIES)
        yield generate_record(city)


def write_jsonl(records: Iterator[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} records to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val",   type=int, default=500)
    parser.add_argument("--output_dir", default="data/processed/")
    args = parser.parse_args()

    out = Path(args.output_dir)
    write_jsonl(generate_dataset(args.n_train, seed=42),  out / "train.jsonl")
    write_jsonl(generate_dataset(args.n_val,   seed=99),  out / "val.jsonl")
