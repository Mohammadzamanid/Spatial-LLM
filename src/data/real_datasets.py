"""
src/data/real_datasets.py

Loaders for REAL geospatial datasets (not synthetic).

Primary source: GeoNames (https://www.geonames.org) — a free, authoritative
gazetteer of real-world places with verified coordinates, population, country,
elevation, and timezone. The `cities15000` export contains ~25,000 cities
with population > 15,000.

We turn this real data into spatial QA pairs:
  - "Which country contains the city at (lat, lon)?"  → real country code
  - "What is the population near (lat, lon)?"          → real population bucket
  - "Which timezone applies at (lat, lon)?"            → real IANA timezone
  - "What is the elevation at (lat, lon)?"             → real elevation

Also supports loading geo-QA datasets from the HuggingFace Hub.

All network access happens at call time, so this works on Colab/Kaggle (where
the hosts are reachable) without affecting import.
"""

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Iterator, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEONAMES_URL = "https://download.geonames.org/export/dump/{name}.zip"

# Official GeoNames tab-separated column order (19 fields)
GEONAMES_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]

# ISO country code → human-readable name (subset; extend as needed)
COUNTRY_NAMES = {
    "US": "the United States", "GB": "the United Kingdom", "JP": "Japan",
    "FR": "France", "DE": "Germany", "IR": "Iran", "CN": "China",
    "IN": "India", "BR": "Brazil", "RU": "Russia", "CA": "Canada",
    "AU": "Australia", "EG": "Egypt", "NG": "Nigeria", "ZA": "South Africa",
    "MX": "Mexico", "ID": "Indonesia", "KR": "South Korea", "IT": "Italy",
    "ES": "Spain", "TR": "Turkey", "AR": "Argentina", "SA": "Saudi Arabia",
}


def download_geonames(name: str = "cities15000",
                      cache_dir: str = "data/raw") -> Path:
    """
    Download and extract a GeoNames export. Returns the path to the .txt file.
    `name` options: cities500, cities1000, cities5000, cities15000 (smaller =
    more cities incl. towns; cities15000 ≈ 25k major cities).
    """
    import requests  # imported here so module import never needs network

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    txt_path = cache / f"{name}.txt"
    if txt_path.exists():
        logger.info(f"Using cached {txt_path}")
        return txt_path

    url = GEONAMES_URL.format(name=name)
    logger.info(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extract(f"{name}.txt", cache)
    logger.info(f"Extracted → {txt_path}")
    return txt_path


def parse_geonames(txt_path: str | Path) -> Iterator[dict]:
    """
    Parse a GeoNames .txt export into dict records.
    Yields dicts keyed by GEONAMES_COLUMNS, with numeric fields cast.
    """
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            # Real GeoNames rows have 19 fields; pad if trailing fields missing
            if len(parts) < 6:
                continue
            if len(parts) < len(GEONAMES_COLUMNS):
                parts = parts + [""] * (len(GEONAMES_COLUMNS) - len(parts))
            rec = dict(zip(GEONAMES_COLUMNS, parts))
            try:
                rec["latitude"] = float(rec["latitude"])
                rec["longitude"] = float(rec["longitude"])
                rec["population"] = int(rec["population"] or 0)
                rec["elevation"] = int(rec["elevation"]) if rec["elevation"] else None
            except ValueError:
                continue
            yield rec


def _population_bucket(pop: int) -> str:
    if pop >= 10_000_000:
        return "a megacity (over 10 million people)"
    if pop >= 1_000_000:
        return "a large city (1–10 million people)"
    if pop >= 100_000:
        return "a mid-sized city (100k–1M people)"
    return "a small city or town (under 100k people)"


def geonames_to_qa(records: Iterator[dict], max_records: Optional[int] = None,
                   task: str = "mixed", elev_threshold: float = 1000.0) -> Iterator[dict]:
    """
    Convert real GeoNames records into spatial QA pairs.
    Each city yields several question types grounded in real attributes.
    """
    import random
    count = 0
    for rec in records:
        lat, lon = rec["latitude"], rec["longitude"]
        cc = rec["country_code"]
        country = COUNTRY_NAMES.get(cc, cc)
        name = rec["name"]
        pop = rec["population"]
        tz = rec["timezone"]
        elev = rec["elevation"]

        templates = [
            (f"Which country contains the location at ({lat:.4f}, {lon:.4f})?",
             f"{country.capitalize()} (the city of {name})."),
            (f"What is the approximate population near ({lat:.4f}, {lon:.4f})?",
             f"{name} is {_population_bucket(pop)}."),
        ]
        if tz:
            templates.append(
                (f"Which timezone applies at ({lat:.4f}, {lon:.4f})?",
                 f"The timezone at {name} is {tz}."))
        if elev is not None:
            templates.append(
                (f"What is the elevation at ({lat:.4f}, {lon:.4f})?",
                 f"{name} sits at approximately {elev} metres above sea level."))

        if task == "elevation":
            # Elevation-threshold classification: depends ONLY on real elevation.
            # A 2D model (lat,lon only) must memorize; a 3D model sees z directly.
            if elev is None:
                continue
            above = elev >= elev_threshold
            q = (f"Is the location at ({lat:.4f}, {lon:.4f}) above "
                 f"{int(elev_threshold)} metres elevation?")
            a = "Yes." if above else "No."
        else:
            q, a = random.choice(templates)
        yield {"question": q, "answer": a, "lat": lat, "lon": lon,
               "city": name, "country": cc, "population": pop,
               "elevation": float(elev) if elev is not None else 0.0}

        count += 1
        if max_records and count >= max_records:
            return


def build_geonames_dataset(
    n_train: int = 8000,
    n_val: int = 1000,
    dataset: str = "cities15000",
    output_dir: str = "data/processed",
    seed: int = 42,
    task: str = "mixed",
):
    """
    End-to-end: download GeoNames, convert to QA, write train/val JSONL.
    This is REAL-WORLD data — actual cities, coordinates, populations.
    """
    import random
    txt = download_geonames(dataset)
    all_records = list(parse_geonames(txt))
    logger.info(f"Parsed {len(all_records):,} real cities from {dataset}")

    random.Random(seed).shuffle(all_records)
    split = int(len(all_records) * 0.85)
    train_src, val_src = all_records[:split], all_records[split:]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # For the elevation task, pick the threshold = median elevation of cities that
    # have elevation data, so the yes/no split is ~50/50 (not 94/6 like the old
    # fixed 1000 m). Computed once on all records and shared across train/val so
    # both use the same decision boundary.
    elev_threshold = 1000.0
    if task == "elevation":
        import statistics
        elevs = [r["elevation"] for r in all_records
                 if r.get("elevation") is not None]
        if elevs:
            elev_threshold = float(statistics.median(elevs))
        logger.info(f"Elevation task: median-based threshold = {elev_threshold:.0f} m "
                    f"(from {len(elevs):,} cities with elevation)")

    def write(records, n, path):
        written = 0
        with open(path, "w", encoding="utf-8") as f:
            for qa in geonames_to_qa(iter(records), max_records=n, task=task,
                                     elev_threshold=elev_threshold):
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                written += 1
        logger.info(f"Wrote {written:,} real QA pairs → {path}")

    write(train_src, n_train, out / "train.jsonl")
    write(val_src, n_val, out / "val.jsonl")


def load_hf_geo_dataset(dataset_id: str, split: str = "train"):
    """
    Load a geo dataset from the HuggingFace Hub. Returns the raw dataset object.
    Example IDs (verify availability on the Hub before use):
        - "AdaptLLM/remote-sensing-visual-instruction"  (remote sensing VQA)
        - geographic QA datasets searchable at huggingface.co/datasets
    Caller is responsible for mapping fields to {question, answer, lat, lon}.
    """
    from datasets import load_dataset
    logger.info(f"Loading HF dataset: {dataset_id} [{split}]")
    return load_dataset(dataset_id, split=split)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--dataset", default="cities15000",
                    choices=["cities500", "cities1000", "cities5000", "cities15000"])
    ap.add_argument("--output_dir", default="data/processed")
    ap.add_argument("--task", default="mixed", choices=["mixed", "elevation"],
                    help="'elevation' = elevation-threshold task (tests 3D coords)")
    args = ap.parse_args()
    build_geonames_dataset(args.n_train, args.n_val, args.dataset, args.output_dir,
                           task=args.task)
