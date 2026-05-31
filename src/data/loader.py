"""
src/data/loader.py
GeoPandas-based loaders for spatial QA datasets.
Expects JSONL records with fields: question, answer, lat, lon, tile_path (optional).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import Point
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


class SpatialQADataset(Dataset):
    """
    Loads spatial question-answering pairs.
    Each record must have: question (str), answer (str), lat (float), lon (float).
    Optional: tile_path (str) — path to a map/satellite tile image.
    """

    TILE_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        jsonl_path: str,
        tile_dir: Optional[str] = None,
        max_text_length: int = 512,
        use_tiles: bool = True,
    ):
        self.records = self._load_jsonl(jsonl_path)
        self.tile_dir = Path(tile_dir) if tile_dir else None
        self.max_text_length = max_text_length
        self.use_tiles = use_tiles
        logger.info(f"Loaded {len(self.records)} records from {jsonl_path}")

    @staticmethod
    def _load_jsonl(path: str) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        coords = torch.tensor(
            [rec["lat"], rec["lon"], rec.get("elevation", 0.0)], dtype=torch.float32
        )

        item = {
            "question": rec["question"],
            "answer": rec["answer"],
            "coords": coords,
        }

        if self.use_tiles and self.tile_dir:
            tile_rel = rec.get("tile_path", "")
            # Guard: skip if tile_path is missing/empty, or resolves to a directory
            if tile_rel and (self.tile_dir / tile_rel).is_file():
                img = Image.open(self.tile_dir / tile_rel).convert("RGB")
                item["pixel_values"] = self.TILE_TRANSFORM(img)
            else:
                # No tile for this record — use a blank one (silent, this is normal
                # for text-only datasets like GeoNames that don't have map tiles)
                item["pixel_values"] = torch.zeros(3, 224, 224)

        return item


def load_geodataframe(path: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON / Shapefile into a GeoDataFrame with WGS84 projection."""
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def coords_to_geodataframe(lats: list[float], lons: list[float]) -> gpd.GeoDataFrame:
    """Convenience: convert coordinate lists to a GeoDataFrame."""
    geometry = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    return gpd.GeoDataFrame({"lat": lats, "lon": lons}, geometry=geometry, crs="EPSG:4326")
