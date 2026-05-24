"""
src/data/tile_fetcher.py
Downloads map tiles (OpenStreetMap / ESRI satellite) for given lat/lon + zoom.
Uses mercantile to compute tile coordinates from geographic coordinates.
"""

import logging
import time
from pathlib import Path

import mercantile
import requests
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)

# Tile URL templates
TILE_PROVIDERS = {
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "esri_satellite": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
}

HEADERS = {"User-Agent": "Spatial-LLM-Research/1.0 (academic use)"}


def fetch_tile(
    lat: float,
    lon: float,
    zoom: int = 15,
    provider: str = "osm",
    save_dir: str = "data/tiles/",
    retries: int = 3,
    delay: float = 1.0,
) -> str | None:
    """
    Fetch a map tile for (lat, lon) at the given zoom level.
    Returns the local file path on success, None on failure.
    """
    tile = mercantile.tile(lon, lat, zoom)
    url_template = TILE_PROVIDERS.get(provider, TILE_PROVIDERS["osm"])
    url = url_template.format(z=tile.z, x=tile.x, y=tile.y)

    save_path = Path(save_dir) / f"{provider}_{zoom}_{tile.z}_{tile.x}_{tile.y}.png"
    if save_path.exists():
        return str(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(save_path)
            logger.debug(f"Saved tile: {save_path}")
            return str(save_path)
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            time.sleep(delay * (attempt + 1))

    logger.error(f"Failed to fetch tile for lat={lat}, lon={lon}, zoom={zoom}")
    return None


def batch_fetch_tiles(
    records: list[dict],
    zoom: int = 15,
    provider: str = "osm",
    save_dir: str = "data/tiles/",
) -> list[dict]:
    """
    Fetch tiles for a list of records with 'lat' and 'lon' keys.
    Adds 'tile_path' to each record in-place.
    """
    for rec in records:
        path = fetch_tile(rec["lat"], rec["lon"], zoom, provider, save_dir)
        rec["tile_path"] = path
    return records
