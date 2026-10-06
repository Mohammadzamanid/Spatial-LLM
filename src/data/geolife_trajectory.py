"""Utilities for building an external real-trajectory benchmark from Microsoft GeoLife.

GeoLife files are expected in the standard layout:
    <root>/Data/<user_id>/Trajectory/*.plt

Each .plt file has six header lines followed by comma-separated points:
latitude, longitude, unused, altitude(feet), date_days, date, time

This module does not download or redistribute GeoLife. It converts GPS traces into
local east/north displacement sequences so Spatial-LLM can be evaluated on real
human trajectories rather than repository-generated random walks.

Splits are USER-disjoint by construction to prevent memorizing an individual's routes.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float
    alt_m: float | None = None


def _step_east_north_m(a: GeoPoint, b: GeoPoint) -> tuple[float, float]:
    """Small-step equirectangular displacement from a -> b in meters."""
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    mean_lat = 0.5 * (lat1 + lat2)
    north = EARTH_RADIUS_M * dlat
    east = EARTH_RADIUS_M * math.cos(mean_lat) * dlon
    return east, north


def read_plt(path: str | Path) -> list[GeoPoint]:
    path = Path(path)
    pts: list[GeoPoint] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < 6:
                continue
            row = line.strip().split(",")
            if len(row) < 2:
                continue
            try:
                lat, lon = float(row[0]), float(row[1])
            except ValueError:
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            alt_m = None
            if len(row) > 3:
                try:
                    alt_ft = float(row[3])
                    if alt_ft > -700:  # GeoLife uses -777 for missing altitude.
                        alt_m = alt_ft * 0.3048
                except ValueError:
                    pass
            pts.append(GeoPoint(lat, lon, alt_m))
    return pts


def trajectory_steps(points: Sequence[GeoPoint],
                     max_step_m: float = 250.0,
                     min_step_m: float = 0.5) -> list[tuple[float, float]]:
    """Convert GPS points to local EN displacements and reject GPS jumps / duplicates."""
    out: list[tuple[float, float]] = []
    for a, b in zip(points, points[1:]):
        east, north = _step_east_north_m(a, b)
        d = math.hypot(east, north)
        if min_step_m <= d <= max_step_m:
            out.append((east, north))
    return out


def _label(dx: float, dy: float) -> dict[str, float | int]:
    dist = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    sector = int(((angle + math.pi) / (2 * math.pi) * 8)) % 8
    return {
        "distance_m": round(dist, 4),
        "bearing_rad": round(angle, 6),
        "bearing_sector_8": sector,
    }


def make_windows(steps: Sequence[tuple[float, float]], length: int,
                 stride: int | None = None) -> Iterator[dict]:
    """Yield fixed-length windows and endpoint labels from displacement sequences."""
    if length < 2:
        raise ValueError("length must be >=2")
    stride = stride or length
    for start in range(0, len(steps) - length + 1, stride):
        win = steps[start:start + length]
        dx = sum(x for x, _ in win)
        dy = sum(y for _, y in win)
        yield {
            "moves_xy_m": [[round(x, 4), round(y, 4)] for x, y in win],
            "length": length,
            "net_dx_m": round(dx, 4),
            "net_dy_m": round(dy, 4),
            **_label(dx, dy),
        }


def user_split(user_ids: Iterable[str],
               train_frac: float = 0.70,
               val_frac: float = 0.15) -> dict[str, str]:
    """Deterministic user-disjoint split by sorted user id."""
    ids = sorted(set(user_ids))
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    out: dict[str, str] = {}
    for i, uid in enumerate(ids):
        out[uid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def build_benchmark(root: str | Path, out_dir: str | Path,
                    lengths: Sequence[int] = (8, 16, 24),
                    max_windows_per_file: int = 128) -> dict[str, int]:
    root = Path(root)
    data_root = root / "Data" if (root / "Data").exists() else root
    users = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    splits = user_split(users)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    handles = {s: (out_dir / f"{s}.jsonl").open("w", encoding="utf-8")
               for s in ("train", "val", "test")}
    counts = {s: 0 for s in handles}
    try:
        for uid in users:
            split = splits[uid]
            traj_dir = data_root / uid / "Trajectory"
            if not traj_dir.exists():
                continue
            for path in sorted(traj_dir.glob("*.plt")):
                pts = read_plt(path)
                steps = trajectory_steps(pts)
                for length in lengths:
                    n_written = 0
                    for rec in make_windows(steps, length=length):
                        rec["user_id"] = uid
                        rec["source_file"] = path.name
                        rec["split"] = split
                        handles[split].write(json.dumps(rec) + "\n")
                        counts[split] += 1
                        n_written += 1
                        if n_written >= max_windows_per_file:
                            break
    finally:
        for h in handles.values():
            h.close()

    meta = {
        "dataset": "Microsoft GeoLife GPS Trajectories",
        "split_policy": "user-disjoint deterministic 70/15/15",
        "lengths": list(lengths),
        "counts": counts,
        "notes": "GPS jumps >250m per sample and duplicate-like steps <0.5m are filtered.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="GeoLife root containing Data/<user>/Trajectory/*.plt")
    ap.add_argument("--out", default="data/geolife_benchmark")
    ap.add_argument("--lengths", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--max_windows_per_file", type=int, default=128)
    args = ap.parse_args()
    counts = build_benchmark(args.root, args.out, args.lengths, args.max_windows_per_file)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
