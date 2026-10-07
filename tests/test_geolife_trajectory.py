import math
from pathlib import Path

from src.data.geolife_trajectory import (
    GeoPoint, read_plt, trajectory_steps, make_windows, user_split
)


def test_geolife_parser_and_window(tmp_path: Path):
    p = tmp_path / "x.plt"
    p.write_text(
        "Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n0,2,255,My Track,0,0,2,8421376\n0\n"
        "39.000000,116.000000,0,100,0,2008-01-01,00:00:00\n"
        "39.000100,116.000000,0,101,0,2008-01-01,00:00:05\n"
        "39.000200,116.000100,0,102,0,2008-01-01,00:00:10\n",
        encoding="utf-8",
    )
    pts = read_plt(p)
    assert len(pts) == 3
    steps = trajectory_steps(pts, max_step_m=100)
    assert len(steps) == 2
    recs = list(make_windows(steps, length=2))
    assert len(recs) == 1
    rec = recs[0]
    assert rec["distance_m"] > 0
    assert -math.pi <= rec["bearing_rad"] <= math.pi
    assert 0 <= rec["bearing_sector_8"] < 8


def test_user_split_is_disjoint_and_deterministic():
    ids = [f"{i:03d}" for i in range(20)]
    a = user_split(ids)
    b = user_split(reversed(ids))
    assert a == b
    assert set(a.values()) == {"train", "val", "test"}
