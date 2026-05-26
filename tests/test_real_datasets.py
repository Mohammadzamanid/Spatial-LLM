"""
tests/test_real_datasets.py
Tests for the real GeoNames dataset loader (parsing + QA generation).
Uses a small fixture in exact GeoNames tab-separated format.
"""
import random
import pytest
from pathlib import Path

from src.data.real_datasets import (
    parse_geonames, geonames_to_qa, _population_bucket, GEONAMES_COLUMNS,
)

# Real cities in exact GeoNames 19-column format
SAMPLE_ROWS = [
    ["1850147", "Tokyo", "Tokyo", "Edo", "35.6895", "139.69171", "P", "PPLC",
     "JP", "", "13", "", "", "", "37977000", "40", "44", "Asia/Tokyo", "2023-03-08"],
    ["2643743", "London", "London", "LON", "51.50853", "-0.12574", "P", "PPLC",
     "GB", "GB", "ENG", "GLA", "", "", "8961989", "25", "14", "Europe/London", "2023-09-06"],
    ["112931", "Tehran", "Tehran", "THR", "35.69439", "51.42151", "P", "PPLC",
     "IR", "", "26", "", "", "", "8154051", "1189", "1191", "Asia/Tehran", "2022-11-10"],
]


@pytest.fixture
def geonames_file(tmp_path):
    p = tmp_path / "cities.txt"
    with open(p, "w", encoding="utf-8") as f:
        for row in SAMPLE_ROWS:
            f.write("\t".join(row) + "\n")
    return p


def test_parse_returns_all_cities(geonames_file):
    records = list(parse_geonames(geonames_file))
    assert len(records) == 3


def test_parsed_fields_typed_correctly(geonames_file):
    records = list(parse_geonames(geonames_file))
    tokyo = records[0]
    assert tokyo["name"] == "Tokyo"
    assert isinstance(tokyo["latitude"], float)
    assert abs(tokyo["latitude"] - 35.6895) < 1e-4
    assert isinstance(tokyo["population"], int)
    assert tokyo["population"] == 37977000
    assert tokyo["country_code"] == "JP"
    assert tokyo["timezone"] == "Asia/Tokyo"


def test_coordinates_in_valid_range(geonames_file):
    for r in parse_geonames(geonames_file):
        assert -90 <= r["latitude"] <= 90
        assert -180 <= r["longitude"] <= 180


def test_qa_generation_has_required_fields(geonames_file):
    random.seed(0)
    records = list(parse_geonames(geonames_file))
    qa_pairs = list(geonames_to_qa(iter(records)))
    assert len(qa_pairs) == 3
    for qa in qa_pairs:
        assert "question" in qa and len(qa["question"]) > 0
        assert "answer" in qa and len(qa["answer"]) > 0
        assert -90 <= qa["lat"] <= 90
        assert -180 <= qa["lon"] <= 180
        assert "city" in qa


def test_qa_max_records_limit(geonames_file):
    records = list(parse_geonames(geonames_file))
    qa = list(geonames_to_qa(iter(records), max_records=2))
    assert len(qa) == 2


def test_population_buckets():
    assert "megacity" in _population_bucket(15_000_000)
    assert "large city" in _population_bucket(5_000_000)
    assert "mid-sized" in _population_bucket(500_000)
    assert "small" in _population_bucket(50_000)


def test_parser_tolerant_of_short_rows(tmp_path):
    """Rows missing trailing fields should still parse (padded)."""
    p = tmp_path / "short.txt"
    # Only 8 fields (missing trailing), but has coords
    short_row = ["123", "Paris", "Paris", "", "48.8566", "2.3522", "P", "PPLC"]
    with open(p, "w") as f:
        f.write("\t".join(short_row) + "\n")
    records = list(parse_geonames(p))
    assert len(records) == 1
    assert records[0]["name"] == "Paris"
    assert abs(records[0]["latitude"] - 48.8566) < 1e-4
