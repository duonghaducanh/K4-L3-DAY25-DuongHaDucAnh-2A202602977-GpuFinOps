"""Shared helpers for missions: locate data + load CSVs as list-of-dicts/pandas."""
from __future__ import annotations
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)  # make `finops` importable when run as a script

# Reports contain unicode (arrows, check marks). On Windows the default console
# codec is cp1252, which raises UnicodeEncodeError on print(); force UTF-8 so the
# missions can run anywhere without a PYTHONIOENCODING env var.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def load_csv(name: str) -> list[dict]:
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def num(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def catalog_by_type() -> dict:
    return {r["gpu_type"]: r for r in load_csv("price_catalog.csv")}
