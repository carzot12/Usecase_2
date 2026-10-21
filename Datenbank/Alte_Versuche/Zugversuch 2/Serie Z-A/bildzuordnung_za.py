#!/usr/bin/env python3
"""Kompatibilitätsstarter für die Bildzuordnung der Serien Z-A bis Z-G."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "bildzuordnung_serien.py"),
    run_name="__main__",
)
