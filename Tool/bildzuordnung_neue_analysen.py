#!/usr/bin/env python3
"""Sortiert eine neue Zugversuchsserie lokal anhand ihrer Nummernfotos."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_SCRIPT = PROJECT_ROOT / "Datenbank" / "Alte_Versuche" / "Zugversuch 2" / "bildzuordnung_serien.py"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CSV_HEADERS = [
    "Prüfkörpernummer",
    "Originalbild",
    "Zuordnung",
    "Prüfstatus",
    "Aufnahmereihenfolge",
    "Nummernfoto",
]


def load_legacy_module():
    """Lädt OCR-, EXIF- und Bildmerkmallogik der Serien Z-A bis Z-G."""

    spec = importlib.util.spec_from_file_location("bildzuordnung_serien", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Vorhandene Bildzuordnungslogik kann nicht geladen werden: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Photo:
    path: Path
    captured_at: datetime
    order: int
    ocr_texts: tuple[str, ...] = ()
    ocr_numbers: tuple[int, ...] = ()
    marker_number: int | None = None
    marker_score: float = 0.0


@dataclass
class Assignment:
    photo: Photo
    specimen: str
    is_number_photo: bool
    status: str
    destination: Path | None = None


def natural_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def load_photos(input_dir: Path, legacy) -> list[Photo]:
    # Ausschließlich direkte Bilddateien lesen. Vorhandene Ausgabeordner wie
    # Sortiert/ werden damit bei jedem erneuten Lauf sicher übersprungen.
    paths = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_number,
    )
    if not paths:
        raise RuntimeError(f"Keine Bilder direkt unter {input_dir} gefunden.")
    dated = [(path, legacy.read_exif_datetime(path)) for path in paths]
    missing = [path.name for path, value in dated if value is None]
    if missing:
        raise RuntimeError("Chronologie nicht zuverlässig; EXIF-Zeit fehlt bei: " + ", ".join(missing))
    times = [value for _, value in dated]
    if len(set(times)) != len(times):
        raise RuntimeError("Chronologie nicht zuverlässig; doppelte EXIF-Zeitstempel gefunden.")
    if [path for path, _ in sorted(dated, key=lambda item: item[1])] != paths:
        raise RuntimeError("EXIF- und Dateinamenreihenfolge widersprechen sich.")
    return [Photo(path, captured_at, order) for order, (path, captured_at) in enumerate(dated, 1)]


def extract_series_numbers(texts: tuple[str, ...]) -> tuple[int, ...]:
    # Leerzeichen statt sichtbarer Trenner erlauben auch den Vision-Fall
    # ("Z-30 -", "03"), bei dem die Kennung in zwei Textblöcke zerfällt.
    normalized = " ".join(texts).upper().replace("–", "-").replace("—", "-").replace("−", "-")
    found: set[int] = set()
    # "30" ist hier die Serienkennung. (?!\d) verhindert Treffer in "307".
    for match in re.finditer(r"(?:Z|7|2|T)\s*[-.]?\s*30(?!\d)[\s\-.:]{0,8}(\d{1,2})(?!\d)", normalized):
        number = int(match.group(1))
        if 1 <= number <= 99:
            found.add(number)
    # Bei Z-30-44 wurde das Z in einen eigenen OCR-Block verschoben. Der
    # Fallback ist absichtlich nur aktiv, wenn die reguläre Erkennung leer ist.
    if not found and "PA" in normalized:
        for match in re.finditer(r"(?:^|\s)-\s*30(?!\d)[\s\-.:]{0,8}(\d{1,2})(?!\d)", normalized):
            number = int(match.group(1))
            if 1 <= number <= 99:
                found.add(number)
    return tuple(sorted(found))


def longest_nondecreasing_markers(photos: list[Photo]) -> set[int]:
    """Filtert zufällig sichtbare Fremdnummern auf Bruchbildern heraus."""

    candidates = [(index, photo.ocr_numbers[0]) for index, photo in enumerate(photos) if len(photo.ocr_numbers) == 1]
    if not candidates:
        raise RuntimeError("Keine eindeutigen Nummernfotos erkannt.")
    scores = [(1, photos[index].marker_score) for index, _ in candidates]
    previous = [-1] * len(candidates)
    for right, (_, right_number) in enumerate(candidates):
        for left in range(right):
            proposed = (scores[left][0] + 1, scores[left][1] + photos[candidates[right][0]].marker_score)
            if candidates[left][1] <= right_number and proposed > scores[right]:
                scores[right] = proposed
                previous[right] = left
    cursor = max(range(len(candidates)), key=scores.__getitem__)
    selected: set[int] = set()
    while cursor >= 0:
        selected.add(candidates[cursor][0])
        cursor = previous[cursor]
    return selected


def recognize_markers(photos: list[Photo], legacy) -> str:
    # Bei neuen Serien ist die Anzahl der Prüfkörper nicht vorab bekannt.
    # Deshalb wird die vorhandene lokale OCR einmal auf alle Fotos angewandt;
    # die Aufnahmereihenfolge filtert anschließend Fremdnummern im Hintergrund.
    with legacy.LocalOCR() as ocr:
        recognized = ocr.recognize([photo.path for photo in photos])
        engine = ocr.engine
    for photo in photos:
        photo.ocr_texts = recognized.get(photo.path.name, ())
        photo.ocr_numbers = extract_series_numbers(photo.ocr_texts)
        if photo.ocr_numbers:
            cyan_ratio, red_ratio = legacy.color_ratios(photo.path)
            photo.marker_score = cyan_ratio * 900 + red_ratio * 350
    selected = longest_nondecreasing_markers(photos)
    for index in selected:
        photos[index].marker_number = photos[index].ocr_numbers[0]
    # Direkt aufeinanderfolgende Tafelfotos derselben Probe beibehalten, auch
    # wenn eine der beiden Vision-Orientierungen eine zusätzliche Fehlzahl las.
    for index, photo in enumerate(photos[:-1]):
        next_marker = photos[index + 1].marker_number
        if photo.marker_number is None and next_marker is not None and next_marker in photo.ocr_numbers:
            photo.marker_number = next_marker
    return engine


def build_assignments(photos: list[Photo]) -> tuple[list[Assignment], list[int]]:
    assignments: list[Assignment] = []
    current: int | None = None
    for photo in photos:
        is_marker = photo.marker_number is not None
        if is_marker:
            current = photo.marker_number
        if current is None:
            assignments.append(Assignment(photo, "UNKLAR", False, "PRUEFEN_OHNE_ZUORDNUNG"))
        else:
            specimen = f"Z-30-{current:02d}"
            assignments.append(
                Assignment(photo, specimen, is_marker, "OK_NUMMERNFOTO" if is_marker else "OK_ZUGEORDNET")
            )

    recognized = sorted({photo.marker_number for photo in photos if photo.marker_number is not None})
    missing = [number for number in range(recognized[0], recognized[-1] + 1) if number not in recognized]
    for specimen in {item.specimen for item in assignments if item.specimen != "UNKLAR"}:
        group = [item for item in assignments if item.specimen == specimen]
        if not any(not item.is_number_photo for item in group):
            for item in group:
                item.status = "PRUEFEN_KEINE_BRUCHBILDER"
    return assignments, missing


def evenly_spaced(items: list[Assignment], limit: int = 3) -> list[Assignment]:
    if len(items) <= limit:
        return items
    indices = (0, (len(items) - 1) // 2, len(items) - 1)
    return [items[index] for index in indices]


def write_outputs(input_dir: Path, assignments: list[Assignment], missing: list[int], engine: str) -> Path:
    output_dir = input_dir / "Sortiert"
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "Zur_Pruefung"
    review_dir.mkdir(exist_ok=True)

    for item in assignments:
        target_root = review_dir if item.status.startswith("PRUEFEN_") else output_dir
        specimen_dir = target_root / item.specimen
        specimen_dir.mkdir(parents=True, exist_ok=True)
        filename = f"Nummer_{item.photo.path.name}" if item.is_number_photo else item.photo.path.name
        destination = specimen_dir / filename
        shutil.copy2(item.photo.path, destination)
        item.destination = destination

    specimens = sorted({item.specimen for item in assignments if item.specimen != "UNKLAR"})
    for specimen in specimens:
        group = [item for item in assignments if item.specimen == specimen and not item.is_number_photo]
        if not group or any(item.status.startswith("PRUEFEN_") for item in group):
            continue
        analysis_dir = output_dir / specimen / "Analyseauswahl"
        analysis_dir.mkdir(exist_ok=True)
        for item in evenly_spaced(group):
            assert item.destination is not None
            link = analysis_dir / item.destination.name
            if not link.exists() and not link.is_symlink():
                os.symlink(Path("..") / item.destination.name, link)

    csv_path = output_dir / "Bildzuordnung.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(CSV_HEADERS)
        for item in assignments:
            assert item.destination is not None
            writer.writerow(
                [item.specimen, item.photo.path.name, item.destination.relative_to(output_dir), item.status,
                 item.photo.order, "JA" if item.is_number_photo else "NEIN"]
            )
        for number in missing:
            writer.writerow([f"Z-30-{number:02d}", "", "", "PRUEFEN_NUMMERNFOTO_FEHLT", "", "NEIN"])

    notes = [
        "Automatisch erzeugte Prüfhinweise",
        "",
        *(f"- Kein Nummernfoto für Z-30-{number:02d} erkannt." for number in missing),
    ]
    (review_dir / "Pruefhinweise.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    manifest = {
        "input_directory": str(input_dir.resolve()),
        "chronology": "EXIF und natürliche Dateinamenfolge stimmen vollständig überein",
        "local_ocr": engine,
        "openai_api_used": False,
        "failure_classification_performed": False,
        "recognized_specimens": specimens,
        "missing_numbers": [f"Z-30-{number:02d}" for number in missing],
        "assigned_images": sum(item.specimen != "UNKLAR" for item in assignments),
        "review_images": sum(item.status.startswith("PRUEFEN_") for item in assignments),
    }
    (output_dir / "Sortierprotokoll.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Ordner mit den unsortierten Originalbildern")
    parser.add_argument("--dry-run", action="store_true", help="Nur lokal erkennen; keine Ausgaben schreiben")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Eingabeordner fehlt: {input_dir}")
    legacy = load_legacy_module()
    photos = load_photos(input_dir, legacy)
    engine = recognize_markers(photos, legacy)
    assignments, missing = build_assignments(photos)
    specimens = sorted({item.specimen for item in assignments if item.specimen != "UNKLAR"})
    review = [item for item in assignments if item.status.startswith("PRUEFEN_")]
    if not args.dry_run:
        csv_path = write_outputs(input_dir, assignments, missing, engine)
        print(f"CSV: {csv_path}")
    print(
        f"{'TEST' if args.dry_run else 'FERTIG'}: {len(specimens)} Prüfkörper, "
        f"{len(assignments) - len(review)}/{len(assignments)} Bilder eindeutig zugeordnet, "
        f"{len(review)} Bilder zur Prüfung."
    )
    if missing:
        print("Manuell prüfen (fehlende Nummernfotos): " + ", ".join(f"Z-30-{number:02d}" for number in missing))


if __name__ == "__main__":
    main()
