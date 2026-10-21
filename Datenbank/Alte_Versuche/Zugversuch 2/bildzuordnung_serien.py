#!/usr/bin/env python3
"""Ordnet Versuchsfotos der Serien Z-A bis Z-G lokal den Prüfkörpern zu."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
SERIES_LETTERS = tuple("ABCDEFG")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CSV_HEADERS = ["Originaldateiname", "Prüfkörpernummer", "Aufnahmereihenfolge", "Dateipfad", "Prüfstatus"]
MINIMUM_CANDIDATES = 45
CYAN_RATIO_THRESHOLD = 0.004
MINIMUM_DIRECT_OCR = 25


@dataclass(frozen=True)
class SeriesContext:
    letter: str

    @property
    def name(self) -> str:
        return f"Z-{self.letter}"

    @property
    def series_dir(self) -> Path:
        return BASE_DIR / f"Serie {self.name}"

    @property
    def image_dir(self) -> Path:
        return self.series_dir / "Bilder"

    @property
    def output_dir(self) -> Path:
        return self.image_dir / "Sortiert"

    @property
    def protocol_csv(self) -> Path:
        return self.series_dir / "Protokoll" / f"Z_{self.letter}.csv"

    @property
    def assignment_csv(self) -> Path:
        return self.output_dir / f"Bildzuordnung_{self.name}.csv"

    @property
    def html_overview(self) -> Path:
        return self.output_dir / "index.html"

    @property
    def manifest(self) -> Path:
        return self.output_dir / ".bildzuordnung_manifest.json"


@dataclass
class Photo:
    path: Path
    captured_at: datetime
    order: int
    cyan_ratio: float = 0.0
    red_ratio: float = 0.0
    ocr_texts: tuple[str, ...] = ()
    ocr_numbers: tuple[int, ...] = ()


@dataclass
class Assignment:
    photo: Photo
    specimen: str
    is_number_photo: bool
    status: str
    destination: Path | None = None


@dataclass
class AnalysisResult:
    context: SeriesContext
    photos: list[Photo]
    expected_specimens: list[str]
    assignments: list[Assignment]
    directly_read: dict[str, bool]
    chronology: str
    ocr_engine: str
    candidate_count: int


VISION_SWIFT = r'''import Foundation
import Vision
import AppKit
for path in CommandLine.arguments.dropFirst() {
    let url = URL(fileURLWithPath: path)
    guard let image = NSImage(contentsOf: url), let data = image.tiffRepresentation,
          let rep = NSBitmapImageRep(data: data), let cgImage = rep.cgImage else { continue }
    do {
        var observations: [[String: Any]] = []
        for orientation: CGImagePropertyOrientation in [.up, .down] {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = false
            request.recognitionLanguages = ["de-DE", "en-US"]
            try VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:]).perform([request])
            observations.append(contentsOf: (request.results ?? []).compactMap { observation in
                guard let candidate = observation.topCandidates(1).first else { return nil }
                return ["text": candidate.string, "confidence": Double(candidate.confidence)]
            })
        }
        let object: [String: Any] = ["filename": url.lastPathComponent, "observations": observations]
        let json = try JSONSerialization.data(withJSONObject: object, options: [])
        print(String(data: json, encoding: .utf8)!)
    } catch {
        let object: [String: Any] = ["filename": url.lastPathComponent, "observations": []]
        let json = try JSONSerialization.data(withJSONObject: object, options: [])
        print(String(data: json, encoding: .utf8)!)
    }
}
'''


class LocalOCR:
    def __init__(self) -> None:
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._vision_executable: Path | None = None
        self.engine = "Tesseract"

    def __enter__(self) -> "LocalOCR":
        if shutil.which("swiftc"):
            try:
                self._temporary = tempfile.TemporaryDirectory(prefix="bildzuordnung_vision_")
                temp = Path(self._temporary.name)
                source, executable = temp / "vision_ocr.swift", temp / "vision_ocr"
                source.write_text(VISION_SWIFT, encoding="utf-8")
                subprocess.run(["swiftc", str(source), "-o", str(executable)], check=True, capture_output=True, text=True)
                self._vision_executable = executable
                self.engine = "macOS Vision"
            except (OSError, subprocess.SubprocessError):
                self._vision_executable = None
        if self._vision_executable is None and not shutil.which("tesseract"):
            raise RuntimeError("Weder macOS Vision noch Tesseract sind lokal verfügbar.")
        return self

    def __exit__(self, *_: object) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()

    def recognize(self, paths: list[Path]) -> dict[str, tuple[str, ...]]:
        if self._vision_executable is None:
            return self._tesseract(paths)
        result = subprocess.run([str(self._vision_executable), *(str(path) for path in paths)], check=True, capture_output=True, text=True)
        output: dict[str, tuple[str, ...]] = {}
        for line in result.stdout.splitlines():
            if line.strip():
                item = json.loads(line)
                output[item["filename"]] = tuple(
                    observation["text"] for observation in item.get("observations", [])
                    if float(observation.get("confidence", 0.0)) >= 0.2
                )
        return output

    @staticmethod
    def _tesseract(paths: list[Path]) -> dict[str, tuple[str, ...]]:
        executable = shutil.which("tesseract")
        if not executable:
            raise RuntimeError("Tesseract ist nicht verfügbar.")
        output: dict[str, tuple[str, ...]] = {}
        with tempfile.TemporaryDirectory(prefix="bildzuordnung_tesseract_") as temp_name:
            temp = Path(temp_name)
            for path in paths:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    width, height = image.size
                    crop = image.crop((int(width * .20), int(height * .16), int(width * .80), int(height * .76)))
                    crop.thumbnail((1800, 1200))
                    prepared = temp / f"{path.stem}.png"
                    crop.save(prepared)
                result = subprocess.run([executable, str(prepared), "stdout", "-l", "eng", "--psm", "11"], capture_output=True, text=True)
                output[path.name] = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        return output


def natural_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def read_exif_datetime(path: Path) -> datetime | None:
    with Image.open(path) as image:
        exif = image.getexif()
        value = exif.get(36867) or exif.get(306)
    try:
        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S") if value else None
    except ValueError:
        return None


def load_photos(context: SeriesContext) -> tuple[list[Photo], str]:
    paths = sorted((path for path in context.image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=natural_number)
    if not paths:
        raise RuntimeError(f"{context.name}: keine Bilder gefunden.")
    dated = [(path, read_exif_datetime(path)) for path in paths]
    missing = [path.name for path, value in dated if value is None]
    if missing:
        raise RuntimeError(f"{context.name}: Chronologie nicht zuverlässig; EXIF-Zeit fehlt bei " + ", ".join(missing))
    exif_times = [value for _, value in dated]
    if len(set(exif_times)) != len(exif_times):
        raise RuntimeError(f"{context.name}: Chronologie nicht zuverlässig; doppelte EXIF-Zeiten.")
    if [path for path, _ in sorted(dated, key=lambda item: item[1])] != [path for path, _ in dated]:
        raise RuntimeError(f"{context.name}: EXIF- und Dateinamenreihenfolge widersprechen sich.")
    return [Photo(path, value, index) for index, (path, value) in enumerate(dated, 1)], "EXIF und Dateinamen stimmen vollständig überein"


def load_protocol_specimens(context: SeriesContext) -> list[str]:
    with context.protocol_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    specimens = [row["Prüfkörper"].strip() for row in rows if row.get("Prüfkörper", "").strip()]
    expected = {f"{context.name}-{number:02d}" for number in range(1, 36)}
    if len(specimens) != 35 or len(set(specimens)) != 35 or set(specimens) != expected:
        raise RuntimeError(f"{context.name}: Protokoll enthält nicht genau die Prüfkörper 01 bis 35.")
    return sorted(specimens, key=lambda value: int(value.rsplit("-", 1)[1]), reverse=True)


def color_ratios(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        image.draft("RGB", (684, 456))
        small = ImageOps.exif_transpose(image).convert("RGB").resize((342, 228))
    array = np.asarray(small)
    center = array[35:180, 60:285]
    red, green, blue = (center[:, :, index].astype(np.int16) for index in range(3))
    cyan = ((green - red > 20) & (blue - red > 5) & (green > 80)).mean()
    red_ink = ((red - green > 45) & (red - blue > 35) & (red > 95)).mean()
    return float(cyan), float(red_ink)


def identify_number_candidates(photos: list[Photo]) -> list[Photo]:
    for photo in photos:
        photo.cyan_ratio, photo.red_ratio = color_ratios(photo.path)
    ranked = sorted(photos, key=lambda item: (item.cyan_ratio, item.red_ratio), reverse=True)
    selected = {photo.path.name for photo in ranked[:MINIMUM_CANDIDATES]}
    selected.update(photo.path.name for photo in photos if photo.cyan_ratio >= CYAN_RATIO_THRESHOLD)
    return [photo for photo in photos if photo.path.name in selected]


def extract_specimen_numbers(texts: Iterable[str]) -> tuple[int, ...]:
    found: set[int] = set()
    for text in texts:
        normalized = text.upper().replace("А", "A").replace("–", "-").replace("—", "-").replace("−", "-")
        pattern = r"(?:Z|2|7|8|E|S)\s*[-+*.]?\s*[A-Z0-9<]\s*[^0-9]{0,6}0?(\d{1,2})(?!\d)"
        for match in re.finditer(pattern, normalized):
            number = int(match.group(1))
            if 1 <= number <= 35:
                found.add(number)
    return tuple(sorted(found))


def align_candidates(candidates: list[Photo], expected_specimens: list[str]) -> dict[str, Photo]:
    if len(candidates) < 35:
        raise RuntimeError(f"Nur {len(candidates)} Nummernfoto-Kandidaten für 35 Prüfkörper erkannt.")

    def scene_score(candidate: Photo) -> int:
        # Nummerntafeln besitzen deutlich mehr türkisfarbene Fläche und rote
        # Beschriftung als die Bruchfotos. Dieses Bildmerkmal erhält bewusst
        # mehr Gewicht als ein zufälliger OCR-Treffer auf einem Bruchbild.
        return min(24, round(candidate.cyan_ratio * 900 + candidate.red_ratio * 350))

    def match_score(candidate: Photo, expected: int) -> int:
        scene = scene_score(candidate)
        if expected in candidate.ocr_numbers:
            return 8 + scene
        if not candidate.ocr_numbers:
            return scene
        # Eine fremde OCR-Zahl ist nur ein schwaches Gegenindiz: Bei
        # durchgestrichenen Beschriftungen werden z. B. "08" und "10" gern
        # zu "0810" zusammengezogen. Das Nummerntafel-Motiv bleibt maßgeblich.
        return -2 + scene

    def solve_order(expected_numbers: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        @lru_cache(maxsize=None)
        def solve(candidate_index: int, expected_index: int) -> tuple[int, tuple[int, ...]]:
            if expected_index == len(expected_numbers):
                return (-(len(candidates) - candidate_index), ())
            if len(candidates) - candidate_index < len(expected_numbers) - expected_index:
                return (-10**9, ())
            assign_score, assign_path = solve(candidate_index + 1, expected_index + 1)
            assign_score += match_score(candidates[candidate_index], expected_numbers[expected_index])
            assign_path = (candidate_index,) + assign_path
            skip_score, skip_path = solve(candidate_index + 1, expected_index)
            skip_score -= 1
            return (assign_score, assign_path) if assign_score >= skip_score else (skip_score, skip_path)

        return solve(0, 0)

    # Die Serien wurden nicht durchgängig in derselben Richtung fotografiert.
    # Deshalb werden beide Zählrichtungen samt möglichem zyklischem Startpunkt
    # geprüft; die OCR- und Bildmerkmale bestimmen die plausible Reihenfolge.
    descending = tuple(range(35, 0, -1))
    ascending = tuple(range(1, 36))
    orders: list[tuple[int, ...]] = []
    for base in (descending, ascending):
        orders.extend(base[offset:] + base[:offset] for offset in range(35))
    best_score, best_indices, best_order = -10**9, (), ()
    for order in orders:
        score, indices = solve_order(order)
        if score > best_score:
            best_score, best_indices, best_order = score, indices, order

    if best_score <= -10**8 or len(best_indices) != 35:
        raise RuntimeError("Nummernfotos konnten nicht plausibel ausgerichtet werden.")
    prefix = expected_specimens[0].rsplit("-", 1)[0]
    return {
        f"{prefix}-{number:02d}": candidates[index]
        for number, index in zip(best_order, best_indices, strict=True)
    }


def build_assignments(photos: list[Photo], expected: list[str], markers: dict[str, Photo]) -> tuple[list[Assignment], dict[str, bool]]:
    marker_by_name = {photo.path.name: specimen for specimen, photo in markers.items()}
    direct = {specimen: int(specimen.rsplit("-", 1)[1]) in photo.ocr_numbers for specimen, photo in markers.items()}
    assignments: list[Assignment] = []
    current = ""
    for photo in photos:
        is_number = photo.path.name in marker_by_name
        if is_number:
            current = marker_by_name[photo.path.name]
        if not current:
            assignments.append(Assignment(photo, "UNKLAR", False, "PRUEFEN_OHNE_ZUORDNUNG"))
            continue
        status = "OK_NUMMERNFOTO" if is_number else "OK_ZUGEORDNET"
        if not direct[current]:
            status = "PRUEFEN_OCR_INFERIERT"
        assignments.append(Assignment(photo, current, is_number, status))
    for specimen in expected:
        group = [item for item in assignments if item.specimen == specimen]
        if not any(not item.is_number_photo for item in group):
            for item in group:
                item.status = "PRUEFEN_KEINE_BRUCHBILDER"
    return assignments, direct


def analyze_series(context: SeriesContext, photos: list[Photo], chronology: str, expected: list[str], ocr: LocalOCR) -> AnalysisResult:
    candidates = identify_number_candidates(photos)
    recognized = ocr.recognize([photo.path for photo in candidates])
    for photo in candidates:
        photo.ocr_texts = recognized.get(photo.path.name, ())
        photo.ocr_numbers = extract_specimen_numbers(photo.ocr_texts)
    markers = align_candidates(candidates, expected)
    assignments, direct = build_assignments(photos, expected, markers)
    direct_count = sum(direct.values())
    assigned = {item.specimen for item in assignments}
    if direct_count < MINIMUM_DIRECT_OCR:
        raise RuntimeError(f"{context.name}: nur {direct_count}/35 Nummern direkt per OCR erkannt; keine Ausgabe erzeugt.")
    if "UNKLAR" in assigned or assigned != set(expected):
        raise RuntimeError(f"{context.name}: nicht alle Fotos konnten lückenlos zugeordnet werden.")
    return AnalysisResult(context, photos, expected, assignments, direct, chronology, ocr.engine, len(candidates))


def clean_previous_generated_files(context: SeriesContext) -> None:
    if not context.manifest.exists():
        return
    try:
        entries = json.loads(context.manifest.read_text(encoding="utf-8")).get("generated_files", [])
    except (json.JSONDecodeError, OSError):
        return
    root = context.output_dir.resolve()
    for relative in entries:
        target = (context.output_dir / relative).resolve()
        if target != root and root in target.parents and (target.is_file() or target.is_symlink()):
            target.unlink()
    for directory in sorted((path for path in context.output_dir.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def destination_for(result: AnalysisResult, assignment: Assignment) -> Path:
    base = (result.context.output_dir / "Zur_Pruefung" / assignment.specimen
            if assignment.status.startswith("PRUEFEN_") else result.context.output_dir / assignment.specimen)
    return base / (f"Nummer{assignment.photo.path.suffix.upper()}" if assignment.is_number_photo else assignment.photo.path.name)


def copy_and_make_previews(result: AnalysisResult) -> list[str]:
    generated: list[str] = []
    for assignment in result.assignments:
        destination = destination_for(result, assignment)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(assignment.photo.path, destination)
        assignment.destination = destination
        generated.append(str(destination.relative_to(result.context.output_dir)))
        preview = (result.context.output_dir / "_Vorschau" / destination.relative_to(result.context.output_dir)).with_suffix(".jpg")
        preview.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(assignment.photo.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((420, 280))
            image.save(preview, format="JPEG", quality=82, optimize=True)
        generated.append(str(preview.relative_to(result.context.output_dir)))
    return generated


def relative_project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def write_csv(result: AnalysisResult) -> None:
    with result.context.assignment_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(CSV_HEADERS)
        for item in result.assignments:
            writer.writerow([item.photo.path.name, item.specimen, item.photo.order,
                             relative_project_path(item.destination or item.photo.path), item.status])


def write_html(result: AnalysisResult) -> None:
    direct_count = sum(result.directly_read.values())
    review_count = sum(item.status.startswith("PRUEFEN_") for item in result.assignments)
    break_count = sum(not item.is_number_photo for item in result.assignments)
    sections = []
    status_labels = {
        "OK_NUMMERNFOTO": "OK", "OK_ZUGEORDNET": "OK",
        "PRUEFEN_OCR_INFERIERT": "Manuell prüfen: OCR-inferiert",
        "PRUEFEN_KEINE_BRUCHBILDER": "Manuell prüfen: keine Bruchbilder",
        "PRUEFEN_OHNE_ZUORDNUNG": "Manuell prüfen: ohne Zuordnung",
    }
    for specimen in sorted(result.expected_specimens, key=lambda value: int(value.rsplit("-", 1)[1])):
        group = [item for item in result.assignments if item.specimen == specimen]
        review = any(item.status.startswith("PRUEFEN_") for item in group)
        cards = []
        for item in group:
            assert item.destination is not None
            full = item.destination.relative_to(result.context.output_dir).as_posix()
            preview = (result.context.output_dir / "_Vorschau" / item.destination.relative_to(result.context.output_dir)).with_suffix(".jpg")
            kind = "Nummernfoto" if item.is_number_photo else "Bruchbild"
            cards.append(f'''<figure class="card"><a href="{html.escape(full)}"><img loading="lazy" src="{html.escape(preview.relative_to(result.context.output_dir).as_posix())}" alt="{html.escape(item.photo.path.name)}"></a><figcaption><strong>{kind}</strong><br>{html.escape(item.photo.path.name)}<br><span>{html.escape(status_labels.get(item.status, item.status))}</span></figcaption></figure>''')
        sections.append(f'''<section class="group{' review' if review else ''}"><h2>{html.escape(specimen)} <span class="badge">{'Manuell prüfen' if review else 'OK'}</span></h2><div class="cards">{''.join(cards)}</div></section>''')
    document = f'''<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bildzuordnung Serie {result.context.name}</title><style>
:root{{font-family:Arial,sans-serif}}body{{margin:0;background:#f4f6f8;color:#1f2933}}header{{background:#17324d;color:white;padding:24px 5vw}}header h1{{margin:0 0 12px;font-size:28px}}.summary{{display:flex;flex-wrap:wrap;gap:12px}}.summary span{{background:#284b6d;border-radius:6px;padding:8px 12px}}main{{width:min(1500px,94vw);margin:24px auto 60px}}.notice{{background:#fff6d8;border-left:5px solid #d99b00;padding:14px 16px;margin-bottom:22px}}.group{{background:white;border-radius:8px;margin:18px 0;padding:16px;box-shadow:0 2px 8px #00000012}}.group.review{{border:3px solid #e0a000}}h2{{margin:0 0 14px;font-size:21px}}.badge{{font-size:12px;font-weight:normal;background:#dcefe3;color:#17603a;padding:4px 8px;border-radius:999px}}.review .badge{{background:#ffe7a3;color:#754c00}}.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}.card{{margin:0;border:1px solid #d6dce2;border-radius:6px;overflow:hidden;background:#fafbfc}}.card img{{width:100%;height:170px;object-fit:cover;display:block;background:#e9edf1}}figcaption{{padding:9px 10px;font-size:13px;line-height:1.45}}figcaption span{{color:#52606d}}</style></head><body><header><h1>Bildzuordnung Serie {result.context.name}</h1><div class="summary"><span>35 Prüfkörper gruppiert</span><span>{direct_count} Nummern direkt per OCR</span><span>{len(result.assignments)} Fotos insgesamt</span><span>{break_count} Bruch-/Übersichtsbilder</span><span>{review_count} Fotos zur Prüfung</span></div></header><main><div class="notice"><strong>Reihenfolge:</strong> {html.escape(result.chronology)}.<br><strong>Lokale OCR:</strong> {html.escape(result.ocr_engine)}. Gelb umrandete Gruppen liegen unter <code>Zur_Pruefung</code>.</div>{''.join(sections)}</main></body></html>'''
    result.context.html_overview.write_text(document, encoding="utf-8")


def write_outputs(result: AnalysisResult) -> None:
    context = result.context
    context.output_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_generated_files(context)
    generated = copy_and_make_previews(result)
    write_csv(result)
    write_html(result)
    generated.extend([context.assignment_csv.name, context.html_overview.name])
    context.manifest.write_text(json.dumps({"generated_files": generated}, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(result: AnalysisResult, dry_run: bool) -> None:
    direct = sum(result.directly_read.values())
    inferred = sorted((specimen for specimen, value in result.directly_read.items() if not value), key=lambda value: int(value.rsplit("-", 1)[1]))
    review = [item for item in result.assignments if item.status.startswith("PRUEFEN_")]
    no_breaks = sorted({item.specimen for item in review if item.status == "PRUEFEN_KEINE_BRUCHBILDER"})
    print(f"{'TEST' if dry_run else 'FERTIG'} {result.context.name}: {len(result.photos)} Fotos, 35 Prüfkörper, OCR direkt {direct}, inferiert {len(inferred)}, Prüfung {len(review)}, Kandidaten {result.candidate_count}")
    if inferred:
        print("  OCR-inferiert: " + ", ".join(inferred))
    if no_breaks:
        print("  Ohne Bruchbilder: " + ", ".join(no_breaks))
    if not dry_run:
        print(f"  CSV: {result.context.assignment_csv}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="+", default=list(SERIES_LETTERS), metavar="BUCHSTABE")
    parser.add_argument("--dry-run", action="store_true", help="Nur analysieren; keine Dateien schreiben.")
    args = parser.parse_args()
    args.series = [value.upper().removeprefix("Z-") for value in args.series]
    invalid = [value for value in args.series if value not in SERIES_LETTERS]
    if invalid:
        parser.error("Ungültige Serie: " + ", ".join(invalid))
    args.series = list(dict.fromkeys(args.series))
    return args


def main() -> None:
    args = parse_arguments()
    contexts = [SeriesContext(letter) for letter in args.series]
    prepared: dict[str, tuple[list[Photo], str, list[str]]] = {}
    for context in contexts:
        photos, chronology = load_photos(context)
        prepared[context.letter] = (photos, chronology, load_protocol_specimens(context))
    with LocalOCR() as ocr:
        results = [analyze_series(context, *prepared[context.letter], ocr) for context in contexts]
    if not args.dry_run:
        for result in results:
            write_outputs(result)
    for result in results:
        print_summary(result, args.dry_run)


if __name__ == "__main__":
    main()
