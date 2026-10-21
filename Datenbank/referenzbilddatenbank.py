#!/usr/bin/env python3
"""Erstellt eine serienübergreifende, manuell prüfbare Referenzbildauswahl."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "Datenbank"
EXPERIMENT_ROOT = DATABASE_DIR / "Alte_Versuche" / "Zugversuch 2"
REFERENCE_ROOT = DATABASE_DIR / "Referenzbilder"
OUTPUT_ROOT = PROJECT_ROOT / "Output"
RULES_PATH = DATABASE_DIR / "klassifikationsregeln.json"
MODES_PATH = DATABASE_DIR / "versagensarten.json"
EXTENDED_MODES_PATH = DATABASE_DIR / "versagensarten_erweitert.json"
MANIFEST_PATH = REFERENCE_ROOT / "referenzbilder_manifest.json"
SERIES_LETTERS = tuple("ABCDEFG")
TARGET_REFERENCES_PER_MODE = 3
MAX_REFERENCES_PER_MODE = 4
EXCLUSION_NOTE_TERMS = ("klemmbereich", "direkt an kb", "direkt am kb", "verloren", "überspeichert")
MANUAL_VISUAL_EXCLUSIONS = {
    "KA|Z-D-12|DSC01388.JPG": "Bruchbereich in der Gesamtaufnahme zu klein",
    "SA|Z-A-32|DSC01446.JPG": "Bruchbereich in der Gesamtaufnahme zu klein",
    "AA|Z-B-01|DSC01995.JPG": "Merkmal in der Gesamtaufnahme nicht eindeutig beurteilbar",
    "Bu|Z-G-27|DSC01117.JPG": "Bruchbereich in der Gesamtaufnahme zu klein",
    "GSF|Z-F-19|DSC01639.JPG": "Faserverlauf in der Gesamtaufnahme nicht ausreichend erkennbar",
    "GSF|Z-F-20|DSC01634.JPG": "Faserverlauf in der Gesamtaufnahme nicht ausreichend erkennbar",
    "HB|Z-E-21|DSC01208.JPG": "Bruchbild in der Gesamtaufnahme nicht ausreichend erkennbar",
    "HB|Z-G-27|DSC01117.JPG": "Bruchbild in der Gesamtaufnahme nicht ausreichend erkennbar",
    "sp HB|Z-A-14|DSC01517.JPG": "Bruchbild in der Gesamtaufnahme nicht ausreichend erkennbar",
    "KB|Z-E-18|DSC01220.JPG": "Klemmbereich in der Gesamtaufnahme nicht eindeutig beurteilbar",
}
MANUAL_CLASS_SPECIMEN_EXCLUSIONS = {
    ("KB", "Z-E-18"): "Einziger dokumentierter KB-Prüfkörper; vorhandene Ansichten erlauben keine eindeutige visuelle Bestätigung",
}
PROTOCOL_COLUMNS = ["Lfd. Nr.", "Prüfkörper", "Maximalkraft [kN]", "Bruchursache", "Anmerkungen"]
ASSIGNMENT_COLUMNS = ["Originaldateiname", "Prüfkörpernummer", "Aufnahmereihenfolge", "Dateipfad", "Prüfstatus"]
CSV_COLUMNS = [
    "Versagensart", "Klassenname", "Vorschlagsrang", "Serie", "Prüfkörper", "Originaldateiname",
    "Ursprünglicher Dateipfad", "Sortierter Dateipfad", "Dokumentierte Bruchursache", "Protokollanmerkung",
    "Bildzuordnungsstatus", "Auswahlgrund", "Technischer Qualitätswert", "Fachlicher Klassenstatus",
    "Klassifikationsregeln-Version", "Vorschlagsstatus", "Sichtbarkeit", "Entscheidung", "Prüfer", "Kommentar",
    "Bereits bestätigte Referenzbilder", "Offene Kandidatenplätze bis Ziel 3",
    "Fehlende bestätigte Referenzbilder bis Ziel 3",
]


@dataclass(frozen=True)
class PhotoRecord:
    series: str
    original_name: str
    specimen: str
    order: int
    sorted_path: Path
    original_path: Path
    assignment_status: str


@dataclass(frozen=True)
class Candidate:
    code: str
    name: str
    series: str
    specimen: str
    photo: PhotoRecord
    protocol_entry: str
    protocol_note: str
    technical_score: float
    image_hash: int
    label_count: int
    professional_status: str
    rules_version: str


@dataclass(frozen=True)
class Proposal:
    identifier: str
    code: str
    name: str
    series: str
    specimen: str
    original_name: str
    original_path: Path
    sorted_path: Path
    assignment_status: str
    protocol_entry: str
    protocol_note: str
    technical_score: float
    technical_reason: str
    professional_status: str
    rules_version: str
    preview_path: Path


def project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader.fieldnames or []), list(reader)


def extract_codes(text: str, known_codes: Iterable[str]) -> list[str]:
    """Extrahiert Kürzel, ohne HB aus sp HB oder Spl HB doppelt zu lesen."""
    masked = text
    found: list[tuple[int, str]] = []
    for code in sorted(known_codes, key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-zÄÖÜäöüß]){re.escape(code)}(?![A-Za-zÄÖÜäöüß])")
        for match in list(pattern.finditer(masked)):
            found.append((match.start(), code))
            masked = masked[: match.start()] + (" " * (match.end() - match.start())) + masked[match.end() :]
    return [code for _, code in sorted(found)]


def load_rules() -> tuple[str, dict[str, str]]:
    data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return str(data["meta"]["version"]), {
        item["code"]: item["fachlicher_pruefstatus"]["status"] for item in data.get("classes", [])
    }


def load_extended_review_status() -> dict[str, bool]:
    data = json.loads(EXTENDED_MODES_PATH.read_text(encoding="utf-8"))
    return {item["code"]: bool(item["review_required"]) for item in data["failure_modes"]}


def professional_status(code: str, rule_statuses: dict[str, str], extended_review: dict[str, bool]) -> str:
    if code in rule_statuses:
        return rule_statuses[code]
    return "fachlich_offen_nicht_im_sechserfokus_geprueft" if extended_review.get(code, True) else "nicht_im_sechserfokus_geprueft"


def series_paths(letter: str) -> tuple[str, Path, Path, Path]:
    series = f"Z-{letter}"
    series_dir = EXPERIMENT_ROOT / f"Serie {series}"
    return series, series_dir, series_dir / "Protokoll" / f"Z_{letter}.csv", series_dir / "Bilder" / "Sortiert" / f"Bildzuordnung_{series}.csv"


def validate_protocol(series: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    if fields != PROTOCOL_COLUMNS:
        raise RuntimeError(f"{series}: unerwartete Protokollspalten: {fields}")
    expected = {f"{series}-{number:02d}" for number in range(1, 36)}
    specimens = [row.get("Prüfkörper", "").strip() for row in rows]
    if len(rows) != 35 or set(specimens) != expected or len(set(specimens)) != 35:
        raise RuntimeError(f"{series}: Protokoll enthält nicht genau 35 eindeutige Prüfkörper.")


def load_photos(series: str, series_dir: Path, fields: list[str], rows: list[dict[str, str]]) -> list[PhotoRecord]:
    if fields != ASSIGNMENT_COLUMNS:
        raise RuntimeError(f"{series}: unerwartete Bildzuordnungsspalten: {fields}")
    records: list[PhotoRecord] = []
    for row in rows:
        sorted_path = PROJECT_ROOT / row["Dateipfad"]
        original_path = series_dir / "Bilder" / row["Originaldateiname"]
        if not sorted_path.is_file() or not original_path.is_file():
            raise RuntimeError(f"{series}: Bildpfad fehlt: {original_path} oder {sorted_path}")
        records.append(PhotoRecord(series, row["Originaldateiname"], row["Prüfkörpernummer"], int(row["Aufnahmereihenfolge"]), sorted_path, original_path, row["Prüfstatus"]))
    return records


def is_number_photo(record: PhotoRecord) -> bool:
    return record.sorted_path.stem.lower().startswith("nummer") or record.assignment_status == "OK_NUMMERNFOTO"


def image_metrics(path: Path) -> tuple[float, int]:
    """Bewertet nur technische Qualität und Bildähnlichkeit, nie eine Versagensart."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image.thumbnail((640, 480))
        array = np.asarray(image, dtype=np.float32)
        dhash_array = np.asarray(image.resize((9, 8)), dtype=np.int16)
    if array.size == 0:
        return 0.0, 0
    sharpness = float(np.var(np.diff(array, axis=1)) + np.var(np.diff(array, axis=0)))
    exposure_factor = max(0.25, 1.0 - abs(float(array.mean()) - 128.0) / 180.0)
    bits = dhash_array[:, 1:] > dhash_array[:, :-1]
    return sharpness * exposure_factor, sum(int(bit) << index for index, bit in enumerate(bits.ravel()))


def hash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def make_thumbnail(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((720, 480))
        image.save(destination, "JPEG", quality=84, optimize=True)


def reference_directory(code: str) -> Path:
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", code).strip("_")
    matches = [p for p in REFERENCE_ROOT.iterdir() if p.is_dir() and (p.name == prefix or p.name.startswith(prefix + "_"))]
    if len(matches) != 1:
        raise RuntimeError(f"Referenzordner für {code!r} ist nicht eindeutig: {matches}")
    return matches[0]


def load_confirmed(modes: dict[str, str]) -> tuple[dict[str, list[str]], set[str], set[str]]:
    confirmed: dict[str, list[str]] = {code: [] for code in modes}
    sources: set[str] = set()
    ids: set[str] = set()
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for item in manifest.get("reference_sets", []):
            code = item.get("failure_mode")
            reference = item.get("reference_image", {})
            if code not in modes:
                continue
            path = reference.get("copied_path", "")
            if path and path not in confirmed[code]:
                confirmed[code].append(path)
            if reference.get("original_path"):
                sources.add(reference["original_path"])
            if item.get("id"):
                ids.add(item["id"])
        for item in manifest.get("referenzbilder", []):
            code = item.get("versagensart")
            if code not in modes:
                continue
            path = item.get("referenzdateipfad", "")
            if path and path not in confirmed[code]:
                confirmed[code].append(path)
            if item.get("urspruenglicher_dateipfad"):
                sources.add(item["urspruenglicher_dateipfad"])
            if item.get("id"):
                ids.add(item["id"])
    for code in modes:
        for path in sorted(reference_directory(code).iterdir()):
            if path.is_file() and path.name != ".DS_Store":
                relative = project_relative(path)
                if relative not in confirmed[code]:
                    confirmed[code].append(relative)
    return confirmed, sources, ids


def assignment_index() -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for letter in SERIES_LETTERS:
        series, series_dir, _, assignment_path = series_paths(letter)
        fields, rows = read_csv(assignment_path)
        if fields != ASSIGNMENT_COLUMNS:
            raise RuntimeError(f"{series}: unerwartete Bildzuordnungsspalten: {fields}")
        for row in rows:
            original_path = series_dir / "Bilder" / row["Originaldateiname"]
            if not original_path.is_file():
                raise RuntimeError(f"Originalbild fehlt: {original_path}")
            index[row["Prüfkörpernummer"]].append({
                **row,
                "Serie": series,
                "Originalpfad": project_relative(original_path),
            })
    return index


def load_all_series() -> tuple[list[dict[str, str]], list[PhotoRecord], dict[str, object]]:
    protocols: list[dict[str, str]] = []
    photos: list[PhotoRecord] = []
    sources: dict[str, object] = {}
    for letter in SERIES_LETTERS:
        series, series_dir, protocol_path, assignment_path = series_paths(letter)
        p_fields, p_rows = read_csv(protocol_path)
        a_fields, a_rows = read_csv(assignment_path)
        validate_protocol(series, p_fields, p_rows)
        photos.extend(load_photos(series, series_dir, a_fields, a_rows))
        protocols.extend({**row, "_serie": series} for row in p_rows)
        sources[series] = {
            "protokoll": project_relative(protocol_path), "protokollzeilen": len(p_rows),
            "bildzuordnung": project_relative(assignment_path), "bildzeilen": len(a_rows),
            "statusverteilung": dict(Counter(row["Prüfstatus"] for row in a_rows)),
        }
    return protocols, photos, sources


def build_candidates(protocols: list[dict[str, str]], photos: list[PhotoRecord], modes: dict[str, str], rules_version: str, rule_statuses: dict[str, str], extended_review: dict[str, bool]) -> tuple[dict[str, list[Candidate]], dict[str, list[str]], dict[str, str]]:
    protocol_by_specimen = {row["Prüfkörper"]: row for row in protocols}
    codes_by_specimen = {specimen: extract_codes(row.get("Bruchursache", ""), modes) for specimen, row in protocol_by_specimen.items()}
    documented: dict[str, list[str]] = {code: [] for code in modes}
    for specimen, codes in codes_by_specimen.items():
        for code in codes:
            documented[code].append(specimen)
    records: dict[str, list[PhotoRecord]] = defaultdict(list)
    for photo in photos:
        records[photo.specimen].append(photo)
    candidates: dict[str, list[Candidate]] = {code: [] for code in modes}
    excluded: dict[str, str] = {}
    metric_cache: dict[Path, tuple[float, int]] = {}
    for specimen, row in protocol_by_specimen.items():
        all_records = records.get(specimen, [])
        break_photos = [photo for photo in all_records if not is_number_photo(photo)]
        if not break_photos:
            excluded[specimen] = "Keine zugeordneten Bruchbilder vorhanden"
            continue
        if any(photo.assignment_status.startswith("PRUEFEN_") for photo in all_records):
            excluded[specimen] = "Bildzuordnung muss manuell geprüft werden"
            continue
        note = row.get("Anmerkungen", "")
        if any(term in note.casefold() for term in EXCLUSION_NOTE_TERMS):
            excluded[specimen] = "Hinweis auf Klemmbereich, Prüfartefakt oder fehlende Prüfdaten"
            continue
        codes = codes_by_specimen[specimen]
        for photo in break_photos:
            metric_cache.setdefault(photo.original_path, image_metrics(photo.original_path))
            score, image_hash = metric_cache[photo.original_path]
            for code in codes:
                candidates[code].append(Candidate(code, modes[code], row["_serie"], specimen, photo, row.get("Bruchursache", ""), note, score, image_hash, len(codes), professional_status(code, rule_statuses, extended_review), rules_version))
    return candidates, documented, excluded


def choose_proposals(candidates: dict[str, list[Candidate]], modes: dict[str, str], confirmed: dict[str, list[str]], confirmed_sources: set[str], output_dir: Path) -> list[Proposal]:
    preview_dir = output_dir / "vorschau"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in preview_dir.glob("*.jpg"):
        old.unlink()
    selected: dict[str, list[Candidate]] = {code: [] for code in modes}
    used_sources = set(confirmed_sources)
    mode_order = list(modes)
    for code in sorted(modes, key=lambda c: (len({x.specimen for x in candidates[c]}), mode_order.index(c))):
        slots = max(0, min(MAX_REFERENCES_PER_MODE - len(confirmed[code]), TARGET_REFERENCES_PER_MODE - len(confirmed[code])))
        pool = sorted(
            (
                x for x in candidates[code]
                if f"{code}|{x.specimen}|{x.photo.original_name}" not in MANUAL_VISUAL_EXCLUSIONS
                and (code, x.specimen) not in MANUAL_CLASS_SPECIMEN_EXCLUSIONS
            ),
            key=lambda x: (x.label_count, -x.technical_score, x.specimen, x.photo.order),
        )
        while len(selected[code]) < slots:
            specimens = {x.specimen for x in selected[code]}
            series = {x.series for x in selected[code]}
            eligible = [x for x in pool if x.specimen not in specimens and project_relative(x.photo.original_path) not in used_sources and all(hash_distance(x.image_hash, y.image_hash) >= 5 for y in selected[code])]
            if not eligible:
                break
            different_series = [x for x in eligible if x.series not in series]
            choice = (different_series or eligible)[0]
            selected[code].append(choice)
            used_sources.add(project_relative(choice.photo.original_path))
    proposals: list[Proposal] = []
    reason = "Protokolltreffer; sichere Zuordnung; technische Schärfe/Belichtung; Serien- und Prüfkörperdiversität; keine visuelle Klassenbestätigung"
    for code in modes:
        for item in selected[code]:
            preview = preview_dir / f"{code.replace(' ', '_').replace('(', '').replace(')', '')}__{item.specimen}__{item.photo.original_path.stem}.jpg"
            make_thumbnail(item.photo.original_path, preview)
            proposals.append(Proposal(f"{code}|{item.specimen}|{item.photo.original_name}", code, item.name, item.series, item.specimen, item.photo.original_name, item.photo.original_path, item.photo.sorted_path, item.photo.assignment_status, item.protocol_entry, item.protocol_note, round(item.technical_score, 2), reason, item.professional_status, item.rules_version, preview))
    return proposals


def proposal_to_dict(p: Proposal) -> dict[str, object]:
    return {
        "id": p.identifier, "versagensart": p.code, "klassenname": p.name, "serie": p.series,
        "pruefkoerper": p.specimen, "originaldateiname": p.original_name,
        "urspruenglicher_dateipfad": project_relative(p.original_path), "sortierter_dateipfad": project_relative(p.sorted_path),
        "bildzuordnungsstatus": p.assignment_status, "dokumentierte_bruchursache": p.protocol_entry,
        "protokollanmerkung": p.protocol_note, "auswahlgrund": p.technical_reason,
        "technischer_qualitaetswert": p.technical_score, "fachlicher_klassenstatus": p.professional_status,
        "klassifikationsregeln_version": p.rules_version,
        "vorschlagsstatus": "UNGEPRUEFT_MANUELLE_SICHTPRUEFUNG_ERFORDERLICH",
        "sichtbarkeit": "NICHT_BEWERTET", "entscheidung": "OFFEN", "pruefer": "", "kommentar": "",
    }


def write_template(output_dir: Path, proposals: list[Proposal], rules_version: str, sources: dict[str, object]) -> Path:
    payload = {
        "meta": {"serien": [f"Z-{x}" for x in SERIES_LETTERS], "erstellt_am": date.today().isoformat(), "status": "UNGEPRUEFTE_VORSCHLAEGE", "klassifikationsregeln_version": rules_version, "ziel_referenzbilder_je_klasse": TARGET_REFERENCES_PER_MODE, "maximum_referenzbilder_je_klasse": MAX_REFERENCES_PER_MODE, "quellen": sources, "hinweis": "BESTAETIGT erfordert sichtbare Merkmale, KEINE_TESTDATEN und einen Prüfernamen."},
        "pruefkoerper_status": {specimen: "OFFEN" for specimen in sorted({p.specimen for p in proposals})},
        "vorschlaege": [proposal_to_dict(p) for p in proposals],
    }
    path = output_dir / "Bestaetigungsvorlage_Gesamt.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_overview_csv(output_dir: Path, proposals: list[Proposal], modes: dict[str, str], confirmed: dict[str, list[str]]) -> Path:
    grouped: dict[str, list[Proposal]] = defaultdict(list)
    for proposal in proposals:
        grouped[proposal.code].append(proposal)
    path = output_dir / "Referenzbildvorschlaege_Gesamt.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter=";")
        writer.writeheader()
        for code, name in modes.items():
            rows = grouped.get(code, [])
            missing = max(0, TARGET_REFERENCES_PER_MODE - len(confirmed[code]) - len(rows))
            if not rows:
                writer.writerow({"Versagensart": code, "Klassenname": name, "Vorschlagsstatus": "KEIN_UNGEPRUEFTER_VORSCHLAG", "Bereits bestätigte Referenzbilder": len(confirmed[code]), "Offene Kandidatenplätze bis Ziel 3": missing, "Fehlende bestätigte Referenzbilder bis Ziel 3": max(0, TARGET_REFERENCES_PER_MODE - len(confirmed[code]))})
            for rank, p in enumerate(rows, 1):
                data = proposal_to_dict(p)
                writer.writerow({"Versagensart": code, "Klassenname": name, "Vorschlagsrang": rank, "Serie": p.series, "Prüfkörper": p.specimen, "Originaldateiname": p.original_name, "Ursprünglicher Dateipfad": data["urspruenglicher_dateipfad"], "Sortierter Dateipfad": data["sortierter_dateipfad"], "Dokumentierte Bruchursache": p.protocol_entry, "Protokollanmerkung": p.protocol_note, "Bildzuordnungsstatus": p.assignment_status, "Auswahlgrund": p.technical_reason, "Technischer Qualitätswert": p.technical_score, "Fachlicher Klassenstatus": p.professional_status, "Klassifikationsregeln-Version": p.rules_version, "Vorschlagsstatus": data["vorschlagsstatus"], "Sichtbarkeit": data["sichtbarkeit"], "Entscheidung": data["entscheidung"], "Prüfer": "", "Kommentar": "", "Bereits bestätigte Referenzbilder": len(confirmed[code]), "Offene Kandidatenplätze bis Ziel 3": missing, "Fehlende bestätigte Referenzbilder bis Ziel 3": max(0, TARGET_REFERENCES_PER_MODE - len(confirmed[code]))})
    return path


def html_relative(base: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def write_html(output_dir: Path, proposals: list[Proposal], documented: dict[str, list[str]], excluded: dict[str, str], modes: dict[str, str], confirmed: dict[str, list[str]], rules_version: str) -> Path:
    grouped: dict[str, list[Proposal]] = defaultdict(list)
    for p in proposals:
        grouped[p.code].append(p)
    sections: list[str] = []
    for code, name in modes.items():
        cards: list[str] = []
        for p in grouped.get(code, []):
            preview, original = html_relative(output_dir, p.preview_path), html_relative(output_dir, p.original_path)
            cards.append(f'''<article class="card" data-id="{html.escape(p.identifier)}" data-specimen="{p.specimen}"><a href="{html.escape(original)}"><img src="{html.escape(preview)}" alt="{html.escape(p.specimen)}"></a><div class="card-body"><h3>{p.series} · {p.specimen} · {html.escape(p.original_name)}</h3><p><strong>Dokumentiert:</strong> {html.escape(p.protocol_entry)}<br><strong>Anmerkung:</strong> {html.escape(p.protocol_note or 'Keine')}<br><strong>Bildzuordnung:</strong> {html.escape(p.assignment_status)}</p><p class="warning">Ungeprüft: Die Klasse wurde im Foto nicht automatisch bestätigt.</p><label>Sichtbarkeit<select class="visibility"><option>NICHT_BEWERTET</option><option>SICHTBAR_BESTAETIGT</option><option>NICHT_SICHTBAR</option><option>UNKLAR</option></select></label><label>Testdatenstatus des Prüfkörpers<select class="test-status"><option>OFFEN</option><option>KEINE_TESTDATEN</option><option>TESTDATEN_RESERVIERT</option></select></label><label>Entscheidung<select class="decision"><option>OFFEN</option><option>BESTAETIGT</option><option>ABGELEHNT</option></select></label><label>Prüfer<input class="reviewer" type="text"></label><label>Kommentar<textarea class="comment" rows="2"></textarea></label></div></article>''')
        confirmed_count, proposed_count = len(confirmed[code]), len(grouped.get(code, []))
        missing = max(0, TARGET_REFERENCES_PER_MODE - confirmed_count - proposed_count)
        status = grouped[code][0].professional_status if grouped.get(code) else professional_status(code, {}, {})
        empty = "Kein ungeprüfter Vorschlag. " + ("Klasse in den sieben Protokollen nicht dokumentiert." if not documented[code] else "Sicher zuordenbare Kandidaten fehlen; manuelle Nachsuche erforderlich.")
        sections.append(f'''<section><div class="section-head"><div><h2>{html.escape(code)} – {html.escape(name)}</h2><p>Dokumentierte Prüfkörper: {len(documented[code])} · bereits bestätigt: {confirmed_count} · ungeprüfte Vorschläge: {proposed_count} · offene Kandidatenplätze: {missing} · bestätigte Referenzbilder fehlen: {max(0, TARGET_REFERENCES_PER_MODE - confirmed_count)}</p></div><span>{html.escape(status)}</span></div><div class="cards">{''.join(cards) if cards else f'<p>{html.escape(empty)}</p>'}</div></section>''')
    excluded_html = "".join(f"<li><strong>{html.escape(k)}</strong>: {html.escape(v)}</li>" for k, v in sorted(excluded.items()))
    rejected_html = "".join(f"<li><strong>{html.escape(k)}</strong>: {html.escape(v)}</li>" for k, v in sorted(MANUAL_VISUAL_EXCLUSIONS.items()))
    rejected_html += "".join(
        f"<li><strong>{html.escape(code + '|' + specimen)}</strong>: {html.escape(reason)}</li>"
        for (code, specimen), reason in sorted(MANUAL_CLASS_SPECIMEN_EXCLUSIONS.items())
    )
    embedded = json.dumps(json.loads((output_dir / "Bestaetigungsvorlage_Gesamt.json").read_text(encoding="utf-8")), ensure_ascii=False).replace("</", "<\\/")
    document = f'''<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Referenzbildvorschläge Z-A bis Z-G</title><style>:root{{font-family:Arial,sans-serif;color:#1d2733;background:#f3f5f7}}*{{box-sizing:border-box}}body{{margin:0}}header{{background:#183b56;color:white;padding:28px 5vw}}header h1{{margin:0 0 10px}}main{{width:min(1500px,94vw);margin:24px auto 60px}}.notice{{background:#fff4ce;border-left:5px solid #c58a00;padding:16px;margin:16px 0}}.controls{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}button,a.button{{border:0;border-radius:5px;padding:11px 15px;background:#176b87;color:white;text-decoration:none;font-weight:bold;cursor:pointer}}section{{background:white;border-radius:8px;margin:20px 0;padding:18px;box-shadow:0 2px 8px #0001}}.section-head{{display:flex;justify-content:space-between;gap:20px;align-items:start}}.section-head h2{{margin:0}}.section-head span{{background:#eef1f4;padding:6px 9px;border-radius:999px;font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px;margin-top:14px}}.card{{border:1px solid #ccd5dd;border-radius:7px;overflow:hidden;background:#fafbfc}}.card img{{width:100%;height:225px;object-fit:cover;display:block;background:#e7ebef}}.card-body{{padding:13px}}.card h3{{font-size:16px;margin:0 0 8px}}.card p{{font-size:13px;line-height:1.45}}.warning{{color:#8b5500;font-weight:bold}}label{{display:block;font-size:13px;font-weight:bold;margin-top:9px}}select,input,textarea{{display:block;width:100%;margin-top:4px;padding:8px;border:1px solid #aeb8c2;border-radius:4px;background:white}}.error{{color:#a12622;font-weight:bold}}code{{background:#eef1f4;padding:2px 4px}}</style></head><body><header><h1>Referenzbildvorschläge Z-A bis Z-G</h1><p>Gemeinsame lokale Prüfübersicht · Klassifikationsregeln Version {html.escape(rules_version)} · alle neuen Vorschläge ungeprüft</p></header><main><div class="notice"><strong>Wichtig:</strong> Protokollangabe, sichere Zuordnung und technische Bildqualität erzeugen nur Vorschläge. Das Skript erkennt keine Versagensart zuverlässig im Bild. Bestätigen Sie nur klar sichtbare, charakteristische Merkmale.</div><div class="notice"><strong>Testdaten:</strong> Ein Bild darf nur bestätigt werden, wenn der gesamte Prüfkörper als <code>KEINE_TESTDATEN</code> markiert ist.</div><div class="controls"><button id="export">Bestätigungsdatei herunterladen</button><a class="button" href="Referenzbildvorschlaege_Gesamt.csv">Gesamtübersicht CSV</a><a class="button" href="Bestaetigungsvorlage_Gesamt.json">Unbearbeitete JSON-Vorlage</a></div><p id="message" class="error"></p>{''.join(sections)}<section><h2>Automatisch ausgeschlossene Prüfkörper</h2><ul>{excluded_html or '<li>Keine</li>'}</ul></section><section><h2>Bei der Kontaktbogenprüfung zurückgestellte Fotos</h2><ul>{rejected_html or '<li>Keine</li>'}</ul></section></main><script id="base-data" type="application/json">{embedded}</script><script>const cards=[...document.querySelectorAll('.card')];for(const select of document.querySelectorAll('.test-status')){{select.addEventListener('change',()=>{{const specimen=select.closest('.card').dataset.specimen;for(const peer of document.querySelectorAll(`.card[data-specimen="${{specimen}}"] .test-status`))peer.value=select.value;}})}}document.getElementById('export').addEventListener('click',()=>{{const base=JSON.parse(document.getElementById('base-data').textContent),byId=Object.fromEntries(base.vorschlaege.map(x=>[x.id,x]));let errors=[];for(const card of cards){{const item=byId[card.dataset.id],test=card.querySelector('.test-status').value;item.sichtbarkeit=card.querySelector('.visibility').value;item.entscheidung=card.querySelector('.decision').value;item.pruefer=card.querySelector('.reviewer').value.trim();item.kommentar=card.querySelector('.comment').value.trim();base.pruefkoerper_status[card.dataset.specimen]=test;if(item.entscheidung==='BESTAETIGT'&&(test!=='KEINE_TESTDATEN'||item.sichtbarkeit!=='SICHTBAR_BESTAETIGT'||!item.pruefer))errors.push(`${{item.id}}: Sichtbarkeit, KEINE_TESTDATEN und Prüfer erforderlich.`)}}if(errors.length){{document.getElementById('message').textContent=errors.join(' ');return}}base.meta.status='MANUELL_BEARBEITET';base.meta.exportiert_am=new Date().toISOString();const blob=new Blob([JSON.stringify(base,null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='Bestaetigung_Gesamt.json';a.click();URL.revokeObjectURL(a.href);document.getElementById('message').textContent='';}});</script></body></html>'''
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def write_summary(output_dir: Path, modes: dict[str, str], proposals: list[Proposal], documented: dict[str, list[str]], excluded: dict[str, str], confirmed: dict[str, list[str]], sources: dict[str, object]) -> Path:
    counts = Counter(p.code for p in proposals)
    payload = {"serien": list(sources), "quellen": sources, "versagensarten": {code: {"bezeichnung": name, "dokumentierte_pruefkoerper": len(documented[code]), "bereits_bestaetigt": len(confirmed[code]), "ungepruefte_vorschlaege": counts[code], "offene_kandidatenplaetze_bis_ziel_3": max(0, TARGET_REFERENCES_PER_MODE - len(confirmed[code]) - counts[code]), "fehlende_bestaetigte_referenzbilder_bis_ziel_3": max(0, TARGET_REFERENCES_PER_MODE - len(confirmed[code]))} for code, name in modes.items()}, "vorschlaege_gesamt": len(proposals), "automatisch_ausgeschlossene_pruefkoerper": excluded, "bei_kontaktbogenpruefung_zurueckgestellte_fotos": MANUAL_VISUAL_EXCLUSIONS, "bei_kontaktbogenpruefung_zurueckgestellte_klasse_pruefkoerper": {f"{code}|{specimen}": reason for (code, specimen), reason in MANUAL_CLASS_SPECIMEN_EXCLUSIONS.items()}, "referenzbilder_kopiert_durch_diesen_lauf": 0}
    path = output_dir / "Pruefzusammenfassung_Gesamt.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def apply_confirmations(path: Path, copy_confirmed: bool) -> None:
    if not copy_confirmed:
        raise RuntimeError("Bestätigte Bilder werden nur mit --copy-confirmed kopiert.")
    data = json.loads(path.read_text(encoding="utf-8"))
    current_version, _ = load_rules()
    if str(data["meta"]["klassifikationsregeln_version"]) != current_version:
        raise RuntimeError("Die Regelversion der Bestätigungsdatei ist nicht aktuell.")
    modes = json.loads(MODES_PATH.read_text(encoding="utf-8"))
    confirmed, _, existing_ids = load_confirmed(modes)
    selected = [x for x in data["vorschlaege"] if x.get("entscheidung") == "BESTAETIGT"]
    if not selected:
        raise RuntimeError("Die Bestätigungsdatei enthält keine bestätigten Vorschläge.")
    additions = Counter(x["versagensart"] for x in selected if x.get("id") not in existing_ids)
    for code, amount in additions.items():
        if len(confirmed[code]) + amount > MAX_REFERENCES_PER_MODE:
            raise RuntimeError(f"{code}: Bestätigung würde das Maximum von {MAX_REFERENCES_PER_MODE} überschreiten.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {"meta": {}, "reference_sets": [], "dataset_assignments": {}}
    manifest.setdefault("reference_sets", [])
    manifest.setdefault("dataset_assignments", {})
    assignments = assignment_index()
    pending: list[tuple[dict[str, object], list[dict[str, str]]]] = []
    for item in selected:
        if item["id"] in existing_ids:
            continue
        specimen = item["pruefkoerper"]
        if data["pruefkoerper_status"].get(specimen) != "KEINE_TESTDATEN":
            raise RuntimeError(f"{specimen}: bestätigter Vorschlag ist nicht von Testdaten ausgeschlossen.")
        if item.get("sichtbarkeit") != "SICHTBAR_BESTAETIGT" or not item.get("pruefer", "").strip():
            raise RuntimeError(f"{item['id']}: Sichtbarkeitsbestätigung oder Prüfer fehlt.")
        source = PROJECT_ROOT / item["urspruenglicher_dateipfad"]
        if not source.is_file():
            raise RuntimeError(f"Quelldatei fehlt: {source}")
        rows = assignments.get(specimen, [])
        if not rows or item["originaldateiname"] not in {row["Originaldateiname"] for row in rows}:
            raise RuntimeError(f"{item['id']}: keine vollständige Bildzuordnung für den Prüfkörper.")
        uncertain = [row for row in rows if row["Prüfstatus"].startswith("PRUEFEN_")]
        if uncertain:
            raise RuntimeError(f"{specimen}: Kontextgruppe enthält unsichere Bildzuordnungen.")
        existing_assignment = manifest["dataset_assignments"].get(specimen, {})
        if existing_assignment.get("dataset_role") == "independent_test":
            raise RuntimeError(f"{specimen}: Prüfkörper ist bereits dem unabhängigen Testdatensatz zugeordnet.")
        pending.append((item, rows))

    copied_references = 0
    copied_context = 0
    for item, rows in pending:
        specimen = item["pruefkoerper"]
        class_dir = reference_directory(item["versagensart"])
        reference_dir = class_dir / specimen / "reference_image"
        context_dir = class_dir / specimen / "context_images"
        reference_dir.mkdir(parents=True, exist_ok=True)
        context_dir.mkdir(parents=True, exist_ok=True)
        reference_source = PROJECT_ROOT / item["urspruenglicher_dateipfad"]
        reference_destination = reference_dir / reference_source.name
        shutil.copy2(reference_source, reference_destination)
        context_entries = []
        for row in rows:
            if row["Originaldateiname"] == item["originaldateiname"]:
                continue
            context_source = PROJECT_ROOT / row["Originalpfad"]
            context_destination = context_dir / context_source.name
            shutil.copy2(context_source, context_destination)
            context_entries.append({
                "image_role": "context_image",
                "context_kind": "identification_photo" if row["Prüfstatus"] == "OK_NUMMERNFOTO" else "specimen_view",
                "confirmed": False,
                "default_for_inference": False,
                "assignment_status": row["Prüfstatus"],
                "original_path": row["Originalpfad"],
                "copied_path": project_relative(context_destination),
            })
            copied_context += 1
        reference_set = {
            "id": item["id"],
            "failure_mode": item["versagensart"],
            "class_name": item["klassenname"],
            "specimen": specimen,
            "series": item["serie"],
            "dataset_role": "reference",
            "excluded_from_independent_test": True,
            "classification_rules_version": current_version,
            "confirmation_source": project_relative(path),
            "reference_image": {
                "image_role": "reference_image",
                "confirmed": True,
                "default_for_inference": True,
                "confirmed_by": item["pruefer"],
                "confirmed_at": date.today().isoformat(),
                "visibility": item["sichtbarkeit"],
                "review_comment": item.get("kommentar", ""),
                "original_path": item["urspruenglicher_dateipfad"],
                "copied_path": project_relative(reference_destination),
            },
            "context_images": context_entries,
        }
        manifest["reference_sets"].append(reference_set)
        manifest["dataset_assignments"][specimen] = {
            "dataset_role": "reference",
            "excluded_from_independent_test": True,
            "assigned_at": date.today().isoformat(),
        }
        existing_ids.add(item["id"])
        copied_references += 1
    manifest.setdefault("meta", {}).update({
        "schema_version": "2.0",
        "aktualisiert_am": date.today().isoformat(),
        "klassifikationsregeln_version": current_version,
        "context_images_default_for_inference": False,
    })
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{copied_references} Hauptreferenzbilder und {copied_context} Kontextbilder kopiert; Manifest: {MANIFEST_PATH}")


def generate_all() -> None:
    output_dir = OUTPUT_ROOT / "Referenzbilddatenbank_Gesamt"
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = json.loads(MODES_PATH.read_text(encoding="utf-8"))
    rules_version, rule_statuses = load_rules()
    confirmed, confirmed_sources, _ = load_confirmed(modes)
    protocols, photos, sources = load_all_series()
    candidates, documented, excluded = build_candidates(protocols, photos, modes, rules_version, rule_statuses, load_extended_review_status())
    proposals = choose_proposals(candidates, modes, confirmed, confirmed_sources, output_dir)
    template = write_template(output_dir, proposals, rules_version, sources)
    csv_path = write_overview_csv(output_dir, proposals, modes, confirmed)
    overview = write_html(output_dir, proposals, documented, excluded, modes, confirmed, rules_version)
    summary = write_summary(output_dir, modes, proposals, documented, excluded, confirmed, sources)
    counts = Counter(p.code for p in proposals)
    print(f"Z-A bis Z-G: {len(protocols)} Prüfkörper, {len(photos)} Bildzuordnungen")
    print(f"Ungeprüfte Vorschläge: {len(proposals)} für {sum(bool(counts[c]) for c in modes)} von {len(modes)} Klassen")
    print(f"HTML: {overview}\nCSV: {csv_path}\nBestätigungsvorlage: {template}\nPrüfzusammenfassung: {summary}")
    print("Referenzbilder kopiert: 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="ALL", help="Gemeinsame Auswertung Z-A bis Z-G")
    parser.add_argument("--apply-confirmations", type=Path, help="Manuell exportierte Bestätigungsdatei anwenden")
    parser.add_argument("--copy-confirmed", action="store_true", help="Explizite Freigabe zum Kopieren bestätigter Bilder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.apply_confirmations:
        apply_confirmations(args.apply_confirmations.resolve(), args.copy_confirmed)
    elif args.series.upper().replace("Z-", "") in {"ALL", "A-G"}:
        generate_all()
    else:
        raise RuntimeError("Die Auswertung arbeitet gemeinsam mit Z-A bis Z-G; verwenden Sie --series ALL.")


if __name__ == "__main__":
    main()
