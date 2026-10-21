#!/usr/bin/env python3
"""Erster, abgesicherter Prototyp fuer die KI-Bildanalyse von Holz-Zugproben."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from .openai_config import (
        OpenAIConfigurationError,
        create_openai_client,
        resolve_api_key,
        safe_request_error,
    )
except ImportError:  # Direkter Aufruf als Skript bleibt moeglich.
    from openai_config import (  # type: ignore[no-redef]
        OpenAIConfigurationError,
        create_openai_client,
        resolve_api_key,
        safe_request_error,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "Datenbank"
RULES_PATH = DATABASE_DIR / "klassifikationsregeln.json"
DEFINITIONS_PATH = DATABASE_DIR / "versagensarten_erweitert.json"
MANIFEST_PATH = DATABASE_DIR / "Referenzbilder" / "referenzbilder_manifest.json"
MODEL_VERIFICATION_PATH = Path(__file__).with_name("model_verification.json")
DEFAULT_MOCK_PATH = Path(__file__).with_name("mock_response.json")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "KI_Klassifikation"
ALLOWED_INPUT_ROOT = PROJECT_ROOT / "Pruefdaten" / "Neue_Analysen"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_TEST_IMAGES = 3
MAX_TRANSMITTED_LONG_EDGE = 2048
DEFAULT_REFERENCES_PER_CLASS = 1
CAUSAL_CLAIM = re.compile(
    r"\b(?:verursacht(?:e|en)?|ursächlich|bruchursache|führte\s+zum|löste\s+den\s+bruch\s+aus)\b",
    re.IGNORECASE,
)


class AnalysisError(RuntimeError):
    """Eine Eingabe, Datenbasis oder Modellantwort verletzt die Anforderungen."""


@dataclass(frozen=True)
class ReferenceImage:
    code: str
    path: Path


@dataclass(frozen=True)
class Catalog:
    definitions: dict[str, dict[str, Any]]
    references: tuple[ReferenceImage, ...]
    known_dataset_paths: tuple[Path, ...]
    reference_counts: dict[str, int]
    rule_version: str

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(self.definitions)

    @property
    def codes_without_references(self) -> tuple[str, ...]:
        return tuple(code for code in self.codes if self.reference_counts.get(code, 0) == 0)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"JSON-Datei kann nicht gelesen werden: {path}") from exc


def load_catalog(project_root: Path = PROJECT_ROOT) -> Catalog:
    """Laedt genau 29 Definitionen und nur bestaetigte Hauptreferenzbilder."""

    database = project_root / "Datenbank"
    extended = _read_json(database / "versagensarten_erweitert.json")
    rules = _read_json(database / "klassifikationsregeln.json")
    manifest = _read_json(database / "Referenzbilder" / "referenzbilder_manifest.json")

    modes = extended.get("failure_modes")
    if not isinstance(modes, list) or len(modes) != 29:
        raise AnalysisError("versagensarten_erweitert.json muss genau 29 Klassen enthalten.")
    definitions: dict[str, dict[str, Any]] = {}
    for item in modes:
        code = str(item.get("code", "")).strip()
        if not code or code in definitions:
            raise AnalysisError("Die 29 Klassencodes muessen vorhanden und eindeutig sein.")
        definitions[code] = dict(item)

    validated_rules = rules.get("classes")
    if not isinstance(validated_rules, list):
        raise AnalysisError("klassifikationsregeln.json enthaelt keine Klassenliste.")
    for rule in validated_rules:
        code = str(rule.get("code", "")).strip()
        if code not in definitions:
            raise AnalysisError(f"Klassifikationsregel verwendet unbekannte Klasse: {code!r}")
        definitions[code]["validated_classification_rule"] = dict(rule)

    references: list[ReferenceImage] = []
    known_paths: set[Path] = set()
    counts: Counter[str] = Counter()
    for item in manifest.get("reference_sets", []):
        for image_key in ("reference_image", "context_images"):
            image_items = item.get(image_key, [])
            if isinstance(image_items, dict):
                image_items = [image_items]
            for image in image_items:
                for path_key in ("original_path", "copied_path"):
                    relative = image.get(path_key)
                    if relative:
                        known_paths.add((project_root / relative).resolve())

        reference = item.get("reference_image", {})
        is_confirmed_main = (
            item.get("dataset_role") == "reference"
            and item.get("excluded_from_independent_test") is True
            and reference.get("image_role") == "reference_image"
            and reference.get("confirmed") is True
            and reference.get("default_for_inference") is True
            and reference.get("visibility") == "SICHTBAR_BESTAETIGT"
        )
        if not is_confirmed_main:
            continue
        code = str(item.get("failure_mode", "")).strip()
        if code not in definitions:
            raise AnalysisError(f"Referenzmanifest verwendet unbekannte Klasse: {code!r}")
        path = (project_root / reference["copied_path"]).resolve()
        if not path.is_file():
            raise AnalysisError(f"Bestaetigtes Hauptreferenzbild fehlt: {path}")
        references.append(ReferenceImage(code=code, path=path))
        counts[code] += 1

    return Catalog(
        definitions=definitions,
        references=tuple(references),
        known_dataset_paths=tuple(sorted(known_paths)),
        reference_counts=dict(counts),
        rule_version=str(rules.get("meta", {}).get("version", "unbekannt")),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            if (
                image.format == "GIF"
                and getattr(image, "is_animated", False)
                and getattr(image, "n_frames", 1) > 1
            ):
                raise AnalysisError(f"Animierte GIF-Dateien werden nicht unterstuetzt: {path.name}")
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise AnalysisError(f"Keine lesbare Bilddatei: {path.name}") from exc


def collect_test_images(input_dir: Path, catalog: Catalog) -> list[Path]:
    """Liest 1-3 Bilder und verhindert bekannte Referenzproben im Testdatensatz."""

    resolved_dir = input_dir.expanduser().resolve()
    allowed_root = ALLOWED_INPUT_ROOT.resolve()
    if not resolved_dir.is_relative_to(allowed_root):
        raise AnalysisError(f"Eingabeordner muss unter {allowed_root} liegen.")
    if not resolved_dir.is_dir():
        raise AnalysisError(f"Eingabeordner fehlt: {resolved_dir}")

    # Automatisch sortierte Prüfkörper enthalten alle Originalansichten und
    # zusätzlich eine lokale Auswahl von maximal drei Bruchbildern. So kann der
    # Prüfkörperordner direkt übergeben werden, ohne das Nummernfoto zu senden.
    selection_dir = resolved_dir / "Analyseauswahl"
    image_dir = selection_dir if selection_dir.is_dir() else resolved_dir

    images = sorted(
        path.resolve()
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not 1 <= len(images) <= MAX_TEST_IMAGES:
        raise AnalysisError("Der Eingabeordner muss genau 1 bis 3 unterstuetzte Fotos enthalten.")

    known_paths = set(catalog.known_dataset_paths)
    for path in images:
        if path in known_paths:
            raise AnalysisError(f"Bekanntes Datenbankbild darf nicht als unabhaengiger Test dienen: {path.name}")
        _verify_image(path)

    # Auch umbenannte oder kopierte Datenbankbilder werden als Leakage erkannt.
    test_hashes = {_sha256(path): path for path in images}
    for known in catalog.known_dataset_paths:
        if not known.is_file():
            continue
        known_hash = _sha256(known)
        if known_hash in test_hashes:
            raise AnalysisError(
                "Ein Testbild ist inhaltlich identisch mit einem bekannten Referenz- oder Kontextbild: "
                f"{test_hashes[known_hash].name}"
            )
    return images


def _data_url(path: Path) -> str:
    """Erzeugt eine kostenschonende Bildkopie im RAM; die Originaldatei bleibt unveraendert."""

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((MAX_TRANSMITTED_LONG_EDGE, MAX_TRANSMITTED_LONG_EDGE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def select_references(catalog: Catalog, max_per_class: int) -> tuple[ReferenceImage, ...]:
    if max_per_class < 0:
        raise AnalysisError("max_references_per_class darf nicht negativ sein.")
    selected: list[ReferenceImage] = []
    counts: Counter[str] = Counter()
    for reference in catalog.references:
        if counts[reference.code] >= max_per_class:
            continue
        selected.append(reference)
        counts[reference.code] += 1
    return tuple(selected)


def _prompt_catalog(catalog: Catalog) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for code, definition in catalog.definitions.items():
        item = {
            "code": code,
            "name": definition.get("name"),
            "feature_type": definition.get("feature_type"),
            "definition": definition.get("definition"),
            "visual_features": definition.get("visual_features", []),
            "distinguishing_features": definition.get("distinguishing_features", []),
            "limitations": definition.get("limitations", []),
            "confirmed_reference_count": catalog.reference_counts.get(code, 0),
        }
        if "validated_classification_rule" in definition:
            rule = definition["validated_classification_rule"]
            item["validated_classification_rule"] = {
                key: rule.get(key)
                for key in (
                    "klassentyp",
                    "klassifikationsziel",
                    "eindeutige_visuelle_erkennungsmerkmale",
                    "abgrenzung_zu_aehnlichen_klassen",
                    "benoetigte_fotoansichten",
                    "einschraenkungen_visuelle_erkennung",
                    "nicht_zulaessige_ursachenaussage",
                    "fachlicher_pruefstatus",
                )
            }
        compact.append(item)
    return compact


def build_request_content(
    test_images: Sequence[Path],
    catalog: Catalog,
    references: Sequence[ReferenceImage],
) -> list[dict[str, Any]]:
    """Baut Bildinput ohne Protokolldaten, Probennummern oder Referenzpfade."""

    instructions = {
        "task": "Multi-Label-Sichtung sichtbarer Merkmale an einer unbekannten Holz-Zugprobe",
        "rules": [
            "Verwende ausschliesslich die bereitgestellten 29 Klassencodes oder UNKLAR.",
            "Trenne genau ein primaeres sichtbares Merkmal von moeglichen sekundaeren Merkmalen.",
            "Begruende jede Klasse ausschliesslich mit konkret sichtbaren Bildmerkmalen.",
            "Behaupte niemals, ein Holzmerkmal sei die nachgewiesene Versagens- oder Bruchursache.",
            "Gib UNKLAR aus, wenn Ansichten, Massstab, Schaerfe oder sichtbare Merkmale nicht ausreichen.",
            "Klassen ohne bestaetigtes Referenzbild bleiben zulaessig, muessen aber besonders vorsichtig anhand der Definition beurteilt werden.",
            "Referenzbilder sind nur visuelle Beispiele und keine Daten des unbekannten Pruefkoerpers.",
        ],
        "classes": _prompt_catalog(catalog),
    }
    content: list[dict[str, Any]] = [{
        "type": "input_text",
        "text": json.dumps(instructions, ensure_ascii=False, separators=(",", ":")),
    }]
    for reference in references:
        content.extend((
            {"type": "input_text", "text": f"Bestaetigtes Hauptreferenzbild fuer Klasse {reference.code}."},
            {"type": "input_image", "image_url": _data_url(reference.path), "detail": "high"},
        ))
    content.append({
        "type": "input_text",
        "text": "Ab hier folgen ausschliesslich die 1 bis 3 Bilder des unbekannten Testpruefkoerpers.",
    })
    for index, path in enumerate(test_images, start=1):
        content.extend((
            {"type": "input_text", "text": f"Unbekanntes Testbild {index}."},
            {"type": "input_image", "image_url": _data_url(path), "detail": "high"},
        ))
    return content


def response_schema(codes: Iterable[str]) -> dict[str, Any]:
    code_enum = [*codes, "UNKLAR"]
    feature = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "enum": list(codes)},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["code", "evidence", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["KLASSIFIZIERT", "UNKLAR"]},
            "primary_code": {"type": "string", "enum": code_enum},
            "primary_evidence": {"type": "array", "items": {"type": "string"}},
            "primary_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "secondary_features": {"type": "array", "items": feature},
            "visible_observations": {"type": "array", "items": {"type": "string"}},
            "uncertainty_reasons": {"type": "array", "items": {"type": "string"}},
            "missing_views": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "status", "primary_code", "primary_evidence", "primary_confidence",
            "secondary_features", "visible_observations", "uncertainty_reasons", "missing_views",
        ],
        "additionalProperties": False,
    }


def load_verified_model(model: str, path: Path = MODEL_VERIFICATION_PATH) -> dict[str, Any]:
    verification = _read_json(path)
    entry = next((item for item in verification.get("models", []) if item.get("model") == model), None)
    if not entry or not entry.get("responses_api") or not entry.get("image_input"):
        raise AnalysisError(
            f"Modell {model!r} ist lokal nicht als Responses-/Bildmodell dokumentiert. "
            "Vor einem Live-Test muss Tool/model_verification.json anhand der offiziellen OpenAI Docs aktualisiert werden."
        )
    return {"checked_at": verification.get("checked_at"), **entry}


def call_openai(
    *,
    model: str,
    content: list[dict[str, Any]],
    catalog: Catalog,
    client: Any,
) -> dict[str, Any]:
    try:
        response = client.responses.create(
            model=model,
            instructions=(
                "Du sichtest ausschliesslich sichtbare Merkmale von Holz-Zugproben. "
                "Du gibst keine nachgewiesene Bruchursache aus und haeltst dich strikt an das JSON-Schema."
            ),
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "holz_zugversuch_multilabel",
                    "strict": True,
                    "schema": response_schema(catalog.codes),
                }
            },
            store=False,
        )
        return json.loads(response.output_text)
    except (json.JSONDecodeError, AttributeError) as exc:
        raise AnalysisError("OpenAI-Antwort enthaelt kein lesbares strukturiertes JSON.") from exc
    except Exception as exc:
        raise safe_request_error(exc, model=model) from exc


def _validated_evidence(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise AnalysisError(f"Modellantwort: {field} muss eine Liste nichtleerer Texte sein.")
    for text in value:
        if CAUSAL_CLAIM.search(text):
            raise AnalysisError(f"Modellantwort: unzulaessige Ursachenaussage in {field}.")
    return [item.strip() for item in value]


def validate_and_enrich(raw: Mapping[str, Any], catalog: Catalog) -> dict[str, Any]:
    """Validiert Codes, Multi-Label-Struktur und das Verbot kausaler Aussagen."""

    status = raw.get("status")
    if status not in {"KLASSIFIZIERT", "UNKLAR"}:
        raise AnalysisError("Modellantwort: ungueltiger Status.")
    primary_code = raw.get("primary_code")
    if primary_code not in {*catalog.codes, "UNKLAR"}:
        raise AnalysisError(f"Modellantwort: unbekannter primaerer Code {primary_code!r}.")
    secondary_raw = raw.get("secondary_features")
    if not isinstance(secondary_raw, list):
        raise AnalysisError("Modellantwort: secondary_features muss eine Liste sein.")

    if status == "UNKLAR":
        if primary_code != "UNKLAR" or secondary_raw:
            raise AnalysisError("UNKLAR darf keine primaere oder sekundaere Klasse enthalten.")
        primary = None
    else:
        if primary_code == "UNKLAR":
            raise AnalysisError("KLASSIFIZIERT benoetigt eine der 29 Klassen als primaeres Merkmal.")
        evidence = _validated_evidence(raw.get("primary_evidence"), field="primary_evidence")
        if not evidence:
            raise AnalysisError("Primaeres Merkmal benoetigt bildbezogene Belege.")
        primary = _enrich_feature(
            primary_code,
            evidence,
            raw.get("primary_confidence"),
            catalog,
        )

    secondary: list[dict[str, Any]] = []
    seen = {primary_code}
    for feature in secondary_raw:
        if not isinstance(feature, Mapping):
            raise AnalysisError("Modellantwort: ungueltiges sekundaeres Merkmal.")
        code = feature.get("code")
        if code not in catalog.definitions:
            raise AnalysisError(f"Modellantwort: unbekannter sekundaerer Code {code!r}.")
        if code in seen:
            raise AnalysisError(f"Modellantwort: Klasse {code!r} wurde mehrfach ausgegeben.")
        seen.add(code)
        evidence = _validated_evidence(feature.get("evidence"), field=f"evidence[{code}]")
        if not evidence:
            raise AnalysisError(f"Sekundaeres Merkmal {code!r} benoetigt bildbezogene Belege.")
        secondary.append(_enrich_feature(code, evidence, feature.get("confidence"), catalog))

    list_fields = {}
    for field in ("visible_observations", "uncertainty_reasons", "missing_views"):
        list_fields[field] = _validated_evidence(raw.get(field), field=field)

    return {
        "status": status,
        "primary_feature": primary,
        "secondary_features": secondary,
        **list_fields,
    }


def _enrich_feature(
    code: str,
    evidence: list[str],
    confidence: Any,
    catalog: Catalog,
) -> dict[str, Any]:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise AnalysisError(f"Modellantwort: Konfidenz fuer {code!r} muss zwischen 0 und 1 liegen.")
    definition = catalog.definitions[code]
    reference_count = catalog.reference_counts.get(code, 0)
    return {
        "code": code,
        "name": definition["name"],
        "feature_type": definition["feature_type"],
        "confidence": float(confidence),
        "image_based_reasoning": evidence,
        "has_confirmed_reference_image": reference_count > 0,
        "confirmed_reference_count": reference_count,
        "proven_failure_cause": False,
        "causality_note": (
            "Sichtbares Merkmal beziehungsweise Bruchbild; eine ursächliche Auslösung "
            "des Versagens ist durch die Bildklassifikation nicht nachgewiesen."
        ),
    }


def build_result_document(
    *,
    input_dir: Path,
    test_images: Sequence[Path],
    catalog: Catalog,
    model: str,
    mode: str,
    used_reference_count: int,
    validated_result: dict[str, Any],
    model_verification: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": input_dir.resolve().name,
        "execution": {
            "mode": mode,
            "provider": "openai" if mode == "live" else "mock",
            "model": model,
            "model_verification": model_verification,
            "paid_api_request_performed": mode == "live",
        },
        "inputs": [
            {"filename": path.name, "sha256": _sha256(path)}
            for path in test_images
        ],
        "classification_basis": {
            "allowed_class_count": len(catalog.codes),
            "classification_rules_version": catalog.rule_version,
            "confirmed_main_reference_count_available": len(catalog.references),
            "confirmed_main_reference_count_used": used_reference_count,
            "protocol_failure_cause_transmitted": False,
            "only_confirmed_main_references_used": True,
        },
        "classes_without_confirmed_references": [
            {"code": code, "name": catalog.definitions[code]["name"]}
            for code in catalog.codes_without_references
        ],
        "result": validated_result,
        "scientific_limits": [
            "Die Ausgabe beschreibt sichtbare Merkmale und keine nachgewiesene Versagensursache.",
            "Klassen ohne bestaetigte Hauptreferenzbilder sind gesondert gekennzeichnet.",
            "Das Ergebnis muss fachlich geprueft werden und ist keine Ground-Truth-Annotation.",
        ],
    }


def save_result(document: Mapping[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    analysis_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(document["analysis_id"]))
    destination = output_dir / f"{analysis_id}_{timestamp}.json"
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def run_analysis(
    input_dir: Path,
    *,
    live: bool = False,
    confirm_paid_request: bool = False,
    model: str | None = None,
    mock_response_path: Path = DEFAULT_MOCK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_references_per_class: int = DEFAULT_REFERENCES_PER_CLASS,
    client: Any = None,
) -> Path:
    catalog = load_catalog()
    test_images = collect_test_images(input_dir, catalog)
    selected_model = model or os.getenv("USECASE2_OPENAI_MODEL", "gpt-6.1-sol").strip()
    if not selected_model:
        raise AnalysisError("Keine OpenAI-Modellkennung konfiguriert.")

    if live:
        if not confirm_paid_request:
            raise AnalysisError(
                "Live-Aufruf gesperrt: Zusaetzlich zu --live ist --confirm-paid-request erforderlich."
            )
        verification = load_verified_model(selected_model)
        if client is None:
            resolution = resolve_api_key()
            client = create_openai_client(resolution)
        selected_references = select_references(catalog, max_references_per_class)
        raw = call_openai(
            model=selected_model,
            content=build_request_content(test_images, catalog, selected_references),
            catalog=catalog,
            client=client,
        )
        mode = "live"
    else:
        verification = None
        selected_references = ()
        raw = _read_json(mock_response_path)
        mode = "mock"

    validated = validate_and_enrich(raw, catalog)
    document = build_result_document(
        input_dir=input_dir,
        test_images=test_images,
        catalog=catalog,
        model=selected_model,
        mode=mode,
        used_reference_count=len(selected_references),
        validated_result=validated,
        model_verification=verification,
    )
    return save_result(document, output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Label-Bildanalyse von Holz-Zugproben (standardmaessig kostenfreier Mock-Modus)."
    )
    parser.add_argument("input_dir", type=Path, help="Ordner mit genau 1 bis 3 Fotos")
    parser.add_argument("--model", help="OpenAI-Modell; alternativ USECASE2_OPENAI_MODEL")
    parser.add_argument("--mock-response", type=Path, default=DEFAULT_MOCK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-references-per-class",
        type=int,
        default=DEFAULT_REFERENCES_PER_CLASS,
        help="Anzahl bestaetigter Hauptreferenzen je Klasse (Standard: 1)",
    )
    parser.add_argument("--live", action="store_true", help="Echten Responses-API-Aufruf aktivieren")
    parser.add_argument(
        "--confirm-paid-request",
        action="store_true",
        help="Bestaetigt ausdruecklich, dass dieser Lauf kostenpflichtig sein kann",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = run_analysis(
            args.input_dir,
            live=args.live,
            confirm_paid_request=args.confirm_paid_request,
            model=args.model,
            mock_response_path=args.mock_response,
            output_dir=args.output_dir,
            max_references_per_class=args.max_references_per_class,
        )
    except (AnalysisError, OpenAIConfigurationError, RuntimeError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    print(f"Ergebnis gespeichert: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

