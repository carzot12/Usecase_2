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
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from .openai_config import (
        OpenAIConfigurationError,
        create_openai_client,
        resolve_api_key,
        safe_request_error,
    )
    from .mistral_config import resolve_mistral_api_key, safe_mistral_request_error
except ImportError:  # Direkter Aufruf als Skript bleibt moeglich.
    from openai_config import (  # type: ignore[no-redef]
        OpenAIConfigurationError,
        create_openai_client,
        resolve_api_key,
        safe_request_error,
    )
    from mistral_config import resolve_mistral_api_key, safe_mistral_request_error  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "Datenbank"
RULES_PATH = DATABASE_DIR / "klassifikationsregeln.json"
DEFINITIONS_PATH = DATABASE_DIR / "versagensarten_erweitert.json"
MANIFEST_PATH = DATABASE_DIR / "Referenzbilder" / "referenzbilder_manifest.json"
MODEL_VERIFICATION_PATH = Path(__file__).with_name("model_verification.json")
MISTRAL_MODEL_VERIFICATION_PATH = Path(__file__).with_name("mistral_model_verification.json")
DEFAULT_MOCK_PATH = Path(__file__).with_name("mock_response.json")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "KI_Klassifikation"
ALLOWED_INPUT_ROOT = PROJECT_ROOT / "Pruefdaten" / "Neue_Analysen"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_TEST_IMAGES = 3
MAX_TRANSMITTED_LONG_EDGE = 2048
DEFAULT_OLLAMA_MAX_IMAGE_EDGE = 768
DEFAULT_REFERENCES_PER_CLASS = 1
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 900
DEFAULT_OLLAMA_NUM_CTX = 16384
DEFAULT_MISTRAL_MODEL = "ministral-3b-2512"
DEFAULT_MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MISTRAL_TIMEOUT_SECONDS = 900
SYSTEM_INSTRUCTIONS = (
    "Du sichtest ausschliesslich sichtbare Merkmale von Holz-Zugproben. "
    "Du gibst keine nachgewiesene Bruchursache aus und haeltst dich strikt an das JSON-Schema."
)
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


def collect_test_images(
    input_dir: Path,
    catalog: Catalog,
    *,
    allowed_input_root: Path = ALLOWED_INPUT_ROOT,
) -> list[Path]:
    """Liest 1-3 Bilder und verhindert bekannte Referenzproben im Testdatensatz."""

    resolved_dir = input_dir.expanduser().resolve()
    allowed_root = allowed_input_root.expanduser().resolve()
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


def _jpeg_base64(path: Path, *, max_long_edge: int = MAX_TRANSMITTED_LONG_EDGE) -> str:
    """Erzeugt eine verkleinerte JPEG-Kopie im RAM; das Original bleibt unveraendert."""

    if isinstance(max_long_edge, bool) or not isinstance(max_long_edge, int) or max_long_edge <= 0:
        raise AnalysisError("max_long_edge muss eine positive Ganzzahl sein.")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_long_edge, max_long_edge))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _data_url(path: Path) -> str:
    """Erzeugt eine kostenschonende Data-URL fuer den OpenAI-Bildinput."""

    return f"data:image/jpeg;base64,{_jpeg_base64(path)}"


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


def _analysis_instructions(catalog: Catalog) -> dict[str, Any]:
    """Gemeinsamer fachlicher Prompt fuer OpenAI und Ollama."""

    return {
        "task": "Multi-Label-Sichtung sichtbarer Merkmale an einer unbekannten Holz-Zugprobe",
        "rules": [
            "Verwende ausschliesslich die bereitgestellten 29 Klassencodes oder UNKLAR.",
            "Trenne genau ein primaeres sichtbares Merkmal von moeglichen sekundaeren Merkmalen.",
            "Begruende jede Klasse ausschliesslich mit konkret sichtbaren Bildmerkmalen.",
            "Behaupte niemals, ein Holzmerkmal sei die nachgewiesene Versagens- oder Bruchursache.",
            "Gib UNKLAR aus, wenn Ansichten, Massstab, Schaerfe oder sichtbare Merkmale nicht ausreichen.",
            "Klassen ohne bestaetigtes Referenzbild bleiben zulaessig, muessen aber besonders vorsichtig anhand der Definition beurteilt werden.",
            "Referenzbilder sind nur visuelle Beispiele und keine Daten des unbekannten Pruefkoerpers.",
            "Beschreibe in der Ausgabe ausschliesslich die unbekannten Testbilder, niemals die einzelnen Referenzbilder.",
            "Halte die Ausgabe knapp: hoechstens 3 Belege je Merkmal, 4 sekundaere Merkmale, 6 Beobachtungen sowie je 4 Unsicherheiten und fehlende Ansichten.",
            "Bei Status UNKLAR muss primary_code UNKLAR sein; primary_evidence und secondary_features muessen dann leer sein.",
        ],
        "classes": _prompt_catalog(catalog),
    }


def build_request_content(
    test_images: Sequence[Path],
    catalog: Catalog,
    references: Sequence[ReferenceImage],
) -> list[dict[str, Any]]:
    """Baut Bildinput ohne Protokolldaten, Probennummern oder Referenzpfade."""

    instructions = _analysis_instructions(catalog)
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
    content.append({
        "type": "input_text",
        "text": (
            "Erzeuge jetzt ausschliesslich die knappe JSON-Klassifikation der unbekannten "
            "Testbilder 1 bis 3. Beschreibe keine Referenzbilder."
        ),
    })
    return content


def response_schema(codes: Iterable[str]) -> dict[str, Any]:
    code_enum = [*codes, "UNKLAR"]
    feature = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "enum": list(codes)},
            "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
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
            "secondary_features": {"type": "array", "items": feature, "maxItems": 4},
            "visible_observations": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "uncertainty_reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "missing_views": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
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


def load_verified_mistral_model(
    model: str,
    path: Path = MISTRAL_MODEL_VERIFICATION_PATH,
) -> dict[str, Any]:
    verification = _read_json(path)
    entry = next((item for item in verification.get("models", []) if item.get("model") == model), None)
    if not entry or not entry.get("chat_completions_api") or not entry.get("image_input"):
        raise AnalysisError(
            f"Modell {model!r} ist lokal nicht als Mistral-Chat-/Vision-Modell dokumentiert."
        )
    return {"checked_at": verification.get("checked_at"), **entry}


def call_openai(
    *,
    model: str,
    content: list[dict[str, Any]],
    catalog: Catalog,
    client: Any,
) -> dict[str, Any]:
    raw, _ = call_openai_with_metadata(
        model=model,
        content=content,
        catalog=catalog,
        client=client,
    )
    return raw


def call_openai_with_metadata(
    *,
    model: str,
    content: list[dict[str, Any]],
    catalog: Catalog,
    client: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """OpenAI-Aufruf mit Nutzungsmetadaten fuer reproduzierbare Evaluationen.

    ``call_openai`` bleibt aus Rueckwaertskompatibilitaet unveraendert und gibt
    weiterhin nur das dekodierte Ergebnisobjekt zurueck.
    """

    try:
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
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
        raw = json.loads(response.output_text)
        usage = getattr(response, "usage", None)
        if usage is None:
            usage_data: dict[str, Any] = {}
        elif hasattr(usage, "model_dump"):
            usage_data = usage.model_dump()
        elif isinstance(usage, Mapping):
            usage_data = dict(usage)
        else:
            usage_data = {
                key: getattr(usage, key)
                for key in ("input_tokens", "output_tokens", "total_tokens")
                if getattr(usage, key, None) is not None
            }
        metadata = {
            "response_id": getattr(response, "id", None),
            "response_model": getattr(response, "model", None),
            "usage": usage_data,
            "cost": {
                "currency": "USD",
                "reported_amount": None,
                "note": "Die Responses-API meldet Tokenverbrauch, aber keinen Rechnungsbetrag.",
            },
        }
        return raw, metadata
    except (json.JSONDecodeError, AttributeError) as exc:
        raise AnalysisError("OpenAI-Antwort enthaelt kein lesbares strukturiertes JSON.") from exc
    except Exception as exc:
        raise safe_request_error(exc, model=model) from exc


def build_mistral_messages(
    test_images: Sequence[Path],
    catalog: Catalog,
    references: Sequence[ReferenceImage],
) -> list[dict[str, Any]]:
    """Überträgt den gemeinsamen OpenAI-Bildinhalt verlustfrei ins Mistral-Format."""

    content = []
    for item in build_request_content(test_images, catalog, references):
        if item["type"] == "input_text":
            content.append({"type": "text", "text": item["text"]})
        elif item["type"] == "input_image":
            content.append({"type": "image_url", "image_url": item["image_url"]})
        else:  # Defensive Absicherung bei einer späteren Erweiterung des gemeinsamen Prompts.
            raise AnalysisError(f"Nicht unterstützter Mistral-Inhaltstyp: {item['type']!r}")
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": content},
    ]


def _mistral_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        if text_parts:
            return "".join(text_parts)
    raise AnalysisError("Mistral-Antwort enthält keinen lesbaren Text.")


def call_mistral(
    *,
    model: str,
    messages: list[dict[str, Any]],
    catalog: Catalog,
    api_key: str,
    endpoint: str = DEFAULT_MISTRAL_URL,
    timeout_seconds: int = DEFAULT_MISTRAL_TIMEOUT_SECONDS,
    opener: Any = urlopen,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ruft die Mistral Chat-Completions-API mit Vision und JSON-Schema auf."""

    if endpoint != DEFAULT_MISTRAL_URL:
        raise AnalysisError("Mistral-Endpunkt muss aus Sicherheitsgründen die offizielle HTTPS-API sein.")
    if not api_key.strip():
        raise AnalysisError("Der Mistral-Schlüsselbundwert ist leer.")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4096,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "holz_zugversuch_multilabel",
                # `schema_definition` is the Python SDK field name. The direct
                # HTTP API requires its serialized alias `schema`.
                "schema": response_schema(catalog.codes),
                "strict": True,
            },
        },
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            envelope = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError) as exc:
        raise safe_mistral_request_error(exc, model=model) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError("Mistral-Antwort kann nicht als JSON gelesen werden.") from exc

    try:
        choice = envelope["choices"][0]
        raw = json.loads(_mistral_message_text(choice["message"]["content"]))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise AnalysisError("Mistral-Antwort enthält kein lesbares strukturiertes JSON.") from exc
    if not isinstance(raw, dict):
        raise AnalysisError("Mistral-Antwort muss ein JSON-Objekt enthalten.")

    usage = envelope.get("usage") if isinstance(envelope.get("usage"), Mapping) else {}
    metadata = {
        "request_id": envelope.get("id"),
        "response_model": envelope.get("model"),
        "created": envelope.get("created"),
        "finish_reason": choice.get("finish_reason"),
        "usage": dict(usage),
        "cost": {
            "currency": "USD",
            "reported_amount": None,
            "note": "Die Chat-Completions-Antwort meldet Tokenverbrauch, aber keinen Rechnungsbetrag.",
        },
        "request_options": {
            "temperature": 0,
            "max_tokens": 4096,
            "response_format": "json_schema",
        },
    }
    return raw, metadata


def build_ollama_messages(
    test_images: Sequence[Path],
    catalog: Catalog,
    references: Sequence[ReferenceImage],
    max_image_edge: int = DEFAULT_OLLAMA_MAX_IMAGE_EDGE,
) -> list[dict[str, Any]]:
    """Baut denselben Prompt und dieselbe Bildreihenfolge fuer Ollama."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {
            "role": "user",
            "content": json.dumps(
                _analysis_instructions(catalog),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    for reference in references:
        messages.append({
            "role": "user",
            "content": f"Bestaetigtes Hauptreferenzbild fuer Klasse {reference.code}.",
            "images": [_jpeg_base64(reference.path, max_long_edge=max_image_edge)],
        })
    messages.append({
        "role": "user",
        "content": "Ab hier folgen ausschliesslich die 1 bis 3 Bilder des unbekannten Testpruefkoerpers.",
    })
    for index, path in enumerate(test_images, start=1):
        messages.append({
            "role": "user",
            "content": f"Unbekanntes Testbild {index}.",
            "images": [_jpeg_base64(path, max_long_edge=max_image_edge)],
        })
    messages.append({
        "role": "user",
        "content": (
            "Erzeuge jetzt ausschliesslich die knappe JSON-Klassifikation der unbekannten "
            "Testbilder 1 bis 3. Beschreibe keine Referenzbilder."
        ),
    })
    return messages


def _parse_ollama_json(content: Any) -> dict[str, Any]:
    """Akzeptiert JSON sowie haeufige Markdown-Huellen lokaler Modelle."""

    if not isinstance(content, str) or not content.strip():
        raise AnalysisError("Ollama-Antwort enthaelt keinen Text.")
    candidates = [content.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(content[first_brace:last_brace + 1])
    for candidate in candidates:
        for strict in (True, False):
            try:
                parsed = json.loads(candidate, strict=strict)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    preview = re.sub(r"\s+", " ", content).strip()[:500]
    raise AnalysisError(f"Ollama-Antwort enthaelt kein lesbares strukturiertes JSON: {preview}")


def call_ollama(
    *,
    model: str,
    messages: list[dict[str, Any]],
    catalog: Catalog,
    endpoint: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
    opener: Any = urlopen,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ruft ausschliesslich die lokale Ollama-Chat-API auf."""

    if endpoint.rstrip("/") != DEFAULT_OLLAMA_URL:
        raise AnalysisError("Ollama-Endpunkt muss aus Sicherheitsgruenden http://localhost:11434 sein.")
    if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx <= 0:
        raise AnalysisError("ollama_num_ctx muss eine positive Ganzzahl sein.")
    ollama_options = {
        "temperature": 0,
        "num_predict": 4096,
        "num_ctx": num_ctx,
    }
    payload = {
        "model": model,
        "messages": messages,
        "format": response_schema(catalog.codes),
        "stream": False,
        "think": False,
        "options": ollama_options,
    }
    request = Request(
        f"{DEFAULT_OLLAMA_URL}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            envelope = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        raise AnalysisError(f"Ollama-HTTP-Fehler {exc.code}: {detail[:500]}") from exc
    except URLError as exc:
        raise AnalysisError(
            "Lokales Ollama ist unter http://localhost:11434 nicht erreichbar."
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError("Ollama-Antwort kann nicht gelesen werden.") from exc

    try:
        content = envelope["message"]["content"]
        raw = _parse_ollama_json(content)
    except (KeyError, TypeError) as exc:
        raise AnalysisError("Ollama-Antwort enthaelt kein lesbares strukturiertes JSON.") from exc
    except AnalysisError as exc:
        tail = re.sub(r"\s+", " ", str(content)).strip()[-300:]
        thinking = envelope.get("message", {}).get("thinking", "")
        raise AnalysisError(
            f"{exc} [Zeichen: {len(content)}, Thinking-Zeichen: {len(thinking)}, "
            f"prompt_eval_count: {envelope.get('prompt_eval_count')}, "
            f"eval_count: {envelope.get('eval_count')}, "
            f"done_reason: {envelope.get('done_reason')}, Ende: {tail}]"
        ) from exc
    metadata = {
        key: envelope[key]
        for key in (
            "created_at", "done_reason", "total_duration", "load_duration",
            "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
        )
        if key in envelope
    }
    metadata["request_options"] = dict(ollama_options)
    return raw, metadata


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
    provider: str,
    model: str,
    mode: str,
    duration_seconds: float,
    used_reference_count: int,
    validated_result: dict[str, Any],
    model_verification: dict[str, Any] | None,
    provider_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": input_dir.resolve().name,
        "execution": {
            "mode": mode,
            "provider": provider,
            "model": model,
            "model_verification": model_verification,
            "duration_seconds": round(duration_seconds, 3),
            "provider_metadata": dict(provider_metadata or {}),
            "paid_api_request_performed": provider in {"openai", "mistral"} and mode == "live",
            "images_transmitted_to_openai": provider == "openai" and mode == "live",
            "images_transmitted_to_mistral": provider == "mistral" and mode == "live",
            "images_transmitted_to_cloud": provider in {"openai", "mistral"} and mode == "live",
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
    execution = document.get("execution", {})
    provider = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(execution.get("provider", "unknown")))
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(execution.get("model", "unknown")))
    destination = output_dir / f"{analysis_id}_{provider}_{model}_{timestamp}.json"
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
    provider: str = "openai",
    model: str | None = None,
    mock_response_path: Path = DEFAULT_MOCK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_references_per_class: int = DEFAULT_REFERENCES_PER_CLASS,
    ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
    ollama_max_image_edge: int = DEFAULT_OLLAMA_MAX_IMAGE_EDGE,
    allowed_input_root: Path = ALLOWED_INPUT_ROOT,
    client: Any = None,
    ollama_opener: Any = urlopen,
    mistral_opener: Any = urlopen,
    mistral_key_resolver: Any = resolve_mistral_api_key,
) -> Path:
    started_at = time.perf_counter()
    catalog = load_catalog()
    test_images = collect_test_images(
        input_dir,
        catalog,
        allowed_input_root=allowed_input_root,
    )
    if provider not in {"openai", "ollama", "mistral"}:
        raise AnalysisError("Provider muss 'openai', 'ollama' oder 'mistral' sein.")
    default_models = {
        "ollama": os.getenv("USECASE2_OLLAMA_MODEL", "llava:latest"),
        "openai": os.getenv("USECASE2_OPENAI_MODEL", "gpt-6.1-sol"),
        "mistral": DEFAULT_MISTRAL_MODEL,
    }
    default_model = default_models[provider]
    selected_model = (model or default_model).strip()
    if not selected_model:
        raise AnalysisError(f"Keine Modellkennung fuer Provider {provider!r} konfiguriert.")

    provider_metadata: dict[str, Any] = {}
    if provider == "ollama":
        verification = None
        selected_references = select_references(catalog, max_references_per_class)
        raw, provider_metadata = call_ollama(
            model=selected_model,
            messages=build_ollama_messages(
                test_images,
                catalog,
                selected_references,
                max_image_edge=ollama_max_image_edge,
            ),
            catalog=catalog,
            num_ctx=ollama_num_ctx,
            opener=ollama_opener,
        )
        mode = "local"
        result_provider = "ollama"
    elif provider == "mistral" and live:
        if not confirm_paid_request:
            raise AnalysisError(
                "Live-Aufruf gesperrt: Zusätzlich zu --live ist --confirm-paid-request erforderlich."
            )
        verification = load_verified_mistral_model(selected_model)
        selected_references = select_references(catalog, max_references_per_class)
        resolution = mistral_key_resolver()
        raw, provider_metadata = call_mistral(
            model=selected_model,
            messages=build_mistral_messages(test_images, catalog, selected_references),
            catalog=catalog,
            api_key=resolution.value,
            opener=mistral_opener,
        )
        provider_metadata["credential_source"] = resolution.source
        provider_metadata["keychain_service"] = resolution.service
        mode = "live"
        result_provider = "mistral"
    elif provider == "openai" and live:
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
        result_provider = "openai"
    elif provider == "mistral":
        verification = load_verified_mistral_model(selected_model)
        selected_references = select_references(catalog, max_references_per_class)
        raw = _read_json(mock_response_path)
        provider_metadata = {
            "network_request_performed": False,
            "keychain_accessed": False,
            "usage": {},
            "cost": {
                "currency": "USD",
                "reported_amount": 0,
                "note": "Mock-Test ohne API-Aufruf.",
            },
        }
        mode = "mock"
        result_provider = "mistral"
    else:
        verification = None
        selected_references = ()
        raw = _read_json(mock_response_path)
        mode = "mock"
        result_provider = "mock"

    validated = validate_and_enrich(raw, catalog)
    duration_seconds = time.perf_counter() - started_at
    document = build_result_document(
        input_dir=input_dir,
        test_images=test_images,
        catalog=catalog,
        provider=result_provider,
        model=selected_model,
        mode=mode,
        duration_seconds=duration_seconds,
        used_reference_count=len(selected_references),
        validated_result=validated,
        model_verification=verification,
        provider_metadata=provider_metadata,
    )
    return save_result(document, output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Label-Bildanalyse von Holz-Zugproben (standardmaessig kostenfreier Mock-Modus)."
    )
    parser.add_argument("input_dir", type=Path, help="Ordner mit genau 1 bis 3 Fotos")
    parser.add_argument(
        "--provider",
        choices=("openai", "ollama", "mistral"),
        default="openai",
        help="Analyseanbieter (openai und mistral bleiben ohne --live im Mock-Modus)",
    )
    parser.add_argument(
        "--model",
        help="Modellkennung, z. B. gpt-6.1-sol, ministral-3b-2512 oder qwen3-vl:4b",
    )
    parser.add_argument("--mock-response", type=Path, default=DEFAULT_MOCK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-references-per-class",
        type=int,
        default=DEFAULT_REFERENCES_PER_CLASS,
        help="Anzahl bestaetigter Hauptreferenzen je Klasse (Standard: 1)",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=DEFAULT_OLLAMA_NUM_CTX,
        metavar="TOKENS",
        help="Ollama-Kontextgroesse in Tokens (Standard: 16384)",
    )
    parser.add_argument(
        "--ollama-max-image-edge",
        type=int,
        default=DEFAULT_OLLAMA_MAX_IMAGE_EDGE,
        metavar="PIXEL",
        help="Maximale Kantenlaenge temporaerer Ollama-Bilder (Standard: 768)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Echten OpenAI- oder Mistral-API-Aufruf aktivieren; für Ollama nicht erforderlich",
    )
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
            provider=args.provider,
            model=args.model,
            mock_response_path=args.mock_response,
            output_dir=args.output_dir,
            max_references_per_class=args.max_references_per_class,
            ollama_num_ctx=args.ollama_num_ctx,
            ollama_max_image_edge=args.ollama_max_image_edge,
        )
    except (AnalysisError, OpenAIConfigurationError, RuntimeError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    print(f"Ergebnis gespeichert: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

