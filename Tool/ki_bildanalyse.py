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
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from .evaluation_variants import ACTIVE_CLASS_CODES, VARIANT_IDS, inspect_variant, memory_status
    from .openai_config import (
        OpenAIConfigurationError,
        create_openai_client,
        resolve_api_key,
        safe_request_error,
    )
    from .mistral_config import resolve_mistral_api_key, safe_mistral_request_error
except ImportError:  # Direkter Aufruf als Skript bleibt moeglich.
    from evaluation_variants import ACTIVE_CLASS_CODES, VARIANT_IDS, inspect_variant, memory_status  # type: ignore[no-redef]
    from openai_config import (  # type: ignore[no-redef]
        OpenAIConfigurationError,
        create_openai_client,
        resolve_api_key,
        safe_request_error,
    )
    from mistral_config import resolve_mistral_api_key, safe_mistral_request_error  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "Datenbank"
EVALUATION_VARIANTS_DIR = DATABASE_DIR / "Evaluationsvarianten"
REFERENCE_MEMORY_DIR = PROJECT_ROOT / "Output" / "KI_Evaluation" / "Referenzgedaechtnis"
RULES_PATH = DATABASE_DIR / "klassifikationsregeln.json"
DEFINITIONS_PATH = DATABASE_DIR / "versagensarten_erweitert.json"
MANIFEST_PATH = DATABASE_DIR / "Referenzbilder" / "referenzbilder_manifest.json"
MODEL_VERIFICATION_PATH = Path(__file__).with_name("model_verification.json")
MISTRAL_MODEL_VERIFICATION_PATH = Path(__file__).with_name("mistral_model_verification.json")
DEFAULT_MOCK_PATH = Path(__file__).with_name("mock_response.json")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "KI_Klassifikation"
ALLOWED_INPUT_ROOT = PROJECT_ROOT / "Pruefdaten" / "Neue_Analysen"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_MISTRAL_IMAGES_PER_REQUEST = 8
# ``None`` bedeutet: Die vorhandene Provider-Implementierung erzwingt keine
# feste Bildanzahl. Kontext-, Token- und Größenlimits gelten weiterhin.
PROVIDER_MAX_TEST_IMAGES_PER_REQUEST: dict[str, int | None] = {
    "ollama": None,
    "openai": None,
    "mistral": MAX_MISTRAL_IMAGES_PER_REQUEST,
}
MAX_TRANSMITTED_LONG_EDGE = 2048
DEFAULT_OLLAMA_MAX_IMAGE_EDGE = 768
DEFAULT_REFERENCES_PER_CLASS = 1
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 900
DEFAULT_OLLAMA_MODEL = "qwen3-vl:4b-instruct"
DEFAULT_OLLAMA_NUM_CTX = 24576
DEFAULT_OLLAMA_NUM_PREDICT = 4096
DEFAULT_MISTRAL_MODEL = "ministral-3b-2512"
DEFAULT_MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MISTRAL_TIMEOUT_SECONDS = 900
PROMPT_VERSION = "5.5-scientific-evidence-gated-multilabel"
SYSTEM_INSTRUCTIONS = (
    "Du klassifizierst sichtbare Versagensmerkmale von Holz-Zugproben anhand aller "
    "gemeinsam bereitgestellten Bilder. Ein Klassenvorschlag ist keine nachgewiesene "
    "Bruchursache. Halte dich strikt an das JSON-Schema und gib niemals eine vollständig "
    "leere Sichtauswertung zurück."
)
CAUSAL_CLAIM = re.compile(
    r"""
    \b(?:
        verursacht(?:e|en)?\s+(?!nicht\s+)
            (?:(?:den|die|das|einen|eine|ein)\s+)?(?:bruch|versagen)(?!\s+nicht\b)
        |(?:bruchursache|versagensursache|ursache\s+des\s+(?:bruchs|versagens))
            \s+(?:ist|war|:|=)\s+
            (?!nicht\b|unklar\b|offen\b|unbekannt\b|aus\b)
        |(?:ist|war)\s+(?!nicht\s+)(?:die\s+)?(?:bruchursache|versagensursache)
        |(?:ist|war)\s+(?!nicht\s+)ursächlich(?:\s+für)?
        |führte\s+(?!nicht\s+)(?:zum|zu\s+dem)\s+(?:bruch|versagen)(?!\s+nicht\b)
        |löste\s+(?!nicht\s+)(?:den|einen)\s+bruch\s+aus
        |hat\s+(?:den|einen)\s+bruch\s+(?!nicht\s+)ausgelöst
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


class AnalysisError(RuntimeError):
    """Eine Eingabe, Datenbasis oder Modellantwort verletzt die Anforderungen."""


class ModelOutputError(AnalysisError):
    """Provider response that cannot be stored as a scientific result."""

    failure_type = "technical_model_output"

    def __init__(self, message: str, *, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


class OllamaOutputError(ModelOutputError):
    """Technischer Ollama-Fehler mit rückwärtskompatiblem Fehlertyp."""


@dataclass(frozen=True)
class ReferenceImage:
    code: str
    path: Path
    specimen_id: str = ""
    description: str = ""


@dataclass(frozen=True)
class Catalog:
    definitions: dict[str, dict[str, Any]]
    references: tuple[ReferenceImage, ...]
    known_dataset_paths: tuple[Path, ...]
    reference_counts: dict[str, int]
    rule_version: str
    reference_database_accessed: bool = True
    active_class_codes: tuple[str, ...] = ()
    additional_visible_codes: tuple[str, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(self.definitions)

    @property
    def codes_without_references(self) -> tuple[str, ...]:
        reference_codes = self.active_class_codes or self.codes
        return tuple(code for code in reference_codes if self.reference_counts.get(code, 0) == 0)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"JSON-Datei kann nicht gelesen werden: {path}") from exc


def load_catalog(
    project_root: Path = PROJECT_ROOT,
    *,
    include_references: bool = True,
    allowed_codes: Sequence[str] | None = None,
    additional_visible_codes: Sequence[str] = (),
) -> Catalog:
    """Lädt Klassendefinitionen und optional bestätigte Hauptreferenzbilder.

    ``include_references=False`` ist eine harte Sicherheitsgrenze für
    referenzfreie Evaluationsläufe: In diesem Fall wird das Referenzmanifest
    weder geöffnet noch werden Referenzbildpfade aufgelöst.
    """

    database = project_root / "Datenbank"
    extended = _read_json(database / "versagensarten_erweitert.json")
    rules = _read_json(database / "klassifikationsregeln.json")

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
    required_rule_fields = {
        "code", "name", "klassentyp", "klassifikationsziel",
        "positive_bildmerkmale", "abgrenzung_zu_aehnlichen_klassen",
        "nicht_vergeben_wenn", "unklar_wenn",
        "einschraenkungen_visuelle_erkennung",
        "nicht_zulaessige_ursachenaussage", "fachlicher_pruefstatus",
        "literaturquellen",
    }
    list_rule_fields = {
        "positive_bildmerkmale", "abgrenzung_zu_aehnlichen_klassen",
        "nicht_vergeben_wenn", "unklar_wenn",
        "einschraenkungen_visuelle_erkennung", "literaturquellen",
    }
    validated_codes: set[str] = set()
    for rule_index, rule in enumerate(validated_rules):
        if not isinstance(rule, dict):
            raise AnalysisError("Jede Klassifikationsregel muss ein JSON-Objekt sein.")
        code = str(rule.get("code", "")).strip()
        if code not in definitions:
            raise AnalysisError(f"Klassifikationsregel verwendet unbekannte Klasse: {code!r}")
        if code in validated_codes:
            raise AnalysisError(f"Klassifikationsregel ist doppelt vorhanden: {code!r}")
        validated_codes.add(code)
        missing = sorted(required_rule_fields - set(rule))
        if missing:
            raise AnalysisError(
                f"Klassifikationsregel {code!r} enthält nicht alle Pflichtfelder: "
                + ", ".join(missing)
            )
        for field in list_rule_fields:
            value = rule[field]
            if not isinstance(value, list) or not value or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise AnalysisError(
                    f"Klassifikationsregel {code!r}: {field} muss eine nichtleere Textliste sein."
                )
        if not isinstance(rule["fachlicher_pruefstatus"], dict):
            raise AnalysisError(
                f"Klassifikationsregel {code!r}: fachlicher_pruefstatus muss ein Objekt sein."
            )
        definitions[code]["validated_classification_rule"] = dict(rule)
        definitions[code]["validated_classification_rule_order"] = rule_index

    known_definition_codes = set(definitions)
    original_definitions = definitions
    active_codes: tuple[str, ...]
    if allowed_codes is not None:
        requested = tuple(str(code).strip() for code in allowed_codes)
        if not requested or len(requested) != len(set(requested)):
            raise AnalysisError("Die erlaubten Klassencodes müssen vorhanden und eindeutig sein.")
        unknown = [code for code in requested if code not in definitions]
        if unknown:
            raise AnalysisError("Unbekannte Klassencodes: " + ", ".join(unknown))
        active_codes = requested
        additional = tuple(str(code).strip() for code in additional_visible_codes)
        if len(additional) != len(set(additional)) or any(code not in original_definitions for code in additional):
            raise AnalysisError("Zusätzliche sichtbare Merkmalscodes müssen vorhanden und eindeutig sein.")
        if set(active_codes) & set(additional):
            raise AnalysisError("Aktive und zusätzliche Merkmalscodes müssen getrennt sein.")
        definitions = {code: original_definitions[code] for code in (*active_codes, *additional)}
    else:
        active_codes = tuple(definitions)
        additional = ()

    references: list[ReferenceImage] = []
    known_paths: set[Path] = set()
    counts: Counter[str] = Counter()
    if not include_references:
        return Catalog(
            definitions=definitions,
            references=(),
            known_dataset_paths=(),
            reference_counts={},
            rule_version=str(rules.get("meta", {}).get("version", "unbekannt")),
            reference_database_accessed=False,
            active_class_codes=active_codes,
            additional_visible_codes=additional,
        )

    manifest = _read_json(database / "Referenzbilder" / "referenzbilder_manifest.json")
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
        if code in additional:
            # Zusätzliche sichtbare Holzmerkmale besitzen bewusst keine
            # Referenzordner und dürfen keine historischen Referenzen laden.
            continue
        if code not in definitions:
            if code in known_definition_codes:
                # Das Manifest bleibt vollständig erhalten. Ein explizit
                # begrenzter Klassensatz lädt nur seine eigenen Referenzen.
                continue
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
        reference_database_accessed=True,
        active_class_codes=active_codes,
        additional_visible_codes=additional,
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
    selected_image_names: Sequence[str] | None = None,
) -> list[Path]:
    """Liest die ausgewählten Fotos und verhindert bekannte Referenzproben im Testdatensatz."""

    resolved_dir = input_dir.expanduser().resolve()
    allowed_root = allowed_input_root.expanduser().resolve()
    if not resolved_dir.is_relative_to(allowed_root):
        raise AnalysisError(f"Eingabeordner muss unter {allowed_root} liegen.")
    if not resolved_dir.is_dir():
        raise AnalysisError(f"Eingabeordner fehlt: {resolved_dir}")

    source_images = {
        path.name: path.resolve()
        for path in resolved_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.casefold().startswith("nummer_")
    }
    if selected_image_names is not None:
        names = list(selected_image_names)
        if not names:
            raise AnalysisError("Mindestens ein Analysebild muss ausgewählt sein.")
        if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
            raise AnalysisError("Die Bildauswahl enthält ungültige oder doppelte Dateinamen.")
        missing = [name for name in names if name not in source_images]
        if missing:
            raise AnalysisError(
                "Ausgewählte Analysebilder wurden im Prüfkörperordner nicht gefunden: "
                + ", ".join(missing)
            )
        images = [source_images[name] for name in names]
    else:
        # Backwards-compatible CLI support for older projects using this sidecar folder.
        selection_dir = resolved_dir / "Analyseauswahl"
        image_dir = selection_dir if selection_dir.is_dir() else resolved_dir
        images = sorted(
            path.resolve()
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    if not images:
        raise AnalysisError("Der Eingabeordner enthält keine Analysebilder.")

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


def _reference_specimen_id(reference: ReferenceImage) -> str:
    """Return the specimen folder used by the confirmed reference manifest layout."""

    if reference.specimen_id:
        return reference.specimen_id
    return reference.path.parents[1].name if len(reference.path.parents) > 1 else ""


def select_references(
    catalog: Catalog,
    max_per_class: int,
    *,
    excluded_specimen_id: str | None = None,
    excluded_hashes: set[str] | None = None,
) -> tuple[ReferenceImage, ...]:
    if max_per_class < 0:
        raise AnalysisError("max_references_per_class darf nicht negativ sein.")
    selected: list[ReferenceImage] = []
    counts: Counter[str] = Counter()
    excluded_hashes = excluded_hashes or set()
    for reference in catalog.references:
        if excluded_specimen_id and _reference_specimen_id(reference) == excluded_specimen_id:
            continue
        if excluded_hashes and _sha256(reference.path) in excluded_hashes:
            continue
        if counts[reference.code] >= max_per_class:
            continue
        selected.append(reference)
        counts[reference.code] += 1
    return tuple(selected)


def direct_variant_references(report: Mapping[str, Any]) -> tuple[ReferenceImage, ...]:
    """Load every confirmed file declared by a direct-reference variant."""

    variant_id = str(report.get("variantId", ""))
    if variant_id not in {"2A", "2B", "3"}:
        raise AnalysisError("Direkte Evaluationsreferenzen sind nur für Variante 2A, 2B und 3 vorgesehen.")
    root = Path(str(report.get("referenceRoot", ""))).resolve()
    by_code: dict[str, list[ReferenceImage]] = {code: [] for code in ACTIVE_CLASS_CODES}
    for item in report.get("references", []):
        if not isinstance(item, Mapping):
            raise AnalysisError(f"Variante {variant_id} enthält einen ungültigen Referenzeintrag.")
        code = str(item.get("code", ""))
        relative = str(item.get("relative_path", ""))
        path = (root / relative).resolve()
        if code not in ACTIVE_CLASS_CODES:
            raise AnalysisError(f"Variante {variant_id} enthält den unbekannten Merkmalscode {code!r}.")
        if not bool(item.get("confirmed")):
            raise AnalysisError(f"Referenzbild für {code} ist nicht fachlich bestätigt.")
        if not path.is_relative_to(root) or not path.is_file():
            raise AnalysisError(
                f"Referenzbild für {code} fehlt oder liegt außerhalb von Variante {variant_id}."
            )
        if str(item.get("sha256", "")) != _sha256(path):
            raise AnalysisError(f"Referenzbild für {code} wurde seit der Strukturprüfung verändert.")
        by_code[code].append(ReferenceImage(
            code, path, str(item.get("specimen", "")), str(item.get("description", "")),
        ))
    missing = [code for code in ACTIVE_CLASS_CODES if not by_code[code]]
    if missing:
        raise AnalysisError(
            f"Variante {variant_id} benötigt bestätigte Referenzbilder für alle zwölf Merkmale: "
            + ", ".join(missing)
        )
    if variant_id == "2A" and any(len(by_code[code]) != 1 for code in ACTIVE_CLASS_CODES):
        raise AnalysisError(f"Variante {variant_id} benötigt genau ein bestätigtes Referenzbild je Merkmal.")
    return tuple(reference for code in ACTIVE_CLASS_CODES for reference in by_code[code])


def limit_mistral_references(
    test_images: Sequence[Path],
    references: Sequence[ReferenceImage],
) -> tuple[ReferenceImage, ...]:
    """Hält mindestens einen Bildplatz für den Prüfkörper im Mistral-Limit frei."""

    if not test_images:
        raise AnalysisError("Mindestens ein Analysebild muss ausgewählt sein.")
    available_slots = MAX_MISTRAL_IMAGES_PER_REQUEST - 1
    return tuple(references[:available_slots])


def _prompt_catalog(catalog: Catalog) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for code, definition in catalog.definitions.items():
        rule = definition.get("validated_classification_rule")
        if rule:
            # Die Anbieter erhalten exakt dieselbe kompakte Fachinformation.
            # Prüfvermerke, Literatur, interne offene Punkte und Ursachenhinweise
            # bleiben ausschließlich in der lokalen Regeldatei.
            item = {
                "code": code,
                "name": rule.get("name") or definition.get("name"),
                "definition": rule.get("klassifikationsziel"),
                "positive_image_features": rule.get("positive_bildmerkmale", []),
                "distinguishing_features": rule.get(
                    "abgrenzung_zu_aehnlichen_klassen", []
                ),
                "uncertainty_rules": {
                    "do_not_assign_when": rule.get("nicht_vergeben_wenn", []),
                    "unclear_when": rule.get("unklar_wenn", []),
                    "visual_limitations": rule.get(
                        "einschraenkungen_visuelle_erkennung", []
                    ),
                },
            }
        else:
            item = {
                "code": code,
                "name": definition.get("name"),
                "definition": definition.get("definition"),
                "review_required": bool(definition.get("review_required", False)),
            }
        if catalog.reference_database_accessed:
            item["confirmed_reference_count"] = catalog.reference_counts.get(code, 0)
        compact.append(item)
    return compact


def _analysis_instructions(catalog: Catalog) -> dict[str, Any]:
    """Gemeinsamer fachlicher Prompt fuer OpenAI und Ollama."""

    rules = [
        "Arbeite in zwei strikt getrennten Phasen: (1) sichtbare Beobachtungen knapp erfassen, (2) erst danach mögliche Klassencodes gleichrangig vorschlagen.",
        "Nenne in Phase 1 Risse und Rissverläufe, Splitter/Auffaserung, mögliche Jahrringtrennungen, Äste und deren Position, Faserrichtung, Bruchflächen und weitere tatsächlich sichtbare Auffälligkeiten.",
        "Untersuche den gesamten sichtbaren Prüfkörper und dokumentiere relevante Merkmale innerhalb und außerhalb der Bruchzone.",
        "Kennzeichne unsichere Beobachtungen ausdrücklich als unsicher. Unterscheide: nicht sichtbar = in den Bildern nicht erkennbar; nicht beurteilbar = Bildqualität, Ausschnitt oder Perspektive erlauben keine verlässliche Entscheidung.",
        f"Verwende ausschliesslich die bereitgestellten {len(catalog.codes)} Klassencodes oder UNKLAR.",
        "Alle bereitgestellten Klassen sind gleichberechtigt. Erzwinge insbesondere weder sp HB noch Spl HB.",
        "Werte alle ausgewählten Testbilder gemeinsam als verschiedene Ansichten desselben Prüfkörpers aus; entscheide nicht bildweise.",
        "Liste alle ausreichend visuell belegten, regelkonformen Merkmale vollständig und gleichberechtigt auf; mehrere Merkmale sind erlaubt, sofern jedes einzeln belegt ist.",
        "Vergib keine Rangfolge, keine Rollen wie primary oder secondary und keine Hauptversagensursache.",
        "Ordne jedem Klassenvorschlag nur eine knappe Sicherheitsstufe hoch, mittel oder unsicher zu; schreibe dazu keine Begründung.",
        "Kennzeichne jeden Klassenvorschlag zusätzlich mit evidence_status sicher_sichtbar oder vermutet. sicher_sichtbar verlangt direkt erkennbare Klassenmerkmale; vermutet ist nur bei einer in der jeweiligen Klassenregel ausdrücklich zugelassenen indirekten Evidenz zulässig.",
        "Prüfe vor jedem Klassenvorschlag einzeln: Ist die charakteristische Struktur tatsächlich sichtbar, ist ihre konkrete Bildstelle eindeutig benennbar, entspricht sie der Definition aus Regelwerk v1.4 und ist eine andere Klasse oder uneindeutige Bildstruktur ausreichend ausgeschlossen?",
        "Gib kein Merkmal vorsorglich aus. Bloße visuelle Ähnlichkeit mit bereitgestellten Referenzen oder eine typische Begleiterscheinung eines anderen Merkmals reichen nicht als Evidenz.",
        "Setze sicher_sichtbar nur, wenn die passende sichtbare Struktur und ihre Lage in visible_features konkret benannt werden können; erfinde keine Bildkoordinaten, Bildstellen oder Begründungen.",
        "Ist eine Zuordnung nicht eindeutig, nimm sie nicht als sicher_sichtbar auf. Nutze vermutet nur bei ausdrücklich regelkonformer indirekter Evidenz; dokumentiere sonst die Unsicherheit ausschließlich in uncertain_points.",
        "Nutze vermutet nicht als Ersatz für eine nicht beurteilbare Ansicht. Insbesondere darf SSA ohne tatsächlich sichtbare Schmalseite nicht vermutet werden.",
        "Erfinde bei unzureichender Bildqualität oder fehlender Sichtbarkeit keine Merkmale.",
        "Behaupte niemals, ein Holzmerkmal sei die nachgewiesene Versagens- oder Bruchursache.",
        "Gib UNKLAR aus, wenn Bildausschnitt, Perspektive oder Schärfe keine verlässliche Beurteilung erlauben; nenne die konkrete Einschränkung in uncertain_points.",
        "Gib KEINE_MERKMALE aus, wenn die Bilder beurteilbar sind, aber keines der definierten Merkmale sichtbar ist.",
        "Verwende KLASSIFIZIERT, sobald mindestens ein regelkonformes sichtbares Merkmal erkannt ist; fehlende Ursachensicherheit ist kein Grund für UNKLAR.",
        "Erfinde keine Winkel, Längen, Anzahlen oder sonstigen Entscheidungsschwellen.",
    ]
    if catalog.reference_database_accessed:
        rules.extend([
            "Klassen ohne bestaetigtes Referenzbild bleiben zulaessig, muessen aber besonders vorsichtig anhand der Definition beurteilt werden.",
            "Referenzbilder sind nur visuelle Beispiele und keine Daten des unbekannten Pruefkoerpers.",
            "Beschreibe in der Ausgabe ausschliesslich die unbekannten Testbilder, niemals die einzelnen Referenzbilder.",
        ])
    else:
        rules.append(
            "Beurteile ausschließlich die ausgewählten unbekannten Prüfkörperbilder und die bereitgestellten Klassenbeschreibungen."
        )
    rules.extend([
        "Gib ausschließlich kurze Stichpunktfragmente aus: keine Absätze, Einleitungen, bildweisen Mini-Gutachten oder wiederholten Informationen.",
        "Nutze observations_short für den Gesamtblick, visible_features für einzelne sichtbare Merkmale und uncertain_points nur für nicht sichere Punkte; wiederhole keinen Inhalt zwischen diesen Listen.",
        "Beginne keinen Stichpunkt mit Testbild, Bild oder In Bild. Nenne einen Bildbezug nur, wenn er zur Unterscheidung notwendig ist.",
        f"Begrenze jeden Text auf höchstens 120 Zeichen und möglichst acht Wörter; höchstens {len(catalog.codes)} Beobachtungen, {len(catalog.codes)} sichtbare Merkmale, {len(catalog.codes)} Klassenvorschläge und 6 Unsicherheiten.",
        "Bei Status UNKLAR oder KEINE_MERKMALE muss class_suggestions leer sein.",
        "Auch bei UNKLAR muss observations_short mindestens eine konkrete sichtbare Beobachtung oder eine konkrete Bildbegrenzung enthalten; gib niemals alle Textlisten leer aus.",
    ])
    return {
        "task": "Gemeinsame visuelle Mehrbildklassifikation aller erkennbaren Merkmale einer unbekannten Holz-Zugprobe",
        "rules": rules,
        "classes": _prompt_catalog(catalog),
    }


def _memory_text(reference_memory: Mapping[str, Any] | None) -> str | None:
    if not reference_memory:
        return None
    return json.dumps({
        "reference_memory": reference_memory.get("batch_outputs", []),
        "instruction": (
            "Nutze dieses zuvor aus dem festgelegten Referenzpool erzeugte Gedächtnis nur als "
            "allgemeines Vergleichswissen. Es enthält keine Daten des unbekannten Prüfkörpers."
        ),
    }, ensure_ascii=False, separators=(",", ":"))


def build_request_content(
    test_images: Sequence[Path],
    catalog: Catalog,
    references: Sequence[ReferenceImage],
    reference_memory: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Baut Bildinput ohne Protokolldaten, Probennummern oder Referenzpfade."""

    instructions = _analysis_instructions(catalog)
    content: list[dict[str, Any]] = [{
        "type": "input_text",
        "text": json.dumps(instructions, ensure_ascii=False, separators=(",", ":")),
    }]
    memory_text = _memory_text(reference_memory)
    if memory_text:
        content.append({"type": "input_text", "text": memory_text})
    for reference in references:
        label = f"Bestaetigtes Referenzbild fuer Klasse {reference.code}."
        if reference.description:
            label += f" Sichtbare Annotation: {reference.description}"
        content.extend((
            {"type": "input_text", "text": label},
            {"type": "input_image", "image_url": _data_url(reference.path), "detail": "high"},
        ))
    content.append({
        "type": "input_text",
        "text": "Ab hier folgen ausschliesslich ausgewaehlte Bilder des unbekannten Testpruefkoerpers.",
        # OpenAI kann damit den unveraenderten Praefix aus Regeln und
        # Variantenreferenzen wiederverwenden. Andere Provider ignorieren das
        # reine Transport-Metadatum; der fachliche Inhalt bleibt identisch.
        "prompt_cache_breakpoint": {"mode": "explicit"},
    })
    for index, path in enumerate(test_images, start=1):
        content.extend((
            {"type": "input_text", "text": f"Unbekanntes Testbild {index}."},
            {"type": "input_image", "image_url": _data_url(path), "detail": "high"},
        ))
    content.append({
        "type": "input_text",
        "text": (
            "Erzeuge jetzt ausschließlich die strukturierte Kurzform aller bereitgestellten "
            "Testbilder gemeinsam. Nur Stichpunktfragmente; alle Klassenvorschläge sind "
            "gleichberechtigte sichtbare Merkmale und keine Hauptversagensursachen."
        ),
    })
    return content


def response_schema(codes: Iterable[str]) -> dict[str, Any]:
    code_list = list(codes)
    short_text = {"type": "string", "minLength": 1, "maxLength": 120}
    suggestion = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "enum": code_list},
            "confidence": {"type": "string", "enum": ["hoch", "mittel", "unsicher"]},
            "evidence_status": {"type": "string", "enum": ["sicher_sichtbar", "vermutet"]},
        },
        "required": ["code", "confidence", "evidence_status"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["KLASSIFIZIERT", "UNKLAR", "KEINE_MERKMALE"]},
            "observations_short": {"type": "array", "items": short_text, "minItems": 1, "maxItems": 8},
            "visible_features": {"type": "array", "items": short_text, "maxItems": 8},
            "class_suggestions": {"type": "array", "items": suggestion, "maxItems": len(code_list)},
            "uncertain_points": {"type": "array", "items": short_text, "maxItems": 6},
        },
        "required": [
            "status", "observations_short", "visible_features", "class_suggestions", "uncertain_points",
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
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
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
        original_response_text = response.output_text
        raw = json.loads(original_response_text)
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
            "prompt_cache": {
                "mode": "explicit",
                "ttl": "30m",
                "breakpoint": "after_rules_and_variant_references_before_test_images",
            },
            "cost": {
                "currency": "USD",
                "reported_amount": None,
                "note": "Die Responses-API meldet Tokenverbrauch, aber keinen Rechnungsbetrag.",
            },
            "original_response_text": original_response_text,
        }
        return raw, metadata
    except (json.JSONDecodeError, AttributeError) as exc:
        response_text = getattr(locals().get("response"), "output_text", None)
        raise ModelOutputError(
            "OpenAI-Antwort enthaelt kein lesbares strukturiertes JSON.",
            diagnostics={
                "reason": "invalid_structured_json",
                "original_response_text": response_text,
            },
        ) from exc
    except ModelOutputError:
        raise
    except Exception as exc:
        raise safe_request_error(exc, model=model) from exc


def build_mistral_messages(
    test_images: Sequence[Path],
    catalog: Catalog,
    references: Sequence[ReferenceImage],
    reference_memory: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Überträgt den gemeinsamen OpenAI-Bildinhalt verlustfrei ins Mistral-Format."""

    image_count = len(test_images) + len(references)
    if image_count > MAX_MISTRAL_IMAGES_PER_REQUEST:
        raise AnalysisError(
            "Mistral-Anfrage enthaelt "
            f"{image_count} Bilder; erlaubt sind maximal {MAX_MISTRAL_IMAGES_PER_REQUEST}."
        )

    content = []
    for item in build_request_content(test_images, catalog, references, reference_memory):
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
    image_count = sum(
        1
        for message in messages
        for item in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(item, Mapping) and item.get("type") == "image_url"
    )
    if image_count > MAX_MISTRAL_IMAGES_PER_REQUEST:
        raise AnalysisError(
            "Mistral-Anfrage enthaelt "
            f"{image_count} Bilder; erlaubt sind maximal {MAX_MISTRAL_IMAGES_PER_REQUEST}."
        )
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
        original_response_text = _mistral_message_text(choice["message"]["content"])
        raw = json.loads(original_response_text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ModelOutputError(
            "Mistral-Antwort enthält kein lesbares strukturiertes JSON.",
            diagnostics={
                "reason": "invalid_structured_json",
                "original_response_text": locals().get("original_response_text"),
            },
        ) from exc
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
            "image_count": image_count,
            "max_images_per_request": MAX_MISTRAL_IMAGES_PER_REQUEST,
        },
        "original_response_text": original_response_text,
    }
    return raw, metadata


def build_ollama_messages(
    test_images: Sequence[Path],
    catalog: Catalog,
    references: Sequence[ReferenceImage],
    max_image_edge: int = DEFAULT_OLLAMA_MAX_IMAGE_EDGE,
    reference_memory: Mapping[str, Any] | None = None,
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
    memory_text = _memory_text(reference_memory)
    if memory_text:
        messages.append({"role": "user", "content": memory_text})
    for reference in references:
        label = f"Bestaetigtes Referenzbild fuer Klasse {reference.code}."
        if reference.description:
            label += f" Sichtbare Annotation: {reference.description}"
        messages.append({
            "role": "user",
            "content": label,
            "images": [_jpeg_base64(reference.path, max_long_edge=max_image_edge)],
        })
    messages.append({
        "role": "user",
        "content": "Ab hier folgen ausschliesslich ausgewaehlte Bilder des unbekannten Testpruefkoerpers.",
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
            "Erzeuge jetzt ausschliesslich die knappe gemeinsame JSON-Klassifikation aller "
            "bereitgestellten Testbilder. Beschreibe keine Referenzbilder. Bei erkannten "
            "Merkmalen liste alle Vorschläge vollständig und gleichberechtigt ohne Rangfolge auf."
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
    num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT,
    opener: Any = urlopen,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ruft ausschliesslich die lokale Ollama-Chat-API auf."""

    if endpoint.rstrip("/") != DEFAULT_OLLAMA_URL:
        raise AnalysisError("Ollama-Endpunkt muss aus Sicherheitsgruenden http://localhost:11434 sein.")
    if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx <= 0:
        raise AnalysisError("ollama_num_ctx muss eine positive Ganzzahl sein.")
    if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
        raise AnalysisError("ollama_num_predict muss eine positive Ganzzahl sein.")
    ollama_options = {
        "temperature": 0,
        "num_predict": num_predict,
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
        if isinstance(exc.reason, TimeoutError):
            raise AnalysisError(
                f"Zeitüberschreitung: Ollama hat innerhalb von {timeout_seconds} Sekunden "
                "nicht geantwortet. Prüfen Sie die Modellauslastung und versuchen Sie es erneut."
            ) from exc
        raise AnalysisError(
            "Lokales Ollama ist unter http://localhost:11434/api/chat nicht erreichbar: "
            f"{exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise AnalysisError(
            f"Zeitüberschreitung: Ollama hat innerhalb von {timeout_seconds} Sekunden "
            "nicht geantwortet. Prüfen Sie die Modellauslastung und versuchen Sie es erneut."
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Ollama-Antwort kann nicht gelesen werden: {exc}") from exc

    message = envelope.get("message")
    if not isinstance(message, Mapping):
        raise OllamaOutputError(
            "Technischer Fehlversuch: Ollama lieferte kein message-Objekt. "
            "Es wurde kein Klassifikationsergebnis gespeichert.",
            diagnostics={"reason": "missing_message", "think_requested": False},
        )
    content = message.get("content", "")
    thinking = message.get("thinking", "")
    content = content if isinstance(content, str) else ""
    thinking = thinking if isinstance(thinking, str) else ""
    diagnostics = {
        "reason": "output_parse_error",
        "think_requested": False,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "content_characters": len(content),
        "thinking_characters": len(thinking),
        "prompt_eval_count": envelope.get("prompt_eval_count"),
        "eval_count": envelope.get("eval_count"),
        "done_reason": envelope.get("done_reason"),
        "original_response_text": content,
        "original_thinking_text": thinking,
    }
    if envelope.get("done_reason") == "length":
        diagnostics["reason"] = "thinking_only_output_limit" if not content.strip() and thinking else "output_limit"
        detail = (
            "ausschließlich internes Thinking und keine Klassifikationsantwort erzeugt"
            if not content.strip() and thinking else
            "das Ausgabelimit vor einer sicher auswertbaren Klassifikationsantwort erreicht"
        )
        raise OllamaOutputError(
            f"Technischer Fehlversuch: {model} hat {detail}. "
            f"Thinking war angefordert: nein; Ausgabelimit: {num_predict}; "
            f"Thinking-Zeichen: {len(thinking)}; prompt_eval_count: "
            f"{envelope.get('prompt_eval_count')}; eval_count: {envelope.get('eval_count')}; "
            "done_reason: length. Es wurde kein Klassifikationsergebnis gespeichert.",
            diagnostics=diagnostics,
        )
    try:
        raw = _parse_ollama_json(content)
    except (KeyError, TypeError) as exc:
        raise AnalysisError("Ollama-Antwort enthaelt kein lesbares strukturiertes JSON.") from exc
    except AnalysisError as exc:
        tail = re.sub(r"\s+", " ", str(content)).strip()[-300:]
        diagnostics["reason"] = "thinking_without_content" if not content.strip() and thinking else "invalid_content"
        raise OllamaOutputError(
            f"Technischer Fehlversuch: {exc} [Zeichen: {len(content)}, "
            f"Thinking-Zeichen: {len(thinking)}, Thinking angefordert: nein, "
            f"prompt_eval_count: {envelope.get('prompt_eval_count')}, "
            f"eval_count: {envelope.get('eval_count')}, "
            f"done_reason: {envelope.get('done_reason')}, Ende: {tail}]. "
            "Es wurde kein Klassifikationsergebnis gespeichert.",
            diagnostics=diagnostics,
        ) from exc
    metadata = {
        key: envelope[key]
        for key in (
            "created_at", "done_reason", "total_duration", "load_duration",
            "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
        )
        if key in envelope
    }
    metadata["response_model"] = envelope.get("model", model)
    metadata["request_options"] = dict(ollama_options)
    metadata["request_options"]["image_count"] = sum(
        len(message.get("images", []))
        for message in messages
        if isinstance(message, Mapping) and isinstance(message.get("images"), list)
    )
    metadata["think"] = False
    metadata["stream"] = False
    metadata["original_response_text"] = content
    return raw, metadata


def _validated_evidence(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise AnalysisError(f"Modellantwort: {field} muss eine Liste nichtleerer Texte sein.")
    for text in value:
        if CAUSAL_CLAIM.search(text):
            raise AnalysisError(f"Modellantwort: unzulaessige Ursachenaussage in {field}.")
    return [item.strip() for item in value]


def _validated_short_entries(value: Any, *, field: str) -> list[str]:
    entries = _validated_evidence(value, field=field)
    if any(len(entry) > 120 for entry in entries):
        raise AnalysisError(f"Modellantwort: {field} enthält einen Eintrag mit mehr als 120 Zeichen.")
    return list(dict.fromkeys(entries))


def validate_and_enrich(raw: Mapping[str, Any], catalog: Catalog) -> dict[str, Any]:
    """Validiert Codes, Multi-Label-Struktur und das Verbot kausaler Aussagen."""

    status = raw.get("status")
    if status not in {"KLASSIFIZIERT", "UNKLAR", "KEINE_MERKMALE"}:
        raise AnalysisError("Modellantwort: ungueltiger Status.")
    if "class_suggestions" in raw:
        observations = _validated_short_entries(
            raw.get("observations_short"), field="observations_short"
        )
        visible_features = _validated_short_entries(
            raw.get("visible_features"), field="visible_features"
        )
        uncertain_points = _validated_short_entries(
            raw.get("uncertain_points"), field="uncertain_points"
        )
        if not observations and not visible_features and not uncertain_points:
            raise AnalysisError(
                "Modellantwort: vollständig leere Sichtauswertung ist kein fachliches Ergebnis."
            )
        suggestions_raw = raw.get("class_suggestions")
        if not isinstance(suggestions_raw, list):
            raise AnalysisError("Modellantwort: class_suggestions muss eine Liste sein.")
        if status in {"UNKLAR", "KEINE_MERKMALE"} and suggestions_raw:
            raise AnalysisError(f"{status} darf keine Klassenvorschläge enthalten.")
        if status == "KLASSIFIZIERT" and not suggestions_raw:
            raise AnalysisError("KLASSIFIZIERT benötigt mindestens einen Klassenvorschlag.")
        if status == "UNKLAR" and not uncertain_points:
            raise AnalysisError("UNKLAR benötigt mindestens einen konkreten Unsicherheitspunkt.")
        confidence_values = {"hoch": 0.85, "mittel": 0.6, "unsicher": 0.35}
        suggestions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, suggestion in enumerate(suggestions_raw, start=1):
            if not isinstance(suggestion, Mapping):
                raise AnalysisError("Modellantwort: ungültiger Klassenvorschlag.")
            code = suggestion.get("code")
            if code not in catalog.definitions:
                raise AnalysisError(f"Modellantwort: unbekannter Klassencode {code!r}.")
            if code in seen:
                raise AnalysisError(f"Modellantwort: Klasse {code!r} wurde mehrfach ausgegeben.")
            seen.add(code)
            confidence_label = suggestion.get("confidence")
            if confidence_label not in confidence_values:
                raise AnalysisError(
                    f"Modellantwort: Sicherheit für {code!r} muss hoch, mittel oder unsicher sein."
                )
            evidence_status = suggestion.get("evidence_status")
            if evidence_status not in {"sicher_sichtbar", "vermutet"}:
                raise AnalysisError(
                    f"Modellantwort: Evidenzstatus für {code!r} muss sicher_sichtbar oder vermutet sein."
                )
            definition = catalog.definitions[code]
            suggestions.append({
                "suggestion_id": f"suggestion-{index:02d}",
                "code": code,
                "name": definition["name"],
                "confidence": confidence_values[confidence_label],
                "confidence_label": confidence_label,
                "evidence_status": evidence_status,
                "proven_failure_cause": False,
            })
        return {
            "status": status,
            "analysis_outcome": (
                "merkmale_erkannt" if status == "KLASSIFIZIERT" else
                "keine_merkmale_erkennbar" if status == "KEINE_MERKMALE" else
                "fachlich_unklar"
            ),
            "observations_short": observations,
            "visible_features": visible_features,
            "class_suggestions": suggestions,
            "suggested_feature_codes": [item["code"] for item in suggestions],
            "uncertain_points": uncertain_points,
        }
    if "detected_features" in raw:
        features_raw = raw.get("detected_features")
        if not isinstance(features_raw, list):
            raise AnalysisError("Modellantwort: detected_features muss eine Liste sein.")
        if status == "UNKLAR" and features_raw:
            raise AnalysisError("UNKLAR darf keine erkannten Merkmale enthalten.")
        if status == "KLASSIFIZIERT" and not features_raw:
            raise AnalysisError("KLASSIFIZIERT benötigt mindestens ein erkanntes Merkmal.")
        features: list[dict[str, Any]] = []
        for index, feature in enumerate(features_raw, start=1):
            if not isinstance(feature, Mapping):
                raise AnalysisError("Modellantwort: ungültiger Merkmalsvorschlag.")
            code = feature.get("code")
            if code not in catalog.definitions:
                raise AnalysisError(f"Modellantwort: unbekannter Klassencode {code!r}.")
            visible_description = str(feature.get("visible_description", "")).strip()
            approximate_location = str(feature.get("approximate_location", "")).strip()
            if not visible_description or not approximate_location:
                raise AnalysisError("Jedes Merkmal benötigt sichtbare Beschreibung und ungefähre Lage.")
            relation = feature.get("fracture_zone_relation")
            if relation not in {"inside", "outside", "overlapping", "unclear"}:
                raise AnalysisError("Ungültige Beziehung zur Bruchzone.")
            reasoning = _validated_evidence(
                feature.get("image_based_reasoning"), field=f"image_based_reasoning[{index}]"
            )
            images = _validated_evidence(feature.get("image_evidence"), field=f"image_evidence[{index}]")
            uncertainties = _validated_evidence(
                feature.get("uncertainty_reasons"), field=f"feature_uncertainty_reasons[{index}]"
            )
            if not reasoning or not images:
                raise AnalysisError("Jedes Merkmal benötigt Bildbegründung und Bildbelege.")
            enriched = _enrich_feature(code, reasoning, feature.get("confidence"), catalog)
            enriched.update({
                "feature_id": f"feature-{index:02d}",
                "visible_description": visible_description,
                "image_evidence": images,
                "approximate_location": approximate_location,
                "fracture_zone_relation": relation,
                "uncertainty_reasons": uncertainties,
            })
            features.append(enriched)
        list_fields = {
            field: _validated_evidence(raw.get(field), field=field)
            for field in ("visible_observations", "uncertainty_reasons", "missing_views")
        }
        return {"status": status, "detected_features": features, **list_fields}

    # Historisches Antwortformat: ausschließlich für vorhandene Dateien und
    # ältere Offline-Fixtures lesbar halten. Neue Provider-Schemata erzeugen es nicht.
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


def _merge_batch_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine validated batch classifications without discarding any batch evidence."""

    def combine(field: str, limit: int) -> list[str]:
        return list(dict.fromkeys(
            text for result in results for text in result.get(field, [])
        ))[:limit]

    if all("class_suggestions" in result for result in results):
        rank = {"hoch": 3, "mittel": 2, "unsicher": 1}
        evidence_rank = {"sicher_sichtbar": 2, "vermutet": 1}
        suggestions_by_code: dict[str, dict[str, Any]] = {}
        for result in results:
            for suggestion in result.get("class_suggestions", []):
                code = str(suggestion.get("code", ""))
                current = suggestions_by_code.get(code)
                candidate_rank = (
                    evidence_rank.get(str(suggestion.get("evidence_status")), 0),
                    rank.get(str(suggestion.get("confidence_label")), 0),
                )
                current_rank = (
                    evidence_rank.get(str((current or {}).get("evidence_status")), 0),
                    rank.get(str((current or {}).get("confidence_label")), 0),
                )
                if current is None or candidate_rank > current_rank:
                    suggestions_by_code[code] = dict(suggestion)
        suggestions = list(suggestions_by_code.values())[:12]
        for index, suggestion in enumerate(suggestions, start=1):
            suggestion["suggestion_id"] = f"suggestion-{index:02d}"
        return {
            "status": "KLASSIFIZIERT" if suggestions else "UNKLAR",
            "observations_short": combine("observations_short", 8),
            "visible_features": combine("visible_features", 8),
            "class_suggestions": suggestions,
            "uncertain_points": combine("uncertain_points", 6),
        }

    if all("detected_features" in result for result in results):
        merged: list[dict[str, Any]] = []
        for batch_index, result in enumerate(results, start=1):
            for feature in result.get("detected_features", []):
                copied = dict(feature)
                copied["feature_id"] = f"feature-{len(merged) + 1:02d}"
                copied["source_batch"] = batch_index
                merged.append(copied)

        return {
            "status": "KLASSIFIZIERT" if merged else "UNKLAR",
            "detected_features": merged[:12],
            "visible_observations": combine("visible_observations", 8),
            "uncertainty_reasons": combine("uncertainty_reasons", 6),
            "missing_views": combine("missing_views", 6),
        }

    primary_candidates = [
        result["primary_feature"]
        for result in results
        if result.get("primary_feature") is not None
    ]
    primary = max(primary_candidates, key=lambda item: item["confidence"]) if primary_candidates else None
    candidates = [
        feature
        for result in results
        for feature in (
            ([result["primary_feature"]] if result.get("primary_feature") is not None else [])
            + result.get("secondary_features", [])
        )
        if primary is None or feature["code"] != primary["code"]
    ]
    best_by_code: dict[str, dict[str, Any]] = {}
    for feature in candidates:
        existing = best_by_code.get(feature["code"])
        if existing is None:
            best_by_code[feature["code"]] = dict(feature)
            continue
        best = feature if feature["confidence"] > existing["confidence"] else existing
        evidence = list(dict.fromkeys(
            existing["image_based_reasoning"] + feature["image_based_reasoning"]
        ))
        best_by_code[feature["code"]] = {**best, "image_based_reasoning": evidence[:3]}
    secondary = sorted(
        best_by_code.values(), key=lambda item: item["confidence"], reverse=True
    )[:4]

    def combine(field: str, limit: int) -> list[str]:
        return list(dict.fromkeys(
            text for result in results for text in result.get(field, [])
        ))[:limit]

    return {
        "status": "KLASSIFIZIERT" if primary else "UNKLAR",
        "primary_feature": primary,
        "secondary_features": secondary,
        "visible_observations": combine("visible_observations", 6),
        "uncertainty_reasons": combine("uncertainty_reasons", 4),
        "missing_views": combine("missing_views", 4),
    }


def merge_partial_classifications(
    results: Sequence[Mapping[str, Any]], *, complete: bool
) -> dict[str, Any]:
    """Merge persisted subrequest results and retain contradictory evidence."""

    if not results:
        raise AnalysisError("Mindestens ein Teilresultat ist für die Zusammenführung erforderlich.")
    merged = _merge_batch_results(results)
    observations_by_code: dict[str, list[dict[str, Any]]] = {}
    for part_index, result in enumerate(results, start=1):
        for suggestion in result.get("class_suggestions", []):
            if not isinstance(suggestion, Mapping) or not suggestion.get("code"):
                continue
            observations_by_code.setdefault(str(suggestion["code"]), []).append({
                "partial_request": part_index,
                "confidence": suggestion.get("confidence"),
                "confidence_label": suggestion.get("confidence_label"),
                "evidence_status": suggestion.get("evidence_status"),
            })
    conflicts: list[dict[str, Any]] = []
    statuses = [str(result.get("status", "")) for result in results]
    if len(set(statuses)) > 1:
        conflicts.append({"type": "status_disagreement", "partial_statuses": statuses})
    for code, observations in observations_by_code.items():
        states = {
            (item.get("confidence_label"), item.get("evidence_status"))
            for item in observations
        }
        if len(states) > 1:
            conflicts.append({
                "type": "suggestion_disagreement", "code": code,
                "observations": observations,
            })
    merged["merge_status"] = "vollstaendig" if complete else "teilweise"
    merged["partial_request_count"] = len(results)
    merged["partial_evidence_by_code"] = observations_by_code
    merged["merge_conflicts"] = conflicts
    return merged


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
    excluded_image_names: Sequence[str] = (),
    run_id: str | None = None,
    project_id: str | None = None,
    prompt_hash: str | None = None,
    reference_records: Sequence[Mapping[str, Any]] = (),
    settings: Mapping[str, Any] | None = None,
    evaluation_variant: str | None = None,
    test_series: Mapping[str, Any] | None = None,
    model_output: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_run_id = run_id or uuid.uuid4().hex
    return {
        "schema_version": (
            "4.2" if "class_suggestions" in validated_result else
            "3.0" if "detected_features" in validated_result else "2.0"
        ),
        "run_id": resolved_run_id,
        "project_id": project_id,
        "specimen_id": input_dir.resolve().name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": input_dir.resolve().name,
        "test_series": dict(test_series) if test_series else None,
        "execution": {
            "mode": mode,
            "provider": provider,
            "model": model,
            "requested_model": model,
            "actual_model": (provider_metadata or {}).get("response_model") or model,
            "model_verification": model_verification,
            "duration_seconds": round(duration_seconds, 3),
            "provider_metadata": dict(provider_metadata or {}),
            "settings": dict(settings or {}),
            "paid_api_request_performed": provider in {"openai", "mistral"} and mode == "live",
            "images_transmitted_to_openai": provider == "openai" and mode == "live",
            "images_transmitted_to_mistral": provider == "mistral" and mode == "live",
            "images_transmitted_to_cloud": provider in {"openai", "mistral"} and mode == "live",
        },
        "inputs": [
            {"filename": path.name, "sha256": _sha256(path)}
            for path in test_images
        ],
        "image_selection": {
            "included": [path.name for path in test_images],
            "excluded": list(excluded_image_names),
        },
        "classification_basis": {
            "prompt_version": PROMPT_VERSION,
            "prompt_hash_sha256": prompt_hash,
            "evaluation_variant": evaluation_variant,
            "allowed_class_count": len(catalog.active_class_codes or catalog.codes),
            "allowed_class_codes": list(catalog.active_class_codes or catalog.codes),
            "active_reference_feature_codes": list(catalog.active_class_codes),
            "additional_visible_feature_codes": list(catalog.additional_visible_codes),
            "classification_rules_version": catalog.rule_version,
            "active_class_set_version": f"klassifikationsregeln/{catalog.rule_version}",
            "active_class_descriptions_sha256": hashlib.sha256(
                json.dumps(
                    _prompt_catalog(catalog),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "reference_database_accessed": catalog.reference_database_accessed,
            "confirmed_main_reference_count_available": len(catalog.references),
            "confirmed_main_reference_count_used": used_reference_count,
            "protocol_failure_cause_transmitted": False,
            "ground_truth_transmitted": False,
            "previous_results_transmitted": False,
            "expert_assessments_transmitted": False,
            "only_confirmed_main_references_used": True,
            "references": [dict(item) for item in reference_records],
        },
        "classes_without_confirmed_references": [
            {"code": code, "name": catalog.definitions[code]["name"]}
            for code in catalog.codes_without_references
        ],
        "result": validated_result,
        "model_output": dict(model_output or {}),
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
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(document.get("run_id", uuid.uuid4().hex)))
    destination = output_dir / f"{analysis_id}_{provider}_{model}_{timestamp}_{run_id}.json"
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
    provider: str = "ollama",
    model: str | None = None,
    mock_response_path: Path = DEFAULT_MOCK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_references_per_class: int = DEFAULT_REFERENCES_PER_CLASS,
    ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
    ollama_num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT,
    ollama_max_image_edge: int = DEFAULT_OLLAMA_MAX_IMAGE_EDGE,
    allowed_input_root: Path = ALLOWED_INPUT_ROOT,
    selected_image_names: Sequence[str] | None = None,
    client: Any = None,
    ollama_opener: Any = urlopen,
    mistral_opener: Any = urlopen,
    mistral_key_resolver: Any = resolve_mistral_api_key,
    progress_callback: Callable[[str], None] | None = None,
    run_id: str | None = None,
    project_id: str | None = None,
    evaluation_variant: str | None = None,
    allowed_class_codes: Sequence[str] | None = None,
    test_series: Mapping[str, Any] | None = None,
    evaluation_memory_path: Path | None = None,
    evaluation_variants_dir: Path = EVALUATION_VARIANTS_DIR,
) -> Path:
    started_at = time.perf_counter()
    if evaluation_variant not in {None, *VARIANT_IDS}:
        raise AnalysisError("Evaluationsvariante muss 1, 2A, 2B oder 3 sein.")
    documented_catalog = load_catalog(include_references=False)
    documented_codes = ACTIVE_CLASS_CODES
    missing_definitions = [code for code in documented_codes if code not in documented_catalog.definitions]
    if missing_definitions:
        raise AnalysisError("Merkmalsdefinitionen fehlen: " + ", ".join(missing_definitions))
    requested_codes = tuple(allowed_class_codes or documented_codes)
    if not documented_codes or requested_codes != documented_codes:
        raise AnalysisError(
            "Neue Analysen verwenden ausschließlich die vollständig in "
            "Merkmalskatalog dokumentierten aktiven Merkmale: "
            + ", ".join(documented_codes)
        )
    reference_memory: dict[str, Any] | None = None
    direct_references: tuple[ReferenceImage, ...] = ()
    variant_report: dict[str, Any] | None = None
    if evaluation_variant in VARIANT_IDS:
        variant_report = inspect_variant(evaluation_variants_dir, evaluation_variant)
        if not variant_report["ready"]:
            raise AnalysisError(f"Evaluationsvariante {evaluation_variant} ist NICHT BEREIT.")
    if evaluation_variant == "1":
        catalog = load_catalog(include_references=False, allowed_codes=documented_codes)
        max_references_per_class = 0
    elif evaluation_variant in {"2A", "2B", "3"}:
        catalog = load_catalog(include_references=False, allowed_codes=documented_codes)
        max_references_per_class = 0
        if evaluation_memory_path is not None:
            raise AnalysisError(f"Variante {evaluation_variant} verwendet kein Referenzgedächtnis.")
        direct_references = direct_variant_references(variant_report)
    else:
        catalog = load_catalog(allowed_codes=documented_codes)
    test_images = collect_test_images(
        input_dir,
        catalog,
        allowed_input_root=allowed_input_root,
        selected_image_names=selected_image_names,
    )
    available_image_names = {
        path.name
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.casefold().startswith("nummer_")
    }
    excluded_image_names = sorted(available_image_names - {path.name for path in test_images})
    test_image_hashes = {_sha256(path) for path in test_images}

    if direct_references:
        specimen_id = input_dir.resolve().name
        same_specimen = [
            reference.code for reference in direct_references
            if _reference_specimen_id(reference) == specimen_id
        ]
        duplicate_images = [
            reference.code for reference in direct_references
            if _sha256(reference.path) in test_image_hashes
        ]
        if same_specimen or duplicate_images:
            details = []
            if same_specimen:
                details.append("gleicher Prüfkörper: " + ", ".join(same_specimen))
            if duplicate_images:
                details.append("bytegleiches Testbild: " + ", ".join(duplicate_images))
            raise AnalysisError(
                f"Variante {evaluation_variant} wurde wegen möglicher Datenleckage gesperrt ("
                + "; ".join(details) + ")."
            )

    def eligible_references() -> tuple[ReferenceImage, ...]:
        if direct_references:
            return direct_references
        return select_references(
            catalog,
            max_references_per_class,
            excluded_specimen_id=input_dir.resolve().name,
            excluded_hashes=test_image_hashes,
        )
    if provider not in {"openai", "ollama", "mistral"}:
        raise AnalysisError("Provider muss 'openai', 'ollama' oder 'mistral' sein.")
    default_models = {
        "ollama": os.getenv("USECASE2_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        "openai": os.getenv("USECASE2_OPENAI_MODEL", "gpt-6.1-sol"),
        "mistral": DEFAULT_MISTRAL_MODEL,
    }
    default_model = default_models[provider]
    selected_model = (model or default_model).strip()
    if not selected_model:
        raise AnalysisError(f"Keine Modellkennung fuer Provider {provider!r} konfiguriert.")
    if evaluation_variant == "3" and (provider != "openai" or selected_model != "gpt-6.1-sol"):
        raise AnalysisError(
            "Variante 3 ist wissenschaftlich ausschließlich für OpenAI mit "
            "dem Modell gpt-6.1-sol freigegeben."
        )
    if (
        bool(direct_references)
        and provider == "mistral"
        and len(direct_references) + 1 > MAX_MISTRAL_IMAGES_PER_REQUEST
    ):
        raise AnalysisError(
            f"Variante {evaluation_variant} ist mit Mistral derzeit nicht startbereit: Die vorhandene "
            f"Provider-Implementierung erlaubt höchstens {MAX_MISTRAL_IMAGES_PER_REQUEST} "
            f"Bilder pro Anfrage, benötigt werden {len(direct_references)} Referenzbilder "
            "plus mindestens ein Testbild."
        )

    def report_progress(status: str) -> None:
        if progress_callback is not None:
            progress_callback(status)

    report_progress("Modell wird geladen")

    provider_metadata: dict[str, Any] = {}
    validated_result: dict[str, Any] | None = None
    if provider == "ollama":
        verification = None
        selected_references = eligible_references()
        messages = build_ollama_messages(
            test_images,
            catalog,
            selected_references,
            max_image_edge=ollama_max_image_edge,
            reference_memory=reference_memory,
        )
        report_progress("Bilder werden analysiert")
        raw, provider_metadata = call_ollama(
            model=selected_model,
            messages=messages,
            catalog=catalog,
            num_ctx=ollama_num_ctx,
            num_predict=ollama_num_predict,
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
        selected_references = limit_mistral_references(test_images, eligible_references())
        resolution = mistral_key_resolver()
        batch_size = MAX_MISTRAL_IMAGES_PER_REQUEST - len(selected_references)
        result_batches: list[dict[str, Any]] = []
        batch_metadata: list[dict[str, Any]] = []
        for offset in range(0, len(test_images), batch_size):
            batch = test_images[offset:offset + batch_size]
            raw, metadata = call_mistral(
                model=selected_model,
                messages=build_mistral_messages(batch, catalog, selected_references, reference_memory),
                catalog=catalog,
                api_key=resolution.value,
                opener=mistral_opener,
            )
            validated_batch = validate_and_enrich(raw, catalog)
            result_batches.append(validated_batch)
            batch_metadata.append({
                "batch": len(batch_metadata) + 1,
                "images": [path.name for path in batch],
                "detected_codes": [
                    feature["code"] for feature in validated_batch.get("detected_features", [])
                ],
                "provider_metadata": metadata,
            })
        validated_result = _merge_batch_results(result_batches)
        provider_metadata = {
            **(batch_metadata[0]["provider_metadata"] if len(batch_metadata) == 1 else {}),
            "multi_stage": len(batch_metadata) > 1,
            "image_batches": batch_metadata,
        }
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
        selected_references = eligible_references()
        raw, provider_metadata = call_openai_with_metadata(
            model=selected_model,
            content=build_request_content(test_images, catalog, selected_references, reference_memory),
            catalog=catalog,
            client=client,
        )
        mode = "live"
        result_provider = "openai"
    elif provider == "mistral":
        verification = load_verified_mistral_model(selected_model)
        selected_references = limit_mistral_references(test_images, eligible_references())
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
        selected_references = eligible_references() if direct_references else ()
        raw = _read_json(mock_response_path)
        mode = "mock"
        result_provider = "mock"

    report_progress("Ergebnis wird verarbeitet")
    original_response_text = provider_metadata.pop("original_response_text", None)
    try:
        validated = validated_result or validate_and_enrich(raw, catalog)
    except AnalysisError as exc:
        error_class = OllamaOutputError if provider == "ollama" else ModelOutputError
        provider_name = "Qwen" if provider == "ollama" else provider.capitalize()
        raise error_class(
            f"Technischer Fehlversuch: Die {provider_name}-Antwort war lesbares JSON, "
            f"entsprach aber nicht dem vereinbarten Schema 4.2 ({exc}). Es wurde kein "
            "fachliches UNKLAR-Ergebnis gespeichert.",
            diagnostics={
                "reason": "schema_validation_error",
                "think_requested": False if provider == "ollama" else None,
                "original_response_text": original_response_text,
                "validation_error": str(exc),
            },
        ) from exc
    duration_seconds = time.perf_counter() - started_at
    reference_records = [
        {
            "id": f"{reference.code}:{reference.path.name}",
            "code": reference.code,
            "filename": reference.path.name,
            "specimen_id": _reference_specimen_id(reference),
            "description": reference.description,
            "path": str(reference.path.resolve()),
            "sha256": _sha256(reference.path),
        }
        for reference in selected_references
    ]
    prompt_manifest = {
        "prompt_version": PROMPT_VERSION,
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "analysis_instructions": _analysis_instructions(catalog),
        "response_schema": response_schema(catalog.codes),
        "test_images": [
            {"filename": path.name, "sha256": _sha256(path)} for path in test_images
        ],
        "references": reference_records,
        "reference_memory_key": reference_memory.get("memory_key") if reference_memory else None,
        "reference_delivery_mode": (
            "direct_images_per_analysis" if direct_references
            else "reference_memory" if reference_memory else "none"
        ),
    }
    prompt_hash = hashlib.sha256(
        json.dumps(
            prompt_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    settings = {
        "evaluation_variant": evaluation_variant,
        "reference_memory_key": reference_memory.get("memory_key") if reference_memory else None,
        "reference_memory_path": str(evaluation_memory_path) if evaluation_memory_path else None,
        "reference_delivery_mode": (
            "direct_images_per_analysis" if direct_references
            else "reference_memory" if reference_memory else "none"
        ),
        "evaluation_manifest_hash": reference_memory.get("manifest_hash") if reference_memory else (
            variant_report.get("manifestHash") if evaluation_variant in VARIANT_IDS else None
        ),
        "reference_batches": reference_memory.get("reference_batches", []) if reference_memory else [],
        "reference_preparation_provider": reference_memory.get("provider") if reference_memory else None,
        "reference_preparation_model": reference_memory.get("model") if reference_memory else None,
        "reference_preparation_prompt_version": reference_memory.get("prompt_version") if reference_memory else None,
        "reference_descriptions_csv_sha256": reference_memory.get("descriptions_csv_sha256") if reference_memory else None,
        "general_class_descriptions_source": reference_memory.get("general_class_descriptions_source") if reference_memory else None,
        "max_references_per_class": max_references_per_class,
        "ollama_num_ctx": ollama_num_ctx if provider == "ollama" else None,
        "ollama_num_predict": ollama_num_predict if provider == "ollama" else None,
        "ollama_max_image_edge": ollama_max_image_edge if provider == "ollama" else None,
        "ollama_timeout_seconds": (
            DEFAULT_OLLAMA_TIMEOUT_SECONDS if provider == "ollama" else None
        ),
        "ollama_think": False if provider == "ollama" else None,
        "ollama_stream": False if provider == "ollama" else None,
        "ollama_temperature": 0 if provider == "ollama" else None,
    }
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
        excluded_image_names=excluded_image_names,
        run_id=run_id,
        project_id=project_id,
        prompt_hash=prompt_hash,
        reference_records=reference_records,
        settings=settings,
        evaluation_variant=evaluation_variant,
        test_series=test_series,
        model_output={
            "original_response_text": original_response_text,
            "parsed_response": raw,
            "validation_status": "valid",
        } if original_response_text is not None else {},
    )
    return save_result(document, output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Label-Bildanalyse von Holz-Zugproben (standardmaessig kostenfreier Mock-Modus)."
    )
    parser.add_argument("input_dir", type=Path, help="Ordner mit einem oder mehreren Fotos")
    parser.add_argument(
        "--provider",
        choices=("openai", "ollama", "mistral"),
        default="ollama",
        help="Analyseanbieter (openai und mistral bleiben ohne --live im Mock-Modus)",
    )
    parser.add_argument(
        "--model",
        help="Modellkennung, z. B. gpt-6.1-sol, ministral-3b-2512 oder qwen3-vl:4b-instruct",
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
        "--evaluation-variant",
        choices=VARIANT_IDS,
        help="Wissenschaftliche Evaluationsvariante 1, 2A, 2B oder 3",
    )
    parser.add_argument("--prepare-reference-memory", action="store_true",
                        help="Referenzgedächtnis vorbereiten (mit --dry-run nur planen)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur Struktur und Aufrufplan prüfen; kein Provider-Aufruf")
    parser.add_argument("--test-series-id", help="Eindeutige ID der wissenschaftlichen Testreihe")
    parser.add_argument("--evaluation-memory", type=Path,
                        help="Vorbereitetes Referenzgedächtnis für Variante 3")
    parser.add_argument(
        "--allowed-class-code",
        dest="allowed_class_codes",
        action="append",
        help="Optionaler Klassencode; die vollständige dokumentierte Regelliste muss übereinstimmen",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=DEFAULT_OLLAMA_NUM_CTX,
        metavar="TOKENS",
        help="Ollama-Kontextgroesse in Tokens (Standard: 24576)",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=DEFAULT_OLLAMA_NUM_PREDICT,
        help="Maximales lokales Generationsbudget (Standard: 4096)",
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
        if args.dry_run:
            if not args.evaluation_variant:
                raise AnalysisError("--dry-run erfordert --evaluation-variant.")
            report = inspect_variant(EVALUATION_VARIANTS_DIR, args.evaluation_variant)
            selected_model = args.model or {
                "ollama": DEFAULT_OLLAMA_MODEL, "openai": "gpt-6.1-sol",
                "mistral": DEFAULT_MISTRAL_MODEL,
            }[args.provider]
            report["memory"] = {
                "status": "NICHT ERFORDERLICH", "prepared": True,
            }
            report["dryRun"] = True
            report["providerCalled"] = False
            report["testSeriesId"] = args.test_series_id
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready"] else 3
        if args.prepare_reference_memory:
            raise AnalysisError(
                "Die Vorbereitung führt Modellaufrufe aus und muss in der App nach Prüfung der "
                "Kontrollübersicht ausdrücklich gestartet werden. Verwenden Sie hier zunächst --dry-run."
            )
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
            ollama_num_predict=args.ollama_num_predict,
            ollama_max_image_edge=args.ollama_max_image_edge,
            evaluation_variant=args.evaluation_variant,
            allowed_class_codes=args.allowed_class_codes,
            test_series={"series_id": args.test_series_id} if args.test_series_id else None,
            evaluation_memory_path=args.evaluation_memory,
        )
    except (AnalysisError, OpenAIConfigurationError, RuntimeError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    print(f"Ergebnis gespeichert: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
