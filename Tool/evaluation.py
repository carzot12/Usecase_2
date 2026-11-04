#!/usr/bin/env python3
"""Reproduzierbare Evaluation der Bildklassifikation ohne Ground-Truth-Leakage.

Der Standardaufruf erstellt nur einen Laufplan. Modellaufrufe werden erst mit
``--execute`` aktiviert; Cloudanbieter benoetigen zusaetzlich die beiden bereits
bekannten Live-Schutzschalter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ki_bildanalyse import (
    DEFAULT_OLLAMA_MAX_IMAGE_EDGE,
    MAX_MISTRAL_IMAGES_PER_REQUEST,
    DEFAULT_OLLAMA_NUM_CTX,
    PROJECT_ROOT,
    SYSTEM_INSTRUCTIONS,
    AnalysisError,
    Catalog,
    _data_url,
    _jpeg_base64,
    _prompt_catalog,
    _read_json,
    _sha256,
    _verify_image,
    call_mistral,
    call_ollama,
    call_openai_with_metadata,
    load_catalog,
    load_verified_mistral_model,
    load_verified_model,
    validate_and_enrich,
)
from .mistral_config import resolve_mistral_api_key
from .openai_config import OpenAIConfigurationError, create_openai_client, resolve_api_key


EVALUATION_DIR = PROJECT_ROOT / "Evaluation"
GROUND_TRUTH_PATH = EVALUATION_DIR / "ground_truth_manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output" / "KI_Evaluation"
VARIANTS = ("A", "B", "C", "D")
VARIANT_DESCRIPTIONS = {
    "A": "Nur Testbilder und erlaubte Klassencodes; keine Beschreibungen, keine Referenzen.",
    "B": "Testbilder und Klassenbeschreibungen; keine Referenzbilder.",
    "C": "Testbilder und bestaetigte Referenzbilder; keine Klassenbeschreibungen.",
    "D": "Zwei getrennte Modellaufrufe: neutrale Merkmalsextraktion, dann regelgestuetzte Klassifikation.",
}
CONFIRMED_VISIBILITY = "SICHTBAR_BESTAETIGT"
UNSCORED_AUXILIARY_CODES = frozenset({"KB"})
RESULT_FIELDS = (
    "case_id", "specimen_id", "provider", "model", "model_version",
    "variant", "active_class_set_version", "active_class_descriptions_sha256",
    "expected_primary_code", "predicted_primary_code", "evaluation_predicted_code", "outcome",
    "expected_codes", "predicted_feature_codes", "matched_codes", "missed_codes", "additional_codes",
    "secure_visible_feature_codes", "suspected_feature_codes", "undocumented_evidence_feature_codes",
    "unscored_feature_codes", "secure_evaluation_predicted_code", "secure_outcome",
    "secondary_features", "reasoning", "visible_observations", "confidence",
    "duration_seconds", "input_tokens", "output_tokens", "total_tokens",
    "reported_cost", "cost_currency", "reference_count", "reference_specimens",
    "error_type", "error_message",
)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    specimen_id: str
    expected_code: str
    image_paths: tuple[Path, ...]
    image_sha256: tuple[str, ...]
    confirmation_source: str


@dataclass(frozen=True)
class ConfirmedReference:
    code: str
    specimen_id: str
    path: Path
    sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_confirmed_reference(item: Mapping[str, Any]) -> bool:
    reference = item.get("reference_image")
    return bool(
        isinstance(reference, Mapping)
        and item.get("dataset_role") == "reference"
        and item.get("excluded_from_independent_test") is True
        and reference.get("image_role") == "reference_image"
        and reference.get("confirmed") is True
        and reference.get("default_for_inference") is True
        and reference.get("visibility") == CONFIRMED_VISIBILITY
        and str(reference.get("confirmed_by", "")).strip()
        and str(reference.get("confirmed_at", "")).strip()
        and str(item.get("confirmation_source", "")).strip()
    )


def build_ground_truth_document(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Leitet Ground Truth ausschliesslich aus bestaetigten Hauptreferenzen ab."""

    source = project_root / "Datenbank" / "Referenzbilder" / "referenzbilder_manifest.json"
    catalog = load_catalog(project_root)
    manifest = _read_json(source)
    cases: list[dict[str, Any]] = []
    seen_specimens: dict[str, str] = {}
    for item in manifest.get("reference_sets", []):
        if not isinstance(item, Mapping) or not _is_confirmed_reference(item):
            continue
        specimen = str(item.get("specimen", "")).strip()
        code = str(item.get("failure_mode", "")).strip()
        if not specimen or code not in catalog.definitions:
            raise AnalysisError("Bestaetigte Referenz ohne gueltigen Pruefkoerper oder Klassencode.")
        if specimen in seen_specimens and seen_specimens[specimen] != code:
            raise AnalysisError(f"Widerspruechliche Ground Truth fuer {specimen}.")
        seen_specimens[specimen] = code
        reference = item["reference_image"]
        relative = Path(str(reference["copied_path"]))
        absolute = (project_root / relative).resolve()
        _verify_image(absolute)
        cases.append({
            "case_id": f"confirmed-{specimen}",
            "specimen_id": specimen,
            "expected_primary_code": code,
            "expected_primary_name": catalog.definitions[code]["name"],
            "test_images": [{"path": relative.as_posix(), "sha256": _sha256(absolute)}],
            "label_confirmation": {
                "source": item["confirmation_source"],
                "confirmed_by": reference["confirmed_by"],
                "confirmed_at": reference["confirmed_at"],
                "visibility": reference["visibility"],
            },
        })
    if not cases:
        raise AnalysisError("Keine fachlich bestaetigten Hauptreferenzen fuer Ground Truth gefunden.")
    cases.sort(key=lambda item: (item["expected_primary_code"], item["specimen_id"]))
    distribution = Counter(item["expected_primary_code"] for item in cases)
    return {
        "schema_version": "1.0",
        "dataset_name": "confirmed-reference-leave-one-specimen-out-v1",
        "dataset_kind": "internal_leave_one_specimen_out",
        "created_at": _utc_now(),
        "source_manifest": source.relative_to(project_root).as_posix(),
        "source_manifest_sha256": _sha256(source),
        "classification_rules_version": catalog.rule_version,
        "ground_truth_policy": {
            "accepted": [
                "Nur als reference markierte Hauptreferenzbilder.",
                "confirmed=true, default_for_inference=true und Sichtbarkeit SICHTBAR_BESTAETIGT.",
                "Bestaetigende Person, Datum und Bestaetigungsquelle muessen vorhanden sein.",
            ],
            "rejected": [
                "Unbestaetigte Kontextbilder.",
                "Protokollklassen ohne explizite Bildbestaetigung.",
                "KI-Vorschlaege und bestehende Modellergebnisse.",
            ],
            "reference_exclusion": "Je Fall werden alle Bilder desselben Pruefkoerpers ausgeschlossen.",
        },
        "limitations": [
            "Dies ist eine interne Leave-one-Pruefkoerper-out-Evaluation und kein unabhaengiger Testdatensatz.",
            "Je Pruefkoerper ist derzeit nur das bestaetigte Hauptreferenzbild als Testbild zugelassen.",
            "Die 26 bestaetigten Faelle decken nur 11 der 29 Klassen ab.",
            "Klassen mit nur einem bestaetigten Pruefkoerper besitzen im jeweiligen Holdout keine gleichklassige Referenz.",
        ],
        "case_count": len(cases),
        "covered_class_count": len(distribution),
        "class_distribution": dict(sorted(distribution.items())),
        "cases": cases,
    }


def write_ground_truth(document: Mapping[str, Any], destination: Path = GROUND_TRUTH_PATH) -> Path:
    if destination.exists():
        raise AnalysisError(f"Ground-Truth-Datei existiert bereits und wird nicht ueberschrieben: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def load_ground_truth(path: Path = GROUND_TRUTH_PATH, project_root: Path = PROJECT_ROOT) -> list[EvaluationCase]:
    document = _read_json(path)
    source = (project_root / str(document.get("source_manifest", ""))).resolve()
    if not source.is_file() or _sha256(source) != document.get("source_manifest_sha256"):
        raise AnalysisError("Ground Truth passt nicht mehr zum Referenzmanifest; manuelle Pruefung erforderlich.")
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for item in document.get("cases", []):
        specimen = str(item.get("specimen_id", ""))
        if not specimen or specimen in seen:
            raise AnalysisError("Ground Truth enthaelt fehlende oder doppelte Pruefkoerper-IDs.")
        seen.add(specimen)
        images = item.get("test_images")
        if not isinstance(images, list) or not images:
            raise AnalysisError(f"Ground Truth fuer {specimen} enthaelt keine Bilder.")
        paths: list[Path] = []
        hashes: list[str] = []
        for image in images:
            absolute = (project_root / str(image["path"])).resolve()
            _verify_image(absolute)
            digest = _sha256(absolute)
            if digest != image.get("sha256"):
                raise AnalysisError(f"Ground-Truth-Bild wurde seit der Freigabe veraendert: {absolute}")
            paths.append(absolute)
            hashes.append(digest)
        cases.append(EvaluationCase(
            case_id=str(item["case_id"]), specimen_id=specimen,
            expected_code=str(item["expected_primary_code"]),
            image_paths=tuple(paths), image_sha256=tuple(hashes),
            confirmation_source=str(item["label_confirmation"]["source"]),
        ))
    return cases


def load_confirmed_references(project_root: Path = PROJECT_ROOT) -> tuple[ConfirmedReference, ...]:
    manifest = _read_json(project_root / "Datenbank" / "Referenzbilder" / "referenzbilder_manifest.json")
    references: list[ConfirmedReference] = []
    for item in manifest.get("reference_sets", []):
        if not isinstance(item, Mapping) or not _is_confirmed_reference(item):
            continue
        reference = item["reference_image"]
        path = (project_root / str(reference["copied_path"])).resolve()
        references.append(ConfirmedReference(
            code=str(item["failure_mode"]), specimen_id=str(item["specimen"]),
            path=path, sha256=_sha256(path),
        ))
    return tuple(references)


def select_holdout_references(
    case: EvaluationCase,
    references: Sequence[ConfirmedReference],
    max_per_class: int,
) -> tuple[ConfirmedReference, ...]:
    if max_per_class < 0:
        raise AnalysisError("max_references_per_class darf nicht negativ sein.")
    counts: Counter[str] = Counter()
    selected: list[ConfirmedReference] = []
    test_hashes = set(case.image_sha256)
    for reference in references:
        if reference.specimen_id == case.specimen_id or reference.sha256 in test_hashes:
            continue
        if counts[reference.code] >= max_per_class:
            continue
        selected.append(reference)
        counts[reference.code] += 1
    if any(item.specimen_id == case.specimen_id or item.sha256 in test_hashes for item in selected):
        raise AnalysisError(f"Referenz-Leakage fuer {case.specimen_id} erkannt.")
    return tuple(selected)


def _minimal_instructions(catalog: Catalog, *, staged: bool = False) -> dict[str, Any]:
    rules = [
        "Verwende nur die erlaubten Klassencodes oder UNKLAR.",
        "Beschreibe nur sichtbare Merkmale und behaupte keine Bruchursache.",
        "Behandle alle erlaubten Klassen gleichberechtigt und gib alle passenden Klassenvorschläge aus.",
        "Eine optionale primary-Empfehlung ist nur ein visueller Vorschlag und keine nachgewiesene Bruchursache.",
        "Kennzeichne jeden Vorschlag als sicher_sichtbar oder vermutet; confidence bleibt davon getrennt.",
        "Nutze UNKLAR, wenn die Bilder keine eindeutige Klassifikation erlauben.",
        "Gib ausschliesslich das vorgegebene JSON aus.",
    ]
    if staged:
        rules = [
            "Stufe 1: Erfasse zuerst neutral sichtbare Geometrie, Lage, Faserverlauf, Bruchtextur und fehlende Ansichten.",
            "Stufe 2: Vergleiche erst danach diese Beobachtungen mit den Regeln und nenne alle passenden Klassen.",
            "Keine Klassifikation darf ohne einen sichtbaren Beleg aus Stufe 1 erfolgen.",
            *rules,
        ]
    return {
        "task": "Klassifikation sichtbarer Merkmale an einer Holz-Zugprobe",
        "allowed_codes": list(catalog.codes),
        "rules": rules,
    }


def _class_descriptions(catalog: Catalog) -> list[dict[str, Any]]:
    """Beschreibungen fuer B, bewusst ohne validierte Entscheidungsregeln."""

    return [{
        "code": code,
        "name": definition.get("name"),
        "feature_type": definition.get("feature_type"),
        "definition": definition.get("definition"),
        "visual_features": definition.get("visual_features", []),
        "distinguishing_features": definition.get("distinguishing_features", []),
        "limitations": definition.get("limitations", []),
    } for code, definition in catalog.definitions.items()]


def build_variant_content(
    case: EvaluationCase,
    variant: str,
    catalog: Catalog,
    references: Sequence[ConfirmedReference],
) -> list[dict[str, Any]]:
    if variant not in VARIANTS:
        raise AnalysisError(f"Unbekannte Evaluationsvariante: {variant}")
    if variant == "B":
        instructions = {**_minimal_instructions(catalog), "classes": _class_descriptions(catalog)}
    elif variant == "D":
        instructions = _minimal_instructions(catalog, staged=True)
    else:
        instructions = _minimal_instructions(catalog)
    content: list[dict[str, Any]] = [{
        "type": "input_text",
        "text": json.dumps(instructions, ensure_ascii=False, separators=(",", ":")),
    }]
    if variant == "C":
        for reference in references:
            content.extend((
                {"type": "input_text", "text": f"Bestaetigtes Hauptreferenzbild fuer Klasse {reference.code}."},
                {"type": "input_image", "image_url": _data_url(reference.path), "detail": "high"},
            ))
    content.append({"type": "input_text", "text": "Ab hier folgen nur Bilder des unbekannten Testpruefkoerpers."})
    for index, path in enumerate(case.image_paths, start=1):
        content.extend((
            {"type": "input_text", "text": f"Unbekanntes Testbild {index}."},
            {"type": "input_image", "image_url": _data_url(path), "detail": "high"},
        ))
    content.append({"type": "input_text", "text": "Klassifiziere jetzt ausschliesslich die unbekannten Testbilder."})
    return content


def build_feature_extraction_content(
    case: EvaluationCase,
    catalog: Catalog,
) -> list[dict[str, Any]]:
    """Stufe 1 von D: Beobachtungen ohne Klassenwissen oder Referenzen."""

    instructions = {
        "task": "Extrahiere ausschliesslich neutrale sichtbare Bildmerkmale.",
        "rules": [
            "Nenne Geometrie, Lage, Faserverlauf, Bruchtextur, Schaerfe und fehlende Ansichten.",
            "Verwende keine Klassenbezeichnung und keinen Klassencode.",
            "Behaupte keine Bruchursache.",
            "Setze status auf UNKLAR und class_suggestions auf eine leere Liste.",
            "Schreibe die Merkmalsextraktion ausschließlich nach observations_short und visible_features.",
            "Nenne in uncertain_points, dass die Klassenzuordnung erst in Stufe 2 erfolgt.",
        ],
        "schema_note": "Die UNKLAR-Felder dienen nur der gemeinsamen strukturierten Ausgabeschnittstelle.",
    }
    content: list[dict[str, Any]] = [{
        "type": "input_text",
        "text": json.dumps(instructions, ensure_ascii=False, separators=(",", ":")),
    }]
    for index, path in enumerate(case.image_paths, start=1):
        content.extend((
            {"type": "input_text", "text": f"Unbekanntes Testbild {index}."},
            {"type": "input_image", "image_url": _data_url(path), "detail": "high"},
        ))
    return content


def build_rule_classification_content(
    observations: Sequence[str],
    catalog: Catalog,
) -> list[dict[str, Any]]:
    """Stufe 2 von D: regelgestuetzte Klassifikation ohne erneuten Bildzugriff."""

    instructions = {
        **_minimal_instructions(catalog),
        "task": "Klassifiziere nur die in Stufe 1 extrahierten sichtbaren Merkmale.",
        "classes_and_rules": _prompt_catalog(catalog),
        "stage_1_visible_observations": list(observations),
        "rules_for_stage_2": [
            "Nutze keine Information, die nicht in stage_1_visible_observations enthalten ist.",
            "Belege die Klassifikation mit den Beobachtungen aus Stufe 1.",
            "Gib UNKLAR aus, wenn die Beobachtungen keine regelkonforme Zuordnung erlauben.",
        ],
    }
    return [{
        "type": "input_text",
        "text": json.dumps(instructions, ensure_ascii=False, separators=(",", ":")),
    }]


def _to_mistral_messages(content: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for item in content:
        if item["type"] == "input_text":
            converted.append({"type": "text", "text": item["text"]})
        else:
            converted.append({"type": "image_url", "image_url": item["image_url"]})
    return [{"role": "system", "content": SYSTEM_INSTRUCTIONS}, {"role": "user", "content": converted}]


def _to_ollama_messages(
    content: Sequence[Mapping[str, Any]],
    *,
    max_image_edge: int,
    image_lookup: Mapping[str, Path],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    for item in content:
        if item["type"] == "input_text":
            messages.append({"role": "user", "content": item["text"]})
            continue
        url = str(item["image_url"])
        path = image_lookup.get(url)
        if path is None:
            raise AnalysisError("Interne Bildzuordnung fuer Ollama fehlt.")
        messages.append({
            "role": "user", "content": "Bild.",
            "images": [_jpeg_base64(path, max_long_edge=max_image_edge)],
        })
    return messages


def _usage_values(metadata: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    usage = metadata.get("usage") if isinstance(metadata.get("usage"), Mapping) else {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", metadata.get("prompt_eval_count")))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", metadata.get("eval_count")))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _merge_stage_metadata(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usages = [_usage_values(stage) for stage in stages]

    def summed(index: int) -> int | None:
        values = [value[index] for value in usages]
        return sum(values) if values and all(isinstance(value, int) for value in values) else None

    costs = [
        stage.get("cost", {}).get("reported_amount")
        for stage in stages
        if isinstance(stage.get("cost"), Mapping)
    ]
    reported_cost = (
        sum(costs) if costs and all(isinstance(value, (int, float)) for value in costs) else None
    )
    currency = next((
        stage["cost"].get("currency")
        for stage in stages
        if isinstance(stage.get("cost"), Mapping) and stage["cost"].get("currency")
    ), None)
    return {
        "response_model": next((stage.get("response_model") for stage in reversed(stages) if stage.get("response_model")), None),
        "usage": {
            "input_tokens": summed(0), "output_tokens": summed(1), "total_tokens": summed(2),
        },
        "cost": {"reported_amount": reported_cost, "currency": currency},
        "stages": [dict(stage) for stage in stages],
    }


def _prediction_items(validated: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not validated:
        return []
    suggestions = validated.get("class_suggestions")
    if isinstance(suggestions, list):
        return [dict(item) for item in suggestions if isinstance(item, Mapping) and item.get("code")]
    detected = validated.get("detected_features")
    if isinstance(detected, list):
        return [dict(item) for item in detected if isinstance(item, Mapping) and item.get("code")]
    primary = validated.get("primary_feature")
    secondary = validated.get("secondary_features")
    return (
        ([dict(primary)] if isinstance(primary, Mapping) and primary.get("code") else [])
        + [dict(item) for item in secondary or [] if isinstance(item, Mapping) and item.get("code")]
    )


def classify_outcome(expected: str, validated: Mapping[str, Any] | None, error_type: str = "") -> str:
    if error_type:
        return error_type
    if not validated or validated.get("status") == "UNKLAR":
        return "unclassifiable"
    predicted = {str(item.get("code")) for item in _prediction_items(validated)}
    scored = predicted - UNSCORED_AUXILIARY_CODES
    if expected in scored:
        return "correct"
    if scored:
        return "wrong_classification"
    return "unscored_auxiliary_only" if predicted & UNSCORED_AUXILIARY_CODES else "unclassifiable"


def classify_secure_outcome(
    expected: str, validated: Mapping[str, Any] | None, error_type: str = ""
) -> str:
    if error_type:
        return error_type
    if not validated or validated.get("status") == "UNKLAR":
        return "unclassifiable"
    items = _prediction_items(validated)
    secure = {
        str(item.get("code")) for item in items
        if item.get("evidence_status") == "sicher_sichtbar"
    }
    scored = secure - UNSCORED_AUXILIARY_CODES
    if expected in scored:
        return "correct"
    if scored:
        return "wrong_classification"
    if secure & UNSCORED_AUXILIARY_CODES:
        return "unscored_auxiliary_only"
    if any(item.get("evidence_status") in {"vermutet", None} for item in items):
        return "excluded_no_secure_evidence"
    return "unclassifiable"


def _evaluation_predicted_code(expected: str, feature_codes: Sequence[str], primary_code: str) -> str:
    scored = [code for code in feature_codes if code not in UNSCORED_AUXILIARY_CODES]
    if expected in scored:
        return expected
    if primary_code and primary_code not in UNSCORED_AUXILIARY_CODES:
        return primary_code
    if scored:
        return scored[0]
    if any(code in UNSCORED_AUXILIARY_CODES for code in feature_codes) or primary_code in UNSCORED_AUXILIARY_CODES:
        return "UNSCORED"
    return "UNKLAR"


def _error_type(exc: Exception) -> str:
    text = str(exc).lower()
    return "invalid_json" if "json" in text or "modellantwort" in text else "api_error"


def execute_job(
    *,
    case: EvaluationCase,
    variant: str,
    catalog: Catalog,
    references: Sequence[ConfirmedReference],
    provider: str,
    model: str,
    ollama_num_ctx: int,
    ollama_max_image_edge: int,
    client: Any = None,
    mistral_api_key: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw: Mapping[str, Any] | None = None
    validated: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = {}
    error_type = ""
    error_message = ""
    image_lookup = {_data_url(item.path): item.path for item in references}
    image_lookup.update({_data_url(path): path for path in case.image_paths})

    def invoke(content: list[dict[str, Any]]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if provider == "openai":
            return call_openai_with_metadata(
                model=model, content=content, catalog=catalog, client=client,
            )
        if provider == "mistral":
            return call_mistral(
                model=model, messages=_to_mistral_messages(content), catalog=catalog,
                api_key=mistral_api_key or "",
            )
        if provider == "ollama":
            return call_ollama(
                model=model,
                messages=_to_ollama_messages(
                    content, max_image_edge=ollama_max_image_edge, image_lookup=image_lookup,
                ),
                catalog=catalog, num_ctx=ollama_num_ctx,
            )
        raise AnalysisError(f"Unbekannter Provider: {provider}")

    try:
        if variant == "D":
            stage_1_raw, stage_1_metadata = invoke(build_feature_extraction_content(case, catalog))
            stage_1 = validate_and_enrich(stage_1_raw, catalog)
            stage_1_observations = (
                stage_1.get("observations_short") or stage_1.get("visible_observations") or []
            )
            if stage_1["status"] != "UNKLAR" or not stage_1_observations:
                raise AnalysisError("Merkmalsextraktion D muss UNKLAR und sichtbare Beobachtungen liefern.")
            raw, stage_2_metadata = invoke(
                build_rule_classification_content(stage_1_observations, catalog)
            )
            metadata = _merge_stage_metadata((stage_1_metadata, stage_2_metadata))
            validated = validate_and_enrich(raw, catalog)
            validated["visible_observations"] = list(stage_1_observations)
        else:
            variant_references = references
            if provider == "mistral" and variant == "C":
                available_reference_slots = max(
                    0,
                    MAX_MISTRAL_IMAGES_PER_REQUEST - len(case.image_paths),
                )
                variant_references = references[:available_reference_slots]
            content = build_variant_content(case, variant, catalog, variant_references)
            raw, metadata = invoke(content)
            validated = validate_and_enrich(raw, catalog)
    except Exception as exc:  # Ein einzelner Fehler darf den Evaluationslauf nicht abbrechen.
        error_type = _error_type(exc)
        error_message = re.sub(r"\s+", " ", str(exc)).strip()[:1000]
    duration = time.perf_counter() - started
    prediction_items = _prediction_items(validated)
    features = prediction_items
    primary = next(
        (item for item in prediction_items if item.get("suggestion_role") == "primary"),
        (validated or {}).get("primary_feature") or {},
    )
    feature_codes = [str(item.get("code")) for item in prediction_items if item.get("code")]
    secure_feature_codes = [
        str(item.get("code")) for item in prediction_items
        if item.get("code") and item.get("evidence_status") == "sicher_sichtbar"
    ]
    suspected_feature_codes = [
        str(item.get("code")) for item in prediction_items
        if item.get("code") and item.get("evidence_status") == "vermutet"
    ]
    undocumented_evidence_codes = [
        str(item.get("code")) for item in prediction_items
        if item.get("code") and item.get("evidence_status") not in {"sicher_sichtbar", "vermutet"}
    ]
    predicted_code = primary.get("code") or (feature_codes[0] if feature_codes else ("UNKLAR" if validated else "ERROR"))
    scored_feature_codes = [code for code in feature_codes if code not in UNSCORED_AUXILIARY_CODES]
    unscored_feature_codes = sorted(set(feature_codes) & UNSCORED_AUXILIARY_CODES)
    evaluation_predicted_code = _evaluation_predicted_code(case.expected_code, feature_codes, str(primary.get("code", "")))
    secure_primary_code = str(primary.get("code", "")) if primary.get("evidence_status") == "sicher_sichtbar" else ""
    secure_evaluation_predicted_code = _evaluation_predicted_code(
        case.expected_code, secure_feature_codes, secure_primary_code
    )
    auxiliary_only = evaluation_predicted_code == "UNSCORED"
    input_tokens, output_tokens, total_tokens = _usage_values(metadata)
    cost = metadata.get("cost") if isinstance(metadata.get("cost"), Mapping) else {}
    class_descriptions_hash = hashlib.sha256(json.dumps(
        _prompt_catalog(catalog), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "case_id": case.case_id,
        "specimen_id": case.specimen_id,
        "provider": provider,
        "model": model,
        "model_version": metadata.get("response_model") or model,
        "variant": variant,
        "active_class_set_version": f"klassifikationsregeln/{catalog.rule_version}",
        "active_class_descriptions_sha256": class_descriptions_hash,
        "expected_primary_code": case.expected_code,
        "predicted_primary_code": predicted_code,
        "evaluation_predicted_code": evaluation_predicted_code,
        "expected_codes": [case.expected_code],
        "predicted_feature_codes": feature_codes or ([primary.get("code")] if primary.get("code") else []),
        "secure_visible_feature_codes": secure_feature_codes,
        "suspected_feature_codes": suspected_feature_codes,
        "undocumented_evidence_feature_codes": undocumented_evidence_codes,
        "matched_codes": [case.expected_code] if case.expected_code in scored_feature_codes else [],
        "missed_codes": [] if case.expected_code in scored_feature_codes or auxiliary_only else [case.expected_code],
        "additional_codes": sorted(set(scored_feature_codes) - {case.expected_code}),
        "unscored_feature_codes": unscored_feature_codes,
        "outcome": classify_outcome(case.expected_code, validated, error_type),
        "secure_evaluation_predicted_code": secure_evaluation_predicted_code,
        "secure_outcome": classify_secure_outcome(case.expected_code, validated, error_type),
        "secondary_features": (validated or {}).get("secondary_features", []),
        "reasoning": primary.get("image_based_reasoning", []) or [
            reason for item in features for reason in item.get("image_based_reasoning", [])
        ],
        "visible_observations": (validated or {}).get("visible_observations", []),
        "confidence": primary.get("confidence"),
        "duration_seconds": round(duration, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reported_cost": cost.get("reported_amount"),
        "cost_currency": cost.get("currency"),
        "reference_count": len(references),
        "reference_specimens": [item.specimen_id for item in references],
        "error_type": error_type,
        "error_message": error_message,
        "provider_metadata": dict(metadata),
        "raw_response": dict(raw) if isinstance(raw, Mapping) else None,
    }


def compute_metrics(
    records: Sequence[Mapping[str, Any]], *, evidence_scope: str = "all"
) -> dict[str, Any]:
    if evidence_scope not in {"all", "secure_visible"}:
        raise ValueError("evidence_scope muss all oder secure_visible sein.")
    outcome_field = "outcome" if evidence_scope == "all" else "secure_outcome"
    prediction_field = (
        "evaluation_predicted_code" if evidence_scope == "all"
        else "secure_evaluation_predicted_code"
    )
    excluded_outcomes = {"unscored_auxiliary_only"}
    if evidence_scope == "secure_visible":
        excluded_outcomes.add("excluded_no_secure_evidence")
    if not records:
        return {"record_count": 0, "scored_record_count": 0, "unscored_record_count": 0,
                "coverage_rate": None, "excluded_record_count": 0,
                "accuracy": None, "accuracy_evaluated_only": None,
                "effective_correct_rate_all_cases": None, "macro_f1": None,
                "evidence_scope": evidence_scope}
    scored_records = [row for row in records if row.get(outcome_field) not in excluded_outcomes]
    excluded_records = [row for row in records if row.get(outcome_field) in excluded_outcomes]
    coverage_rate = len(scored_records) / len(records)
    if not scored_records:
        return {"record_count": len(records), "scored_record_count": 0,
                "unscored_record_count": len(records), "coverage_rate": 0.0,
                "excluded_record_count": len(excluded_records),
                "excluded_outcome_counts": dict(Counter(str(row.get(outcome_field)) for row in excluded_records)),
                "accuracy": None, "accuracy_evaluated_only": None,
                "effective_correct_rate_all_cases": 0.0, "macro_f1": None,
                "outcome_counts": dict(Counter(str(row.get(outcome_field)) for row in records)),
                "evidence_scope": evidence_scope}
    expected_label_set = {str(row["expected_primary_code"]) for row in scored_records}
    expected_labels = sorted(expected_label_set)
    matrix_labels = sorted(expected_label_set | {
        str(row.get(prediction_field, row.get("predicted_primary_code", "ERROR")))
        for row in scored_records
    })
    matrix = {expected: {predicted: 0 for predicted in matrix_labels} for expected in expected_labels}
    for row in scored_records:
        predicted = str(row.get(prediction_field, row.get("predicted_primary_code", "ERROR")))
        matrix[str(row["expected_primary_code"])][predicted] += 1
    class_metrics: dict[str, Any] = {}
    f1_values: list[float] = []
    correct_total = 0
    for label in expected_labels:
        tp = sum(
            row.get("expected_primary_code") == label
            and row.get(prediction_field, row.get("predicted_primary_code")) == label
            for row in scored_records
        )
        support = sum(row.get("expected_primary_code") == label for row in scored_records)
        predicted = sum(
            row.get(prediction_field, row.get("predicted_primary_code")) == label
            for row in scored_records
        )
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        correct_total += tp
        class_metrics[label] = {
            "support": support, "correct": tp, "hit_rate": round(recall, 6),
            "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
        }
    outcome_counts = Counter(str(row.get(outcome_field)) for row in records)
    accuracy = correct_total / len(scored_records)
    return {
        "record_count": len(records),
        "scored_record_count": len(scored_records),
        "unscored_record_count": len(records) - len(scored_records),
        "coverage_rate": round(coverage_rate, 6),
        "excluded_record_count": len(excluded_records),
        "excluded_outcome_counts": dict(Counter(str(row.get(outcome_field)) for row in excluded_records)),
        "accuracy": round(accuracy, 6),
        "accuracy_evaluated_only": round(accuracy, 6),
        "effective_correct_rate_all_cases": round(correct_total / len(records), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "evidence_scope": evidence_scope,
        "covered_expected_classes": expected_labels,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "per_class": class_metrics,
        "confusion_matrix": {"rows_expected": expected_labels, "columns_predicted": matrix_labels, "values": matrix},
        "interpretation_note": (
            "Kennzahlen beziehen sich nur auf die im Ground-Truth-Datensatz vertretenen Klassen; "
            "UNKLAR und Fehler zaehlen in der Gesamtgenauigkeit als nicht korrekt. "
            "Reine KB-Vorschlaege werden als unscored_auxiliary_only separat ausgewiesen und nicht gewertet. "
            + (
                "Die Hauptauswertung berücksichtigt sichere und vermutete Vorschläge."
                if evidence_scope == "all" else
                "Die Zusatzsicht berücksichtigt nur sicher sichtbare Vorschläge. Coverage, ausgeschlossene Fälle und effective_correct_rate_all_cases müssen gemeinsam mit accuracy_evaluated_only berichtet werden, damit der Ausschluss vermuteter Vorschläge die Güte nicht künstlich verbessert erscheinen lässt."
            )
        ),
    }


def _run_directory(output_dir: Path, provider: str, model: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    destination = output_dir / f"{stamp}_{provider}_{safe_model}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_run_artifacts(
    destination: Path,
    run_config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    metrics_by_variant = {
        variant: compute_metrics([row for row in records if row.get("variant") == variant])
        for variant in VARIANTS
    }
    secure_visible_metrics_by_variant = {
        variant: compute_metrics(
            [row for row in records if row.get("variant") == variant],
            evidence_scope="secure_visible",
        )
        for variant in VARIANTS
    }
    document = {
        "schema_version": "1.1", "run_config": dict(run_config),
        "records": list(records), "metrics_by_variant": metrics_by_variant,
        "secure_visible_metrics_by_variant": secure_visible_metrics_by_variant,
    }
    (destination / "evaluation_results.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with (destination / "evaluation_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(record.get(key), ensure_ascii=False) if isinstance(record.get(key), (list, dict)) else record.get(key)
                for key in RESULT_FIELDS
            })
    with (destination / "confusion_matrices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evidence_scope", "variant", "expected", "predicted", "count"])
        for evidence_scope, grouped_metrics in (
            ("all", metrics_by_variant),
            ("secure_visible", secure_visible_metrics_by_variant),
        ):
            for variant, metrics in grouped_metrics.items():
                matrix = metrics.get("confusion_matrix", {})
                for expected, values in matrix.get("values", {}).items():
                    for predicted, count in values.items():
                        writer.writerow([evidence_scope, variant, expected, predicted, count])


def build_plan(
    cases: Sequence[EvaluationCase],
    references: Sequence[ConfirmedReference],
    *, provider: str, model: str, max_references_per_class: int,
) -> list[dict[str, Any]]:
    jobs = []
    for case in cases:
        holdout = select_holdout_references(case, references, max_references_per_class)
        for variant in VARIANTS:
            used = holdout if variant == "C" else ()
            jobs.append({
                "case_id": case.case_id, "specimen_id": case.specimen_id,
                "expected_primary_code": case.expected_code, "variant": variant,
                "variant_description": VARIANT_DESCRIPTIONS[variant],
                "provider": provider, "model": model,
                "test_image_count": len(case.image_paths),
                "reference_count": len(used),
                "reference_specimens": [item.specimen_id for item in used],
                "same_specimen_reference_present": any(item.specimen_id == case.specimen_id for item in used),
                "model_request_count": 2 if variant == "D" else 1,
            })
    return jobs


def run_evaluation(args: argparse.Namespace) -> Path:
    complete_catalog = load_catalog(include_references=False)
    active_codes = tuple(code for code, definition in sorted(
        complete_catalog.definitions.items(),
        key=lambda item: item[1].get("validated_classification_rule_order", 10**9),
    ) if "validated_classification_rule" in definition)
    catalog = load_catalog(allowed_codes=active_codes)
    all_cases = load_ground_truth(args.ground_truth)
    cases = [case for case in all_cases if case.expected_code in catalog.definitions]
    references = tuple(
        reference for reference in load_confirmed_references()
        if reference.code in catalog.definitions
    )
    destination = _run_directory(args.output_dir, args.provider, args.model)
    request_parameters: dict[str, Any]
    if args.provider == "ollama":
        request_parameters = {
            "temperature": 0, "num_predict": 4096, "num_ctx": args.ollama_num_ctx,
            "max_image_edge": args.ollama_max_image_edge, "structured_output": True,
        }
    elif args.provider == "mistral":
        request_parameters = {"temperature": 0, "max_tokens": 4096, "structured_output": True}
    else:
        request_parameters = {"temperature": "provider_default", "store": False, "structured_output": True}
    run_config = {
        "prompt_protocol_version": "1.0",
        "created_at": _utc_now(), "provider": args.provider, "model": args.model,
        "variants": list(VARIANTS), "case_count": len(cases),
        "full_ground_truth_case_count": len(all_cases),
        "active_class_codes": list(catalog.codes),
        "planned_model_request_count": len(cases) * 5,
        "max_references_per_class": args.max_references_per_class,
        "ollama_num_ctx": args.ollama_num_ctx,
        "ollama_max_image_edge": args.ollama_max_image_edge,
        "request_parameters": request_parameters,
        "ground_truth_path": str(args.ground_truth.resolve()),
        "ground_truth_sha256": _sha256(args.ground_truth),
        "evaluation_code_sha256": _sha256(Path(__file__).resolve()),
        "class_definitions_sha256": _sha256(PROJECT_ROOT / "Datenbank" / "versagensarten_erweitert.json"),
        "classification_rules_sha256": _sha256(PROJECT_ROOT / "Datenbank" / "klassifikationsregeln.json"),
        "reference_manifest_sha256": _sha256(
            PROJECT_ROOT / "Datenbank" / "Referenzbilder" / "referenzbilder_manifest.json"
        ),
        "classification_rules_version": catalog.rule_version,
        "ground_truth_classification_rules_version": str(
            _read_json(args.ground_truth).get("classification_rules_version", "unbekannt")
        ),
        "ground_truth_rules_version_matches_active": str(
            _read_json(args.ground_truth).get("classification_rules_version", "unbekannt")
        ) == catalog.rule_version,
        "execution_requested": args.execute,
        "paid_api_request_performed": False,
    }
    jobs = build_plan(
        cases, references, provider=args.provider, model=args.model,
        max_references_per_class=args.max_references_per_class,
    )
    (destination / "run_plan.json").write_text(
        json.dumps({"run_config": run_config, "jobs": jobs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.execute:
        return destination
    if args.provider in {"openai", "mistral"} and not (args.live and args.confirm_paid_request):
        raise AnalysisError("Cloud-Evaluation gesperrt: --live und --confirm-paid-request sind erforderlich.")
    client = None
    mistral_key = None
    if args.provider == "openai":
        load_verified_model(args.model)
        client = create_openai_client(resolve_api_key())
    elif args.provider == "mistral":
        load_verified_mistral_model(args.model)
        mistral_key = resolve_mistral_api_key().value
    records: list[dict[str, Any]] = []
    for case in cases:
        holdout = select_holdout_references(case, references, args.max_references_per_class)
        for variant in VARIANTS:
            record = execute_job(
                case=case, variant=variant, catalog=catalog,
                references=holdout if variant == "C" else (),
                provider=args.provider, model=args.model,
                ollama_num_ctx=args.ollama_num_ctx,
                ollama_max_image_edge=args.ollama_max_image_edge,
                client=client, mistral_api_key=mistral_key,
            )
            records.append(record)
            write_run_artifacts(destination, run_config, records)
    run_config["paid_api_request_performed"] = args.provider in {"openai", "mistral"}
    write_run_artifacts(destination, run_config, records)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduzierbare Leave-one-Pruefkoerper-out-Evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Ground Truth einmalig aus bestaetigten Referenzen erzeugen")
    prepare.add_argument("--output", type=Path, default=GROUND_TRUTH_PATH)
    prepare.add_argument("--validate-only", action="store_true")
    run = subparsers.add_parser("run", help="Laufplan erzeugen oder Evaluation explizit ausfuehren")
    run.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--provider", choices=("openai", "mistral", "ollama"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--max-references-per-class", type=int, default=1)
    run.add_argument("--ollama-num-ctx", type=int, default=DEFAULT_OLLAMA_NUM_CTX)
    run.add_argument("--ollama-max-image-edge", type=int, default=DEFAULT_OLLAMA_MAX_IMAGE_EDGE)
    run.add_argument("--execute", action="store_true", help="Ohne diesen Schalter wird nur ein kostenfreier Plan erstellt")
    run.add_argument("--live", action="store_true")
    run.add_argument("--confirm-paid-request", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            document = build_ground_truth_document()
            if args.validate_only:
                print(json.dumps({
                    "case_count": document["case_count"],
                    "covered_class_count": document["covered_class_count"],
                    "class_distribution": document["class_distribution"],
                }, ensure_ascii=False, indent=2))
                return 0
            destination = write_ground_truth(document, args.output)
        else:
            destination = run_evaluation(args)
    except (AnalysisError, OpenAIConfigurationError, RuntimeError, OSError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    print(f"Gespeichert: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
