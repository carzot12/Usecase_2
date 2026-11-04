from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

from desktop.reference_store import ReferenceStore
from Tool.ki_bildanalyse import (
    PROMPT_VERSION, PROVIDER_MAX_TEST_IMAGES_PER_REQUEST, SUPPORTED_SUFFIXES,
    load_catalog, merge_partial_classifications, run_analysis,
)
from Tool.evaluation_variants import (
    ACTIVE_CLASS_CODES, CLASS_DIRECTORIES, VARIANT_IDS, control_token, inspect_variant, memory_status,
    prepare_reference_memory as write_reference_memory,
    reference_inventory_preview, update_reference_inventories,
)


ROOT = Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    """Locate mutable project data outside a frozen application bundle."""

    configured = os.getenv("EKI_PROJECT_ROOT", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        ROOT,
        Path.home() / "Desktop/Masterarbeit/Usecase_2",
    ]
    for candidate in candidates:
        if candidate and (candidate / "Pruefdaten").is_dir():
            return candidate.resolve()
    return ROOT


WORKSPACE_ROOT = _workspace_root()
EXAMPLE = WORKSPACE_ROOT / "Pruefdaten/Neue_Analysen/Test_001/Sortiert"
APP_DATA = WORKSPACE_ROOT / "Output/EKI_Bildanalyse/Projekte"
EVALUATION_VARIANTS_DIR = WORKSPACE_ROOT / "Datenbank/Evaluationsvarianten"
EVALUATION_MEMORY_DIR = WORKSPACE_ROOT / "Output/KI_Evaluation/Referenzgedaechtnis"
EVALUATION_PROMPT_VERSION = PROMPT_VERSION
MODEL_CONFIGS: dict[tuple[str, str], tuple[int, int]] = {
    ("ollama", "qwen3-vl:4b-instruct"): (24576, 4096),
    ("openai", "gpt-6.1-sol"): (16384, 4096),
    ("mistral", "ministral-3b-2512"): (16384, 4096),
}
DEFAULT_SELECTED_IMAGE_COUNT = 3
TERMINAL_BATCH_JOB_STATUSES = frozenset({
    "succeeded", "partially_succeeded", "failed", "cancelled",
})
MAX_AUTOMATIC_PART_RETRIES = 2


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError("Bitte einen gültigen Namen eingeben.")
    return cleaned


def _provider_failure_policy(error: Exception) -> dict[str, Any]:
    """Classify a redacted provider failure for bounded retry or run pause."""

    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    status_codes = {
        int(value) for item in chain
        for value in (getattr(item, "status_code", None), getattr(item, "code", None))
        if isinstance(value, int)
    }
    texts = " ".join(str(item) for item in chain).casefold()
    headers = next((getattr(item, "headers", None) for item in chain if getattr(item, "headers", None)), None)
    retry_after: float | None = None
    if headers is not None:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        try:
            retry_after = max(0.0, float(raw)) if raw is not None else None
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(raw))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                retry_after = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                retry_after = None
    billing_terms = (
        "insufficient_quota", "billing", "payment required", "credit balance",
        "quota exceeded", "abrechnung", "guthaben",
    )
    if 402 in status_codes or (
        status_codes.intersection({403, 429}) and any(term in texts for term in billing_terms)
    ):
        return {"action": "pause", "reason": "billing_or_quota", "retryAfterSeconds": retry_after}
    temporary = bool(status_codes.intersection({408, 409, 425, 429, 500, 502, 503, 504}))
    temporary = temporary or any(
        term in texts for term in (
            "timeout", "timed out", "zeitüberschreitung", "temporarily unavailable",
            "connection reset", "rate limit", "too many requests",
        )
    )
    if temporary:
        return {"action": "retry", "reason": "temporary_provider_error", "retryAfterSeconds": retry_after}
    return {"action": "fail", "reason": "permanent_technical_error", "retryAfterSeconds": None}


def _partition_provider_images(image_names: list[str], provider: str) -> list[list[str]]:
    if not image_names:
        return []
    limit = PROVIDER_MAX_TEST_IMAGES_PER_REQUEST[provider]
    chunk_size = limit or len(image_names)
    return [image_names[offset:offset + chunk_size]
            for offset in range(0, len(image_names), chunk_size)]


def _result_features(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return an equal-weight display view without mutating stored AI output."""

    definitions: dict[str, dict[str, Any]] | None = None

    def unique_codes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal definitions
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            code = item.get("code")
            if not isinstance(code, str) or not code.strip() or code in seen:
                continue
            seen.add(code)
            copied = dict(item)
            if not copied.get("name"):
                if definitions is None:
                    definitions = load_catalog(include_references=False).definitions
                definition = definitions.get(code, {})
                copied["name"] = definition.get("name") or "Bezeichnung nicht dokumentiert"
            unique.append(copied)
        return unique

    suggestions = result.get("class_suggestions")
    if isinstance(suggestions, list):
        return unique_codes([{
            **item,
            "feature_id": item.get("suggestion_id", f"suggestion-{index:02d}"),
            "visible_description": "",
            "image_evidence": [],
            "approximate_location": "",
            "fracture_zone_relation": "unclear",
            "uncertainty_reasons": [],
        } for index, item in enumerate(suggestions, start=1) if isinstance(item, dict)])
    current = result.get("detected_features")
    if isinstance(current, list):
        return unique_codes([dict(item) for item in current if isinstance(item, dict)])
    features: list[dict[str, Any]] = []
    primary = result.get("primary_feature")
    historic = ([primary] if isinstance(primary, dict) else []) + [
        item for item in result.get("secondary_features", []) if isinstance(item, dict)
    ]
    for index, item in enumerate(historic, start=1):
        features.append({
            **item,
            "feature_id": f"legacy-feature-{index:02d}",
            "legacy_role": "primary" if index == 1 and isinstance(primary, dict) else "secondary",
            "visible_description": item.get("visible_description") or "Im historischen Format nicht separat erfasst.",
            "image_evidence": item.get("image_evidence", []),
            "approximate_location": item.get("approximate_location", "Im historischen Format nicht erfasst."),
            "fracture_zone_relation": item.get("fracture_zone_relation", "unclear"),
            "uncertainty_reasons": item.get("uncertainty_reasons", []),
        })
    return unique_codes(features)


def _primary_result_feature(result: dict[str, Any]) -> dict[str, Any]:
    suggestions = result.get("class_suggestions")
    if isinstance(suggestions, list):
        primary = next((
            item for item in suggestions
            if isinstance(item, dict) and item.get("suggestion_role") == "primary"
        ), None)
        if primary is not None:
            return dict(primary)
    primary = result.get("primary_feature")
    return dict(primary) if isinstance(primary, dict) else {}


class DesktopBridge:
    """Allow-listed bridge; project files and analysis stay in Python."""

    def __init__(self) -> None:
        APP_DATA.mkdir(parents=True, exist_ok=True)
        self.project_root: Path | None = None
        self.project_name = ""
        self.jobs: dict[str, dict[str, Any]] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.evaluation_plans: dict[str, dict[str, Any]] = {}
        self.session_id = uuid.uuid4().hex
        self._lock = threading.RLock()

    @staticmethod
    def _z30_expected_specimens() -> list[str]:
        return [f"Z-30-{number:02d}" for number in (*range(1, 30), *range(31, 51))]

    def _reference_store(self) -> ReferenceStore:
        return ReferenceStore(WORKSPACE_ROOT, load_catalog(include_references=False).definitions)

    def bootstrap(self, _payload: Any = None) -> dict[str, Any]:
        definitions = load_catalog(include_references=False).definitions
        # Startup must depend only on the classification rules bundled with the app.
        # The editable reference store lives below Desktop and can legitimately be
        # unavailable to a Finder-launched process until the user opens that area.
        # Reading the full catalog here used to make that optional access fatal.
        active_codes = list(ACTIVE_CLASS_CODES)
        reference_configured = bool(active_codes)
        return {"ok": True, "exampleAvailable": EXAMPLE.is_dir(), "examplePath": str(EXAMPLE),
                "models": [
                    {"provider": provider, "model": model, "local": provider == "ollama"}
                    for provider, model in MODEL_CONFIGS
                ],
                "classes": [{"code": code, "name": definitions[code]["name"]}
                            for code in active_codes],
                "evaluation": {
                    "activeVariant": "1",
                    "ready": reference_configured,
                    "reason": (
                        "Die zwölf dokumentierten Referenzmerkmale sind aktiv."
                        if reference_configured else
                        "In klassifikationsregeln.json sind keine eindeutigen Referenzmerkmale dokumentiert."
                    ),
                }}

    def _project_data(self) -> Path:
        if not self.project_name:
            raise RuntimeError("Kein Projekt geöffnet.")
        path = APP_DATA / _safe_name(self.project_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _selection_path(self, specimen: str) -> Path:
        folder = self._project_data() / "Bildauswahl"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{_safe_name(specimen)}.json"

    @staticmethod
    def _default_image_selection(image_names: list[str]) -> set[str]:
        if len(image_names) <= DEFAULT_SELECTED_IMAGE_COUNT:
            return set(image_names)
        indices = {
            round(index * (len(image_names) - 1) / (DEFAULT_SELECTED_IMAGE_COUNT - 1))
            for index in range(DEFAULT_SELECTED_IMAGE_COUNT)
        }
        return {image_names[index] for index in sorted(indices)}

    def _selected_image_names(self, specimen: str, folder: Path) -> set[str]:
        image_names = [path.name for path in self._images(folder)]
        available = set(image_names)
        selection_path = self._selection_path(specimen)
        if selection_path.is_file():
            saved = json.loads(selection_path.read_text(encoding="utf-8"))
            previous = set(saved.get("knownImages", []))
            selected = set(saved.get("selectedImages", [])) & available
            selected.update(available - previous)
            customized = bool(saved.get("selectionCustomized", False))
            if not customized and len(selected) > DEFAULT_SELECTED_IMAGE_COUNT:
                selected = self._default_image_selection(image_names)
        else:
            legacy_folder = folder / "Analyseauswahl"
            legacy_names = {
                path.name for path in legacy_folder.iterdir()
                if path.is_file() or path.is_symlink()
            } if legacy_folder.is_dir() else set()
            selected = (legacy_names & available) if legacy_names else self._default_image_selection(image_names)
            customized = bool(legacy_names)
        self._write_image_selection(specimen, selected, available, customized=customized)
        return selected

    def _confirmed_image_selection(self, specimen: str, folder: Path) -> tuple[set[str], str]:
        """Read an existing manual selection without creating or expanding it."""

        available = {path.name for path in self._images(folder)}
        selection_path = self._selection_path(specimen)
        if selection_path.is_file():
            saved = json.loads(selection_path.read_text(encoding="utf-8"))
            selected = set(saved.get("selectedImages", []))
            unknown = selected - available
            new_images = available - set(saved.get("knownImages", []))
            if unknown:
                raise ValueError(
                    f"{specimen}: ausgewählte Dateien fehlen: {', '.join(sorted(unknown))}."
                )
            if new_images:
                raise ValueError(
                    f"{specimen}: {len(new_images)} neue Bilder wurden noch nicht manuell geprüft."
                )
            if not bool(saved.get("selectionCustomized", False)):
                raise ValueError(f"{specimen}: Bildauswahl wurde noch nicht manuell bestätigt.")
            if not selected:
                raise ValueError(f"{specimen}: kein Analysebild ist manuell ausgewählt.")
            return selected, "gespeicherte_manuelle_auswahl"
        legacy_folder = folder / "Analyseauswahl"
        if legacy_folder.is_dir():
            selected = {
                path.name for path in legacy_folder.iterdir()
                if (path.is_file() or path.is_symlink()) and path.name in available
            }
            if selected:
                return selected, "bestehende_analyseauswahl"
        raise ValueError(f"{specimen}: keine bestätigte manuelle Bildauswahl gefunden.")

    def _write_image_selection(
        self,
        specimen: str,
        selected: set[str],
        available: set[str],
        *,
        customized: bool,
    ) -> None:
        document = {
            "specimen": specimen,
            "selectedImages": sorted(selected),
            "knownImages": sorted(available),
            "selectionCustomized": customized,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            target = self._selection_path(specimen)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)

    def open_example(self, _payload: Any = None) -> dict[str, Any]:
        return self.open_project_path(str(EXAMPLE))

    def choose_project(self, _payload: Any = None) -> dict[str, Any]:
        import webview
        chosen = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        if not chosen:
            return {"ok": False, "cancelled": True}
        return self.open_project_path(str(chosen[0] if isinstance(chosen, (list, tuple)) else chosen))

    def open_project_path(self, path: str) -> dict[str, Any]:
        root = Path(path).expanduser().resolve()
        if root.name != "Sortiert" and (root / "Sortiert").is_dir():
            root = root / "Sortiert"
        if not root.is_dir():
            return {"ok": False, "error": "Der Projektordner wurde nicht gefunden."}
        self.project_root = root
        self.project_name = root.parent.name if root.name == "Sortiert" else root.name
        self._save_manifest()
        return {"ok": True, "project": self._snapshot()}

    def create_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            name = _safe_name(str(payload.get("name", "")))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        root = WORKSPACE_ROOT / "Pruefdaten/Neue_Analysen" / name / "Sortiert"
        root.mkdir(parents=True, exist_ok=True)
        return self.open_project_path(str(root))

    def _save_manifest(self) -> None:
        target = self._project_data() / "project.json"
        target.write_text(json.dumps({"name": self.project_name, "source": str(self.project_root),
                                      "updatedAt": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")

    def _specimens(self) -> list[Path]:
        if not self.project_root:
            return []
        return sorted((p for p in self.project_root.iterdir() if p.is_dir() and p.name != "Zur_Pruefung"),
                      key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)])

    @staticmethod
    def _images(folder: Path) -> list[Path]:
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                      and not p.name.casefold().startswith("nummer_"))

    @staticmethod
    def _all_images(folder: Path) -> list[Path]:
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )

    @staticmethod
    def _is_identification_image(path: Path) -> bool:
        return path.name.casefold().startswith("nummer_")

    def _history_visibility_path(self) -> Path:
        folder = self._project_data() / "Historie"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "sichtbarkeit.json"

    def _history_visibility(self) -> dict[str, Any]:
        path = self._history_visibility_path()
        if not path.is_file():
            return {"schemaVersion": "1.0", "hiddenRuns": {}}
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schemaVersion": "1.0", "hiddenRuns": {}}
        hidden = document.get("hiddenRuns", {}) if isinstance(document, dict) else {}
        return {
            "schemaVersion": "1.0",
            "hiddenRuns": hidden if isinstance(hidden, dict) else {},
        }

    def _write_history_visibility(self, document: dict[str, Any]) -> None:
        target = self._history_visibility_path()
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _latest_result(self, specimen: str) -> Path | None:
        results = [record for record in self._result_records(specimen) if not record["hidden"]]
        return Path(results[-1]["path"]) if results else None

    @staticmethod
    def _run_id(document: dict[str, Any], path: Path) -> str:
        return str(document.get("run_id") or (
            "legacy-" + hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
        ))

    def _result_records(self, specimen: str) -> list[dict[str, Any]]:
        folders = [self._project_data() / "Ergebnisse"]
        scientific_root = self._project_data() / "Wissenschaftliche_Laeufe"
        if scientific_root.is_dir():
            folders.extend(scientific_root.glob("*/Ergebnisse"))
        hidden_runs = self._history_visibility()["hiddenRuns"]
        records: list[dict[str, Any]] = []
        paths = [
            path for folder in folders if folder.is_dir()
            for path in folder.glob(f"{_safe_name(specimen)}_*.json")
        ]
        for path in paths:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(document, dict):
                    continue
                run_id = self._run_id(document, path)
                result = document.get("result", {})
                primary = _primary_result_feature(result)
                features = _result_features(result)
                execution = document.get("execution", {})
                metadata = execution.get("provider_metadata", {})
                cost = metadata.get("cost", {}) if isinstance(metadata, dict) else {}
                series = document.get("test_series")
                series = series if isinstance(series, dict) else {}
                visibility = hidden_runs.get(run_id, {})
                visibility = visibility if isinstance(visibility, dict) else {}
                records.append({
                    "runId": run_id,
                    "specimenId": specimen,
                    "path": str(path),
                    "createdAt": document.get("created_at", ""),
                    "provider": execution.get("provider", "unknown"),
                    "model": execution.get("model", "unknown"),
                    "status": result.get("status", ""),
                    "primaryCode": primary.get("code", ""),
                    "primaryConfidence": primary.get("confidence"),
                    "secondaryCodes": [item.get("code") for item in result.get("secondary_features", [])],
                    "detectedCodes": [item.get("code") for item in features if item.get("code")],
                    "featureCount": len(features),
                    "durationSeconds": execution.get("duration_seconds"),
                    "usage": metadata.get("usage", {}) if isinstance(metadata, dict) else {},
                    "cost": cost,
                    "promptHash": document.get("classification_basis", {}).get("prompt_hash_sha256"),
                    "promptVersion": document.get("classification_basis", {}).get("prompt_version"),
                    "evaluationVariant": document.get("classification_basis", {}).get("evaluation_variant"),
                    "seriesId": series.get("series_id"),
                    "seriesName": series.get("name"),
                    "resultGroup": "scientific" if series.get("series_id") else "historical",
                    "hidden": run_id in hidden_runs,
                    "hiddenAt": visibility.get("hiddenAt"),
                    "visibleObservations": result.get(
                        "observations_short", result.get("visible_observations", [])
                    ),
                    "assessment": self._assessment(run_id),
                    "legacy": not any(
                        field in result for field in ("class_suggestions", "detected_features")
                    ),
                })
            except (OSError, json.JSONDecodeError, AttributeError):
                continue
        records.sort(key=lambda item: (item["createdAt"], item["path"]))
        return records

    def _result_by_run_id(self, run_id: str) -> tuple[Path, dict[str, Any]] | None:
        for specimen in self._specimens():
            for record in self._result_records(specimen.name):
                if record["runId"] == run_id:
                    path = Path(record["path"])
                    return path, json.loads(path.read_text(encoding="utf-8"))
        return None

    def _assessment(self, run_id: str) -> dict[str, Any] | None:
        folder = self._project_data() / "Bewertungen"
        current = folder / f"{_safe_name(run_id)}.review.json"
        legacy = folder / f"{_safe_name(run_id)}.json"
        path = current if current.is_file() else legacy
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _legacy_assessment(self, run_id: str) -> dict[str, Any] | None:
        path = self._project_data() / "Bewertungen" / f"{_safe_name(run_id)}.json"
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        return document if document.get("schemaVersion") != "2.0" else None

    def _snapshot(self) -> dict[str, Any]:
        return {"name": self.project_name, "path": str(self.project_root), "specimens": [
            {"id": p.name, "imageCount": len(self._all_images(p)),
             "analysisImageCount": len(self._images(p)),
             "identificationImageCount": len(self._all_images(p)) - len(self._images(p)),
             "runCount": len([record for record in self._result_records(p.name) if not record["hidden"]]),
             "hasResult": self._latest_result(p.name) is not None}
            for p in self._specimens()]}

    def project_snapshot(self, _payload: Any = None) -> dict[str, Any]:
        return {"ok": True, "project": self._snapshot()}

    @staticmethod
    def _image_data(path: Path, edge: int = 720) -> str:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((edge, edge))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=84)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    def specimen_detail(self, payload: dict[str, Any]) -> dict[str, Any]:
        specimen = _safe_name(str(payload.get("specimen", "")))
        folder = (self.project_root / specimen).resolve() if self.project_root else Path()
        if not self.project_root or folder.parent != self.project_root.resolve() or not folder.is_dir():
            return {"ok": False, "error": "Prüfkörper wurde nicht gefunden."}
        source_images = self._all_images(folder)
        selected_names = self._selected_image_names(specimen, folder)
        images = [
            {
                "name": p.name,
                "src": self._image_data(p),
                "analysisSelected": p.name in selected_names,
                "analysisEligible": not self._is_identification_image(p),
                "isIdentification": self._is_identification_image(p),
            }
            for p in source_images
        ]
        latest = self._latest_result(specimen)
        result = json.loads(latest.read_text()) if latest else None
        history = self._result_records(specimen)
        correction_path = self._project_data() / "Korrekturen" / f"{specimen}.json"
        correction = json.loads(correction_path.read_text()) if correction_path.is_file() else None
        return {"ok": True, "specimen": specimen, "images": images, "result": result,
                "resultPath": str(latest) if latest else None, "correction": correction,
                "history": history}

    def project_results(self, _payload: Any = None) -> dict[str, Any]:
        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        records = [
            record
            for specimen in self._specimens()
            for record in self._result_records(specimen.name)
        ]
        records.sort(key=lambda item: (item.get("createdAt", ""), item["runId"]), reverse=True)
        return {"ok": True, "results": records, "series": self.test_series_overview()}

    def set_result_visibility(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist display visibility without modifying analysis or review artifacts."""

        run_id = str(payload.get("runId", "")).strip()
        if not run_id:
            return {"ok": False, "error": "Analyselauf wurde nicht angegeben."}
        found = self._result_by_run_id(run_id)
        if found is None:
            return {"ok": False, "error": "Analyselauf wurde nicht gefunden."}
        hidden = bool(payload.get("hidden", True))
        visibility = self._history_visibility()
        hidden_runs = visibility["hiddenRuns"]
        if hidden:
            hidden_runs[run_id] = {
                "hiddenAt": datetime.now(timezone.utc).isoformat(),
                "reason": "manual",
            }
        else:
            hidden_runs.pop(run_id, None)
        with self._lock:
            self._write_history_visibility(visibility)
        return {"ok": True, "runId": run_id, "hidden": hidden}

    def reset_test_history(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Hide development runs only; scientific series and source files remain untouched."""

        specimen = str(payload.get("specimen", "")).strip()
        records = [
            record
            for item in self._specimens()
            for record in self._result_records(item.name)
            if (not specimen or record["specimenId"] == specimen)
            and record["resultGroup"] == "historical"
            and not record["hidden"]
        ]
        visibility = self._history_visibility()
        hidden_at = datetime.now(timezone.utc).isoformat()
        for record in records:
            visibility["hiddenRuns"][record["runId"]] = {
                "hiddenAt": hidden_at,
                "reason": "development_history_reset",
            }
        with self._lock:
            self._write_history_visibility(visibility)
        return {
            "ok": True,
            "hiddenCount": len(records),
            "scientificRunsChanged": 0,
        }

    def _delete_result_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Delete only discovered KI result JSON files and preserve every other artifact."""

        results_root = (self._project_data() / "Ergebnisse").resolve()
        visibility = self._history_visibility()
        deleted: list[str] = []
        preserved_assessments = 0
        with self._lock:
            for record in records:
                path = Path(record["path"]).resolve()
                if path.parent != results_root or not path.is_file():
                    continue
                run_id = str(record["runId"])
                if self._assessment(run_id) is not None:
                    preserved_assessments += 1
                path.unlink()
                visibility["hiddenRuns"].pop(run_id, None)
                deleted.append(run_id)
            self._write_history_visibility(visibility)
        return {
            "ok": True,
            "deletedCount": len(deleted),
            "deletedRunIds": deleted,
            "preservedAssessmentCount": preserved_assessments,
            "preservedScopes": [
                "original_images", "reference_database", "expert_assessments",
                "official_historical_data", "batch_manifests", "test_series",
            ],
        }

    def delete_analysis_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Permanently delete one selected KI run result after explicit confirmation."""

        if not bool(payload.get("confirmDeletion")):
            return {"ok": False, "error": "Das endgültige Löschen wurde nicht bestätigt."}
        run_id = str(payload.get("runId", "")).strip()
        found = self._result_by_run_id(run_id) if run_id else None
        if found is None:
            return {"ok": False, "error": "Analyselauf wurde nicht gefunden."}
        path, document = found
        record = {
            "runId": self._run_id(document, path),
            "path": str(path),
        }
        return self._delete_result_records([record])

    def delete_specimen_analysis_runs(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Delete all KI result files for one specimen, never source or review data."""

        if not bool(payload.get("confirmDeletion")):
            return {"ok": False, "error": "Das endgültige Löschen wurde nicht bestätigt."}
        try:
            specimen = _safe_name(str(payload.get("specimen", "")))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        records = self._result_records(specimen)
        return self._delete_result_records(records)

    def _write_technical_failure(
        self,
        *,
        run_id: str,
        specimen: str,
        provider: str,
        model: str,
        selected_images: list[str],
        error: Exception,
        output_folder: Path | None = None,
    ) -> Path:
        folder = output_folder or (self._project_data() / "Technische_Fehlversuche")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{_safe_name(run_id)}.json"
        document = {
            "schemaVersion": "1.0",
            "runId": run_id,
            "specimenId": specimen,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "classification": "technischer_analysefehler",
            "scientificResult": False,
            "provider": provider,
            "model": model,
            "selectedImages": list(selected_images),
            "error": str(error) or repr(error),
            "failureType": getattr(error, "failure_type", "technical_execution"),
            "diagnostics": getattr(error, "diagnostics", {}),
        }
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, target)
        return target

    def _test_series_folder(self) -> Path:
        folder = self._project_data() / "Testreihen"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _active_test_series(self) -> dict[str, Any] | None:
        active_path = self._test_series_folder() / "active.json"
        if not active_path.is_file():
            return None
        pointer = json.loads(active_path.read_text(encoding="utf-8"))
        series_id = _safe_name(str(pointer.get("seriesId", "")))
        manifest_path = self._test_series_folder() / f"{series_id}.json"
        if not manifest_path.is_file():
            return None
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        return document if isinstance(document, dict) else None

    def test_series_overview(self, _payload: Any = None) -> dict[str, Any]:
        if not self.project_root:
            return {"active": None, "items": []}
        folder = self._test_series_folder()
        items = []
        for path in folder.glob("*.json"):
            if path.name == "active.json":
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(document, dict) and document.get("seriesId"):
                    items.append(document)
            except (OSError, json.JSONDecodeError):
                continue
        items.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
        active = self._active_test_series()
        return {"active": active, "items": items}

    def begin_test_series(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        definitions = load_catalog(include_references=False)
        active_codes = list(ACTIVE_CLASS_CODES)
        if any(code not in definitions.definitions for code in active_codes) or definitions.rule_version != "1.4":
            return {"ok": False, "error": "Die wissenschaftliche Testreihe erfordert Regelversion 1.4 und zwölf Merkmale."}
        series_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        name = str(payload.get("name", "")).strip() or f"Wissenschaftliche Testreihe {now[:10]}"
        document = {
            "schemaVersion": "1.0",
            "seriesId": series_id,
            "name": name,
            "category": "scientific_evaluation",
            "createdAt": now,
            "classificationRulesVersion": definitions.rule_version,
            "activeClassCodes": active_codes,
            "activeClassCount": len(active_codes),
        }
        folder = self._test_series_folder()
        with self._lock:
            _atomic_target = folder / f"{series_id}.json"
            _atomic_temp = _atomic_target.with_suffix(".tmp")
            _atomic_temp.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(_atomic_temp, _atomic_target)
            pointer = folder / "active.json"
            pointer_temp = pointer.with_suffix(".tmp")
            pointer_temp.write_text(json.dumps({"seriesId": series_id}, indent=2) + "\n", encoding="utf-8")
            os.replace(pointer_temp, pointer)
        return {"ok": True, "series": document, "results": []}

    def _ensure_test_series_for_plan(
        self, provider: str, model: str,
    ) -> tuple[dict[str, Any], bool]:
        """Reuse a compatible v1.4 series or create a traceable successor."""

        active = self._active_test_series()
        compatible = bool(
            active
            and active.get("classificationRulesVersion") == "1.4"
            and active.get("activeClassCodes") == list(ACTIVE_CLASS_CODES)
        )
        frozen = active.get("frozenConfiguration") if isinstance(active, dict) else None
        if compatible and isinstance(frozen, dict):
            compatible = (
                provider == frozen.get("provider") and model == frozen.get("model")
            )
        if compatible:
            return active, False
        created = self.begin_test_series({
            "name": (
                "Wissenschaftliche Testreihe v1.4 · "
                f"{provider}/{model} · {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')}"
            ),
        })
        if not created.get("ok"):
            raise RuntimeError(created.get("error", "Testreihe konnte nicht angelegt werden."))
        return created["series"], True

    def evaluation_overview(self, _payload: Any = None) -> dict[str, Any]:
        definitions = load_catalog(include_references=False).definitions
        active_classes = []
        for code in ACTIVE_CLASS_CODES:
            definition = definitions[code]
            rule = definition.get("validated_classification_rule", {})
            active_classes.append({
                "code": code,
                "name": definition.get("name", ""),
                "description": rule.get("klassifikationsziel") or definition.get("definition", ""),
                "reviewRequired": code in {"Bu", "AA", "SSA", "KB"},
            })
        variants = []
        for variant_id in VARIANT_IDS:
            try:
                variants.append(inspect_variant(EVALUATION_VARIANTS_DIR, variant_id))
            except Exception as exc:
                variants.append({"variantId": variant_id, "name": f"Variante {variant_id}",
                                 "ready": False, "readiness": "NICHT BEREIT", "errors": [str(exc)]})
        variant1 = variants[0]
        return {
            "ok": True,
            "activeVariant": "1",
            "classes": active_classes,
            "variant1": {
                "status": variant1["readiness"], "referenceImages": 0,
                "referenceDatabaseAccessed": False,
                "description": variant1.get("description", "Referenzfreie Klassifikation"),
                "openPoint": "Variante 1 liest keine Referenzbilder oder Referenzgedächtnisse.",
            },
            "variants": variants,
            "requestedVariants": [{"id": item["variantId"], "status": item["readiness"]}
                                  for item in variants],
            "legacyVariants": [
                {"id": "A", "description": "Nur Testbilder und Klassencodes"},
                {"id": "B", "description": "Testbilder und Klassenbeschreibungen"},
                {"id": "C", "description": "Testbilder und bestätigte Referenzbilder"},
                {"id": "D", "description": "Zweistufige Merkmalsextraktion und Klassifikation"},
            ],
        }

    def scientific_series_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Plan any selected specimens through the scientific evaluation path."""

        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        variant_id = str(payload.get("variant", "1"))
        provider = str(payload.get("provider", "ollama"))
        model = str(payload.get("model", "qwen3-vl:4b-instruct"))
        if variant_id not in VARIANT_IDS:
            return {"ok": False, "error": "Variante muss 1, 2, 2a oder 3 sein."}
        if (provider, model) not in MODEL_CONFIGS:
            return {"ok": False, "error": "Unbekannte Provider-/Modellkombination."}
        try:
            active_series, series_created = self._ensure_test_series_for_plan(provider, model)
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            variant = inspect_variant(EVALUATION_VARIANTS_DIR, variant_id)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        folders = {path.name: path for path in self._specimens()}
        requested = payload.get("specimenIds")
        if requested is None:
            specimen_ids = sorted(folders)
        elif not isinstance(requested, list):
            return {"ok": False, "error": "Die Prüfkörperauswahl ist ungültig."}
        else:
            specimen_ids = list(dict.fromkeys(str(item) for item in requested))
        if not specimen_ids:
            return {"ok": False, "error": "Bitte mindestens einen Prüfkörper auswählen."}
        unknown = [specimen for specimen in specimen_ids if specimen not in folders]
        if unknown:
            return {
                "ok": False,
                "error": f"Prüfkörperordner nicht gefunden: {', '.join(unknown)}.",
            }
        selections: list[dict[str, Any]] = []
        selection_errors: list[dict[str, str]] = []
        frozen_selection = active_series.get("frozenImageSelection")
        for specimen in specimen_ids:
            images: list[dict[str, str]] = []
            source = "nicht_bestaetigt"
            selection_error = ""
            try:
                if isinstance(frozen_selection, dict):
                    stored = frozen_selection.get(specimen)
                    if isinstance(stored, list) and stored:
                        for item in stored:
                            name = str(item.get("filename", "")) if isinstance(item, dict) else ""
                            path = folders[specimen] / name
                            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
                            if not path.is_file() or digest != item.get("sha256"):
                                raise ValueError(
                                    f"fixiertes Bild fehlt oder wurde verändert: {name}."
                                )
                            images.append({"filename": name, "sha256": digest})
                        source = "fixierte_testreihenauswahl"
                    else:
                        selected, source = self._confirmed_image_selection(
                            specimen, folders[specimen]
                        )
                        images = [{
                            "filename": name,
                            "sha256": hashlib.sha256(
                                (folders[specimen] / name).read_bytes()
                            ).hexdigest(),
                        } for name in sorted(selected)]
                else:
                    selected, source = self._confirmed_image_selection(specimen, folders[specimen])
                    images = [{
                        "filename": name,
                        "sha256": hashlib.sha256(
                            (folders[specimen] / name).read_bytes()
                        ).hexdigest(),
                    } for name in sorted(selected)]
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                selection_error = str(exc)
                selection_errors.append({"specimen": specimen, "error": selection_error})
            z30_match = re.fullmatch(r"Z-30-(\d{2})", specimen)
            selections.append({
                "specimen": specimen,
                "normalizedSpecimenId": (
                    f"Z-30-01-{z30_match.group(1)}" if z30_match else specimen
                ),
                "selectionSource": source,
                "selectedImageCount": len(images),
                "images": images,
                **({"selectionError": selection_error} if selection_error else {}),
            })
        selected_hashes = {
            image["sha256"] for item in selections for image in item["images"]
        }
        overlapping_references = [
            item for item in variant.get("references", [])
            if isinstance(item, dict) and item.get("sha256") in selected_hashes
        ]
        self.list_batches()
        active_conflicts = []
        with self._lock:
            for batch in self.batches.values():
                for job in batch.get("jobs", []):
                    if (
                        job.get("specimen") in specimen_ids
                        and job.get("status") in {"running", "queued"}
                    ):
                        active_conflicts.append({
                            "batchId": batch.get("batchId"),
                            "jobId": job.get("jobId"),
                            "specimen": job.get("specimen"),
                            "provider": job.get("provider"),
                            "model": job.get("model"),
                            "status": job.get("status"),
                        })
        catalog = load_catalog(include_references=False)
        context, prediction_budget = MODEL_CONFIGS[(provider, model)]
        provider_image_limit = PROVIDER_MAX_TEST_IMAGES_PER_REQUEST[provider]
        rules_path = WORKSPACE_ROOT / "Datenbank/klassifikationsregeln.json"
        immutable = {
            "seriesId": active_series["seriesId"],
            "variant": variant_id,
            "provider": provider,
            "model": model,
            "modelParameters": {
                "temperature": 0,
                "num_ctx": context,
                "num_predict": prediction_budget,
                "think": False,
                "stream": False,
            },
            "providerImageLimit": provider_image_limit,
            "promptVersion": PROMPT_VERSION,
            "classificationRulesVersion": catalog.rule_version,
            "classificationRulesSha256": hashlib.sha256(rules_path.read_bytes()).hexdigest(),
            "variantManifestSha256": variant.get("manifestHash"),
            "activeClassCodes": list(ACTIVE_CLASS_CODES),
            "specimens": selections,
            "groundTruthIncluded": False,
            "protocolsIncluded": False,
            "previousResultsIncluded": False,
            "expertAssessmentsIncluded": False,
        }
        plan_token = hashlib.sha256(
            json.dumps(immutable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        plan = {
            **immutable,
            "planToken": plan_token,
            "readyToStart": bool(
                variant.get("ready")
                and not selection_errors
                and not overlapping_references
                and not active_conflicts
            ),
            "variantReady": bool(variant.get("ready")),
            "variantReadiness": variant.get("readiness", "NICHT BEREIT"),
            "variantErrors": list(variant.get("errors", [])),
            "seriesCreatedAutomatically": series_created,
            "testSeries": active_series,
            "specimenCount": len(selections),
            "selectedImageCount": sum(item["selectedImageCount"] for item in selections),
            "estimatedPartialRequestCount": sum(
                len(_partition_provider_images(
                    [image["filename"] for image in item["images"]], provider
                ))
                for item in selections
            ),
            "referenceImageCount": 0 if variant_id == "1" else variant.get("referenceCount", 0),
            "missingConfirmations": selection_errors,
            "overlappingReferences": overlapping_references,
            "activeConflicts": active_conflicts,
            "possibleCosts": provider in {"openai", "mistral"},
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self.evaluation_plans[plan_token] = plan
        return {"ok": True, "plan": plan}

    def start_scientific_series(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Freeze a reviewed plan and run its specimens sequentially."""

        token = str(payload.get("planToken", ""))
        with self._lock:
            plan = self.evaluation_plans.get(token)
        if not plan:
            return {"ok": False, "error": "Bitte die Bildauswahl unmittelbar vor dem Start erneut prüfen."}
        if not plan.get("readyToStart"):
            return {"ok": False, "error": "Der geprüfte Plan ist noch nicht startbereit."}
        refreshed = self.scientific_series_plan({
            "variant": plan["variant"], "provider": plan["provider"], "model": plan["model"],
            "specimenIds": [item["specimen"] for item in plan["specimens"]],
        })
        if not refreshed.get("ok") or refreshed["plan"]["planToken"] != token:
            return {"ok": False, "error": "Bildauswahl oder Laufkonfiguration hat sich seit der Prüfung geändert."}
        if plan["provider"] in {"openai", "mistral"} and not bool(payload.get("confirmCloud")):
            return {"ok": False, "needsConfirmation": True, "error": "Der Cloud-Lauf benötigt eine ausdrückliche Kostenfreigabe."}
        self.list_batches()
        for item in plan["specimens"]:
            conflict = self._active_analysis_conflict(
                item["specimen"], plan["provider"], plan["model"]
            )
            if conflict:
                return {
                    "ok": False,
                    "conflictStatus": conflict["status"],
                    "error": self._conflict_error(
                        conflict, item["specimen"], plan["model"]
                    ),
                }
        series_run_id = uuid.uuid4().hex
        run_folder = self._project_data() / "Wissenschaftliche_Laeufe" / series_run_id
        request_limit = PROVIDER_MAX_TEST_IMAGES_PER_REQUEST[plan["provider"]]
        jobs = []
        for item in plan["specimens"]:
            selected_names = [image["filename"] for image in item["images"]]
            chunks = _partition_provider_images(selected_names, plan["provider"])
            jobs.append({
                "jobId": uuid.uuid4().hex,
                "runId": uuid.uuid4().hex,
                "specimen": item["specimen"],
                "normalizedSpecimenId": item["normalizedSpecimenId"],
                "provider": plan["provider"],
                "model": plan["model"],
                "selectedImages": selected_names,
                "selectedImageRecords": item["images"],
                "selectedImageCount": len(selected_names),
                "providerImageLimit": request_limit,
                "partialRequestCount": len(chunks),
                "parts": [{
                    "partId": f"teil-{index:02d}",
                    "runId": uuid.uuid4().hex,
                    "images": names,
                    "imageCount": len(names),
                    "status": "queued",
                    "attemptCount": 0,
                } for index, names in enumerate(chunks, start=1)],
                "status": "queued",
                "stage": "queued",
                "mergeStatus": "ausstehend",
            })
        batch = {
            "schemaVersion": "2.0",
            "kind": "scientific_evaluation_series",
            "batchId": series_run_id,
            "seriesRunId": series_run_id,
            "testSeries": self._active_test_series(),
            "projectId": self.project_name,
            "projectPath": str(self.project_root),
            "projectData": str(self._project_data()),
            "runFolder": str(run_folder),
            "evaluationVariant": plan["variant"],
            "provider": plan["provider"],
            "model": plan["model"],
            "modelParameters": plan["modelParameters"],
            "providerImageLimit": request_limit,
            "maxAutomaticPartRetries": MAX_AUTOMATIC_PART_RETRIES,
            "promptVersion": plan["promptVersion"],
            "classificationRulesVersion": plan["classificationRulesVersion"],
            "classificationRulesSha256": plan["classificationRulesSha256"],
            "variantManifestSha256": plan["variantManifestSha256"],
            "activeClassCodes": plan["activeClassCodes"],
            "groundTruthIncluded": False,
            "protocolsIncluded": False,
            "previousResultsIncluded": False,
            "expertAssessmentsIncluded": False,
            "selectionFrozenAt": datetime.now(timezone.utc).isoformat(),
            "jobCount": len(jobs),
            "cloudRequestCount": (
                sum(job["partialRequestCount"] for job in jobs)
                if plan["provider"] in {"openai", "mistral"} else 0
            ),
            "jobs": jobs,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "confirmCloud": bool(payload.get("confirmCloud")),
            "sequential": True,
            "sessionId": self.session_id,
        }
        run_folder.mkdir(parents=True, exist_ok=False)
        series_path = self._test_series_folder() / f"{plan['seriesId']}.json"
        series = dict(batch["testSeries"] or {})
        frozen_selection = dict(series.get("frozenImageSelection", {}))
        frozen_selection.update({
            item["specimen"]: item["images"] for item in plan["specimens"]
        })
        series["frozenImageSelection"] = frozen_selection
        series["frozenConfiguration"] = {
            key: plan[key] for key in (
                "provider", "model", "modelParameters", "promptVersion",
                "classificationRulesVersion", "classificationRulesSha256",
            )
        }
        temporary = series_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(series, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, series_path)
        batch["testSeries"] = series
        with self._lock:
            self.batches[series_run_id] = batch
            self._write_batch(batch)
            self.evaluation_plans.pop(token, None)
        threading.Thread(target=self._run_batch, args=(series_run_id,), daemon=True).start()
        return {"ok": True, "seriesRunId": series_run_id, "batch": self._batch_snapshot(batch)}

    def scientific_series_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.batch_status({"batchId": payload.get("seriesRunId")})

    def resume_scientific_series(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.resume_batch({
            "batchId": payload.get("seriesRunId"),
            "confirmCloud": payload.get("confirmCloud", False),
            "retryFailed": True,
        })

    def local_model_diagnostics(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return safe, read-only diagnostics for the selected local Qwen run."""

        model = str(payload.get("model", "qwen3-vl:4b-instruct")).strip()
        specimen = str(payload.get("specimen", "")).strip()
        installed: list[dict[str, Any]] = []
        endpoint_error = ""
        try:
            with urlopen("http://localhost:11434/api/tags", timeout=3) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            installed = envelope.get("models", []) if isinstance(envelope, dict) else []
        except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            endpoint_error = str(exc)
        model_entry = next(
            (item for item in installed if isinstance(item, dict) and item.get("name") == model),
            None,
        )
        latest: dict[str, Any] | None = None
        if self.project_root and specimen:
            for record in reversed(self._result_records(specimen)):
                if record.get("provider") != "ollama":
                    continue
                try:
                    document = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                execution = document.get("execution", {})
                result = document.get("result", {})
                output = document.get("model_output", {})
                metadata = execution.get("provider_metadata", {})
                latest = {
                    "runId": document.get("run_id"),
                    "createdAt": document.get("created_at"),
                    "requestedModel": execution.get("requested_model") or execution.get("model"),
                    "actualModel": execution.get("actual_model") or metadata.get("response_model"),
                    "imageCount": len(document.get("inputs", [])),
                    "activeClassCount": document.get("classification_basis", {}).get(
                        "allowed_class_count"
                    ),
                    "promptVersion": document.get("classification_basis", {}).get("prompt_version"),
                    "status": result.get("status"),
                    "analysisOutcome": result.get("analysis_outcome"),
                    "doneReason": metadata.get("done_reason"),
                    "promptEvalCount": metadata.get("prompt_eval_count"),
                    "evalCount": metadata.get("eval_count"),
                    "validationStatus": output.get("validation_status"),
                    "rawResponse": output.get("original_response_text"),
                }
                break
        return {
            "ok": True,
            "endpoint": "http://localhost:11434/api/chat",
            "endpointReachable": not endpoint_error,
            "endpointError": endpoint_error,
            "model": model,
            "modelInstalled": model_entry is not None,
            "modelDigest": model_entry.get("digest") if model_entry else None,
            "capabilities": model_entry.get("capabilities", []) if model_entry else [],
            "selectedImageCount": len(self._selected_image_names(
                specimen, self.project_root / specimen
            )) if self.project_root and specimen and (self.project_root / specimen).is_dir() else 0,
            "activeClassCount": len(ACTIVE_CLASS_CODES),
            "activeClassCodes": list(ACTIVE_CLASS_CODES),
            "latestRun": latest,
        }

    def evaluation_run_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        active_series = self._active_test_series()
        if not active_series:
            return {
                "ok": False,
                "error": "Bitte zuerst mit ‚Neue Testreihe beginnen‘ eine wissenschaftliche Testreihe anlegen.",
            }
        try:
            specimen = _safe_name(str(payload.get("specimen", "")))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        folder = (self.project_root / specimen).resolve()
        if folder.parent != self.project_root.resolve() or not folder.is_dir():
            return {"ok": False, "error": "Prüfkörper wurde nicht gefunden."}
        variant_id = str(payload.get("variant", "1"))
        provider = str(payload.get("provider", "ollama"))
        model = str(payload.get("model", "qwen3-vl:4b-instruct"))
        if variant_id not in VARIANT_IDS:
            return {"ok": False, "error": "Variante muss 1, 2, 2a oder 3 sein."}
        if (provider, model) not in MODEL_CONFIGS:
            return {"ok": False, "error": "Unbekannte Provider-/Modellkombination."}
        try:
            variant = inspect_variant(EVALUATION_VARIANTS_DIR, variant_id)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        definitions = load_catalog(include_references=False).definitions
        selected_images = sorted(self._selected_image_names(specimen, folder))
        if not selected_images:
            return {"ok": False, "error": "Bitte mindestens ein Analysebild auswählen."}
        classes = []
        image_records = [{"filename": name, "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()}
                         for name in selected_images]
        for code in ACTIVE_CLASS_CODES:
            definition = definitions[code]
            rule = definition.get("validated_classification_rule", {})
            classes.append({
                "code": code,
                "name": rule.get("name") or definition.get("name", ""),
                "description": rule.get("klassifikationsziel") or definition.get("definition", ""),
                "reviewStatus": (
                    rule.get("fachlicher_pruefstatus", {}).get("status", "")
                    if rule else ("Fachlich zu prüfen" if code in {"Bu", "AA", "SSA", "KB"} else "Definition vorgegeben")
                ),
            })
        plan_token = control_token(report=variant, specimen=specimen, images=image_records,
                                   provider=provider, model=model,
                                   test_series_id=active_series["seriesId"])
        memory = ({"prepared": True, "status": "NICHT ERFORDERLICH", "path": None}
                  if variant_id == "1" else memory_status(
                      EVALUATION_MEMORY_DIR, report=variant, provider=provider, model=model,
                      prompt_version=EVALUATION_PROMPT_VERSION))
        plan = {
            "variant": variant_id,
            "specimen": specimen,
            "images": selected_images,
            "imageCount": len(selected_images),
            "provider": provider,
            "model": model,
            "classes": classes,
            "referenceImages": variant.get("referenceCount", 0),
            "referenceBatches": variant.get("referenceBatchCount", 0),
            "referenceDatabaseAccessed": variant_id != "1",
            "countsByClass": variant.get("countsByClass", {}),
            "missingByClass": variant.get("missingByClass", {}),
            "descriptionsComplete": variant.get("descriptionsComplete", True),
            "sourceLinksComplete": variant.get("sourceLinksComplete", True),
            "reviewRequired": variant.get("reviewRequired", False),
            "reviewComplete": variant.get("reviewComplete", True),
            "annotatedImages": variant.get("referenceCount", 0) if variant_id == "3" else 0,
            "structureReady": variant.get("ready", False),
            "readiness": variant.get("readiness"),
            "errors": variant.get("errors", []),
            "referenceRoot": variant.get("referenceRoot"),
            "memory": memory,
            "expectedModelCalls": variant.get("modelCallCount", 1),
            "expectedCloudCalls": variant.get("cloudCallCount", 1) if provider in {"openai", "mistral"} else 0,
            "groundTruthIncluded": False,
            "previousResultsIncluded": False,
            "assessmentsIncluded": False,
            "testSeries": active_series,
            "planToken": plan_token,
        }
        with self._lock:
            self.evaluation_plans[plan_token] = {
                "specimen": specimen,
                "images": selected_images,
                "provider": provider,
                "model": model,
                "variant": variant_id,
                "manifestHash": variant.get("manifestHash"),
                "memoryPath": memory.get("path") if memory.get("prepared") else None,
                "classCodes": list(ACTIVE_CLASS_CODES),
                "seriesId": active_series["seriesId"],
            }
        return {"ok": True, "plan": plan}

    def prepare_evaluation_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Start the explicitly requested local preparation without blocking pywebview."""

        variant_id = str(payload.get("variant", ""))
        provider = str(payload.get("provider", "ollama"))
        model = str(payload.get("model", "qwen3-vl:4b-instruct"))
        if variant_id not in {"2", "2a", "3"}:
            return {"ok": False, "error": "Nur Variante 2, 2a und 3 benötigen ein Referenzgedächtnis."}
        if (provider, model) not in MODEL_CONFIGS:
            return {"ok": False, "error": "Unbekannte Provider-/Modellkombination."}
        if provider != "ollama":
            return {"ok": False, "error": (
                "Cloud-Referenzvorbereitung ist aus Sicherheitsgründen noch gesperrt. "
                "Es wurde kein API-Aufruf ausgeführt."
            )}
        try:
            report = inspect_variant(EVALUATION_VARIANTS_DIR, variant_id)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        if not report.get("ready"):
            return {"ok": False, "error": "Die Referenzstruktur ist NICHT BEREIT."}
        job_id = uuid.uuid4().hex
        self.jobs[job_id] = {
            "state": "running", "stage": "preparing_reference_memory",
            "status": "Referenzwissen wird vorbereitet", "provider": provider,
            "model": model, "startedAt": datetime.now(timezone.utc).isoformat(),
        }

        def analyze_batch(batch: Any, index: int) -> dict[str, Any]:
            messages: list[dict[str, Any]] = [{
                "role": "system",
                "content": (
                    "Erzeuge ausschließlich ein knappes JSON-Referenzgedächtnis zu sichtbaren "
                    "Holzmerkmalen. Keine Prüfkörperbewertung und keine Ursachenbehauptung."
                ),
            }]
            root = Path(str(report["referenceRoot"]))
            for item in batch:
                path = root / item["relative_path"]
                text = f"Referenz {item['reference_id']}; Klasse {item['code']}."
                if item.get("description"):
                    text += f" Fachlich geprüfte Beschreibung: {item['description']}"
                messages.append({"role": "user", "content": text,
                                 "images": [base64.b64encode(path.read_bytes()).decode("ascii")]})
            messages.append({"role": "user", "content": (
                "Fasse nur klassenbezogene sichtbare Muster dieses Batches als JSON mit den "
                "Feldern batch, classes und visual_patterns zusammen."
            )})
            request = Request("http://localhost:11434/api/chat", data=json.dumps({
                "model": model, "messages": messages, "stream": False, "think": False,
                "format": "json", "options": {"temperature": 0, "num_ctx": 24576,
                                                  "num_predict": 4096},
            }).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=900) as response:
                response_document = json.loads(response.read().decode("utf-8"))
            content = response_document.get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(f"Ollama lieferte für Referenzbatch {index} keinen Text.")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw_text": content}
            return {"batch": index, "response": parsed,
                    "done_reason": response_document.get("done_reason"),
                    "model": response_document.get("model") or model}

        def worker() -> None:
            try:
                destination = write_reference_memory(
                    EVALUATION_MEMORY_DIR, report=report, provider=provider, model=model,
                    prompt_version=EVALUATION_PROMPT_VERSION, analyze_batch=analyze_batch,
                )
                with self._lock:
                    self.jobs[job_id].update(state="complete", stage="complete",
                                             status="Referenzwissen ist vorbereitet.",
                                             path=str(destination), finishedAt=datetime.now(timezone.utc).isoformat())
            except Exception as exc:
                with self._lock:
                    self.jobs[job_id].update(state="error", stage="error",
                                             status="Referenzvorbereitung fehlgeschlagen.",
                                             error=str(exc), finishedAt=datetime.now(timezone.utc).isoformat())

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "jobId": job_id}

    def reference_catalog(self, _payload: Any = None) -> dict[str, Any]:
        try:
            catalog = load_catalog(include_references=False)
            variant2 = inspect_variant(EVALUATION_VARIANTS_DIR, "2")
            variant2a = inspect_variant(EVALUATION_VARIANTS_DIR, "2a")
            variant3 = inspect_variant(EVALUATION_VARIANTS_DIR, "3")
            inventory = reference_inventory_preview(EVALUATION_VARIANTS_DIR)
            def images_for(report: dict[str, Any], variant_id: str) -> list[dict[str, Any]]:
                root = Path(str(report["referenceRoot"]))
                stored = {
                    str(item["reference_id"]): item
                    for item in inventory["variants"][variant_id]["inventory"]
                }
                duplicate_lookup = {
                    str(item["duplicate"]): str(item["first"])
                    for item in report.get("duplicateHashes", [])
                }
                result = []
                for item in report.get("references", []):
                    path = (root / str(item["relative_path"])).resolve()
                    if not path.is_file() or not path.is_relative_to(root):
                        continue
                    metadata = stored.get(str(item["reference_id"]), {})
                    result.append({
                        **item, "variant": variant_id, "name": path.name,
                        "src": self._image_data(path, edge=720), "absolutePath": str(path),
                        "confirmed": bool(item.get("confirmed", metadata.get("confirmed", False))),
                        "description": str(item.get("description", metadata.get("description", ""))),
                        "duplicateOf": duplicate_lookup.get(str(item["reference_id"])),
                    })
                return result

            images2 = images_for(variant2, "2")
            images2a = images_for(variant2a, "2a")
            images3 = images_for(variant3, "3")
            source_images = {str(item["reference_id"]): item for item in images2a}
            for annotation in images3:
                source = source_images.get(str(annotation.get("source_reference_id", "")))
                annotation["sourceImage"] = ({
                    "name": source["name"], "src": source["src"],
                    "reference_id": source["reference_id"], "sha256": source["sha256"],
                    "relative_path": source["relative_path"],
                } if source else None)
            missing_descriptions = set(variant3.get("missingDescriptions", []))
            unknown_sources = set(variant3.get("unknownSourceReferenceIds", []))
            unreviewed = set(variant3.get("unreviewedReferenceIds", []))
            classes = []
            for code in ACTIVE_CLASS_CODES:
                definition = catalog.definitions[code]
                rule = definition.get("validated_classification_rule", {})
                class2 = [item for item in images2 if item["code"] == code]
                class2a = [item for item in images2a if item["code"] == code]
                class3 = [item for item in images3 if item["code"] == code]
                def quantity_ready(report: dict[str, Any], count: int) -> bool:
                    requirement = report["featureRequirements"][code]
                    minimum = requirement.get("minimum") or 0
                    maximum = requirement.get("maximum")
                    return count >= minimum and (maximum is None or count <= maximum)
                ready2 = quantity_ready(variant2, len(class2))
                ready2a = quantity_ready(variant2a, len(class2a))
                ready3 = quantity_ready(variant3, len(class3)) and not any(
                    item["reference_id"] in missing_descriptions
                    or item.get("source_reference_id") in unknown_sources
                    or (variant3.get("reviewRequired") and item["reference_id"] in unreviewed)
                    for item in class3
                )
                classes.append({
                    "code": code, "name": definition.get("name", ""),
                    "definition": rule.get("klassifikationsziel") or definition.get("definition", ""),
                    "reviewRequired": code in {"Bu", "AA", "SSA", "KB"},
                    "folderVariant2": str(Path(str(variant2["referenceRoot"])) / CLASS_DIRECTORIES[code]),
                    "folderVariant2a": str(Path(str(variant2a["referenceRoot"])) / CLASS_DIRECTORIES[code]),
                    "folderVariant3": str(Path(str(variant3["referenceRoot"])) / CLASS_DIRECTORIES[code]),
                    "countVariant2": len(class2), "countVariant2a": len(class2a),
                    "countVariant3": len(class3),
                    "requirements": {
                        "2": variant2["featureRequirements"][code],
                        "2a": variant2a["featureRequirements"][code],
                        "3": variant3["featureRequirements"][code],
                    },
                    "readiness": {"2": ready2, "2a": ready2a, "3": ready3},
                    "variant3DescriptionStatus": {
                        "complete": bool(class3) and not any(item["reference_id"] in missing_descriptions for item in class3),
                        "missingCount": sum(item["reference_id"] in missing_descriptions for item in class3),
                        "totalCount": len(class3),
                    },
                    "variant3SourceStatus": {
                        "complete": bool(class3) and not any(item.get("source_reference_id") in unknown_sources for item in class3),
                    },
                    "variant3ReviewStatus": {
                        "required": bool(variant3.get("reviewRequired")),
                        "complete": bool(class3) and not any(item["reference_id"] in unreviewed for item in class3),
                    },
                    "ready": ready2 and ready2a and ready3,
                    "imagesVariant2": class2, "imagesVariant2a": class2a, "imagesVariant3": class3,
                })
            return {
                "ok": True, "configured": True, "activeCodes": list(ACTIVE_CLASS_CODES),
                "classes": classes,
                "variantStatus": {"2": variant2, "2a": variant2a, "3": variant3},
                "refreshPreview": inventory, "storageRoot": str(EVALUATION_VARIANTS_DIR),
                "note": (
                    "Variante 3 verwendet ausschließlich manuell annotierte Kopien und verknüpft "
                    "sie über source_reference_id mit unveränderten Originalen aus Variante 2a."
                ),
            }
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            return {"ok": False, "error": f"Referenzdatenbank kann nicht gelesen werden: {exc}"}

    def open_evaluation_reference_folder(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Open an existing allow-listed folder without modifying its contents."""
        variant = str(payload.get("variant", ""))
        code = str(payload.get("code", ""))
        if variant not in {"2", "2a", "3"} or code not in ACTIVE_CLASS_CODES:
            return {"ok": False, "error": "Referenzordner ist nicht freigegeben."}
        root = (EVALUATION_VARIANTS_DIR / f"Variante_{variant}").resolve()
        folder = (root / CLASS_DIRECTORIES[code]).resolve()
        if folder.parent != root or not folder.is_dir():
            return {"ok": False, "error": "Referenzordner wurde nicht gefunden."}
        try:
            subprocess.run(["open", str(folder)], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            return {"ok": False, "error": f"Ordner konnte nicht im Finder geöffnet werden: {exc}"}
        return {"ok": True, "path": str(folder)}

    def open_variant3_descriptions(self, _payload: Any = None) -> dict[str, Any]:
        """Open only the allow-listed annotation description table for manual editing."""

        root = (EVALUATION_VARIANTS_DIR / "Variante_3").resolve()
        path = (root / "descriptions.csv").resolve()
        if path.parent != root or not path.is_file():
            return {"ok": False, "error": "descriptions.csv wurde nicht gefunden."}
        try:
            subprocess.run(["open", str(path)], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            return {"ok": False, "error": f"Beschreibungsdatei konnte nicht geöffnet werden: {exc}"}
        return {"ok": True, "path": str(path)}

    def refresh_evaluation_reference_database(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            update = update_reference_inventories(
                EVALUATION_VARIANTS_DIR,
                confirmed=bool((payload or {}).get("confirmManifestUpdate", False)),
            )
            catalog = self.reference_catalog()
            return {"ok": True, "update": update, "catalog": catalog}
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            return {"ok": False, "error": f"Referenzdatenbank konnte nicht aktualisiert werden: {exc}"}

    def save_active_reference_codes(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            codes = payload.get("codes", [])
            if not isinstance(codes, list):
                raise ValueError("Die Klassenauswahl ist ungültig.")
            self._reference_store().save_active_codes(
                [str(code) for code in codes],
                confirm_replace=bool(payload.get("confirmReplace")),
            )
            return self.reference_catalog()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc)}

    def import_reference_images(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            paths = payload.get("paths", [])
            if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                raise ValueError("Die ausgewählten Dateipfade sind ungültig.")
            result = self._reference_store().import_images(
                code=str(payload.get("code", "")),
                paths=[Path(path) for path in paths],
                specimen_id=str(payload.get("specimenId", "")),
                description=str(payload.get("description", "")),
                confirmed=bool(payload.get("confirmed")),
            )
            return {**result, "catalog": self.reference_catalog()}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc)}

    def choose_reference_images(self, payload: dict[str, Any]) -> dict[str, Any]:
        import webview

        chosen = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=("Bilder (*.jpg;*.jpeg;*.png;*.webp)",),
        )
        if not chosen:
            return {"ok": False, "cancelled": True}
        return self.import_reference_images({**payload, "paths": list(chosen)})

    def update_reference_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self._reference_store().update_entry(
                str(payload.get("id", "")),
                code=str(payload.get("code", "")),
                specimen_id=str(payload.get("specimenId", "")),
                description=str(payload.get("description", "")),
                confirmed=bool(payload.get("confirmed")),
                confirm_confirmed_change=bool(payload.get("confirmConfirmedChange")),
            )
            return {**result, "catalog": self.reference_catalog()}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc)}

    def result_detail(self, payload: dict[str, Any]) -> dict[str, Any]:
        found = self._result_by_run_id(str(payload.get("runId", "")))
        if not found:
            return {"ok": False, "error": "Analyselauf wurde nicht gefunden."}
        path, document = found
        run_id = self._run_id(document, path)
        specimen = _safe_name(str(document.get("specimen_id", "")))
        specimen_folder = (self.project_root / specimen).resolve() if self.project_root else Path()
        run_images: list[dict[str, str]] = []
        if (
            self.project_root
            and specimen_folder.parent == self.project_root.resolve()
            and specimen_folder.is_dir()
        ):
            available = {image.name: image for image in self._all_images(specimen_folder)}
            for item in document.get("inputs", []):
                filename = item.get("filename") if isinstance(item, dict) else None
                image = available.get(filename) if isinstance(filename, str) else None
                if image is not None:
                    run_images.append({"name": image.name, "src": self._image_data(image)})
        return {"ok": True, "runId": run_id, "path": str(path), "result": document,
                "displayFeatures": _result_features(document.get("result", {})),
                "assessment": self._assessment(run_id), "legacyAssessment": self._legacy_assessment(run_id),
                "runImages": run_images}

    def save_image_selection(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        try:
            specimen = _safe_name(str(payload.get("specimen", "")))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        folder = (self.project_root / specimen).resolve()
        if folder.parent != self.project_root.resolve() or not folder.is_dir():
            return {"ok": False, "error": "Prüfkörper wurde nicht gefunden."}
        names = payload.get("selectedImages")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            return {"ok": False, "error": "Die Bildauswahl ist ungültig."}
        if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
            return {"ok": False, "error": "Die Bildauswahl enthält ungültige oder doppelte Dateinamen."}
        available = {path.name for path in self._images(folder)}
        if not set(names) <= available:
            return {"ok": False, "error": "Die Bildauswahl enthält nicht zugeordnete Dateien."}
        self._write_image_selection(specimen, set(names), available, customized=True)
        return {"ok": True, "selectedImages": sorted(names)}

    def create_specimen(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        try:
            specimen = _safe_name(str(payload.get("specimen", "")))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        folder = self.project_root / specimen
        if folder.exists():
            return {"ok": False, "error": "Diese Prüfkörpernummer existiert bereits."}
        folder.mkdir()
        self._save_manifest()
        return {"ok": True, "project": self._snapshot(), "specimen": specimen}

    def import_images(self, payload: dict[str, Any]) -> dict[str, Any]:
        import webview
        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        specimen = _safe_name(str(payload.get("specimen", "")))
        folder = self.project_root / specimen
        folder.mkdir(exist_ok=True)
        chosen = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True,
                                                       file_types=("Bilder (*.jpg;*.jpeg;*.png;*.webp)",))
        if not chosen:
            return {"ok": False, "cancelled": True}
        imported = []
        for raw in chosen:
            source = Path(raw)
            target = folder / source.name
            counter = 2
            while target.exists():
                target = folder / f"{source.stem}_{counter}{source.suffix}"
                counter += 1
            shutil.copy2(source, target)
            imported.append(target.name)
        return {"ok": True, "imported": imported}

    def _active_analysis_conflict(
        self, specimen: str, provider: str, model: str, *, ignore_batch_id: str = ""
    ) -> dict[str, Any] | None:
        for existing_id, job in self.jobs.items():
            if (job.get("state") == "running" and job.get("specimen") == specimen
                    and job.get("provider") == provider and job.get("model") == model):
                return {"kind": "single", "id": existing_id, "status": "running", "job": job}
        for batch in self.batches.values():
            if batch.get("batchId") == ignore_batch_id:
                continue
            for job in batch.get("jobs", []):
                if (job.get("status") in {"running", "queued"}
                        and job.get("specimen") == specimen
                        and job.get("provider") == provider and job.get("model") == model):
                    return {
                        "kind": "batch", "id": batch.get("batchId", ""),
                        "status": job.get("status"), "job": job,
                    }
        return None

    @staticmethod
    def _conflict_error(conflict: dict[str, Any], specimen: str, model: str) -> str:
        status = conflict["status"]
        state = "läuft tatsächlich" if status == "running" else "wartet auf Ausführung"
        kind = "Einzelanalyse" if conflict["kind"] == "single" else "Batch-Auftrag"
        started = conflict["job"].get("startedAt")
        suffix = f" Startzeit: {started}." if started else ""
        article = "eine" if conflict["kind"] == "single" else "ein"
        return (
            f"Für {specimen} {state} bereits {article} {kind} mit {model}. "
            f"Der vorhandene Auftrag wird nicht dupliziert.{suffix}"
        )

    def start_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        try:
            specimen = _safe_name(str(payload.get("specimen", "")))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        provider = str(payload.get("provider", "ollama")).strip()
        model = str(payload.get("model", "qwen3-vl:4b-instruct")).strip()
        legacy_choices = {
            "qwen": ("ollama", "qwen3-vl:4b-instruct"),
            "openai": ("openai", "gpt-6.1-sol"),
            "mistral": ("mistral", "ministral-3b-2512"),
        }
        if model in legacy_choices and "provider" not in payload:
            provider, model = legacy_choices[model]
        confirm = bool(payload.get("confirmPaid"))
        specimen_folder = (self.project_root / specimen).resolve()
        if specimen_folder.parent != self.project_root.resolve() or not specimen_folder.is_dir():
            return {"ok": False, "error": "Prüfkörper wurde nicht gefunden."}
        selected_images = self._selected_image_names(specimen, specimen_folder)
        if not selected_images:
            return {"ok": False, "error": "Bitte wählen Sie mindestens ein Analysebild aus."}
        if (provider, model) not in MODEL_CONFIGS:
            return {"ok": False, "error": f"Unbekannte Provider-/Modellkombination: {provider} / {model}"}
        context, prediction_budget = MODEL_CONFIGS[(provider, model)]
        evaluation_variant = str(payload.get("evaluationVariant", "")).strip() or None
        test_series: dict[str, Any] | None = None
        allowed_class_codes = payload.get("allowedClassCodes")
        if allowed_class_codes is None:
            allowed_class_codes = list(ACTIVE_CLASS_CODES)
        evaluation_memory_path: Path | None = None
        if evaluation_variant in VARIANT_IDS:
            test_series = self._active_test_series()
            if not test_series:
                return {"ok": False, "error": "Bitte zuerst eine wissenschaftliche Testreihe beginnen."}
            try:
                variant_report = inspect_variant(EVALUATION_VARIANTS_DIR, evaluation_variant)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            if not variant_report.get("ready"):
                missing = list((variant_report.get("missingByClass") or {}).keys())
                details: list[str] = []
                if missing:
                    details.append(
                        "Für folgende Merkmale fehlen Referenzbilder: " + ", ".join(missing) + "."
                    )
                details.extend(str(item) for item in variant_report.get("errors", []) if item)
                suffix = " " + " ".join(details) if details else ""
                return {
                    "ok": False,
                    "error": (
                        f"Variante {evaluation_variant} ist noch nicht bereit."
                        f"{suffix} Es wurde kein Modellaufruf gestartet."
                    ),
                }
            allowed_class_codes = list(ACTIVE_CLASS_CODES)
            plan_token = str(payload.get("evaluationPlanToken", ""))
            with self._lock:
                planned = self.evaluation_plans.get(plan_token)
            expected = {
                "specimen": specimen,
                "images": sorted(selected_images),
                "provider": provider,
                "model": model,
                "variant": evaluation_variant,
                "manifestHash": variant_report.get("manifestHash"),
                "memoryPath": planned.get("memoryPath") if planned else None,
                "classCodes": allowed_class_codes,
                "seriesId": test_series["seriesId"],
            }
            if not planned or planned != expected:
                return {
                    "ok": False,
                    "error": (
                        "Bitte unmittelbar vor dem Pilotlauf eine aktuelle "
                        "Kontrollübersicht erstellen und prüfen."
                    ),
                }
            if evaluation_variant != "1":
                if not planned.get("memoryPath"):
                    return {"ok": False, "error": "Bitte zuerst das Referenzgedächtnis vorbereiten."}
                evaluation_memory_path = Path(planned["memoryPath"])
            with self._lock:
                self.evaluation_plans.pop(plan_token, None)
        if provider in {"openai", "mistral"} and not confirm:
            return {"ok": False, "needsConfirmation": True,
                    "error": f"{provider.title()} erfordert eine ausdrückliche Kostenbestätigung."}
        job_id = uuid.uuid4().hex
        run_id = uuid.uuid4().hex
        with self._lock:
            conflict = self._active_analysis_conflict(specimen, provider, model)
            if conflict:
                return {
                    "ok": False,
                    "jobId": conflict["id"] if conflict["kind"] == "single" else None,
                    "batchId": conflict["id"] if conflict["kind"] == "batch" else None,
                    "conflictStatus": conflict["status"],
                    "error": self._conflict_error(conflict, specimen, model),
                }
            self.jobs[job_id] = {
                "state": "running", "stage": "model_loading", "status": "Modell wird geladen",
                "provider": provider, "model": model, "specimen": specimen,
                "startedAt": datetime.now(timezone.utc).isoformat(),
            }

        def worker() -> None:
            stages = {
                "Modell wird geladen": "model_loading",
                "Bilder werden analysiert": "analyzing_images",
                "Ergebnis wird verarbeitet": "processing_result",
            }

            def progress(status: str) -> None:
                with self._lock:
                    self.jobs[job_id].update(status=status, stage=stages[status])

            try:
                destination = run_analysis(self.project_root / specimen, provider=provider, model=model,
                                           live=provider in {"openai", "mistral"},
                                           confirm_paid_request=confirm,
                                           ollama_num_ctx=context,
                                           ollama_num_predict=prediction_budget,
                                           max_references_per_class=1,
                                           allowed_input_root=self.project_root,
                                           selected_image_names=sorted(selected_images),
                                           output_dir=self._project_data() / "Ergebnisse",
                                           progress_callback=progress,
                                           run_id=run_id, project_id=self.project_name,
                                           evaluation_variant=evaluation_variant,
                                           allowed_class_codes=allowed_class_codes,
                                           test_series={
                                               "series_id": test_series["seriesId"],
                                               "name": test_series["name"],
                                               "category": test_series["category"],
                                               "classification_rules_version": test_series["classificationRulesVersion"],
                                               "active_class_codes": test_series["activeClassCodes"],
                                           } if test_series else None,
                                           evaluation_memory_path=evaluation_memory_path,
                                           evaluation_variants_dir=EVALUATION_VARIANTS_DIR)
                with self._lock:
                    self.jobs[job_id].update(
                        state="complete", stage="complete", status="Analyse abgeschlossen.",
                        path=str(destination), result=json.loads(destination.read_text()),
                        finishedAt=datetime.now(timezone.utc).isoformat(),
                    )
            except Exception as exc:
                failure_path = self._write_technical_failure(
                    run_id=run_id,
                    specimen=specimen,
                    provider=provider,
                    model=model,
                    selected_images=sorted(selected_images),
                    error=exc,
                )
                with self._lock:
                    self.jobs[job_id].update(
                        state="error", stage="error", status="Analyse fehlgeschlagen.",
                        error=str(exc) or repr(exc),
                        failureType=getattr(exc, "failure_type", "technical_execution"),
                        scientificResult=False,
                        failureDiagnostics=getattr(exc, "diagnostics", {}),
                        failureRecordPath=str(failure_path),
                        finishedAt=datetime.now(timezone.utc).isoformat(),
                    )

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "jobId": job_id}

    def analysis_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            job = self.jobs.get(str(payload.get("jobId", "")))
            snapshot = dict(job) if job else None
        return {"ok": bool(snapshot), **(snapshot or {"error": "Analyseauftrag nicht gefunden."})}

    def _planned_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.project_root:
            raise RuntimeError("Kein Projekt geöffnet.")
        specimen_ids = payload.get("specimenIds")
        model_items = payload.get("models")
        if not isinstance(specimen_ids, list) or not specimen_ids:
            raise ValueError("Bitte mindestens einen Prüfkörper auswählen.")
        if not isinstance(model_items, list) or not model_items:
            raise ValueError("Bitte mindestens ein Modell auswählen.")
        available_specimens = {path.name: path for path in self._specimens()}
        specimens = []
        seen_specimens: set[str] = set()
        for raw in specimen_ids:
            specimen = _safe_name(str(raw))
            if specimen not in available_specimens:
                raise ValueError(f"Prüfkörper wurde nicht gefunden: {specimen}")
            if specimen in seen_specimens:
                continue
            seen_specimens.add(specimen)
            selected = sorted(self._selected_image_names(specimen, available_specimens[specimen]))
            if not selected:
                raise ValueError(f"Für {specimen} ist kein Analysebild ausgewählt.")
            specimens.append((specimen, selected))
        models: list[tuple[str, str]] = []
        for item in model_items:
            if not isinstance(item, dict):
                raise ValueError("Die Modellauswahl ist ungültig.")
            key = (str(item.get("provider", "")), str(item.get("model", "")))
            if key not in MODEL_CONFIGS:
                raise ValueError(f"Unbekannte Provider-/Modellkombination: {key[0]} / {key[1]}")
            if key not in models:
                models.append(key)
        jobs: list[dict[str, Any]] = []
        cloud_requests = 0
        for specimen, selected in specimens:
            for provider, model in models:
                planned_requests = len(selected) if provider == "mistral" else 1
                if provider in {"openai", "mistral"}:
                    cloud_requests += planned_requests
                jobs.append({
                    "jobId": uuid.uuid4().hex,
                    "runId": uuid.uuid4().hex,
                    "specimen": specimen,
                    "provider": provider,
                    "model": model,
                    "selectedImages": selected,
                    "plannedRequestCount": planned_requests,
                    "status": "queued",
                    "stage": "queued",
                })
        return {
            "projectId": self.project_name,
            "projectPath": str(self.project_root),
            "projectData": str(self._project_data()),
            "jobCount": len(jobs),
            "cloudRequestCount": cloud_requests,
            "jobs": jobs,
        }

    def plan_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "plan": self._planned_batch(payload)}
        except (RuntimeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    def _batch_path(self, batch_id: str) -> Path:
        folder = self._project_data() / "Batches"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{_safe_name(batch_id)}.json"

    def _write_batch(self, batch: dict[str, Any]) -> None:
        batch["updatedAt"] = datetime.now(timezone.utc).isoformat()
        target = Path(batch["projectData"]) / "Batches" / f"{batch['batchId']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
        if batch.get("kind") == "scientific_evaluation_series" and batch.get("runFolder"):
            manifest = Path(batch["runFolder"]) / "run_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest_temp = manifest.with_suffix(".tmp")
            manifest_temp.write_text(
                json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(manifest_temp, manifest)

    @staticmethod
    def _batch_snapshot(batch: dict[str, Any]) -> dict[str, Any]:
        """Return the display view without discarding the persisted batch plan."""

        snapshot = json.loads(json.dumps(batch))
        dismissed = set(snapshot.get("dismissedJobIds", []))
        all_jobs = snapshot.get("jobs", [])
        snapshot["jobs"] = [job for job in all_jobs if job.get("jobId") not in dismissed]
        snapshot["dismissedJobCount"] = len(all_jobs) - len(snapshot["jobs"])
        return snapshot

    @staticmethod
    def _refresh_batch_status(batch: dict[str, Any]) -> None:
        statuses = {job.get("status") for job in batch.get("jobs", [])}
        if "paused" in statuses:
            batch["status"] = "paused"
        elif "running" in statuses or "queued" in statuses:
            batch["status"] = "running"
        elif "interrupted" in statuses:
            batch["status"] = "interrupted"
        elif "failed" in statuses or "partially_succeeded" in statuses:
            batch["status"] = "completed_with_errors"
        else:
            batch["status"] = "completed"

    def _merge_scientific_job(self, batch: dict[str, Any], job: dict[str, Any]) -> Path | None:
        successful = [part for part in job.get("parts", []) if part.get("status") == "succeeded"]
        if not successful:
            return None
        documents = [
            json.loads(Path(part["resultPath"]).read_text(encoding="utf-8"))
            for part in successful
        ]
        complete = len(successful) == len(job.get("parts", []))
        merged_result = merge_partial_classifications(
            [document.get("result", {}) for document in documents], complete=complete
        )
        merge_folder = Path(batch["runFolder"]) / "Zusammenfuehrung"
        merge_folder.mkdir(parents=True, exist_ok=True)
        summary_path = merge_folder / f"{job['specimen']}_{job['runId']}.json"
        summary = {
            "schemaVersion": "1.0",
            "specimenId": job["specimen"],
            "runId": job["runId"],
            "selectedImageCount": job["selectedImageCount"],
            "partialRequestCount": job["partialRequestCount"],
            "successfulPartialRequestCount": len(successful),
            "model": job["model"],
            "provider": job["provider"],
            "mergeStatus": "vollstaendig" if complete else "teilweise",
            "parts": [{
                "partId": part["partId"], "images": part["images"],
                "status": part["status"], "attemptCount": part.get("attemptCount", 0),
                "resultPath": part.get("resultPath"), "error": part.get("error"),
            } for part in job["parts"]],
            "result": merged_result,
        }
        temporary = summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, summary_path)
        job["partialMergePath"] = str(summary_path)
        job["mergeStatus"] = summary["mergeStatus"]
        job["mergeConflictCount"] = len(merged_result.get("merge_conflicts", []))
        if not complete:
            return None
        document = dict(documents[0])
        document["run_id"] = job["runId"]
        document["created_at"] = datetime.now(timezone.utc).isoformat()
        document["inputs"] = list(job["selectedImageRecords"])
        document["image_selection"] = {
            "included": list(job["selectedImages"]), "excluded": [],
        }
        document["result"] = merged_result
        execution = dict(document.get("execution", {}))
        execution["duration_seconds"] = round(sum(
            float(item.get("execution", {}).get("duration_seconds", 0) or 0)
            for item in documents
        ), 3)
        metadata = dict(execution.get("provider_metadata", {}))
        metadata["partial_requests"] = [{
            "part_id": part["partId"], "images": part["images"],
            "attempt_count": part.get("attemptCount", 0), "result_path": part["resultPath"],
        } for part in successful]
        execution["provider_metadata"] = metadata
        document["execution"] = execution
        document["series_processing"] = {
            "selected_image_count": job["selectedImageCount"],
            "partial_request_count": job["partialRequestCount"],
            "provider_image_limit": job.get("providerImageLimit"),
            "merge_status": "vollstaendig",
            "merge_conflict_count": job["mergeConflictCount"],
        }
        result_folder = Path(batch["runFolder"]) / "Ergebnisse"
        result_folder.mkdir(parents=True, exist_ok=True)
        target = result_folder / f"{job['specimen']}_{job['provider']}_{job['model'].replace(':', '_')}_{job['runId']}.json"
        if target.exists():
            raise RuntimeError(f"Vollständiges Ergebnis existiert bereits und wird nicht überschrieben: {target}")
        target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target

    def _run_scientific_job(self, batch: dict[str, Any], job: dict[str, Any]) -> str:
        provider = job["provider"]
        model = job["model"]
        context, prediction_budget = MODEL_CONFIGS[(provider, model)]
        for part in job.get("parts", []):
            if part.get("status") == "succeeded":
                continue
            if job.get("cancellationRequested") or batch.get("cancellationRequested"):
                part.update(status="cancelled", error=None)
                continue
            while True:
                part["attemptCount"] = int(part.get("attemptCount", 0)) + 1
                part.update(
                    status="running", error=None,
                    startedAt=datetime.now(timezone.utc).isoformat(),
                )
                job.update(stage="analyzing_images", statusText=f"{part['partId']} wird analysiert")
                with self._lock:
                    self._write_batch(batch)
                try:
                    destination = run_analysis(
                        Path(batch["projectPath"]) / job["specimen"],
                        provider=provider, model=model,
                        live=provider in {"openai", "mistral"},
                        confirm_paid_request=batch["confirmCloud"],
                        ollama_num_ctx=context, ollama_num_predict=prediction_budget,
                        max_references_per_class=1,
                        allowed_input_root=Path(batch["projectPath"]),
                        selected_image_names=list(part["images"]),
                        output_dir=Path(batch["runFolder"]) / "Teilanfragen" / job["specimen"],
                        run_id=part["runId"], project_id=batch["projectId"],
                        allowed_class_codes=list(ACTIVE_CLASS_CODES),
                        evaluation_variant=batch.get("evaluationVariant"),
                        test_series={
                            "series_id": batch.get("testSeries", {}).get("seriesId"),
                            "name": batch.get("testSeries", {}).get("name"),
                            "category": "scientific_evaluation",
                            "series_run_id": batch.get("seriesRunId"),
                            "parent_run_id": job["runId"],
                            "partial_request_id": part["partId"],
                            "classification_rules_version": batch.get("classificationRulesVersion"),
                            "active_class_codes": list(ACTIVE_CLASS_CODES),
                        },
                        evaluation_variants_dir=EVALUATION_VARIANTS_DIR,
                    )
                    part.update(
                        status="succeeded", resultPath=str(destination),
                        finishedAt=datetime.now(timezone.utc).isoformat(), error=None,
                    )
                    with self._lock:
                        self._write_batch(batch)
                    break
                except Exception as exc:
                    policy = _provider_failure_policy(exc)
                    part.update(
                        error=str(exc) or repr(exc), failurePolicy=policy,
                        failureType=getattr(exc, "failure_type", "technical_execution"),
                        finishedAt=datetime.now(timezone.utc).isoformat(),
                    )
                    if policy["action"] == "retry" and part["attemptCount"] <= MAX_AUTOMATIC_PART_RETRIES:
                        delay = policy.get("retryAfterSeconds")
                        delay = float(delay) if delay is not None else float(2 ** (part["attemptCount"] - 1))
                        part.update(status="retry_wait", retryDelaySeconds=delay)
                        with self._lock:
                            self._write_batch(batch)
                        time.sleep(delay)
                        continue
                    if policy["action"] == "pause":
                        part["status"] = "paused"
                        job.update(status="paused", stage="paused", mergeStatus="pausiert")
                        batch.update(
                            status="paused", pauseReason=policy["reason"],
                            pausedAt=datetime.now(timezone.utc).isoformat(),
                        )
                        with self._lock:
                            self._write_batch(batch)
                        return "paused"
                    part["status"] = "failed"
                    failure_path = self._write_technical_failure(
                        run_id=part["runId"], specimen=job["specimen"],
                        provider=provider, model=model, selected_images=list(part["images"]),
                        error=exc,
                        output_folder=Path(batch["runFolder"]) / "Technische_Fehlversuche",
                    )
                    part["failureRecordPath"] = str(failure_path)
                    with self._lock:
                        self._write_batch(batch)
                    break
            if job.get("cancellationRequested") or batch.get("cancellationRequested"):
                for remaining in job.get("parts", []):
                    if remaining.get("status") not in {"succeeded", "failed"}:
                        remaining.update(status="cancelled", error=None)
                break
        if job.get("cancellationRequested") or batch.get("cancellationRequested"):
            job.update(
                status="cancelled", stage="cancelled",
                statusText="Kontrolliert nach der laufenden Teilanfrage abgebrochen",
                mergeStatus="abgebrochen",
                finishedAt=datetime.now(timezone.utc).isoformat(),
            )
            with self._lock:
                self._write_batch(batch)
            return "cancelled"
        successful_count = sum(part.get("status") == "succeeded" for part in job.get("parts", []))
        result_path = self._merge_scientific_job(batch, job) if successful_count else None
        if successful_count == len(job.get("parts", [])):
            job.update(status="succeeded", stage="complete", resultPath=str(result_path))
        elif successful_count:
            job.update(status="partially_succeeded", stage="partial", resultPath=None)
        else:
            job.update(status="failed", stage="error", mergeStatus="fehlgeschlagen")
        job["successfulPartialRequestCount"] = successful_count
        job["finishedAt"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._write_batch(batch)
        return job["status"]

    def start_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            plan = self._planned_batch(payload)
        except (RuntimeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        if plan["cloudRequestCount"] and not bool(payload.get("confirmCloud")):
            return {
                "ok": False,
                "needsConfirmation": True,
                "cloudRequestCount": plan["cloudRequestCount"],
                "error": (
                    f"{plan['cloudRequestCount']} Cloud-Anfragen benötigen eine ausdrückliche "
                    "Kosten- und Bildübertragungsfreigabe."
                ),
            }
        # Load persisted batches first; jobs left by an earlier process are normalized
        # to "interrupted" and therefore never behave like live locks.
        self.list_batches()
        batch_id = uuid.uuid4().hex
        batch = {
            **plan,
            "batchId": batch_id,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "confirmCloud": bool(payload.get("confirmCloud")),
            "sequential": True,
            "sessionId": self.session_id,
        }
        with self._lock:
            for planned_job in plan["jobs"]:
                conflict = self._active_analysis_conflict(
                    planned_job["specimen"], planned_job["provider"], planned_job["model"]
                )
                if conflict:
                    return {
                        "ok": False,
                        "conflictStatus": conflict["status"],
                        "jobId": conflict["id"] if conflict["kind"] == "single" else None,
                        "batchId": conflict["id"] if conflict["kind"] == "batch" else None,
                        "error": self._conflict_error(
                            conflict, planned_job["specimen"], planned_job["model"]
                        ),
                    }
            self.batches[batch_id] = batch
            self._write_batch(batch)
        threading.Thread(target=self._run_batch, args=(batch_id,), daemon=True).start()
        return {"ok": True, "batchId": batch_id, "batch": batch}

    def _run_batch(self, batch_id: str) -> None:
        while True:
            with self._lock:
                batch = self.batches.get(batch_id)
                if not batch:
                    return
                job = next((item for item in batch["jobs"] if item["status"] == "queued"), None)
                if job is None:
                    self._refresh_batch_status(batch)
                    self._write_batch(batch)
                    return
                job.update(
                    status="running",
                    stage="model_loading",
                    error=None,
                    startedAt=datetime.now(timezone.utc).isoformat(),
                    finishedAt=None,
                    statusText="Modell wird geladen",
                )
                self._write_batch(batch)

            if batch.get("kind") == "scientific_evaluation_series":
                outcome = self._run_scientific_job(batch, job)
                if outcome == "paused":
                    return
                continue

            provider = job["provider"]
            model = job["model"]
            context, prediction_budget = MODEL_CONFIGS[(provider, model)]

            def progress(status: str) -> None:
                stages = {
                    "Modell wird geladen": "model_loading",
                    "Bilder werden analysiert": "analyzing_images",
                    "Ergebnis wird verarbeitet": "processing_result",
                }
                with self._lock:
                    job["stage"] = stages[status]
                    job["statusText"] = status
                    self._write_batch(batch)

            try:
                destination = run_analysis(
                    Path(batch["projectPath"]) / job["specimen"],
                    provider=provider,
                    model=model,
                    live=provider in {"openai", "mistral"},
                    confirm_paid_request=batch["confirmCloud"],
                    ollama_num_ctx=context,
                    ollama_num_predict=prediction_budget,
                    max_references_per_class=1,
                    allowed_input_root=Path(batch["projectPath"]),
                    selected_image_names=list(job["selectedImages"]),
                    output_dir=(
                        Path(batch["runFolder"]) / "Ergebnisse"
                        if batch.get("kind") == "scientific_evaluation_series"
                        else Path(batch["projectData"]) / "Ergebnisse"
                    ),
                    progress_callback=progress,
                    run_id=job["runId"],
                    project_id=batch["projectId"],
                    allowed_class_codes=list(ACTIVE_CLASS_CODES),
                    evaluation_variant=batch.get("evaluationVariant"),
                    test_series={
                        "series_id": batch.get("testSeries", {}).get("seriesId"),
                        "name": batch.get("testSeries", {}).get("name"),
                        "category": "scientific_evaluation",
                        "series_run_id": batch.get("seriesRunId"),
                        "classification_rules_version": batch.get("classificationRulesVersion"),
                        "active_class_codes": list(ACTIVE_CLASS_CODES),
                    } if batch.get("kind") == "scientific_evaluation_series" else None,
                    evaluation_variants_dir=EVALUATION_VARIANTS_DIR,
                )
                document = json.loads(destination.read_text(encoding="utf-8"))
                with self._lock:
                    job.update(
                        status="succeeded", stage="complete", resultPath=str(destination),
                        detectedCodes=[item.get("code") for item in _result_features(document.get("result", {}))],
                        finishedAt=datetime.now(timezone.utc).isoformat(),
                    )
                    self._write_batch(batch)
            except Exception as exc:
                failure_path = self._write_technical_failure(
                    run_id=job["runId"],
                    specimen=job["specimen"],
                    provider=provider,
                    model=model,
                    selected_images=list(job["selectedImages"]),
                    error=exc,
                    output_folder=(
                        Path(batch["runFolder"]) / "Technische_Fehlversuche"
                        if batch.get("kind") == "scientific_evaluation_series" else None
                    ),
                )
                with self._lock:
                    job.update(
                        status="failed",
                        stage="error",
                        error=str(exc) or repr(exc),
                        failureType=getattr(exc, "failure_type", "technical_execution"),
                        scientificResult=False,
                        failureDiagnostics=getattr(exc, "diagnostics", {}),
                        failureRecordPath=str(failure_path),
                        finishedAt=datetime.now(timezone.utc).isoformat(),
                    )
                    self._write_batch(batch)

    def batch_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        batch_id = str(payload.get("batchId", ""))
        with self._lock:
            batch = self.batches.get(batch_id)
        if batch is None:
            # Going through list_batches is important: raw persisted "running" and
            # "queued" states belong to a dead app session and become interrupted.
            self.list_batches()
            with self._lock:
                batch = self.batches.get(batch_id)
        with self._lock:
            snapshot = self._batch_snapshot(batch) if batch else None
        return {"ok": bool(snapshot), "batch": snapshot,
                **({} if snapshot else {"error": "Batch wurde nicht gefunden."})}

    def list_batches(self, _payload: Any = None) -> dict[str, Any]:
        folder = self._project_data() / "Batches"
        batches: list[dict[str, Any]] = []
        if folder.is_dir():
            for path in folder.glob("*.json"):
                try:
                    stored = json.loads(path.read_text(encoding="utf-8"))
                    batch_id = stored["batchId"]
                    with self._lock:
                        batch = self.batches.get(batch_id, stored)
                        if batch_id not in self.batches:
                            if any(
                                job.get("status") in {"running", "queued"}
                                for job in batch.get("jobs", [])
                            ):
                                # Nach einem App-Neustart existiert für gespeicherte
                                # running/queued-Einträge kein Worker mehr. Sie dürfen
                                # daher nicht als aktive Sitzung wiederhergestellt werden.
                                for job in batch["jobs"]:
                                    if job.get("status") in {"running", "queued"}:
                                        job.update(status="interrupted", stage="interrupted")
                                    for part in job.get("parts", []):
                                        if part.get("status") in {"running", "retry_wait"}:
                                            part["status"] = "interrupted"
                            self._refresh_batch_status(batch)
                            self._write_batch(batch)
                        self.batches[batch_id] = batch
                        batches.append(self._batch_snapshot(batch))
                except (OSError, json.JSONDecodeError, KeyError):
                    continue
        batches.sort(
            key=lambda item: (
                item.get("status") in {"running", "interrupted"},
                item.get("createdAt", ""),
            ),
            reverse=True,
        )
        return {"ok": True, "batches": batches}

    def dismiss_batch_jobs(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Hide terminal job rows only; analysis artifacts and batch jobs remain stored."""

        status = self.batch_status(payload)
        if not status["ok"]:
            return status
        batch = self.batches[status["batch"]["batchId"]]
        requested = {str(item) for item in payload.get("jobIds", [])}
        if not requested:
            return {"ok": False, "error": "Bitte mindestens einen Anzeigeeintrag auswählen."}
        known = {job.get("jobId"): job for job in batch.get("jobs", [])}
        invalid = [job_id for job_id in requested if job_id not in known]
        protected = [
            job_id for job_id in requested
            if job_id in known and known[job_id].get("status") not in TERMINAL_BATCH_JOB_STATUSES
        ]
        if invalid:
            return {"ok": False, "error": "Mindestens ein Batch-Auftrag wurde nicht gefunden."}
        if protected:
            return {
                "ok": False,
                "error": "Laufende, wartende oder unterbrochene Aufträge können nicht ausgeblendet werden.",
            }
        with self._lock:
            dismissed = set(str(item) for item in batch.get("dismissedJobIds", []))
            dismissed.update(requested)
            batch["dismissedJobIds"] = sorted(dismissed)
            self._write_batch(batch)
            snapshot = self._batch_snapshot(batch)
        return {"ok": True, "batch": snapshot, "dismissedCount": len(requested)}

    def clear_batch_history(self, _payload: Any = None) -> dict[str, Any]:
        """Hide every terminal job row in this project, without deleting any artifact."""

        loaded = self.list_batches()
        if not loaded["ok"]:
            return loaded
        dismissed_count = 0
        with self._lock:
            for batch in self.batches.values():
                if batch.get("projectId") != self.project_name:
                    continue
                dismissed = set(str(item) for item in batch.get("dismissedJobIds", []))
                before = len(dismissed)
                dismissed.update(
                    str(job["jobId"])
                    for job in batch.get("jobs", [])
                    if job.get("status") in TERMINAL_BATCH_JOB_STATUSES
                )
                if len(dismissed) != before:
                    batch["dismissedJobIds"] = sorted(dismissed)
                    dismissed_count += len(dismissed) - before
                    self._write_batch(batch)
        refreshed = self.list_batches()
        return {
            "ok": True,
            "batches": refreshed["batches"],
            "dismissedCount": dismissed_count,
        }

    def cancel_batch_jobs(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = self.batch_status(payload)
        if not status["ok"]:
            return status
        batch = self.batches[status["batch"]["batchId"]]
        requested = set(str(item) for item in payload.get("jobIds", []))
        with self._lock:
            for job in batch["jobs"]:
                if requested and job["jobId"] not in requested:
                    continue
                if job["status"] in {"queued", "interrupted", "paused"}:
                    job.update(status="cancelled", stage="cancelled")
                elif job["status"] == "running":
                    job["cancellationRequested"] = True
                    job["statusText"] = (
                        "Abbruch angefordert; die laufende Teilanfrage wird noch abgeschlossen"
                    )
            if not requested and any(
                job.get("status") == "running" for job in batch.get("jobs", [])
            ):
                batch["cancellationRequested"] = True
            self._refresh_batch_status(batch)
            self._write_batch(batch)
        return {"ok": True, "batch": self._batch_snapshot(batch)}

    def resume_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = self.batch_status(payload)
        if not status["ok"]:
            return status
        batch = self.batches[status["batch"]["batchId"]]
        retry_failed = bool(payload.get("retryFailed"))
        with self._lock:
            resumable_statuses = {"interrupted"}
            if retry_failed:
                resumable_statuses.update({"failed", "partially_succeeded", "paused"})
            resumable = [job for job in batch["jobs"] if job["status"] in resumable_statuses]
            for job in resumable:
                conflict = self._active_analysis_conflict(
                    job["specimen"], job["provider"], job["model"],
                    ignore_batch_id=batch["batchId"],
                )
                if conflict:
                    return {
                        "ok": False,
                        "conflictStatus": conflict["status"],
                        "error": self._conflict_error(conflict, job["specimen"], job["model"]),
                    }
            for job in batch["jobs"]:
                if job["status"] not in resumable_statuses:
                    continue
                job.pop("cancellationRequested", None)
                for part in job.get("parts", []):
                    if part.get("status") != "succeeded":
                        part.update(status="queued", error=None)
                job.update(status="queued", stage="queued", error=None)
            if not any(job["status"] == "queued" for job in batch["jobs"]):
                return {"ok": False, "error": "Der Batch enthält keine fortsetzbaren Aufträge."}
            batch["status"] = "running"
            batch.pop("cancellationRequested", None)
            batch.pop("pauseReason", None)
            batch.pop("pausedAt", None)
            batch["sessionId"] = self.session_id
            self._write_batch(batch)
        threading.Thread(target=self._run_batch, args=(batch["batchId"],), daemon=True).start()
        return {"ok": True, "batchId": batch["batchId"], "batch": self._batch_snapshot(batch)}

    def save_correction(self, payload: dict[str, Any]) -> dict[str, Any]:
        specimen = _safe_name(str(payload.get("specimen", "")))
        codes = set(ACTIVE_CLASS_CODES)
        primary = str(payload.get("primaryCode", "")).strip()
        secondary = [str(x) for x in payload.get("secondaryCodes", [])]
        if primary not in codes or any(code not in codes or code == primary for code in secondary):
            return {"ok": False, "error": "Bitte gültige, eindeutige Klassencodes wählen."}
        folder = self._project_data() / "Korrekturen"
        folder.mkdir(parents=True, exist_ok=True)
        document = {"specimen": specimen, "primaryCode": primary, "secondaryCodes": secondary,
                    "note": str(payload.get("note", "")).strip(), "confirmed": True,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "originalResult": str(self._latest_result(specimen) or "")}
        target = folder / f"{specimen}.json"
        target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        return {"ok": True, "correction": document, "path": str(target)}

    def export_scientific_series(self, payload: dict[str, Any]) -> dict[str, Any]:
        series_run_id = str(payload.get("seriesRunId", ""))
        status = self.batch_status({"batchId": series_run_id})
        if not status.get("ok"):
            return status
        batch = self.batches[status["batch"]["batchId"]]
        if batch.get("kind") != "scientific_evaluation_series":
            return {"ok": False, "error": "Der gewählte Lauf ist keine wissenschaftliche Serienprüfung."}
        export_folder = Path(batch["runFolder"]) / "Export"
        export_folder.mkdir(parents=True, exist_ok=True)
        wide_rows: list[list[Any]] = []
        detail_rows: list[list[Any]] = []
        counts = {"succeeded": 0, "failed": 0, "notEvaluated": 0, "secureSuggestions": 0}
        for job in batch["jobs"]:
            result: dict[str, Any] = {}
            document: dict[str, Any] = {}
            if job.get("status") == "succeeded" and job.get("resultPath"):
                document = json.loads(Path(job["resultPath"]).read_text(encoding="utf-8"))
                result = document.get("result", {})
            elif job.get("status") == "partially_succeeded" and job.get("partialMergePath"):
                document = json.loads(Path(job["partialMergePath"]).read_text(encoding="utf-8"))
                result = document.get("result", {})
            features = _result_features(result)
            by_code = {str(item.get("code")): item for item in features}
            analysis_status = str(result.get("status", ""))
            if job.get("status") == "failed":
                export_status = "fehlgeschlagen"
                feature_values = [""] * len(ACTIVE_CLASS_CODES)
                counts["failed"] += 1
            elif job.get("status") == "partially_succeeded":
                export_status = "teilweise_erfolgreich_nicht_hauptauswertung"
                feature_values = [""] * len(ACTIVE_CLASS_CODES)
                counts["notEvaluated"] += 1
            elif job.get("status") != "succeeded":
                export_status = "nicht_analysiert"
                feature_values = [""] * len(ACTIVE_CLASS_CODES)
                counts["notEvaluated"] += 1
            elif analysis_status == "KLASSIFIZIERT":
                export_status = "erfolgreich_klassifiziert"
                feature_values = [1 if code in by_code else 0 for code in ACTIVE_CLASS_CODES]
                counts["succeeded"] += 1
            elif analysis_status == "KEINE_MERKMALE":
                export_status = "keine_merkmale_erkannt"
                feature_values = [0] * len(ACTIVE_CLASS_CODES)
                counts["succeeded"] += 1
            else:
                export_status = "unklar"
                feature_values = [""] * len(ACTIVE_CLASS_CODES)
                counts["notEvaluated"] += 1
            secure = [code for code, item in by_code.items() if item.get("evidence_status") == "sicher_sichtbar"]
            suspected = [code for code, item in by_code.items() if item.get("evidence_status") == "vermutet"]
            counts["secureSuggestions"] += len(secure)
            execution = document.get("execution", {})
            basis = document.get("classification_basis", {})
            wide_rows.append([
                job["specimen"], job["normalizedSpecimenId"], export_status,
                *feature_values, "|".join(by_code), "|".join(secure), "|".join(suspected),
                "|".join(str(part.get("error", "")) for part in job.get("parts", []) if part.get("error")) or job.get("error", ""),
                job["runId"], batch["evaluationVariant"],
                execution.get("provider", batch["provider"]),
                execution.get("actual_model", batch["model"]),
                basis.get("prompt_version", batch["promptVersion"]),
                basis.get("classification_rules_version", batch["classificationRulesVersion"]),
                job.get("selectedImageCount", len(job.get("selectedImages", []))),
                job.get("partialRequestCount", 1),
                job.get("successfulPartialRequestCount", int(job.get("status") == "succeeded")),
                job.get("mergeStatus", ""), job.get("mergeConflictCount", 0),
            ])
            evaluable = export_status in {"erfolgreich_klassifiziert", "keine_merkmale_erkannt"}
            for code in ACTIVE_CLASS_CODES:
                feature = by_code.get(code, {})
                detail_rows.append([
                    job["specimen"], job["normalizedSpecimenId"], code,
                    ("Ja" if code in by_code else "Nein") if evaluable else "",
                    feature.get("confidence_label", ""), feature.get("evidence_status", ""),
                    execution.get("provider", batch["provider"]),
                    execution.get("actual_model", batch["model"]),
                    batch["evaluationVariant"], basis.get("prompt_version", batch["promptVersion"]),
                    basis.get("classification_rules_version", batch["classificationRulesVersion"]),
                    job["runId"], document.get("created_at", ""),
                    "ja" if export_status in {"fehlgeschlagen", "teilweise_erfolgreich_nicht_hauptauswertung"} else "nein",
                    "|".join(str(part.get("error", "")) for part in job.get("parts", []) if part.get("error")) or job.get("error", ""),
                    export_status,
                    job.get("selectedImageCount", len(job.get("selectedImages", []))),
                    job.get("partialRequestCount", 1),
                    job.get("successfulPartialRequestCount", int(job.get("status") == "succeeded")),
                    job.get("mergeStatus", ""), job.get("mergeConflictCount", 0),
                ])
        wide_path = export_folder / "ergebnisse_eine_zeile_je_pruefkoerper.csv"
        detail_path = export_folder / "ergebnisse_je_pruefkoerper_und_merkmal.csv"
        with wide_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow([
                "Pruefkoerper_ID", "Normalisierte_Pruefkoerper_ID", "Status", *ACTIVE_CLASS_CODES,
                "Vorgeschlagene_Merkmale", "Sicher_sichtbare_Merkmale", "Vermutete_Merkmale",
                "Technischer_Fehler", "Run_ID", "Variante", "Provider", "Modell",
                "Promptversion", "Regelwerksversion", "Ausgewaehlte_Bilder_Anzahl",
                "Teilanfragen_Anzahl", "Erfolgreiche_Teilanfragen_Anzahl",
                "Zusammenfuehrungsstatus", "Widersprueche_Anzahl",
            ])
            writer.writerows(wide_rows)
        with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow([
                "Pruefkoerper_ID", "Normalisierte_Pruefkoerper_ID", "Merkmalscode",
                "Vorgeschlagen", "Confidence", "Evidence_Status", "Provider", "Modell",
                "Variante", "Promptversion", "Regelwerksversion", "Run_ID", "Zeitstempel",
                "Technischer_Fehlerstatus", "Technischer_Fehler", "Analyse_Status",
                "Ausgewaehlte_Bilder_Anzahl", "Teilanfragen_Anzahl",
                "Erfolgreiche_Teilanfragen_Anzahl", "Zusammenfuehrungsstatus",
                "Widersprueche_Anzahl",
            ])
            writer.writerows(detail_rows)
        coverage = counts["succeeded"] / len(batch["jobs"]) if batch["jobs"] else 0
        metadata_path = export_folder / "export_manifest.json"
        metadata_path.write_text(json.dumps({
            "schemaVersion": "1.0",
            "seriesRunId": series_run_id,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "mainEvaluationPolicy": "Alle Vorschläge einschließlich evidence_status=vermutet und confidence=unsicher zählen als erkannt.",
            "secureOnlyPolicy": "Nur evidence_status=sicher_sichtbar; ausgeschlossene Vorschläge und Abdeckung müssen separat berichtet werden.",
            "evaluableSpecimenCount": counts["succeeded"],
            "excludedOrNotEvaluatedSpecimenCount": len(batch["jobs"]) - counts["succeeded"],
            "specimenCoverageRate": coverage,
            "groundTruthIncluded": False,
            "files": [wide_path.name, detail_path.name],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "ok": True, "wideCsvPath": str(wide_path), "detailCsvPath": str(detail_path),
            "manifestPath": str(metadata_path), "coverageRate": coverage,
        }

    def save_assessment(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = str(payload.get("runId", ""))
        found = self._result_by_run_id(run_id)
        if not found:
            return {"ok": False, "error": "Analyselauf wurde nicht gefunden."}
        result_path, result = found
        codes = set(load_catalog().codes)
        if "featureReviews" in payload:
            reviews = payload.get("featureReviews")
            if not isinstance(reviews, list):
                return {"ok": False, "error": "Die Merkmalsprüfungen sind ungültig."}
            source_features = _result_features(result.get("result", {}))
            source_ids = {str(item.get("feature_id")) for item in source_features}
            validated_reviews: list[dict[str, Any]] = []
            for index, review in enumerate(reviews, start=1):
                if not isinstance(review, dict):
                    return {"ok": False, "error": f"Merkmalsprüfung {index} ist ungültig."}
                status = str(review.get("status", "unreviewed"))
                if status not in {"unreviewed", "confirmed", "rejected", "corrected", "added"}:
                    return {"ok": False, "error": f"Prüfstatus für Merkmal {index} ist ungültig."}
                source_id = str(review.get("sourceFeatureId", "")).strip()
                if status == "added":
                    source_id = ""
                elif source_id not in source_ids:
                    return {"ok": False, "error": f"KI-Merkmal {index} wurde nicht gefunden."}
                code = str(review.get("code", "")).strip()
                if status != "rejected" and code not in codes:
                    return {"ok": False, "error": f"Klassencode für Merkmal {index} ist ungültig."}
                validated_reviews.append({
                    "reviewId": str(review.get("reviewId") or f"review-{index:02d}"),
                    "sourceFeatureId": source_id or None,
                    "status": status,
                    "originalCode": str(review.get("originalCode", "")).strip() or None,
                    "code": code or None,
                    "visibleDescription": str(review.get("visibleDescription", "")).strip(),
                    "note": str(review.get("note", "")).strip(),
                })
            expert_primary = str(payload.get("expertPrimaryCode", "")).strip()
            if expert_primary and expert_primary not in codes:
                return {"ok": False, "error": "Das manuell ausgewählte Primärversagen ist ungültig."}
            reviewed_by = str(payload.get("reviewedBy", "")).strip()
            if expert_primary and not reviewed_by:
                return {"ok": False, "error": "Bitte die bewertende Person angeben."}
            updated_at = datetime.now(timezone.utc).isoformat()
            document = {
                "schemaVersion": "2.0",
                "runId": run_id,
                "projectId": self.project_name,
                "specimenId": result.get("specimen_id", result.get("analysis_id")),
                "featureReviews": validated_reviews,
                "expertPrimaryCause": {
                    "code": expert_primary or None,
                    "noUniqueCauseEstablished": not bool(expert_primary),
                    "rationale": str(payload.get("primaryRationale", "")).strip(),
                    "decisionType": "manual_expert_selection",
                    "selectedBy": reviewed_by or None,
                    "selectedAt": updated_at if expert_primary else None,
                },
                "manualPrimaryFailure": {
                    "code": expert_primary or None,
                    "selectedBy": reviewed_by or None,
                    "selectedAt": updated_at if expert_primary else None,
                    "decisionType": "manual_expert_selection",
                },
                "resultPath": str(result_path),
                "updatedAt": updated_at,
            }
            folder = self._project_data() / "Bewertungen"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{_safe_name(run_id)}.review.json"
            target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return {"ok": True, "assessment": document, "path": str(target)}

        # Legacy API remains readable/writeable for historic clients only.
        primary = str(payload.get("primaryCode", "")).strip()
        secondary = [str(item) for item in payload.get("secondaryCodes", [])]
        if primary and primary not in codes:
            return {"ok": False, "error": "Die fachliche Hauptklasse ist ungültig."}
        if any(code not in codes or code == primary for code in secondary):
            return {"ok": False, "error": "Die fachlichen Sekundärklassen sind ungültig."}
        verdict = str(payload.get("verdict", "unreviewed"))
        if verdict not in {"unreviewed", "confirmed", "partially_correct", "incorrect", "unclear"}:
            return {"ok": False, "error": "Die fachliche Bewertung ist ungültig."}
        document = {
            "runId": run_id,
            "projectId": self.project_name,
            "specimenId": result.get("specimen_id", result.get("analysis_id")),
            "verdict": verdict,
            "primaryCode": primary,
            "secondaryCodes": secondary,
            "note": str(payload.get("note", "")).strip(),
            "resultPath": str(result_path),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        folder = self._project_data() / "Bewertungen"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{_safe_name(run_id)}.json"
        target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"ok": True, "assessment": document, "path": str(target)}

    def _generated_export_files(self) -> list[dict[str, Any]]:
        roots: list[tuple[str, Path]] = [("Projekt", self._project_data() / "Export")]
        scientific_root = self._project_data() / "Wissenschaftliche_Laeufe"
        if scientific_root.is_dir():
            roots.extend(
                (path.parent.name, path)
                for path in scientific_root.glob("*/Export")
                if path.is_dir()
            )
        files: list[dict[str, Any]] = []
        for run_id, root in roots:
            if not root.is_dir():
                continue
            resolved_root = root.resolve()
            for path in sorted(root.iterdir()):
                if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
                    continue
                resolved = path.resolve()
                if resolved.parent != resolved_root:
                    continue
                files.append({
                    "exportId": hashlib.sha256(str(resolved).encode("utf-8")).hexdigest(),
                    "name": path.name,
                    "path": str(path),
                    "seriesRunId": "" if run_id == "Projekt" else run_id,
                    "sizeBytes": path.stat().st_size,
                    "modifiedAt": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                })
        files.sort(key=lambda item: item["modifiedAt"], reverse=True)
        return files

    def list_export_files(self, _payload: Any = None) -> dict[str, Any]:
        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        return {"ok": True, "files": self._generated_export_files()}

    def delete_export_files(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not bool(payload.get("confirmDeletion")):
            return {"ok": False, "error": "Das Löschen muss ausdrücklich bestätigt werden."}
        inventory = self._generated_export_files()
        by_id = {item["exportId"]: item for item in inventory}
        requested = (
            set(by_id) if bool(payload.get("deleteAll"))
            else {str(item) for item in payload.get("exportIds", [])}
        )
        if not requested:
            return {"ok": False, "error": "Bitte mindestens eine Exportdatei auswählen."}
        unknown = requested - set(by_id)
        if unknown:
            return {"ok": False, "error": "Mindestens eine Exportdatei wurde nicht gefunden."}
        deleted: list[str] = []
        for export_id in requested:
            path = Path(by_id[export_id]["path"])
            # The inventory only exposes direct CSV/JSON children of the two
            # allow-listed export locations. Recheck immediately before unlinking.
            if path.suffix.lower() not in {".csv", ".json"} or not path.is_file():
                return {"ok": False, "error": f"Exportdatei ist nicht mehr gültig: {path.name}"}
            path.unlink()
            deleted.append(path.name)
        return {
            "ok": True,
            "deleted": sorted(deleted),
            "deletedCount": len(deleted),
            "files": self._generated_export_files(),
        }

    def export_overview(self, _payload: Any = None) -> dict[str, Any]:
        target = self._project_data() / f"{_safe_name(self.project_name)}_Uebersicht.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["Prüfkörper", "Run_ID", "Ergebnisgruppe", "Testreihen_ID", "Testreihenname",
                             "Zeitstempel", "Provider", "Modell", "Status",
                             "KI_Merkmalscodes_gleichrangig", "KI_Merkmalsanzahl", "Historische_Primärklasse",
                             "Historische_Sekundärklassen", "Laufzeit_s", "Prompt_Hash",
                             "Merkmalsprüfungen_JSON", "Manuell_ausgewähltes_Primärversagen",
                             "Bewertet_durch", "Bewertet_am", "Fachliche_Begründung"])
            for specimen in self._specimens():
                records = self._result_records(specimen.name)
                if not records:
                    writer.writerow([specimen.name, "", "", "", "", "NICHT_ANALYSIERT"]); continue
                for record in records:
                    assessment = self._assessment(record["runId"]) or {}
                    manual = assessment.get("manualPrimaryFailure", assessment.get("expertPrimaryCause", {}))
                    writer.writerow([
                        specimen.name, record["runId"], record["resultGroup"], record["seriesId"] or "",
                        record["seriesName"] or "Historischer Entwicklungstest",
                        record["createdAt"], record["provider"],
                        record["model"], record["status"], "|".join(record["detectedCodes"]),
                        record["featureCount"], record["primaryCode"],
                        "|".join(filter(None, record["secondaryCodes"])), record["durationSeconds"],
                        record["promptHash"], json.dumps(assessment.get("featureReviews", []), ensure_ascii=False),
                        manual.get("code", ""), manual.get("selectedBy", ""), manual.get("selectedAt", ""),
                        assessment.get("expertPrimaryCause", {}).get("rationale", assessment.get("note", "")),
                    ])
        return {"ok": True, "path": str(target)}

    def export_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        found = self._result_by_run_id(str(payload.get("runId", "")))
        if not found:
            return {"ok": False, "error": "Analyselauf wurde nicht gefunden."}
        source, document = found
        run_id = self._run_id(document, source)
        folder = self._project_data() / "Export"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{_safe_name(run_id)}_Einzelanalyse.json"
        target.write_text(json.dumps({"analysis": document, "assessment": self._assessment(run_id)},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"ok": True, "path": str(target)}

    def export_comparison(self, payload: dict[str, Any]) -> dict[str, Any]:
        specimen = _safe_name(str(payload.get("specimen", "")))
        run_ids = [str(item) for item in payload.get("runIds", [])]
        if not 1 <= len(run_ids) <= 3 or len(run_ids) != len(set(run_ids)):
            return {"ok": False, "error": "Bitte ein bis drei eindeutige Analyseläufe auswählen."}
        rows = []
        documents = []
        for run_id in run_ids:
            found = self._result_by_run_id(run_id)
            if not found:
                return {"ok": False, "error": f"Analyselauf wurde nicht gefunden: {run_id}"}
            path, document = found
            document_specimen = str(document.get("specimen_id", document.get("analysis_id", "")))
            if document_specimen != specimen:
                return {"ok": False, "error": "Alle Vergleichsläufe müssen zum selben Prüfkörper gehören."}
            result = document.get("result", {})
            primary = _primary_result_feature(result)
            features = _result_features(result)
            execution = document.get("execution", {})
            assessment = self._assessment(run_id)
            manual = (assessment or {}).get(
                "manualPrimaryFailure", (assessment or {}).get("expertPrimaryCause", {})
            )
            rows.append([
                specimen, run_id, execution.get("provider"), execution.get("model"),
                primary.get("code", ""), primary.get("confidence", ""),
                primary.get("evidence_status", "nicht_dokumentiert"),
                "|".join(item.get("code", "") for item in result.get("secondary_features", [])),
                execution.get("duration_seconds", ""),
                document.get("classification_basis", {}).get("prompt_hash_sha256", ""),
                (assessment or {}).get("verdict", ""), (assessment or {}).get("primaryCode", ""),
            ])
            documents.append({"analysis": document, "assessment": assessment, "path": str(path)})
        folder = self._project_data() / "Export"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"{specimen}_Modellvergleich_{stamp}"
        json_target = folder / f"{base}.json"
        csv_target = folder / f"{base}.csv"
        json_target.write_text(json.dumps({"specimen": specimen, "runs": documents},
                                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with csv_target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["Prüfkörper", "Run_ID", "Provider", "Modell", "Primärklasse",
                             "Konfidenz", "Evidenzstatus", "Sekundärklassen", "Laufzeit_s", "Prompt_Hash",
                             "Bewertung", "Ground_Truth_Primärklasse"])
            writer.writerows(rows)
        return {"ok": True, "jsonPath": str(json_target), "csvPath": str(csv_target)}

    def export_selected_runs(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_ids = [str(item) for item in payload.get("runIds", [])]
        if not run_ids or len(run_ids) != len(set(run_ids)):
            return {"ok": False, "error": "Bitte mindestens einen eindeutigen Analyselauf auswählen."}
        documents: list[dict[str, Any]] = []
        rows: list[list[Any]] = []
        for run_id in run_ids:
            found = self._result_by_run_id(run_id)
            if not found:
                return {"ok": False, "error": f"Analyselauf wurde nicht gefunden: {run_id}"}
            path, document = found
            assessment = self._assessment(run_id)
            manual = (assessment or {}).get(
                "manualPrimaryFailure", (assessment or {}).get("expertPrimaryCause", {})
            )
            result = document.get("result", {})
            primary = _primary_result_feature(result)
            features = _result_features(result)
            execution = document.get("execution", {})
            basis = document.get("classification_basis", {})
            selection = document.get("image_selection", {})
            references = basis.get("references", [])
            series = document.get("test_series") if isinstance(document.get("test_series"), dict) else {}
            rows.append([
                document.get("project_id", self.project_name),
                document.get("specimen_id", document.get("analysis_id", "")),
                document.get("created_at", ""), run_id,
                "scientific" if series.get("series_id") else "historical",
                series.get("series_id", ""), series.get("name", "Historischer Entwicklungstest"),
                basis.get("evaluation_variant") or "Standard",
                execution.get("provider", ""), execution.get("model", ""),
                "|".join(selection.get("included", [])), len(references),
                "|".join(item.get("filename", "") for item in references),
                basis.get("prompt_version", ""),
                "|".join(result.get("observations_short", result.get("visible_observations", []))),
                "|".join(str(item.get("code", "")) for item in features),
                json.dumps(features, ensure_ascii=False),
                "|".join(str(item.get("code", "")) for item in features if item.get("evidence_status") == "sicher_sichtbar"),
                "|".join(str(item.get("code", "")) for item in features if item.get("evidence_status") == "vermutet"),
                primary.get("code", ""),
                "|".join(item.get("code", "") for item in result.get("secondary_features", [])),
                (assessment or {}).get("primaryCode", ""),
                "|".join((assessment or {}).get("secondaryCodes", [])),
                manual.get("code", ""), manual.get("selectedBy", ""), manual.get("selectedAt", ""),
                json.dumps((assessment or {}).get("featureReviews", []), ensure_ascii=False),
                (assessment or {}).get("note", ""), (assessment or {}).get("verdict", ""),
                execution.get("duration_seconds", ""), "",
            ])
            documents.append({
                "analysis": document,
                "assessment": assessment,
                "sourcePath": str(path),
            })
        folder = self._project_data() / "Export"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base = f"{_safe_name(self.project_name)}_Bildanalysen_{stamp}"
        json_target = folder / f"{base}.json"
        csv_target = folder / f"{base}.csv"
        json_target.write_text(json.dumps({
            "project": self.project_name,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "runs": documents,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with csv_target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow([
                "Projekt", "Prüfkörper", "Analysedatum", "Run_ID", "Ergebnisgruppe",
                "Testreihen_ID", "Testreihenname", "Evaluationsvariante",
                "Provider", "Modell", "Prüfkörperbilder", "Referenzbildanzahl",
                "Referenzbilder", "Promptversion", "KI_sichtbare_Merkmale",
                "KI_Merkmalscodes_gleichrangig", "KI_Merkmale_JSON",
                "KI_sicher_sichtbare_Merkmale", "KI_vermutete_Merkmale", "Historischer_KI_Primärvorschlag",
                "Historische_KI_Sekundärvorschläge", "Historisch_bestätigtes_Primärversagen",
                "Historisch_bestätigte_weitere_Merkmale", "Manuell_ausgewähltes_Primärversagen",
                "Bewertet_durch", "Bewertet_am", "Merkmalsprüfungen_JSON",
                "Fachliche_Begründung", "Prüfstatus",
                "Laufzeit_s", "Fehlermeldung",
            ])
            writer.writerows(rows)
        return {"ok": True, "jsonPath": str(json_target), "csvPath": str(csv_target)}

    def export_specimen(self, payload: dict[str, Any]) -> dict[str, Any]:
        specimen = _safe_name(str(payload.get("specimen", "")))
        result_path = self._latest_result(specimen)
        if not result_path:
            return {"ok": False, "error": "Für diesen Prüfkörper liegt noch kein Ergebnis vor."}
        result = json.loads(result_path.read_text())
        correction_path = self._project_data() / "Korrekturen" / f"{specimen}.json"
        correction = json.loads(correction_path.read_text()) if correction_path.is_file() else None
        folder = self._project_data() / "Export"
        folder.mkdir(parents=True, exist_ok=True)
        json_target = folder / f"{specimen}_Analyse_und_Korrektur.json"
        json_target.write_text(json.dumps({"kiOriginal": result, "fachlicheKorrektur": correction},
                                          ensure_ascii=False, indent=2) + "\n")
        csv_target = folder / f"{specimen}_Analyse.csv"
        primary = _primary_result_feature(result["result"])
        with csv_target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["Prüfkörper", "Modell", "Laufzeit_s", "KI_Primärklasse",
                             "KI_Konfidenz", "KI_Evidenzstatus", "Einbezogene_Bilder", "Ausgeschlossene_Bilder",
                             "Korrektur_Primärklasse", "Korrektur_Sekundärklassen"])
            writer.writerow([specimen, result["execution"]["model"],
                             result["execution"].get("duration_seconds", ""), primary.get("code", ""),
                             primary.get("confidence", ""),
                             primary.get("evidence_status", "nicht_dokumentiert"),
                             "|".join(result.get("image_selection", {}).get("included", [])),
                             "|".join(result.get("image_selection", {}).get("excluded", [])),
                             (correction or {}).get("primaryCode", ""),
                             "|".join((correction or {}).get("secondaryCodes", []))])
        return {"ok": True, "jsonPath": str(json_target), "csvPath": str(csv_target)}
