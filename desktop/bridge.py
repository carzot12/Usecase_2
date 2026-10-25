from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from Tool.ki_bildanalyse import SUPPORTED_SUFFIXES, load_catalog, run_analysis


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
MODEL_CONFIGS: dict[tuple[str, str], tuple[int, int]] = {
    ("ollama", "qwen3-vl:4b"): (24576, 4096),
    ("ollama", "llava:latest"): (16384, 4096),
    ("openai", "gpt-6.1-sol"): (16384, 4096),
    ("mistral", "ministral-3b-2512"): (16384, 4096),
}
DEFAULT_SELECTED_IMAGE_COUNT = 3


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError("Bitte einen gültigen Namen eingeben.")
    return cleaned


class DesktopBridge:
    """Allow-listed bridge; project files and analysis stay in Python."""

    def __init__(self) -> None:
        APP_DATA.mkdir(parents=True, exist_ok=True)
        self.project_root: Path | None = None
        self.project_name = ""
        self.jobs: dict[str, dict[str, Any]] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def bootstrap(self, _payload: Any = None) -> dict[str, Any]:
        return {"ok": True, "exampleAvailable": EXAMPLE.is_dir(), "examplePath": str(EXAMPLE),
                "models": [
                    {"provider": provider, "model": model, "local": provider == "ollama"}
                    for provider, model in MODEL_CONFIGS
                ],
                "classes": [{"code": c, "name": d["name"]}
                            for c, d in load_catalog().definitions.items()]}

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
        self._selection_path(specimen).write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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
                      and not p.name.startswith("Nummer_"))

    def _latest_result(self, specimen: str) -> Path | None:
        results = self._result_records(specimen)
        return Path(results[-1]["path"]) if results else None

    @staticmethod
    def _run_id(document: dict[str, Any], path: Path) -> str:
        return str(document.get("run_id") or (
            "legacy-" + hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
        ))

    def _result_records(self, specimen: str) -> list[dict[str, Any]]:
        folder = self._project_data() / "Ergebnisse"
        records: list[dict[str, Any]] = []
        if not folder.is_dir():
            return records
        for path in folder.glob(f"{_safe_name(specimen)}_*.json"):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(document, dict):
                    continue
                run_id = self._run_id(document, path)
                result = document.get("result", {})
                primary = result.get("primary_feature") or {}
                execution = document.get("execution", {})
                metadata = execution.get("provider_metadata", {})
                cost = metadata.get("cost", {}) if isinstance(metadata, dict) else {}
                records.append({
                    "runId": run_id,
                    "path": str(path),
                    "createdAt": document.get("created_at", ""),
                    "provider": execution.get("provider", "unknown"),
                    "model": execution.get("model", "unknown"),
                    "status": result.get("status", ""),
                    "primaryCode": primary.get("code", ""),
                    "primaryConfidence": primary.get("confidence"),
                    "secondaryCodes": [item.get("code") for item in result.get("secondary_features", [])],
                    "durationSeconds": execution.get("duration_seconds"),
                    "usage": metadata.get("usage", {}) if isinstance(metadata, dict) else {},
                    "cost": cost,
                    "promptHash": document.get("classification_basis", {}).get("prompt_hash_sha256"),
                    "legacy": "run_id" not in document,
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
        path = self._project_data() / "Bewertungen" / f"{_safe_name(run_id)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _snapshot(self) -> dict[str, Any]:
        return {"name": self.project_name, "path": str(self.project_root), "specimens": [
            {"id": p.name, "imageCount": len(self._images(p)),
             "runCount": len(self._result_records(p.name)),
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
        source_images = self._images(folder)
        selected_names = self._selected_image_names(specimen, folder)
        images = [
            {"name": p.name, "src": self._image_data(p), "analysisSelected": p.name in selected_names}
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

    def result_detail(self, payload: dict[str, Any]) -> dict[str, Any]:
        found = self._result_by_run_id(str(payload.get("runId", "")))
        if not found:
            return {"ok": False, "error": "Analyselauf wurde nicht gefunden."}
        path, document = found
        run_id = self._run_id(document, path)
        return {"ok": True, "runId": run_id, "path": str(path), "result": document,
                "assessment": self._assessment(run_id)}

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

    def start_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.project_root:
            return {"ok": False, "error": "Kein Projekt geöffnet."}
        specimen = _safe_name(str(payload.get("specimen", "")))
        provider = str(payload.get("provider", "ollama")).strip()
        model = str(payload.get("model", "qwen3-vl:4b")).strip()
        legacy_choices = {
            "qwen": ("ollama", "qwen3-vl:4b"),
            "llava": ("ollama", "llava:latest"),
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
        if provider in {"openai", "mistral"} and not confirm:
            return {"ok": False, "needsConfirmation": True,
                    "error": f"{provider.title()} erfordert eine ausdrückliche Kostenbestätigung."}
        job_id = uuid.uuid4().hex
        run_id = uuid.uuid4().hex
        self.jobs[job_id] = {
            "state": "running", "stage": "model_loading", "status": "Modell wird geladen",
            "provider": provider, "model": model, "specimen": specimen,
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
                                           run_id=run_id, project_id=self.project_name)
                with self._lock:
                    self.jobs[job_id].update(
                        state="complete", stage="complete", status="Analyse abgeschlossen.",
                        path=str(destination), result=json.loads(destination.read_text()),
                    )
            except Exception as exc:
                with self._lock:
                    self.jobs[job_id].update(
                        state="error", stage="error", status="Analyse fehlgeschlagen.",
                        error=str(exc) or repr(exc),
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
        for raw in specimen_ids:
            specimen = _safe_name(str(raw))
            if specimen not in available_specimens:
                raise ValueError(f"Prüfkörper wurde nicht gefunden: {specimen}")
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
        batch_id = uuid.uuid4().hex
        batch = {
            **plan,
            "batchId": batch_id,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "confirmCloud": bool(payload.get("confirmCloud")),
            "sequential": True,
        }
        with self._lock:
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
                    statuses = {item["status"] for item in batch["jobs"]}
                    batch["status"] = "completed_with_errors" if "failed" in statuses else "completed"
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
                    output_dir=Path(batch["projectData"]) / "Ergebnisse",
                    progress_callback=progress,
                    run_id=job["runId"],
                    project_id=batch["projectId"],
                )
                document = json.loads(destination.read_text(encoding="utf-8"))
                with self._lock:
                    job.update(
                        status="succeeded", stage="complete", resultPath=str(destination),
                        primaryCode=(document.get("result", {}).get("primary_feature") or {}).get("code"),
                        finishedAt=datetime.now(timezone.utc).isoformat(),
                    )
                    self._write_batch(batch)
            except Exception as exc:
                with self._lock:
                    job.update(
                        status="failed",
                        stage="error",
                        error=str(exc) or repr(exc),
                        finishedAt=datetime.now(timezone.utc).isoformat(),
                    )
                    self._write_batch(batch)

    def batch_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        batch_id = str(payload.get("batchId", ""))
        with self._lock:
            batch = self.batches.get(batch_id)
            if batch is None:
                path = self._batch_path(batch_id)
                if path.is_file():
                    batch = json.loads(path.read_text(encoding="utf-8"))
                    self.batches[batch_id] = batch
            snapshot = json.loads(json.dumps(batch)) if batch else None
        return {"ok": bool(snapshot), "batch": snapshot,
                **({} if snapshot else {"error": "Batch wurde nicht gefunden."})}

    def list_batches(self, _payload: Any = None) -> dict[str, Any]:
        folder = self._project_data() / "Batches"
        batches = []
        if folder.is_dir():
            for path in sorted(folder.glob("*.json"), reverse=True):
                try:
                    batch = json.loads(path.read_text(encoding="utf-8"))
                    if any(job.get("status") == "running" for job in batch.get("jobs", [])):
                        for job in batch["jobs"]:
                            if job.get("status") == "running":
                                job.update(status="interrupted", stage="interrupted")
                        batch["status"] = "interrupted"
                        self._write_batch(batch)
                    self.batches[batch["batchId"]] = batch
                    batches.append(batch)
                except (OSError, json.JSONDecodeError, KeyError):
                    continue
        return {"ok": True, "batches": batches}

    def cancel_batch_jobs(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = self.batch_status(payload)
        if not status["ok"]:
            return status
        batch = self.batches[status["batch"]["batchId"]]
        requested = set(str(item) for item in payload.get("jobIds", []))
        with self._lock:
            for job in batch["jobs"]:
                if job["status"] in {"queued", "interrupted"} and (not requested or job["jobId"] in requested):
                    job.update(status="cancelled", stage="cancelled")
            self._write_batch(batch)
        return {"ok": True, "batch": batch}

    def resume_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = self.batch_status(payload)
        if not status["ok"]:
            return status
        batch = self.batches[status["batch"]["batchId"]]
        retry_failed = bool(payload.get("retryFailed"))
        with self._lock:
            for job in batch["jobs"]:
                if job["status"] == "interrupted" or (retry_failed and job["status"] == "failed"):
                    job.update(status="queued", stage="queued", error=None)
            if not any(job["status"] == "queued" for job in batch["jobs"]):
                return {"ok": False, "error": "Der Batch enthält keine fortsetzbaren Aufträge."}
            batch["status"] = "running"
            self._write_batch(batch)
        threading.Thread(target=self._run_batch, args=(batch["batchId"],), daemon=True).start()
        return {"ok": True, "batchId": batch["batchId"], "batch": batch}

    def save_correction(self, payload: dict[str, Any]) -> dict[str, Any]:
        specimen = _safe_name(str(payload.get("specimen", "")))
        codes = set(load_catalog().codes)
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

    def save_assessment(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = str(payload.get("runId", ""))
        found = self._result_by_run_id(run_id)
        if not found:
            return {"ok": False, "error": "Analyselauf wurde nicht gefunden."}
        result_path, result = found
        codes = set(load_catalog().codes)
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

    def export_overview(self, _payload: Any = None) -> dict[str, Any]:
        target = self._project_data() / f"{_safe_name(self.project_name)}_Uebersicht.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["Prüfkörper", "Run_ID", "Zeitstempel", "Provider", "Modell", "Status",
                             "Primärklasse", "Sekundärklassen", "Konfidenz", "Laufzeit_s",
                             "Prompt_Hash", "Bewertung", "Ground_Truth_Primärklasse"])
            for specimen in self._specimens():
                records = self._result_records(specimen.name)
                if not records:
                    writer.writerow([specimen.name, "", "", "", "", "NICHT_ANALYSIERT"]); continue
                for record in records:
                    assessment = self._assessment(record["runId"]) or {}
                    writer.writerow([
                        specimen.name, record["runId"], record["createdAt"], record["provider"],
                        record["model"], record["status"], record["primaryCode"],
                        "|".join(filter(None, record["secondaryCodes"])), record["primaryConfidence"],
                        record["durationSeconds"], record["promptHash"], assessment.get("verdict", ""),
                        assessment.get("primaryCode", ""),
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
            primary = result.get("primary_feature") or {}
            execution = document.get("execution", {})
            assessment = self._assessment(run_id)
            rows.append([
                specimen, run_id, execution.get("provider"), execution.get("model"),
                primary.get("code", ""), primary.get("confidence", ""),
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
                             "Konfidenz", "Sekundärklassen", "Laufzeit_s", "Prompt_Hash",
                             "Bewertung", "Ground_Truth_Primärklasse"])
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
        primary = result["result"].get("primary_feature") or {}
        with csv_target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["Prüfkörper", "Modell", "Laufzeit_s", "KI_Primärklasse",
                             "KI_Konfidenz", "Einbezogene_Bilder", "Ausgeschlossene_Bilder",
                             "Korrektur_Primärklasse", "Korrektur_Sekundärklassen"])
            writer.writerow([specimen, result["execution"]["model"],
                             result["execution"].get("duration_seconds", ""), primary.get("code", ""),
                             primary.get("confidence", ""),
                             "|".join(result.get("image_selection", {}).get("included", [])),
                             "|".join(result.get("image_selection", {}).get("excluded", [])),
                             (correction or {}).get("primaryCode", ""),
                             "|".join((correction or {}).get("secondaryCodes", []))])
        return {"ok": True, "jsonPath": str(json_target), "csvPath": str(csv_target)}
