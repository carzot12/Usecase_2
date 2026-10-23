from __future__ import annotations

import base64
import csv
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
        self._lock = threading.RLock()

    def bootstrap(self, _payload: Any = None) -> dict[str, Any]:
        return {"ok": True, "exampleAvailable": EXAMPLE.is_dir(), "examplePath": str(EXAMPLE),
                "classes": [{"code": c, "name": d["name"]}
                            for c, d in load_catalog().definitions.items()]}

    def _project_data(self) -> Path:
        if not self.project_name:
            raise RuntimeError("Kein Projekt geöffnet.")
        path = APP_DATA / _safe_name(self.project_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

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
        folder = self._project_data() / "Ergebnisse"
        files = sorted(folder.glob(f"{specimen}_*.json")) if folder.is_dir() else []
        return files[-1] if files else None

    def _snapshot(self) -> dict[str, Any]:
        return {"name": self.project_name, "path": str(self.project_root), "specimens": [
            {"id": p.name, "imageCount": len(self._images(p)), "hasResult": self._latest_result(p.name) is not None}
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
        analysis_dir = folder / "Analyseauswahl"
        selected_paths = {
            p.resolve()
            for p in analysis_dir.iterdir()
            if analysis_dir.is_dir() and (p.is_file() or p.is_symlink())
        } if analysis_dir.is_dir() else set()
        source_images = self._images(folder)
        source_images.sort(key=lambda p: (p.resolve() not in selected_paths, p.name))
        images = [
            {"name": p.name, "src": self._image_data(p), "analysisSelected": p.resolve() in selected_paths}
            for p in source_images
        ]
        latest = self._latest_result(specimen)
        result = json.loads(latest.read_text()) if latest else None
        correction_path = self._project_data() / "Korrekturen" / f"{specimen}.json"
        correction = json.loads(correction_path.read_text()) if correction_path.is_file() else None
        return {"ok": True, "specimen": specimen, "images": images, "result": result,
                "resultPath": str(latest) if latest else None, "correction": correction}

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
        choice = str(payload.get("model", "qwen"))
        confirm = bool(payload.get("confirmPaid"))
        configs = {"qwen": ("ollama", "qwen3-vl:4b", 24576),
                   "llava": ("ollama", "llava:latest", 16384),
                   "openai": ("openai", "gpt-6.1-sol", 16384),
                   "mistral": ("mistral", "ministral-3b-2512", 16384)}
        if choice not in configs:
            return {"ok": False, "error": "Unbekanntes Modell."}
        provider, model, context = configs[choice]
        if provider in {"openai", "mistral"} and not confirm:
            return {"ok": False, "needsConfirmation": True,
                    "error": f"{provider.title()} erfordert eine ausdrückliche Kostenbestätigung."}
        job_id = uuid.uuid4().hex
        self.jobs[job_id] = {"state": "running", "status": "Bilder und Referenzen werden vorbereitet …",
                             "model": model, "specimen": specimen}

        def worker() -> None:
            try:
                self.jobs[job_id]["status"] = f"{model} analysiert 3 Prüfkörperbilder und Referenzen …"
                destination = run_analysis(self.project_root / specimen, provider=provider, model=model,
                                           live=provider in {"openai", "mistral"},
                                           confirm_paid_request=confirm,
                                           ollama_num_ctx=context, max_references_per_class=1,
                                           allowed_input_root=self.project_root,
                                           output_dir=self._project_data() / "Ergebnisse")
                self.jobs[job_id].update(state="complete", status="Analyse abgeschlossen.", path=str(destination),
                                         result=json.loads(destination.read_text()))
            except Exception as exc:
                self.jobs[job_id].update(state="error", status="Analyse fehlgeschlagen.", error=str(exc))

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "jobId": job_id}

    def analysis_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        job = self.jobs.get(str(payload.get("jobId", "")))
        return {"ok": bool(job), **(job or {"error": "Analyseauftrag nicht gefunden."})}

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

    def export_overview(self, _payload: Any = None) -> dict[str, Any]:
        target = self._project_data() / f"{_safe_name(self.project_name)}_Uebersicht.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["Prüfkörper", "Status", "Primärklasse", "Modell", "Laufzeit_s"])
            for specimen in self._specimens():
                result_path = self._latest_result(specimen.name)
                if not result_path:
                    writer.writerow([specimen.name, "NICHT_ANALYSIERT", "", "", ""]); continue
                data = json.loads(result_path.read_text()); result = data["result"]
                primary = result.get("primary_feature") or {}
                writer.writerow([specimen.name, result["status"], primary.get("code", ""),
                                 data["execution"]["model"], data["execution"].get("duration_seconds", "")])
        return {"ok": True, "path": str(target)}

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
                             "KI_Konfidenz", "Korrektur_Primärklasse", "Korrektur_Sekundärklassen"])
            writer.writerow([specimen, result["execution"]["model"],
                             result["execution"].get("duration_seconds", ""), primary.get("code", ""),
                             primary.get("confidence", ""), (correction or {}).get("primaryCode", ""),
                             "|".join((correction or {}).get("secondaryCodes", []))])
        return {"ok": True, "jsonPath": str(json_target), "csvPath": str(csv_target)}
