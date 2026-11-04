from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


SUPPORTED_REFERENCE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError("Bitte einen gültigen Wert eingeben.")
    return cleaned


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class ReferenceStore:
    """Editable reference metadata stored next to copied reference images.

    The store never deletes source images. Existing unregistered images are exposed as
    unconfirmed records and become manifest entries only after an explicit edit.
    """

    def __init__(self, project_root: Path, definitions: Mapping[str, Mapping[str, Any]]) -> None:
        self.project_root = project_root.resolve()
        self.root = self.project_root / "Datenbank/Referenzbilder_neu"
        self.config_path = self.root / "aktive_hauptversagensarten.json"
        self.manifest_path = self.root / "referenzbilder_manifest.json"
        self.definitions = {str(code): dict(value) for code, value in definitions.items()}

    @property
    def documented_codes(self) -> list[str]:
        documented = [
            (code, definition) for code, definition in self.definitions.items()
            if isinstance(definition.get("validated_classification_rule"), Mapping)
        ]
        documented.sort(
            key=lambda item: item[1].get("validated_classification_rule_order", 10**9)
        )
        return [code for code, _definition in documented]

    def _config(self) -> dict[str, Any]:
        if not self.config_path.is_file():
            return {
                "schemaVersion": "1.0",
                "requiredCount": len(self.documented_codes),
                "selectedCodes": self.documented_codes,
                "configured": bool(self.documented_codes),
            }
        document = json.loads(self.config_path.read_text(encoding="utf-8"))
        stored = document.get("selectedCodes", [])
        stored_codes = stored if isinstance(stored, list) else []
        # klassifikationsregeln.json ist die verbindliche Quelle. Neue, dort
        # fachlich validierte Klassen werden damit ohne Codeänderung aktiv.
        return {
            **document,
            "requiredCount": len(self.documented_codes),
            "selectedCodes": self.documented_codes,
            "storedCodes": stored_codes,
            "configurationOutOfSync": stored_codes != self.documented_codes,
            "configured": bool(self.documented_codes),
        }

    def save_active_codes(
        self, codes: Sequence[str], *, confirm_replace: bool = False
    ) -> dict[str, Any]:
        normalized = [str(code).strip() for code in codes]
        if normalized != self.documented_codes or len(set(normalized)) != len(normalized):
            raise ValueError(
                "Die aktive Auswahl muss exakt den in klassifikationsregeln.json "
                "dokumentierten Hauptversagenscodes entsprechen."
            )
        unknown = [code for code in normalized if code not in self.definitions]
        if unknown:
            raise ValueError("Unbekannte Versagenscodes: " + ", ".join(unknown))
        current = self._config()
        if current.get("configured") and current.get("selectedCodes") != normalized and not confirm_replace:
            raise ValueError(
                "Die aktive fachliche Klassenauswahl ist bereits gespeichert. "
                "Eine Änderung muss ausdrücklich bestätigt werden."
            )
        document = {
            "schemaVersion": "1.0",
            "requiredCount": len(self.documented_codes),
            "selectedCodes": normalized,
            "configured": True,
            "updatedAt": _now(),
            "note": "Fachlich festgelegte Hauptversagensarten für Holz-Zugversuche.",
        }
        _atomic_json(self.config_path, document)
        return document

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"schemaVersion": "1.0", "entries": []}
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        entries = document.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("Das Referenzbildmanifest enthält keine gültige Eintragsliste.")
        return {**document, "entries": entries}

    def _write_manifest(self, entries: Sequence[Mapping[str, Any]]) -> None:
        _atomic_json(self.manifest_path, {
            "schemaVersion": "1.0",
            "updatedAt": _now(),
            "entries": list(entries),
        })

    def _class_directory(self, code: str) -> Path:
        prefix = _safe_component(code)
        self.root.mkdir(parents=True, exist_ok=True)
        candidates = sorted(
            path for path in self.root.iterdir()
            if path.is_dir() and (path.name == prefix or path.name.startswith(prefix + "_"))
        )
        if candidates:
            return candidates[0]
        name = _safe_component(str(self.definitions[code].get("name", code)))
        return self.root / f"{prefix}_{name}"

    def _infer_code(self, path: Path) -> str | None:
        try:
            folder = path.relative_to(self.root).parts[0]
        except (ValueError, IndexError):
            return None
        matches = [
            code for code in self.definitions
            if folder == _safe_component(code) or folder.startswith(_safe_component(code) + "_")
        ]
        return max(matches, key=len) if matches else None

    def _paths(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            path.resolve() for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_REFERENCE_SUFFIXES
        )

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.project_root))

    def _unregistered_entry(self, path: Path) -> dict[str, Any] | None:
        code = self._infer_code(path)
        if not code:
            return None
        relative = self._relative(path)
        return {
            "id": "unregistered-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20],
            "code": code,
            "specimenId": "",
            "description": "",
            "confirmed": False,
            "confirmedBy": "",
            "confirmedAt": "",
            "originalFilename": path.name,
            "sourcePath": "",
            "storedPath": relative,
            "sha256": _sha256(path),
            "registered": False,
            "createdAt": "",
            "updatedAt": "",
        }

    def entries(self) -> list[dict[str, Any]]:
        manifest_entries = [dict(item) for item in self._manifest()["entries"]]
        registered_paths = {
            str((self.project_root / str(item.get("storedPath", ""))).resolve())
            for item in manifest_entries if item.get("storedPath")
        }
        for path in self._paths():
            if str(path) not in registered_paths:
                entry = self._unregistered_entry(path)
                if entry:
                    manifest_entries.append(entry)
        first_by_hash: dict[str, str] = {}
        for entry in manifest_entries:
            sha = str(entry.get("sha256", ""))
            if sha in first_by_hash:
                entry["duplicateOf"] = first_by_hash[sha]
            elif sha:
                first_by_hash[sha] = str(entry.get("id", ""))
        return manifest_entries

    def catalog(self) -> dict[str, Any]:
        config = self._config()
        entries = self.entries()
        active = config.get("selectedCodes", []) if config.get("configured") else []
        classes = []
        for code in active:
            definition = self.definitions[code]
            references = [item for item in entries if item.get("code") == code]
            classes.append({
                "code": code,
                "name": definition.get("name", ""),
                "definition": definition.get("definition", ""),
                "featureType": definition.get("feature_type", ""),
                "reviewRequired": bool(definition.get("review_required")),
                "visualFeatures": definition.get("visual_features", []),
                "sources": definition.get("sources", []),
                "references": references,
                "confirmedCount": sum(bool(item.get("confirmed")) for item in references),
            })
        return {
            "ok": True,
            "configured": bool(config.get("configured")),
            "requiredCount": len(self.documented_codes),
            "activeCodes": active,
            "documentedCodes": self.documented_codes,
            "availableClasses": [
                {"code": code, "name": value.get("name", "")}
                for code, value in self.definitions.items()
            ],
            "classes": classes,
            "storageRoot": str(self.root),
            "manifestPath": str(self.manifest_path),
            "configurationPath": str(self.config_path),
            "message": (
                "In klassifikationsregeln.json sind keine Hauptversagensklassen dokumentiert."
                if not config.get("configured") else ""
            ),
        }

    @staticmethod
    def _verify_image(path: Path) -> None:
        if path.suffix.lower() not in SUPPORTED_REFERENCE_SUFFIXES:
            raise ValueError(f"Nicht unterstütztes Bildformat: {path.name}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Bilddatei ist nicht lesbar: {path.name}") from exc

    def import_images(
        self,
        *,
        code: str,
        paths: Sequence[Path],
        specimen_id: str,
        description: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        config = self._config()
        if not config.get("configured") or code not in config.get("selectedCodes", []):
            raise ValueError("Die Klasse gehört nicht zur aktiven fachlichen Klassenauswahl.")
        specimen = _safe_component(specimen_id)
        if not paths:
            raise ValueError("Bitte mindestens ein Referenzbild auswählen.")
        existing = self.entries()
        known_hashes = {str(item.get("sha256")): item for item in existing}
        manifest = self._manifest()
        manifest_entries = [dict(item) for item in manifest["entries"]]
        destination = self._class_directory(code) / specimen
        destination.mkdir(parents=True, exist_ok=True)
        imported: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        for raw in paths:
            source = raw.expanduser().resolve()
            if not source.is_file():
                raise ValueError(f"Bilddatei wurde nicht gefunden: {source}")
            self._verify_image(source)
            sha = _sha256(source)
            if sha in known_hashes:
                duplicates.append({
                    "source": str(source),
                    "sha256": sha,
                    "existingId": known_hashes[sha].get("id"),
                    "existingPath": known_hashes[sha].get("storedPath"),
                })
                continue
            target = destination / source.name
            index = 2
            while target.exists():
                target = destination / f"{source.stem}_{index}{source.suffix}"
                index += 1
            shutil.copy2(source, target)
            entry = {
                "id": uuid.uuid4().hex,
                "code": code,
                "specimenId": specimen_id.strip(),
                "description": description.strip(),
                "confirmed": bool(confirmed),
                "confirmedBy": "App-Benutzerin" if confirmed else "",
                "confirmedAt": _now() if confirmed else "",
                "originalFilename": source.name,
                "sourcePath": str(source),
                "storedPath": self._relative(target),
                "sha256": sha,
                "registered": True,
                "createdAt": _now(),
                "updatedAt": _now(),
            }
            manifest_entries.append(entry)
            imported.append(entry)
            known_hashes[sha] = entry
        if imported:
            self._write_manifest(manifest_entries)
        return {"ok": True, "imported": imported, "duplicates": duplicates}

    def update_entry(
        self,
        entry_id: str,
        *,
        code: str,
        specimen_id: str,
        description: str,
        confirmed: bool,
        confirm_confirmed_change: bool = False,
    ) -> dict[str, Any]:
        config = self._config()
        if code not in config.get("selectedCodes", []):
            raise ValueError("Die Zielklasse gehört nicht zur aktiven fachlichen Klassenauswahl.")
        all_entries = self.entries()
        current = next((item for item in all_entries if item.get("id") == entry_id), None)
        if current is None:
            raise ValueError("Referenzbildeintrag wurde nicht gefunden.")
        protected_change = bool(current.get("confirmed")) and (
            current.get("code") != code
            or current.get("specimenId", "") != specimen_id.strip()
            or current.get("description", "") != description.strip()
            or not confirmed
        )
        if protected_change and not confirm_confirmed_change:
            raise ValueError(
                "Der bereits bestätigte Eintrag darf nur nach ausdrücklicher Bestätigung geändert werden."
            )
        manifest_entries = [dict(item) for item in self._manifest()["entries"]]
        manifest_entry = next((item for item in manifest_entries if item.get("id") == entry_id), None)
        if manifest_entry is None:
            manifest_entry = {key: value for key, value in current.items() if key != "duplicateOf"}
            manifest_entry["id"] = uuid.uuid4().hex
            manifest_entry["registered"] = True
            manifest_entry["createdAt"] = _now()
            manifest_entries.append(manifest_entry)
        manifest_entry.update({
            "code": code,
            "specimenId": specimen_id.strip(),
            "description": description.strip(),
            "confirmed": bool(confirmed),
            "confirmedBy": "App-Benutzerin" if confirmed else "",
            "confirmedAt": _now() if confirmed else "",
            "updatedAt": _now(),
        })
        self._write_manifest(manifest_entries)
        return {"ok": True, "entry": manifest_entry}
