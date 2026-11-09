"""Scientific evaluation variants, kept separate from the legacy A--D evaluation.

This module only inspects explicitly configured evaluation reference folders.  It
never reads protocols, ground truth, previous model results, or expert reviews.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_RULES_PATH = Path(__file__).resolve().parents[1] / "Datenbank/klassifikationsregeln.json"


def load_active_class_codes(path: Path = _RULES_PATH) -> tuple[str, ...]:
    """Load the single active feature order shared by UI, prompts and evaluation."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Aktive Merkmalsklassen können nicht gelesen werden: {path}") from exc
    codes = tuple(str(code).strip() for code in document.get("meta", {}).get(
        "active_class_codes", ()
    ))
    if not codes or len(codes) != len(set(codes)):
        raise RuntimeError("meta.active_class_codes muss eine nichtleere eindeutige Liste sein.")
    return codes


ACTIVE_CLASS_CODES = load_active_class_codes()
# Compatibility export for older callers. These codes are no longer secondary;
# all twelve entries above are equal reference features.
ADDITIONAL_VISIBLE_CODES: tuple[str, ...] = ()
VARIANT_IDS = ("1", "2A", "2B", "3")
MAX_REFERENCE_BATCH_SIZE = 8
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
CLASS_DIRECTORIES = {
    "GSF": "GSF", "LSF": "LSF", "KA": "KA", "SA": "SA",
    "sp HB": "sp_HB", "Spl HB": "Spl_HB", "RiSch": "RiSch", "LR": "LR",
    "Bu": "Bu", "AA": "AA", "SSA": "SSA", "KB": "KB",
}
Z30_NAME_PATTERN = re.compile(r"(?i)(?:^|[^a-z0-9])z[-_ ]?30(?:[^0-9]|$)")


class VariantConfigurationError(RuntimeError):
    pass


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def _z30_test_image_hashes() -> frozenset[str]:
    """Hashes of the 49 scientific Z-30 test specimens (30 has no folder)."""

    test_root = _RULES_PATH.parents[1] / "Pruefdaten"
    hashes: set[str] = set()
    valid_names = {f"Z-30-{number:02d}" for number in (*range(1, 30), *range(31, 51))}
    for folder in test_root.rglob("Z-30-*"):
        if not folder.is_dir() or folder.name not in valid_names:
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                hashes.add(_file_hash(path))
    return frozenset(hashes)


def _test_data_conflicts(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    test_hashes = _z30_test_image_hashes()
    conflicts: list[dict[str, str]] = []
    for item in records:
        relative_path = str(item.get("relative_path", ""))
        reasons: list[str] = []
        if Z30_NAME_PATTERN.search(Path(relative_path).name):
            reasons.append("Dateiname verweist auf Z-30")
        if str(item.get("sha256", "")) in test_hashes:
            reasons.append("SHA-256 entspricht einem Bild der 49 Z-30-Testprüfkörper")
        if reasons:
            conflicts.append({"relative_path": relative_path, "reason": "; ".join(reasons)})
    return conflicts


def _read_manifest(root: Path, variant_id: str) -> tuple[Path, dict[str, Any]]:
    if variant_id not in VARIANT_IDS:
        raise VariantConfigurationError(f"Unbekannte Evaluationsvariante: {variant_id}")
    path = root / f"Variante_{variant_id}" / "variant_manifest.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VariantConfigurationError(f"Variantenmanifest kann nicht gelesen werden: {path}") from exc
    if document.get("variant_id") != variant_id:
        raise VariantConfigurationError(f"variant_id in {path} stimmt nicht mit {variant_id} überein.")
    manifest_codes = tuple(document.get("active_class_codes", ()))
    if manifest_codes not in {ACTIVE_CLASS_CODES, ACTIVE_CLASS_CODES[:8]}:
        raise VariantConfigurationError("Das Variantenmanifest muss die zwölf Merkmale in festgelegter Reihenfolge enthalten.")
    return path, document


def feature_requirements(manifest: Mapping[str, Any]) -> dict[str, dict[str, int | None]]:
    """Return validated per-feature requirements with legacy-field fallback."""

    configured = manifest.get("reference_requirements_by_feature", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise VariantConfigurationError("reference_requirements_by_feature muss ein Objekt sein.")
    result: dict[str, dict[str, int | None]] = {}
    for code in ACTIVE_CLASS_CODES:
        raw = configured.get(code, {})
        if not isinstance(raw, Mapping):
            raise VariantConfigurationError(f"Referenzanforderung für {code} muss ein Objekt sein.")
        values: dict[str, int | None] = {}
        for key, legacy_key in (
            ("minimum", "min_references_per_class"),
            ("target", "target_references_per_class"),
            ("maximum", "max_references_per_class"),
        ):
            value = raw.get(key, manifest.get(legacy_key))
            if value is None:
                values[key] = None
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VariantConfigurationError(f"{code}: {key} muss eine nichtnegative Ganzzahl oder null sein.")
            else:
                values[key] = value
        minimum = values["minimum"] or 0
        maximum = values["maximum"]
        target = values["target"]
        if maximum is not None and maximum < minimum:
            raise VariantConfigurationError(f"{code}: maximum darf nicht kleiner als minimum sein.")
        if target is not None and (target < minimum or (maximum is not None and target > maximum)):
            raise VariantConfigurationError(f"{code}: target liegt außerhalb von minimum/maximum.")
        result[code] = values
    unknown = sorted(set(configured) - set(ACTIVE_CLASS_CODES))
    if unknown:
        raise VariantConfigurationError("Unbekannte Merkmale in reference_requirements_by_feature: " + ", ".join(unknown))
    return result


def _atomic_manifest(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class ReferenceRecord:
    reference_id: str
    code: str
    path: Path
    relative_path: str
    sha256: str
    description: str = ""
    series: str = ""
    specimen: str = ""
    original_path: str = ""
    confirmed: bool = False
    reviewed_by: str = ""
    reviewed_at: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id, "code": self.code,
            "relative_path": self.relative_path, "sha256": self.sha256,
            "description": self.description, "series": self.series,
            "specimen": self.specimen, "original_path": self.original_path,
            "confirmed": self.confirmed, "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
        }


ANNOTATION_CSV_FIELDS = (
    "reference_id", "source_reference_id", "source_relative_path", "source_sha256",
    "class_code", "annotated_path", "description", "description_source",
    "description_status", "reviewed_by", "reviewed_at",
)

REFERENCE_CSV_FIELDS = (
    "merkmal", "dateiname", "versuchsserie", "pruefkoerper", "originalpfad",
    "begruendung", "geprueft_von", "geprueft_am",
)


def _reference_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read the semicolon separated provenance and review ledger."""

    if not path.is_file():
        return [], [f"Referenznachweis fehlt: {path}"]
    result: list[dict[str, str]] = []
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            if tuple(reader.fieldnames or ()) != REFERENCE_CSV_FIELDS:
                errors.append(
                    "referenzen.csv muss exakt diese Spalten besitzen: "
                    + ";".join(REFERENCE_CSV_FIELDS)
                )
            for line, row in enumerate(reader, start=2):
                normalized = {
                    field: str(row.get(field, "") or "").strip()
                    for field in REFERENCE_CSV_FIELDS
                }
                normalized["csv_line"] = str(line)
                result.append(normalized)
    except (OSError, csv.Error) as exc:
        errors.append(f"Referenznachweis kann nicht gelesen werden: {exc}")
    return result, errors


def _annotation_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], [f"Beschreibungsdatei fehlt: {path}"]
    result: list[dict[str, str]] = []
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ANNOTATION_CSV_FIELDS:
                errors.append(
                    "descriptions.csv muss exakt diese Spalten besitzen: "
                    + ",".join(ANNOTATION_CSV_FIELDS)
                )
            for line, row in enumerate(reader, start=2):
                normalized = {field: str(row.get(field, "") or "").strip() for field in ANNOTATION_CSV_FIELDS}
                normalized["csv_line"] = str(line)
                result.append(normalized)
    except (OSError, csv.Error) as exc:
        errors.append(f"Beschreibungsdatei kann nicht gelesen werden: {exc}")
    return result, errors


def _inspect_annotated_variant(
    base_root: Path, manifest_path: Path, manifest: Mapping[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    reference_root = manifest_path.parent.resolve()
    report["referenceRoot"] = str(reference_root)
    report["sourceVariant"] = str(manifest.get("source_variant", ""))
    report["generalClassDescriptionsSource"] = "Datenbank/klassifikationsregeln.json"
    description_path = (manifest_path.parent / str(manifest.get("descriptions_file", ""))).resolve()
    rows, csv_errors = _annotation_rows(description_path)
    report["errors"].extend(csv_errors)
    report["descriptionsFile"] = str(description_path)
    report["descriptionsCsvSha256"] = _file_hash(description_path) if description_path.is_file() else None

    expected_source_ids_raw = manifest.get("expected_source_reference_ids", [])
    expected_source_ids = {
        str(item) for item in expected_source_ids_raw if isinstance(item, str) and item.strip()
    }
    if bool(manifest.get("require_source_reference_link")) and not expected_source_ids:
        report["errors"].append(
            "expected_source_reference_ids fehlt; Variante 3 benötigt einen eigenen Herkunftsnachweis."
        )

    actual_files: dict[str, Path] = {}
    for code in ACTIVE_CLASS_CODES:
        folder = reference_root / CLASS_DIRECTORIES[code]
        if not folder.is_dir():
            report["errors"].append(f"Klassenordner fehlt: {folder}")
            continue
        for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file() or path.name.startswith("."):
                continue
            relative = path.relative_to(reference_root).as_posix()
            if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                report["unsupportedFiles"].append(relative)
            else:
                actual_files[relative] = path

    seen_ids: set[str] = set()
    used_source_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: dict[str, str] = {}
    csv_paths: set[str] = set()
    records: list[dict[str, Any]] = []
    missing_descriptions: list[str] = []
    unknown_sources: list[str] = []
    missing_source_provenance: list[str] = []
    descriptions_for_review: list[str] = []
    unreviewed: list[str] = []
    csv_without_images: list[str] = []
    require_review = bool(manifest.get("require_review", False))

    for row in rows:
        line = row["csv_line"]
        reference_id = row["reference_id"]
        source_id = row["source_reference_id"]
        code = row["class_code"]
        relative = Path(row["annotated_path"]).as_posix() if row["annotated_path"] else ""
        if not reference_id:
            report["errors"].append(f"CSV-Zeile {line}: reference_id fehlt.")
        elif reference_id in seen_ids:
            report["errors"].append(f"CSV-Zeile {line}: reference_id ist doppelt: {reference_id}")
        seen_ids.add(reference_id)
        if code not in ACTIVE_CLASS_CODES:
            report["errors"].append(f"CSV-Zeile {line}: unbekannter class_code {code!r}.")
        if not row["description"]:
            missing_descriptions.append(reference_id or f"Zeile {line}")
        if row["description_status"] not in {
            "pocket_verified", "user_verified", "rule_v1.4_fallback",
        }:
            descriptions_for_review.append(reference_id or f"Zeile {line}")
        if source_id not in expected_source_ids:
            unknown_sources.append(source_id or f"Zeile {line}")
        elif source_id in used_source_ids:
            report["errors"].append(f"CSV-Zeile {line}: source_reference_id ist doppelt: {source_id}")
        else:
            used_source_ids.add(source_id)
        if not row["source_relative_path"] or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
            missing_source_provenance.append(reference_id or f"Zeile {line}")
        if require_review and (not row["reviewed_by"] or not row["reviewed_at"]):
            unreviewed.append(reference_id or f"Zeile {line}")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            report["errors"].append(f"CSV-Zeile {line}: annotated_path ist ungültig.")
            continue
        csv_paths.add(relative)
        if relative in seen_paths:
            report["errors"].append(f"CSV-Zeile {line}: annotierte Datei ist doppelt zugeordnet: {relative}")
        seen_paths.add(relative)
        annotated = (reference_root / relative).resolve()
        if annotated.parent != (reference_root / CLASS_DIRECTORIES.get(code, "__invalid__")).resolve():
            report["errors"].append(f"CSV-Zeile {line}: annotated_path liegt nicht im Ordner der Klasse {code}.")
        if relative not in actual_files:
            csv_without_images.append(relative)
            continue
        if annotated.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            report["errors"].append(f"CSV-Zeile {line}: nicht unterstütztes Bildformat: {relative}")
            continue
        annotated_sha = _file_hash(annotated)
        if annotated_sha in seen_hashes:
            report["duplicateHashes"].append({
                "sha256": annotated_sha, "first": seen_hashes[annotated_sha], "duplicate": reference_id,
            })
        else:
            seen_hashes[annotated_sha] = reference_id
        if row["source_sha256"] and annotated_sha == row["source_sha256"]:
            report["errors"].append(f"{reference_id}: Annotation und dokumentiertes Original sind bytegleich.")
        records.append({
            "reference_id": reference_id, "source_reference_id": source_id,
            "code": code, "class_code": code, "relative_path": relative,
            "annotated_path": relative, "sha256": annotated_sha,
            "source_relative_path": row["source_relative_path"],
            "source_sha256": row["source_sha256"],
            "description": row["description"],
            "description_source": row["description_source"],
            "description_status": row["description_status"],
            "reviewed_by": row["reviewed_by"],
            "reviewed_at": row["reviewed_at"], "confirmed": bool(row["reviewed_by"] and row["reviewed_at"]),
        })
        if code in report["countsByClass"]:
            report["countsByClass"][code] += 1

    images_without_csv = sorted(set(actual_files) - csv_paths)
    report["references"] = records
    report["referenceCount"] = len(records)
    report["missingDescriptions"] = missing_descriptions
    report["unknownSourceReferenceIds"] = unknown_sources
    report["missingSourceProvenance"] = missing_source_provenance
    report["descriptionReviewRequiredIds"] = descriptions_for_review
    report["imagesWithoutCsv"] = images_without_csv
    report["csvWithoutImages"] = csv_without_images
    report["unreviewedReferenceIds"] = unreviewed
    report["descriptionsComplete"] = not missing_descriptions and bool(records)
    report["sourceLinksComplete"] = (
        not unknown_sources and not missing_source_provenance and bool(records)
    )
    missing_source_annotations = sorted(expected_source_ids - used_source_ids)
    report["missingSourceAnnotations"] = missing_source_annotations
    report["testDataConflicts"] = _test_data_conflicts(records)
    report["reviewRequired"] = require_review
    report["reviewComplete"] = not unreviewed and bool(records)
    requirements = feature_requirements(manifest)
    for code, count in report["countsByClass"].items():
        minimum = requirements[code]["minimum"] or 0
        maximum = requirements[code]["maximum"]
        target = requirements[code]["target"]
        if count < minimum:
            report["missingByClass"][code] = minimum - count
        if maximum is not None and count > maximum:
            report["errors"].append(f"{code}: {count} Annotationen überschreiten das Maximum {maximum}.")
        if target is not None and count < target:
            report["warnings"].append(f"{code}: Zielbestand noch nicht erreicht: {count} von {target}.")
    if missing_descriptions:
        report["errors"].append(f"Für {len(missing_descriptions)} Annotationen fehlt die Bildbeschreibung.")
    if descriptions_for_review:
        report["errors"].append(
            f"Für {len(descriptions_for_review)} Annotationen ist die Beschreibungsquelle noch manuell zu prüfen."
        )
    if unknown_sources:
        report["errors"].append(
            f"{len(unknown_sources)} source_reference_id-Werte gehören nicht zum dokumentierten 2B-Herkunftsbestand."
        )
    if missing_source_provenance:
        report["errors"].append(
            f"Für {len(missing_source_provenance)} Annotationen fehlt der eigenständige Herkunftshash oder -pfad."
        )
    if missing_source_annotations:
        report["warnings"].append(
            f"{len(missing_source_annotations)} Bilder aus Variante 2B werden in Variante 3 nicht verwendet."
        )
    if report["testDataConflicts"]:
        report["errors"].append("Referenzbilder aus der Z-30-Testserie sind ausgeschlossen.")
    if images_without_csv:
        report["warnings"].append(
            f"{len(images_without_csv)} Bilddateien ohne CSV-Eintrag werden nicht als Referenz verwendet."
        )
    if csv_without_images:
        report["errors"].append(f"{len(csv_without_images)} CSV-Einträge besitzen keine Bilddatei.")
    if unreviewed:
        report["errors"].append(f"{len(unreviewed)} Annotationen sind nicht vollständig fachlich geprüft.")
    if report["unsupportedFiles"]:
        report["errors"].append("Nicht unterstützte Dateien in Variante 3 gefunden.")
    if report["duplicateHashes"]:
        report["errors"].append("Doppelte SHA-256-Hashes bei annotierten Bildern gefunden.")
    report["referenceBatchCount"] = 0
    report["directReferenceImageCount"] = len(records)
    report["modelCallCount"] = 1
    report["cloudCallCount"] = report["modelCallCount"]
    ready = bool(manifest.get("enabled")) and not report["errors"] and not report["missingByClass"]
    report["ready"] = ready
    report["readiness"] = "BEREIT" if ready else "NICHT BEREIT"
    return report


def inspect_variant(base_root: Path, variant_id: str) -> dict[str, Any]:
    """Return a deterministic, JSON-safe readiness report without mutating data."""

    manifest_path, manifest = _read_manifest(base_root, variant_id)
    report: dict[str, Any] = {
        "variantId": variant_id, "name": manifest.get("name", f"Variante {variant_id}"),
        "description": manifest.get("notes", ""), "manifestPath": str(manifest_path),
        "manifestHash": _canonical_hash(manifest), "enabled": bool(manifest.get("enabled")),
        "revisionId": manifest.get("revision_id"),
        "revisionStatus": manifest.get("revision_status"),
        "requiredProvider": manifest.get("provider"),
        "requiredModel": manifest.get("model"),
        "promptVersion": manifest.get("prompt_version"),
        "classificationRulesVersion": manifest.get("classification_rules_version"),
        "referenceMode": manifest.get("reference_mode"), "referenceRoot": None,
        "activeClassCodes": list(ACTIVE_CLASS_CODES), "countsByClass": {c: 0 for c in ACTIVE_CLASS_CODES},
        "referenceCount": 0, "references": [], "unsupportedFiles": [], "duplicateHashes": [],
        "missingByClass": {}, "descriptionsRequired": bool(manifest.get("descriptions_required")),
        "descriptionsComplete": variant_id != "3", "errors": [], "warnings": [],
        "testDataConflicts": [],
        "featureRequirements": feature_requirements(manifest),
    }
    if variant_id == "1":
        report.update({"ready": bool(manifest.get("enabled")), "referenceBatchCount": 0,
                       "modelCallCount": 1, "cloudCallCount": 1,
                       "readiness": "BEREIT" if manifest.get("enabled") else "NICHT BEREIT"})
        return report

    if variant_id == "3":
        return _inspect_annotated_variant(base_root, manifest_path, manifest, report)

    reference_root_value = str(manifest.get("reference_root", "")).strip()
    if not reference_root_value:
        report["errors"].append("reference_root fehlt im Manifest.")
        reference_root = manifest_path.parent
    else:
        reference_root = (manifest_path.parent / reference_root_value).resolve()
    report["referenceRoot"] = str(reference_root)
    reference_csv = manifest_path.parent / "referenzen.csv"
    rows, csv_errors = _reference_rows(reference_csv)
    report["referenceEvidenceFile"] = str(reference_csv)
    report["referenceEvidenceSha256"] = (
        _file_hash(reference_csv) if reference_csv.is_file() else None
    )
    report["errors"].extend(csv_errors)
    metadata_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        line = row["csv_line"]
        code = row["merkmal"]
        filename = row["dateiname"]
        key = (code, filename)
        if code not in ACTIVE_CLASS_CODES:
            report["errors"].append(
                f"referenzen.csv Zeile {line}: unbekanntes Merkmal {code!r}."
            )
        if not filename or Path(filename).name != filename:
            report["errors"].append(
                f"referenzen.csv Zeile {line}: dateiname muss ein einfacher Dateiname sein."
            )
        if key in metadata_by_key:
            report["errors"].append(
                f"referenzen.csv Zeile {line}: Zuordnung ist doppelt: {code}/{filename}"
            )
        metadata_by_key[key] = row
        if bool(row["geprueft_von"]) != bool(row["geprueft_am"]):
            report["errors"].append(
                f"referenzen.csv Zeile {line}: geprueft_von und geprueft_am müssen gemeinsam gesetzt sein."
            )
        for field in ("versuchsserie", "pruefkoerper", "originalpfad"):
            if not row[field]:
                report["errors"].append(
                    f"referenzen.csv Zeile {line}: {field} fehlt."
                )
    records: list[ReferenceRecord] = []
    seen_hashes: dict[str, str] = {}
    image_keys: set[tuple[str, str]] = set()
    unreviewed: list[str] = []
    for code in ACTIVE_CLASS_CODES:
        folder = reference_root / CLASS_DIRECTORIES[code]
        if not folder.is_dir():
            report["errors"].append(f"Klassenordner fehlt: {folder}")
            continue
        for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                report["unsupportedFiles"].append(str(path))
                continue
            relative = path.relative_to(reference_root).as_posix()
            reference_id = f"{code}:{relative}"
            sha256 = _file_hash(path)
            key = (code, path.name)
            image_keys.add(key)
            metadata = metadata_by_key.get(key, {})
            confirmed = bool(metadata.get("geprueft_von") and metadata.get("geprueft_am"))
            if not confirmed:
                unreviewed.append(reference_id)
            original_path = str(metadata.get("originalpfad", ""))
            if original_path:
                project_root = base_root.parents[1].resolve()
                candidate = (project_root / original_path).resolve()
                if not candidate.is_relative_to(project_root) or not candidate.is_file():
                    report["errors"].append(
                        f"{reference_id}: originalpfad ist nicht lesbar oder liegt außerhalb des Projekts."
                    )
                elif _file_hash(candidate) != sha256:
                    report["warnings"].append(
                        f"{reference_id}: Referenz ist eine abweichende Fassung der dokumentierten Originaldatei."
                    )
            if sha256 in seen_hashes:
                first_reference_id = seen_hashes[sha256]
                report["duplicateHashes"].append({
                    "sha256": sha256,
                    "first": first_reference_id,
                    "duplicate": reference_id,
                    "sameClass": first_reference_id.split(":", 1)[0] == code,
                })
            else:
                seen_hashes[sha256] = reference_id
            records.append(ReferenceRecord(
                reference_id, code, path, relative, sha256,
                "",
                str(metadata.get("versuchsserie", "")),
                str(metadata.get("pruefkoerper", "")), original_path, confirmed,
                str(metadata.get("geprueft_von", "")),
                str(metadata.get("geprueft_am", "")),
            ))
            report["countsByClass"][code] += 1

    csv_without_images = sorted(
        f"{code}/{filename}" for code, filename in set(metadata_by_key) - image_keys
    )
    images_without_csv = sorted(
        f"{code}/{filename}" for code, filename in image_keys - set(metadata_by_key)
    )
    report["references"] = [record.public() for record in records]
    report["testDataConflicts"] = _test_data_conflicts(report["references"])
    report["referenceCount"] = len(records)
    report["unreviewedReferenceIds"] = unreviewed
    report["reviewRequired"] = bool(manifest.get("require_review", False))
    report["reviewComplete"] = not unreviewed and bool(records)
    report["imagesWithoutCsv"] = images_without_csv
    report["csvWithoutImages"] = csv_without_images
    requirements = feature_requirements(manifest)
    for code, count in report["countsByClass"].items():
        minimum = requirements[code]["minimum"] or 0
        maximum = requirements[code]["maximum"]
        target = requirements[code]["target"]
        if count < minimum:
            report["missingByClass"][code] = minimum - count
        if maximum is not None and count > maximum:
            report["errors"].append(f"{code}: {count} Referenzen überschreiten das Maximum {maximum}.")
        if target is None:
            report["warnings"].append(f"{code}: Ziel noch nicht fachlich festgelegt.")
        elif count < target:
            report["warnings"].append(f"{code}: Zielbestand noch nicht erreicht: {count} von {target}.")
    target_total = manifest.get("target_reference_total")
    if target_total and len(records) < int(target_total):
        report["warnings"].append(f"Zielgesamtzahl noch nicht erreicht: {len(records)} von {target_total}.")
    if report["unsupportedFiles"]:
        report["errors"].append("Nicht unterstützte Dateien in Referenzordnern gefunden.")
    same_class_duplicates = [
        item for item in report["duplicateHashes"] if item.get("sameClass")
    ]
    cross_class_duplicates = [
        item for item in report["duplicateHashes"] if not item.get("sameClass")
    ]
    if same_class_duplicates:
        report["errors"].append("Doppelte Bild-Hashes im Referenzpool gefunden.")
    if cross_class_duplicates:
        report["warnings"].append(
            f"{len(cross_class_duplicates)} bytegleiche Referenzzuordnungen werden bewusst "
            "für mehrere Merkmale verwendet."
        )
    if report["testDataConflicts"]:
        report["errors"].append("Referenzbilder aus der Z-30-Testserie sind ausgeschlossen.")
    if images_without_csv:
        report["errors"].append(
            f"{len(images_without_csv)} Referenzbilder besitzen keinen Eintrag in referenzen.csv."
        )
    if csv_without_images:
        report["errors"].append(
            f"{len(csv_without_images)} CSV-Einträge besitzen keine Bilddatei."
        )
    if report["reviewRequired"] and unreviewed:
        report["errors"].append(
            f"{len(unreviewed)} Referenzbilder sind noch nicht fachlich bestätigt."
        )
    if variant_id == "2A":
        variant_2b = inspect_variant(base_root, "2B")
        hashes_2b = {str(item["sha256"]) for item in variant_2b.get("references", [])}
        missing_in_2b = [item["relative_path"] for item in report["references"]
                         if str(item["sha256"]) not in hashes_2b]
        report["missingInVariant2B"] = missing_in_2b
        if missing_in_2b:
            report["errors"].append(
                f"{len(missing_in_2b)} Bilder aus Variante 2A sind nicht bytegleich in Variante 2B enthalten."
            )
    if variant_id in {"2A", "2B"}:
        report["referenceBatchCount"] = 0
        report["directReferenceImageCount"] = len(records)
        report["modelCallCount"] = 1
    else:
        report["referenceBatchCount"] = (len(records) + MAX_REFERENCE_BATCH_SIZE - 1) // MAX_REFERENCE_BATCH_SIZE
        report["directReferenceImageCount"] = 0
        report["modelCallCount"] = report["referenceBatchCount"] + 1
    report["cloudCallCount"] = report["modelCallCount"]
    ready = bool(manifest.get("enabled")) and not report["errors"] and not report["missingByClass"]
    report["ready"] = ready
    report["readiness"] = "BEREIT" if ready else "NICHT BEREIT"
    return report


def reference_inventory_preview(base_root: Path) -> dict[str, Any]:
    """Compare variant folders with stored inventories without writing anything."""

    variants: dict[str, Any] = {}
    total_new = total_missing = 0
    for variant_id in ("2A", "2B", "3"):
        manifest_path, manifest = _read_manifest(base_root, variant_id)
        report = inspect_variant(base_root, variant_id)
        inventory_key = "annotation_inventory" if variant_id == "3" else "reference_inventory"
        stored_items = manifest.get(inventory_key, [])
        if not isinstance(stored_items, list):
            stored_items = []
        stored = {
            str(item.get("reference_id")): dict(item)
            for item in stored_items if isinstance(item, Mapping) and item.get("reference_id")
        }
        current = {str(item["reference_id"]): dict(item) for item in report["references"]}
        new_ids = sorted(set(current) - set(stored))
        missing_ids = sorted(set(stored) - set(current))
        compared_fields = (
            (
                "sha256", "source_reference_id", "source_relative_path", "source_sha256",
                "description", "description_source", "description_status",
                "reviewed_by", "reviewed_at",
            )
            if variant_id == "3" else ("sha256",)
        )
        changed_ids = sorted(
            reference_id for reference_id in set(current) & set(stored)
            if any(current[reference_id].get(field) != stored[reference_id].get(field)
                   for field in compared_fields)
        )
        inventory = []
        for reference_id, item in current.items():
            previous = stored.get(reference_id, {})
            inventory.append({
                **item,
                "confirmed": (
                    bool(item.get("confirmed")) if variant_id == "3"
                    else bool(item.get("confirmed", False))
                ),
                "description": (
                    str(item.get("description", "")) if variant_id == "3"
                    else str(item.get("description", previous.get("description", "")))
                ),
            })
        total_new += len(new_ids)
        total_missing += len(missing_ids)
        variants[variant_id] = {
            "manifestPath": str(manifest_path), "newReferenceIds": new_ids,
            "missingReferenceIds": missing_ids, "changedReferenceIds": changed_ids,
            "duplicateHashes": report["duplicateHashes"],
            "unsupportedFiles": report["unsupportedFiles"], "inventory": inventory,
            "missingDescriptions": report.get("missingDescriptions", []),
            "unknownSourceReferenceIds": report.get("unknownSourceReferenceIds", []),
            "imagesWithoutCsv": report.get("imagesWithoutCsv", []),
            "csvWithoutImages": report.get("csvWithoutImages", []),
            "hasChanges": bool(new_ids or missing_ids or changed_ids),
        }
    return {
        "variants": variants, "newCount": total_new, "missingCount": total_missing,
        "hasChanges": any(item["hasChanges"] for item in variants.values()),
        "written": False,
    }


def update_reference_inventories(base_root: Path, *, confirmed: bool = False) -> dict[str, Any]:
    """Atomically update manifest inventories only after explicit confirmation."""

    preview = reference_inventory_preview(base_root)
    if not confirmed or not preview["hasChanges"]:
        return preview
    updated: list[str] = []
    for variant_id, item in preview["variants"].items():
        if not item["hasChanges"]:
            continue
        manifest_path, manifest = _read_manifest(base_root, variant_id)
        inventory_key = "annotation_inventory" if variant_id == "3" else "reference_inventory"
        manifest[inventory_key] = item["inventory"]
        manifest["inventory_updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_manifest(manifest_path, manifest)
        updated.append(variant_id)
    return {**reference_inventory_preview(base_root), "written": True, "updatedVariants": updated}


def deterministic_batches(report: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    references = [dict(item) for item in report.get("references", [])]
    return [references[index:index + MAX_REFERENCE_BATCH_SIZE]
            for index in range(0, len(references), MAX_REFERENCE_BATCH_SIZE)]


def memory_key(*, report: Mapping[str, Any], provider: str, model: str,
               prompt_version: str, model_version: str = "") -> str:
    return _canonical_hash({
        "variant": report.get("variantId"), "manifest_hash": report.get("manifestHash"),
        "references": [{"reference_id": item.get("reference_id"), "sha256": item.get("sha256"),
                        "source_reference_id": item.get("source_reference_id"),
                        "source_sha256": item.get("source_sha256"),
                        "description": item.get("description", "")} for item in report.get("references", [])],
        "provider": provider, "model": model, "model_version": model_version,
        "prompt_version": prompt_version,
    })


def memory_status(memory_root: Path, *, report: Mapping[str, Any], provider: str,
                  model: str, prompt_version: str, model_version: str = "") -> dict[str, Any]:
    key = memory_key(report=report, provider=provider, model=model,
                     prompt_version=prompt_version, model_version=model_version)
    path = memory_root / f"{key}.json"
    valid = False
    if path.is_file():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            valid = document.get("memory_key") == key and document.get("status") == "prepared"
        except (OSError, json.JSONDecodeError):
            valid = False
    return {"key": key, "path": str(path), "prepared": valid,
            "status": "BEREIT" if valid else "NICHT VORBEREITET"}


def prepare_reference_memory(memory_root: Path, *, report: Mapping[str, Any], provider: str,
                             model: str, prompt_version: str,
                             analyze_batch: Callable[[Sequence[Mapping[str, Any]], int], Mapping[str, Any]],
                             model_version: str = "") -> Path:
    """Prepare memory with an injected transport; tests can prove behavior without network calls."""

    if report.get("variantId") in {"1", "2A", "2B", "3"}:
        raise VariantConfigurationError(
            f"Variante {report.get('variantId')} besitzt und benötigt kein Referenzgedächtnis."
        )
    if not report.get("ready"):
        raise VariantConfigurationError("Die Variante ist strukturell nicht bereit.")
    batches = deterministic_batches(report)
    outputs = [dict(analyze_batch(batch, index)) for index, batch in enumerate(batches, start=1)]
    key = memory_key(report=report, provider=provider, model=model,
                     prompt_version=prompt_version, model_version=model_version)
    document = {
        "schema_version": "1.0", "status": "prepared", "memory_key": key,
        "created_at": datetime.now(timezone.utc).isoformat(), "variant_id": report.get("variantId"),
        "provider": provider, "model": model, "model_version": model_version,
        "prompt_version": prompt_version, "manifest_hash": report.get("manifestHash"),
        "descriptions_csv_sha256": report.get("descriptionsCsvSha256"),
        "general_class_descriptions_source": report.get("generalClassDescriptionsSource"),
        "reference_batches": batches, "batch_outputs": outputs,
    }
    memory_root.mkdir(parents=True, exist_ok=True)
    target = memory_root / f"{key}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def control_token(*, report: Mapping[str, Any], specimen: str, images: Sequence[Mapping[str, str]],
                  provider: str, model: str, test_series_id: str) -> str:
    return _canonical_hash({"variant": report.get("variantId"), "manifest": report.get("manifestHash"),
                            "references": report.get("references", []), "specimen": specimen,
                            "images": list(images), "provider": provider, "model": model,
                            "test_series_id": test_series_id})
