import base64
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from Tool.evaluation_variants import (
    ACTIVE_CLASS_CODES, ANNOTATION_CSV_FIELDS, CLASS_DIRECTORIES, REFERENCE_CSV_FIELDS,
    control_token, deterministic_batches,
    inspect_variant, memory_status, prepare_reference_memory,
    reference_inventory_preview, update_reference_inventories,
)
from desktop.bridge import DesktopBridge


class EvaluationVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = Path(__file__).parents[1] / "Datenbank/Evaluationsvarianten"
        import shutil
        shutil.copytree(source, self.root / "variants")
        self.base = self.root / "variants"
        # Produktionsbestand und wissenschaftliche Sollzahlen sind veränderliche
        # Forschungsdaten. Die synthetischen Tests verwenden weiterhin einen
        # kleinen, stabilen 2B-Pool mit zwei bis fünf Bildern je Merkmal.
        variant_2b_manifest_path = self.base / "Variante_2B/variant_manifest.json"
        variant_2b_manifest = json.loads(variant_2b_manifest_path.read_text(encoding="utf-8"))
        variant_2b_manifest["reference_requirements_by_feature"] = {
            code: {"minimum": 2, "target": 3, "maximum": 5}
            for code in ACTIVE_CLASS_CODES
        }
        variant_2b_manifest["min_references_per_class"] = 2
        variant_2b_manifest["target_references_per_class"] = 3
        variant_2b_manifest["max_references_per_class"] = 5
        variant_2b_manifest["target_reference_total"] = 36
        variant_2b_manifest.pop("reference_inventory", None)
        variant_2b_manifest_path.write_text(
            json.dumps(variant_2b_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for variant in ("2A", "2B"):
            variant_root = self.base / f"Variante_{variant}"
            for path in variant_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                    path.unlink()
            with (variant_root / "referenzen.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(REFERENCE_CSV_FIELDS), delimiter=";"
                )
                writer.writeheader()
        variant_3_root = self.base / "Variante_3"
        for path in variant_3_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                path.unlink()
        self._write_annotation_csv([])
        variant_3_manifest_path = variant_3_root / "variant_manifest.json"
        variant_3_manifest = json.loads(variant_3_manifest_path.read_text(encoding="utf-8"))
        variant_3_manifest["reference_requirements_by_feature"] = {
            code: {"minimum": 2, "target": 3, "maximum": 5}
            for code in ACTIVE_CLASS_CODES
        }
        variant_3_manifest["target_reference_total"] = 36
        variant_3_manifest.pop("annotation_inventory", None)
        variant_3_manifest_path.write_text(
            json.dumps(variant_3_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fill(self, variant: str, count: int | None = None) -> None:
        count = (2 if variant == "2B" else 1) if count is None else count
        for code in ACTIVE_CLASS_CODES:
            folder = self.base / f"Variante_{variant}" / CLASS_DIRECTORIES[code]
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                content = f"{variant}-{code}-{index}".encode()
                image_path = folder / f"ref-{index}.jpg"
                if not (variant == "2B" and image_path.exists()):
                    image_path.write_bytes(content)
                if variant == "2A":
                    target = self.base / "Variante_2B" / CLASS_DIRECTORIES[code] / f"ref-{index}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
        self._write_reference_csv(variant)
        if variant == "2A":
            self._write_reference_csv("2B")

    def _write_reference_csv(self, variant: str) -> None:
        rows = []
        project_root = self.base.parents[1]
        for code in ACTIVE_CLASS_CODES:
            folder = self.base / f"Variante_{variant}" / CLASS_DIRECTORIES[code]
            for image in sorted(folder.glob("*.jpg")):
                original_relative = Path("test-originals") / variant / CLASS_DIRECTORIES[code] / image.name
                original = project_root / original_relative
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_bytes(image.read_bytes())
                rows.append({
                    "merkmal": code, "dateiname": image.name,
                    "versuchsserie": "TEST", "pruefkoerper": f"TEST-{code}",
                    "originalpfad": original_relative.as_posix(),
                    "begruendung": "Mock-Referenz", "geprueft_von": "Test",
                    "geprueft_am": "2026-11-05",
                })
        path = self.base / f"Variante_{variant}" / "referenzen.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REFERENCE_CSV_FIELDS), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        if variant == "2B":
            source = inspect_variant(self.base, "2B")
            manifest_path = self.base / "Variante_3/variant_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["expected_source_reference_ids"] = [
                item["reference_id"] for item in source["references"]
            ]
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    def _annotate_all(self, *, description: str = "Fachlich markiertes Merkmal") -> list[dict[str, str]]:
        source = inspect_variant(self.base, "2B")
        (self.base / "Variante_3").mkdir(parents=True, exist_ok=True)
        rows = []
        for index, item in enumerate(source["references"], start=1):
            code = item["code"]
            filename = f"V3-{index:03d}.png"
            relative = f"{CLASS_DIRECTORIES[code]}/{filename}"
            (self.base / "Variante_3" / CLASS_DIRECTORIES[code]).mkdir(parents=True, exist_ok=True)
            (self.base / "Variante_3" / relative).write_bytes(
                f"annotated-{item['reference_id']}".encode()
            )
            rows.append({
                "reference_id": f"V3-{index:03d}",
                "source_reference_id": item["reference_id"],
                "source_relative_path": item["relative_path"],
                "source_sha256": item["sha256"],
                "class_code": code, "annotated_path": relative,
                "description": description, "description_source": "Synthetischer Test",
                "description_status": "pocket_verified", "reviewed_by": "Fachperson",
                "reviewed_at": "2026-11-02",
            })
        self._write_annotation_csv(rows)
        return rows

    def _write_annotation_csv(self, rows: list[dict[str, str]]) -> None:
        with (self.base / "Variante_3/descriptions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_CSV_FIELDS))
            writer.writeheader()
            writer.writerows(rows)

    def test_variant_one_is_ready_without_reference_directories(self) -> None:
        for variant in ("2A", "2B", "3"):
            (self.base / f"Variante_{variant}").rename(self.base / f"unreachable_{variant}")
        report = inspect_variant(self.base, "1")
        self.assertTrue(report["ready"])
        self.assertEqual(report["referenceCount"], 0)
        self.assertIsNone(report["referenceRoot"])

    def test_empty_reference_folders_are_not_ready_but_do_not_raise(self) -> None:
        report = inspect_variant(self.base, "2A")
        self.assertFalse(report["ready"])
        self.assertEqual(set(report["missingByClass"]), set(ACTIVE_CLASS_CODES))

    def test_one_shot_accepts_exactly_one_per_class(self) -> None:
        self._fill("2A")
        report = inspect_variant(self.base, "2A")
        self.assertTrue(report["ready"])
        self.assertIn("One-Shot", report["name"])
        self.assertNotIn("Few-Shot", report["name"])
        self.assertEqual(report["referenceCount"], 12)
        self.assertEqual(report["referenceBatchCount"], 0)
        self.assertEqual(report["directReferenceImageCount"], 12)
        self.assertEqual(report["modelCallCount"], 1)
        self.assertTrue(report["ready"])
        self.assertFalse(report["warnings"])

    def test_variant_2a_requires_documented_review_for_every_reference(self) -> None:
        self._fill("2A")
        path = self.base / "Variante_2A/referenzen.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=";"))
        rows[0]["geprueft_von"] = ""
        rows[0]["geprueft_am"] = ""
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REFERENCE_CSV_FIELDS), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)

        report = inspect_variant(self.base, "2A")

        self.assertFalse(report["ready"])
        self.assertFalse(report["reviewComplete"])
        self.assertEqual(len(report["unreviewedReferenceIds"]), 1)
        self.assertTrue(any("noch nicht fachlich bestätigt" in item for item in report["errors"]))

    def test_variants_two_and_two_a_read_only_their_own_folders(self) -> None:
        self._fill("2A")
        self._fill("2B", 2)
        variant2a = inspect_variant(self.base, "2A")
        variant2ab = inspect_variant(self.base, "2B")
        self.assertEqual(variant2a["referenceCount"], 12)
        self.assertEqual(variant2ab["referenceCount"], 24)
        self.assertEqual(variant2ab["referenceBatchCount"], 0)
        self.assertEqual(variant2ab["directReferenceImageCount"], 24)
        self.assertEqual(variant2ab["modelCallCount"], 1)
        self.assertEqual(variant2ab["cloudCallCount"], 1)
        self.assertTrue(all(item["relative_path"].endswith("ref-0.jpg") for item in variant2a["references"]))
        self.assertTrue(any(item["relative_path"].endswith("ref-1.jpg") for item in variant2ab["references"]))

    def test_one_shot_rejects_more_than_one_per_class(self) -> None:
        self._fill("2A", 2)
        report = inspect_variant(self.base, "2A")
        self.assertFalse(report["ready"])
        self.assertTrue(any("Maximum" in error for error in report["errors"]))

    def test_unsupported_files_and_duplicate_hashes_are_reported(self) -> None:
        self._fill("2A")
        ka = self.base / "Variante_2A/KA"
        (ka / "note.txt").write_text("not an image")
        (ka / "duplicate.jpg").write_bytes((ka / "ref-0.jpg").read_bytes())
        report = inspect_variant(self.base, "2A")
        self.assertFalse(report["ready"])
        self.assertTrue(report["unsupportedFiles"])
        self.assertTrue(report["duplicateHashes"])

    def test_cross_class_duplicate_is_a_visible_warning_but_not_an_error(self) -> None:
        self._fill("2B", 2)
        source = self.base / "Variante_2B/GSF/ref-0.jpg"
        duplicate = self.base / "Variante_2B/LSF/ref-0.jpg"
        duplicate.write_bytes(source.read_bytes())
        self._write_reference_csv("2B")

        report = inspect_variant(self.base, "2B")

        self.assertTrue(report["ready"])
        self.assertTrue(report["duplicateHashes"])
        self.assertTrue(any("mehrere Merkmale" in item for item in report["warnings"]))
        self.assertFalse(any("Doppelte Bild-Hashes" in item for item in report["errors"]))

    def test_batches_are_deterministic_and_never_larger_than_eight(self) -> None:
        self._fill("2B", 2)
        first = deterministic_batches(inspect_variant(self.base, "2B"))
        second = deterministic_batches(inspect_variant(self.base, "2B"))
        self.assertEqual(first, second)
        self.assertTrue(all(len(batch) <= 8 for batch in first))

    def test_variant_three_uses_only_annotated_files_and_links_to_2b(self) -> None:
        self._fill("2B")
        report_2b = inspect_variant(self.base, "2B")
        self._annotate_all()
        complete = inspect_variant(self.base, "3")
        self.assertTrue(complete["ready"])
        self.assertTrue(complete["descriptionsComplete"])
        self.assertTrue(complete["sourceLinksComplete"])
        self.assertEqual(complete["generalClassDescriptionsSource"], "Datenbank/klassifikationsregeln.json")
        self.assertTrue(all(item["relative_path"].startswith(CLASS_DIRECTORIES[item["code"]] + "/")
                            for item in complete["references"]))
        self.assertTrue(all(item["reference_id"].startswith("V3-") for item in complete["references"]))
        self.assertEqual(len(complete["references"]), 24)
        self.assertTrue(
            {item["source_reference_id"] for item in complete["references"]}
            .issubset({item["reference_id"] for item in report_2b["references"]})
        )
        self.assertTrue(all(item["sha256"] != item["source_sha256"] for item in complete["references"]))

    def test_variant_three_remains_ready_without_variant_2b_directory(self) -> None:
        self._fill("2B")
        self._annotate_all()
        before = inspect_variant(self.base, "3")
        self.assertTrue(before["ready"])
        (self.base / "Variante_2B").rename(self.base / "Variante_2B_unavailable")
        after = inspect_variant(self.base, "3")
        self.assertTrue(after["ready"])
        self.assertEqual(after["references"], before["references"])

    def test_variant_three_requires_description_and_known_source(self) -> None:
        self._fill("2B")
        rows = self._annotate_all()
        rows[0]["description"] = ""
        rows[1]["source_reference_id"] = "V2A-UNBEKANNT"
        self._write_annotation_csv(rows)
        report = inspect_variant(self.base, "3")
        self.assertFalse(report["ready"])
        self.assertIn(rows[0]["reference_id"], report["missingDescriptions"])
        self.assertIn("V2A-UNBEKANNT", report["unknownSourceReferenceIds"])

    def test_variant_three_review_requirement_is_configurable(self) -> None:
        self._fill("2B")
        rows = self._annotate_all()
        rows[0]["reviewed_by"] = ""
        rows[0]["reviewed_at"] = ""
        self._write_annotation_csv(rows)
        manifest_path = self.base / "Variante_3/variant_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["require_review"] = True
        manifest_path.write_text(json.dumps(manifest))
        report = inspect_variant(self.base, "3")
        self.assertFalse(report["ready"])
        self.assertIn(rows[0]["reference_id"], report["unreviewedReferenceIds"])

    def test_variant_three_accepts_documented_rule_fallback_description(self) -> None:
        self._fill("2B")
        rows = self._annotate_all()
        rows[0]["description"] = "Allgemeine Merkmalsdefinition aus Regelwerk v1.4."
        rows[0]["description_source"] = "Regelwerk v1.4"
        rows[0]["description_status"] = "rule_v1.4_fallback"
        self._write_annotation_csv(rows)
        report = inspect_variant(self.base, "3")
        self.assertTrue(report["ready"])
        self.assertNotIn(rows[0]["reference_id"], report["descriptionReviewRequiredIds"])

    def test_variant_three_rejects_unknown_class_duplicate_ids_and_hashes(self) -> None:
        self._fill("2B")
        rows = self._annotate_all()
        rows[0]["class_code"] = "UNBEKANNT"
        rows[1]["reference_id"] = rows[0]["reference_id"]
        first = self.base / "Variante_3" / rows[0]["annotated_path"]
        second = self.base / "Variante_3" / rows[1]["annotated_path"]
        second.write_bytes(first.read_bytes())
        self._write_annotation_csv(rows)
        report = inspect_variant(self.base, "3")
        self.assertFalse(report["ready"])
        self.assertTrue(any("unbekannter class_code" in error for error in report["errors"]))
        self.assertTrue(any("reference_id ist doppelt" in error for error in report["errors"]))
        self.assertTrue(report["duplicateHashes"])

    def test_variant_three_reports_images_without_csv_and_csv_without_images(self) -> None:
        self._fill("2B")
        rows = self._annotate_all()
        orphan = self.base / "Variante_3/GSF/orphan.png"
        orphan.write_bytes(b"orphan")
        missing_path = rows[0]["annotated_path"]
        (self.base / "Variante_3" / missing_path).unlink()
        report = inspect_variant(self.base, "3")
        self.assertIn("GSF/orphan.png", report["imagesWithoutCsv"])
        self.assertIn(missing_path, report["csvWithoutImages"])
        self.assertFalse(report["ready"])

    def test_variant_three_rejects_annotation_that_is_the_original_file(self) -> None:
        self._fill("2B")
        rows = self._annotate_all()
        annotation = self.base / "Variante_3" / rows[0]["annotated_path"]
        source_item = inspect_variant(self.base, "2B")["references"][0]
        source = self.base / "Variante_2B" / source_item["relative_path"]
        annotation.unlink()
        annotation.symlink_to(source)
        report = inspect_variant(self.base, "3")
        self.assertFalse(report["ready"])
        self.assertTrue(any("bytegleich" in error for error in report["errors"]))

    def test_empty_variant_three_does_not_block_other_variants(self) -> None:
        self._fill("2A")
        self._fill("2B")
        self.assertTrue(inspect_variant(self.base, "1")["ready"])
        self.assertTrue(inspect_variant(self.base, "2A")["ready"])
        self.assertTrue(inspect_variant(self.base, "2B")["ready"])
        self.assertFalse(inspect_variant(self.base, "3")["ready"])

    def test_reference_inventory_update_does_not_touch_frozen_series_or_original_images(self) -> None:
        self._fill("2B")
        self._annotate_all()
        frozen = self.root / "frozen-series.json"
        frozen.write_text('{"status":"frozen"}\n')
        originals = {path: path.read_bytes() for path in (self.base / "Variante_2B").rglob("*.jpg")}
        before = frozen.read_bytes()
        update_reference_inventories(self.base, confirmed=True)
        self.assertEqual(frozen.read_bytes(), before)
        self.assertTrue(all(path.read_bytes() == content for path, content in originals.items()))

    def test_variant_three_direct_trace_keeps_annotation_source_and_csv_hash(self) -> None:
        self._fill("2B")
        rows = self._annotate_all()
        report = inspect_variant(self.base, "3")
        self.assertTrue(report["ready"])
        self.assertEqual(report["referenceBatchCount"], 0)
        self.assertEqual(report["directReferenceImageCount"], 24)
        self.assertEqual([item["reference_id"] for item in report["references"]],
                         [row["reference_id"] for row in rows])
        self.assertTrue(report["descriptionsCsvSha256"])
        self.assertTrue(all(item["source_reference_id"] for item in report["references"]))
        self.assertTrue(all(item["source_relative_path"] for item in report["references"]))
        self.assertTrue(all(item["source_sha256"] for item in report["references"]))

    def test_direct_variants_reject_reference_memory_without_model_call(self) -> None:
        self._fill("2B", 2)
        report = inspect_variant(self.base, "2B")
        calls = []
        with self.assertRaisesRegex(Exception, "kein Referenzgedächtnis"):
            prepare_reference_memory(
                self.root / "memory", report=report, provider="openai", model="gpt-6.1-sol",
                prompt_version="p1",
                analyze_batch=lambda batch, index: calls.append((index, len(batch))) or {"ok": True},
            )
        self.assertEqual(calls, [])

    def test_control_overview_token_becomes_stale_after_image_change(self) -> None:
        report = inspect_variant(self.base, "1")
        common = dict(report=report, specimen="Z-1", provider="ollama", model="qwen", test_series_id="s1")
        first = control_token(images=[{"filename": "a.jpg", "sha256": "1"}], **common)
        second = control_token(images=[{"filename": "b.jpg", "sha256": "2A"}], **common)
        self.assertNotEqual(first, second)

    def test_inventory_preview_never_writes_without_confirmation(self) -> None:
        self._fill("2A")
        manifest = self.base / "Variante_2A/variant_manifest.json"
        before = manifest.read_bytes()
        preview = update_reference_inventories(self.base, confirmed=False)
        self.assertEqual(preview["newCount"], 24)
        self.assertFalse(preview["written"])
        self.assertEqual(manifest.read_bytes(), before)

    def test_confirmed_inventory_update_preserves_images_and_detects_missing(self) -> None:
        self._fill("2A")
        image = self.base / "Variante_2A/GSF/ref-0.jpg"
        image_before = image.read_bytes()
        updated = update_reference_inventories(self.base, confirmed=True)
        self.assertTrue(updated["written"])
        self.assertIn("2A", updated["updatedVariants"])
        self.assertEqual(image.read_bytes(), image_before)
        image.unlink()
        preview = reference_inventory_preview(self.base)
        self.assertEqual(preview["missingCount"], 1)

    def test_reference_ui_catalog_exposes_all_twelve_features_equally(self) -> None:
        catalog = DesktopBridge().reference_catalog()
        self.assertTrue(catalog["ok"])
        self.assertEqual(tuple(catalog["activeCodes"]), ACTIVE_CLASS_CODES)
        self.assertNotIn("additionalFeatures", catalog)
        features = {item["code"]: item for item in catalog["classes"]}
        self.assertEqual(tuple(features), ACTIVE_CLASS_CODES)
        self.assertEqual(
            features["AA"]["definition"],
            "Eine Astansammlung liegt vor, wenn mindestens zwei getrennt erkennbare Äste sichtbar dicht beieinander liegen. Eine Obergrenze der Astzahl oder ein zahlenmäßig festgelegter Maximalabstand wird nicht verlangt.",
        )
        self.assertEqual(
            features["SSA"]["definition"],
            "Ein Schmalseitenast ist ein vollständig sichtbarer Astanschnitt auf einer tatsächlich vorhandenen und eindeutig als Schmalseite erkennbaren Ansicht des Prüfkörpers.",
        )
        self.assertTrue(features["Bu"]["reviewRequired"])
        self.assertEqual(features["Bu"]["requirements"]["2A"]["target"], 1)
        self.assertTrue(features["KB"]["reviewRequired"])
        self.assertEqual(features["KB"]["requirements"]["2A"]["target"], 1)
        for key in ("imagesVariant2A", "imagesVariant2B", "imagesVariant3"):
            preview = next(
                image for feature in features.values() for image in feature[key]
            )
            encoded = preview["src"].split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                self.assertEqual(max(image.size), 640)

    def test_all_twelve_features_and_individual_targets_exist(self) -> None:
        self.assertEqual(
            ACTIVE_CLASS_CODES,
            ("GSF", "LSF", "KA", "SA", "sp HB", "Spl HB", "RiSch", "LR", "Bu", "AA", "SSA", "KB"),
        )
        self.assertFalse(any((self.base / "Variante_1" / folder).exists()
                             for folder in CLASS_DIRECTORIES.values()))
        for variant in ("2A", "2B", "3"):
            report = inspect_variant(self.base, variant)
            for code in ACTIVE_CLASS_CODES:
                self.assertTrue((self.base / f"Variante_{variant}" / CLASS_DIRECTORIES[code]).is_dir())
            self.assertFalse(any(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}
                                 for path in (self.base / f"Variante_{variant}" / CLASS_DIRECTORIES["KB"]).iterdir()))
            expected = 1 if variant == "2A" else 3
            self.assertEqual(report["featureRequirements"]["Bu"]["target"], expected)
            self.assertEqual(report["featureRequirements"]["AA"]["target"], expected)
            self.assertEqual(report["featureRequirements"]["SSA"]["target"], expected)
            self.assertEqual(report["featureRequirements"]["KB"]["target"], expected)
        self.assertEqual(inspect_variant(self.base, "2A")["featureRequirements"]["GSF"]["target"], 1)
        self.assertEqual(inspect_variant(self.base, "2B")["featureRequirements"]["GSF"]["target"], 3)

    def test_legacy_shared_requirement_fields_remain_readable(self) -> None:
        manifest_path = self.base / "Variante_2A/variant_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("reference_requirements_by_feature")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = inspect_variant(self.base, "2A")
        self.assertEqual(report["featureRequirements"]["GSF"], {
            "minimum": 1, "target": 1, "maximum": 1,
        })
        self.assertEqual(report["featureRequirements"]["SSA"], {
            "minimum": 1, "target": 1, "maximum": 1,
        })

    def test_z30_filename_marks_only_the_affected_variant_not_ready(self) -> None:
        self._fill("2B")
        image = self.base / "Variante_2B/KA/REF_KA_Z30_008_B01.jpg"
        image.write_bytes(b"not-a-test-image")
        report = inspect_variant(self.base, "2B")
        self.assertFalse(report["ready"])
        self.assertTrue(report["testDataConflicts"])
        self.assertTrue(inspect_variant(self.base, "1")["ready"])

    def test_z30_sha256_match_marks_variant_not_ready_without_real_test_image(self) -> None:
        self._fill("2B")
        image = self.base / "Variante_2B/KA/ref-0.jpg"
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        with patch("Tool.evaluation_variants._z30_test_image_hashes",
                   return_value=frozenset({image_hash})):
            report = inspect_variant(self.base, "2B")
        self.assertFalse(report["ready"])
        self.assertTrue(any("SHA-256" in item["reason"] for item in report["testDataConflicts"]))

    def test_variant_2a_requires_byte_identical_image_in_2b(self) -> None:
        self._fill("2A")
        (self.base / "Variante_2B/GSF/ref-0.jpg").write_bytes(b"changed")
        report = inspect_variant(self.base, "2A")
        self.assertFalse(report["ready"])
        self.assertIn("GSF/ref-0.jpg", report["missingInVariant2B"])

    def test_variant_three_requires_exactly_one_annotation_per_2b_image(self) -> None:
        self._fill("2B")
        rows = self._annotate_all()
        self._write_annotation_csv(rows[:-1])
        missing_file = self.base / "Variante_3" / rows[-1]["annotated_path"]
        missing_file.unlink()
        report = inspect_variant(self.base, "3")
        self.assertFalse(report["ready"])
        self.assertIn(rows[-1]["source_reference_id"], report["missingSourceAnnotations"])


if __name__ == "__main__":
    unittest.main()
