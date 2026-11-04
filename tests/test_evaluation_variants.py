import csv
import json
import tempfile
import unittest
from pathlib import Path

from Tool.evaluation_variants import (
    ACTIVE_CLASS_CODES, ANNOTATION_CSV_FIELDS, CLASS_DIRECTORIES, control_token, deterministic_batches,
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

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fill(self, variant: str, count: int = 1) -> None:
        for code in ACTIVE_CLASS_CODES:
            folder = self.base / f"Variante_{variant}" / CLASS_DIRECTORIES[code]
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                (folder / f"ref-{index}.jpg").write_bytes(f"{variant}-{code}-{index}".encode())

    def _annotate_all(self, *, description: str = "Fachlich markiertes Merkmal") -> list[dict[str, str]]:
        source = inspect_variant(self.base, "2a")
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
                "class_code": code, "annotated_path": relative,
                "description": description, "reviewed_by": "Fachperson",
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
        for variant in ("2", "2a", "3"):
            (self.base / f"Variante_{variant}").rename(self.base / f"unreachable_{variant}")
        report = inspect_variant(self.base, "1")
        self.assertTrue(report["ready"])
        self.assertEqual(report["referenceCount"], 0)
        self.assertIsNone(report["referenceRoot"])

    def test_empty_reference_folders_are_not_ready_but_do_not_raise(self) -> None:
        report = inspect_variant(self.base, "2")
        self.assertFalse(report["ready"])
        self.assertEqual(set(report["missingByClass"]), set(ACTIVE_CLASS_CODES))

    def test_small_pool_accepts_one_to_four_per_class(self) -> None:
        self._fill("2")
        report = inspect_variant(self.base, "2")
        self.assertTrue(report["ready"])
        self.assertEqual(report["referenceCount"], 12)
        self.assertEqual(report["referenceBatchCount"], 2)
        self.assertTrue(report["ready"])
        self.assertTrue(any("Bu: Ziel noch nicht" in warning for warning in report["warnings"]))

    def test_variants_two_and_two_a_read_only_their_own_folders(self) -> None:
        self._fill("2")
        self._fill("2a", 2)
        variant2 = inspect_variant(self.base, "2")
        variant2a = inspect_variant(self.base, "2a")
        self.assertEqual(variant2["referenceCount"], 12)
        self.assertEqual(variant2a["referenceCount"], 24)
        self.assertTrue(all(item["relative_path"].endswith("ref-0.jpg") for item in variant2["references"]))
        self.assertTrue(any(item["relative_path"].endswith("ref-1.jpg") for item in variant2a["references"]))

    def test_small_pool_rejects_more_than_four_per_class(self) -> None:
        self._fill("2", 5)
        report = inspect_variant(self.base, "2")
        self.assertFalse(report["ready"])
        self.assertTrue(any("Maximum" in error for error in report["errors"]))

    def test_unsupported_files_and_duplicate_hashes_are_reported(self) -> None:
        self._fill("2")
        ka = self.base / "Variante_2/KA"
        (ka / "note.txt").write_text("not an image")
        (ka / "duplicate.jpg").write_bytes((ka / "ref-0.jpg").read_bytes())
        report = inspect_variant(self.base, "2")
        self.assertFalse(report["ready"])
        self.assertTrue(report["unsupportedFiles"])
        self.assertTrue(report["duplicateHashes"])

    def test_batches_are_deterministic_and_never_larger_than_eight(self) -> None:
        self._fill("2a", 2)
        first = deterministic_batches(inspect_variant(self.base, "2a"))
        second = deterministic_batches(inspect_variant(self.base, "2a"))
        self.assertEqual(first, second)
        self.assertTrue(all(len(batch) <= 8 for batch in first))

    def test_variant_three_uses_only_annotated_files_and_links_to_2a(self) -> None:
        self._fill("2a")
        report_2a = inspect_variant(self.base, "2a")
        self._annotate_all()
        complete = inspect_variant(self.base, "3")
        self.assertTrue(complete["ready"])
        self.assertTrue(complete["descriptionsComplete"])
        self.assertTrue(complete["sourceLinksComplete"])
        self.assertEqual(complete["generalClassDescriptionsSource"], "Datenbank/klassifikationsregeln.json")
        self.assertTrue(all(item["relative_path"].startswith(CLASS_DIRECTORIES[item["code"]] + "/")
                            for item in complete["references"]))
        self.assertTrue(all(item["reference_id"].startswith("V3-") for item in complete["references"]))
        self.assertEqual(
            {item["source_reference_id"] for item in complete["references"]},
            {item["reference_id"] for item in report_2a["references"]},
        )
        self.assertTrue(all(item["sha256"] != item["source_sha256"] for item in complete["references"]))

    def test_variant_three_requires_description_and_known_source(self) -> None:
        self._fill("2a")
        rows = self._annotate_all()
        rows[0]["description"] = ""
        rows[1]["source_reference_id"] = "V2A-UNBEKANNT"
        self._write_annotation_csv(rows)
        report = inspect_variant(self.base, "3")
        self.assertFalse(report["ready"])
        self.assertIn(rows[0]["reference_id"], report["missingDescriptions"])
        self.assertIn("V2A-UNBEKANNT", report["unknownSourceReferenceIds"])

    def test_variant_three_review_requirement_is_configurable(self) -> None:
        self._fill("2a")
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

    def test_variant_three_rejects_unknown_class_duplicate_ids_and_hashes(self) -> None:
        self._fill("2a")
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
        self._fill("2a")
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
        self._fill("2a")
        rows = self._annotate_all()
        annotation = self.base / "Variante_3" / rows[0]["annotated_path"]
        source_item = inspect_variant(self.base, "2a")["references"][0]
        source = self.base / "Variante_2a" / source_item["relative_path"]
        annotation.unlink()
        annotation.symlink_to(source)
        report = inspect_variant(self.base, "3")
        self.assertFalse(report["ready"])
        self.assertTrue(any("dieselbe Datei" in error for error in report["errors"]))

    def test_empty_variant_three_does_not_block_other_variants(self) -> None:
        self._fill("2")
        self._fill("2a")
        self.assertTrue(inspect_variant(self.base, "1")["ready"])
        self.assertTrue(inspect_variant(self.base, "2")["ready"])
        self.assertTrue(inspect_variant(self.base, "2a")["ready"])
        self.assertFalse(inspect_variant(self.base, "3")["ready"])

    def test_reference_inventory_update_does_not_touch_frozen_series_or_original_images(self) -> None:
        self._fill("2a")
        self._annotate_all()
        frozen = self.root / "frozen-series.json"
        frozen.write_text('{"status":"frozen"}\n')
        originals = {path: path.read_bytes() for path in (self.base / "Variante_2a").rglob("*.jpg")}
        before = frozen.read_bytes()
        update_reference_inventories(self.base, confirmed=True)
        self.assertEqual(frozen.read_bytes(), before)
        self.assertTrue(all(path.read_bytes() == content for path, content in originals.items()))

    def test_variant_three_memory_trace_keeps_annotation_source_and_csv_hash(self) -> None:
        self._fill("2a")
        rows = self._annotate_all()
        report = inspect_variant(self.base, "3")
        target = prepare_reference_memory(
            self.root / "memory", report=report, provider="fake", model="fake-model",
            prompt_version="test", analyze_batch=lambda batch, index: {"batch": index},
        )
        document = json.loads(target.read_text())
        flattened = [item for batch in document["reference_batches"] for item in batch]
        self.assertEqual([item["reference_id"] for item in flattened],
                         [row["reference_id"] for row in rows])
        self.assertTrue(document["descriptions_csv_sha256"])
        self.assertTrue(all(item["source_reference_id"] for item in flattened))
        self.assertTrue(all(item["source_relative_path"] for item in flattened))
        self.assertTrue(all(item["source_sha256"] for item in flattened))

    def test_memory_uses_mock_transport_and_is_invalidated_by_model(self) -> None:
        self._fill("2")
        report = inspect_variant(self.base, "2")
        calls = []
        path = prepare_reference_memory(
            self.root / "memory", report=report, provider="ollama", model="model-a",
            prompt_version="p1", analyze_batch=lambda batch, index: calls.append((index, len(batch))) or {"ok": True},
        )
        self.assertTrue(path.is_file())
        self.assertEqual(calls, [(1, 8), (2, 4)])
        self.assertTrue(memory_status(self.root / "memory", report=report, provider="ollama",
                                      model="model-a", prompt_version="p1")["prepared"])
        self.assertFalse(memory_status(self.root / "memory", report=report, provider="ollama",
                                       model="model-b", prompt_version="p1")["prepared"])

    def test_control_overview_token_becomes_stale_after_image_change(self) -> None:
        report = inspect_variant(self.base, "1")
        common = dict(report=report, specimen="Z-1", provider="ollama", model="qwen", test_series_id="s1")
        first = control_token(images=[{"filename": "a.jpg", "sha256": "1"}], **common)
        second = control_token(images=[{"filename": "b.jpg", "sha256": "2"}], **common)
        self.assertNotEqual(first, second)

    def test_inventory_preview_never_writes_without_confirmation(self) -> None:
        self._fill("2")
        manifest = self.base / "Variante_2/variant_manifest.json"
        before = manifest.read_bytes()
        preview = update_reference_inventories(self.base, confirmed=False)
        self.assertEqual(preview["newCount"], 12)
        self.assertFalse(preview["written"])
        self.assertEqual(manifest.read_bytes(), before)

    def test_confirmed_inventory_update_preserves_images_and_detects_missing(self) -> None:
        self._fill("2")
        image = self.base / "Variante_2/GSF/ref-0.jpg"
        image_before = image.read_bytes()
        updated = update_reference_inventories(self.base, confirmed=True)
        self.assertTrue(updated["written"])
        self.assertIn("2", updated["updatedVariants"])
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
        self.assertIsNone(features["Bu"]["requirements"]["2"]["target"])
        self.assertTrue(features["KB"]["reviewRequired"])
        self.assertIsNone(features["KB"]["requirements"]["2"]["target"])

    def test_all_twelve_features_and_individual_targets_exist(self) -> None:
        self.assertEqual(
            ACTIVE_CLASS_CODES,
            ("GSF", "LSF", "KA", "SA", "sp HB", "Spl HB", "RiSch", "LR", "Bu", "AA", "SSA", "KB"),
        )
        self.assertFalse(any((self.base / "Variante_1" / folder).exists()
                             for folder in CLASS_DIRECTORIES.values()))
        for variant in ("2", "2a", "3"):
            report = inspect_variant(self.base, variant)
            for code in ACTIVE_CLASS_CODES:
                self.assertTrue((self.base / f"Variante_{variant}" / CLASS_DIRECTORIES[code]).is_dir())
            self.assertFalse(any((self.base / f"Variante_{variant}" / CLASS_DIRECTORIES["KB"]).iterdir()))
            self.assertIsNone(report["featureRequirements"]["Bu"]["target"])
            self.assertIsNone(report["featureRequirements"]["AA"]["target"])
            self.assertIsNone(report["featureRequirements"]["SSA"]["target"])
            self.assertIsNone(report["featureRequirements"]["KB"]["target"])
        self.assertEqual(inspect_variant(self.base, "2")["featureRequirements"]["GSF"]["target"], 2)
        self.assertEqual(inspect_variant(self.base, "2a")["featureRequirements"]["GSF"]["target"], 20)

    def test_legacy_shared_requirement_fields_remain_readable(self) -> None:
        manifest_path = self.base / "Variante_2/variant_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("reference_requirements_by_feature")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = inspect_variant(self.base, "2")
        self.assertEqual(report["featureRequirements"]["GSF"], {
            "minimum": 1, "target": 2, "maximum": 4,
        })
        self.assertEqual(report["featureRequirements"]["SSA"], {
            "minimum": 1, "target": None, "maximum": 4,
        })


if __name__ == "__main__":
    unittest.main()
