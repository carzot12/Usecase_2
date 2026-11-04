from __future__ import annotations

import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import desktop.bridge as bridge_module
from desktop.bridge import (
    DesktopBridge, _partition_provider_images, _provider_failure_policy,
    _result_features, _safe_name,
)
from Tool.evaluation_variants import ACTIVE_CLASS_CODES


class DesktopBridgeTests(unittest.TestCase):
    def test_provider_failure_policy_retries_temporarily_and_pauses_on_billing(self) -> None:
        timeout = _provider_failure_policy(TimeoutError("timed out"))
        self.assertEqual(timeout["action"], "retry")

        class BillingError(RuntimeError):
            status_code = 429

        billing = _provider_failure_policy(BillingError("insufficient_quota: credit balance"))
        self.assertEqual(billing["action"], "pause")
        self.assertEqual(billing["reason"], "billing_or_quota")

    def test_provider_image_limits_match_implemented_request_paths(self) -> None:
        images = [f"bild-{index}" for index in range(18)]
        self.assertEqual([len(part) for part in _partition_provider_images(images, "mistral")], [8, 8, 2])
        self.assertEqual(_partition_provider_images(images, "ollama"), [images])
        self.assertEqual(_partition_provider_images(images, "openai"), [images])

    def test_local_model_diagnostics_reports_endpoint_model_and_twelve_classes(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def read() -> bytes:
                return json.dumps({"models": [{
                    "name": "qwen3-vl:4b-instruct",
                    "digest": "test-digest",
                    "capabilities": ["completion", "vision"],
                }]}).encode("utf-8")

        with patch.object(bridge_module, "urlopen", return_value=FakeResponse()) as request:
            result = DesktopBridge().local_model_diagnostics({
                "model": "qwen3-vl:4b-instruct",
            })

        request.assert_called_once_with("http://localhost:11434/api/tags", timeout=3)
        self.assertTrue(result["endpointReachable"])
        self.assertTrue(result["modelInstalled"])
        self.assertEqual(result["modelDigest"], "test-digest")
        self.assertEqual(result["activeClassCount"], 12)
        self.assertEqual(tuple(result["activeClassCodes"]), ACTIVE_CLASS_CODES)

    def test_display_adapter_deduplicates_new_and_historical_structured_classes(self) -> None:
        current = _result_features({
            "class_suggestions": [
                {"suggestion_id": "one", "code": "KA", "name": "Kantenast"},
                {"suggestion_id": "two", "code": "KA", "name": "Kantenast"},
                {"suggestion_id": "three", "code": "LSF", "name": "Lokale Schrägfaser"},
            ]
        })
        historic = _result_features({
            "primary_feature": {"code": "KA"},
            "secondary_features": [{"code": "LSF"}, {"code": "KA"}],
            "visible_observations": ["Dieser Fließtext darf keine zusätzliche Klasse erzeugen."],
        })
        self.assertEqual([item["code"] for item in current], ["KA", "LSF"])
        self.assertEqual([item["code"] for item in historic], ["KA", "LSF"])
        self.assertEqual([item["name"] for item in historic], ["Kantenast", "Lokale Schrägfaser"])

    def test_scientific_series_plan_uses_exactly_49_confirmed_z30_selections(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Z30_Evaluation" / "Sortiert"
            for specimen in DesktopBridge._z30_expected_specimens():
                folder = root / specimen
                folder.mkdir(parents=True)
                Image.new("RGB", (12, 8), "white").save(folder / "breitseite.jpg")
                Image.new("RGB", (8, 12), "gray").save(folder / "schmalseite.jpg")
                Image.new("RGB", (12, 8), "black").save(folder / "Nummer_1.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                for specimen in DesktopBridge._z30_expected_specimens():
                    self.assertTrue(bridge.save_image_selection({
                        "specimen": specimen,
                        "selectedImages": ["breitseite.jpg", "schmalseite.jpg"],
                    })["ok"])
                self.assertTrue(bridge.begin_test_series({"name": "Z-30 Zero-Shot"})["ok"])
                planned = bridge.scientific_series_plan({
                    "variant": "1", "provider": "ollama", "model": "qwen3-vl:4b-instruct",
                })
            self.assertTrue(planned["ok"])
            plan = planned["plan"]
            self.assertEqual(plan["specimenCount"], 49)
            self.assertEqual(plan["selectedImageCount"], 98)
            self.assertNotIn("Z-30-30", [item["specimen"] for item in plan["specimens"]])
            self.assertEqual(plan["referenceImageCount"], 0)
            self.assertFalse(plan["groundTruthIncluded"])
            self.assertFalse(plan["protocolsIncluded"])
            self.assertEqual(plan["activeClassCodes"], list(ACTIVE_CLASS_CODES))
            self.assertTrue(all(
                [image["filename"] for image in item["images"]]
                == ["breitseite.jpg", "schmalseite.jpg"]
                for item in plan["specimens"]
            ))

    def test_scientific_csv_export_keeps_technical_errors_distinct_from_negative_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            root.mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                run_folder = bridge._project_data() / "Wissenschaftliche_Laeufe/run-test"
                result_folder = run_folder / "Ergebnisse"
                result_folder.mkdir(parents=True)
                classified_path = result_folder / "Z-30-01_test.json"
                classified_path.write_text(json.dumps({
                    "run_id": "run-1", "created_at": "2026-11-04T10:00:00+00:00",
                    "execution": {"provider": "ollama", "actual_model": "qwen3-vl:4b-instruct"},
                    "classification_basis": {"prompt_version": "5.4", "classification_rules_version": "1.4"},
                    "result": {"status": "KLASSIFIZIERT", "class_suggestions": [
                        {"code": "KA", "confidence_label": "unsicher", "evidence_status": "vermutet"},
                    ]},
                }), encoding="utf-8")
                empty_path = result_folder / "Z-30-02_test.json"
                empty_path.write_text(json.dumps({
                    "run_id": "run-2", "created_at": "2026-11-04T10:01:00+00:00",
                    "execution": {"provider": "ollama", "actual_model": "qwen3-vl:4b-instruct"},
                    "classification_basis": {"prompt_version": "5.4", "classification_rules_version": "1.4"},
                    "result": {"status": "KEINE_MERKMALE", "class_suggestions": []},
                }), encoding="utf-8")
                jobs = [
                    {"jobId": "j1", "runId": "run-1", "specimen": "Z-30-01", "normalizedSpecimenId": "Z-30-01-01", "status": "succeeded", "resultPath": str(classified_path)},
                    {"jobId": "j2", "runId": "run-2", "specimen": "Z-30-02", "normalizedSpecimenId": "Z-30-01-02", "status": "succeeded", "resultPath": str(empty_path)},
                    {"jobId": "j3", "runId": "run-3", "specimen": "Z-30-03", "normalizedSpecimenId": "Z-30-01-03", "status": "failed", "error": "künstlicher Fehler"},
                ]
                bridge.batches["run-test"] = {
                    "kind": "scientific_evaluation_series", "batchId": "run-test",
                    "seriesRunId": "run-test", "runFolder": str(run_folder), "jobs": jobs,
                    "evaluationVariant": "1", "provider": "ollama", "model": "qwen3-vl:4b-instruct",
                    "promptVersion": "5.4", "classificationRulesVersion": "1.4",
                }
                exported = bridge.export_scientific_series({"seriesRunId": "run-test"})
            self.assertTrue(exported["ok"])
            with Path(exported["wideCsvPath"]).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter=";"))
            self.assertEqual(rows[0]["KA"], "1")
            self.assertEqual(rows[0]["Vermutete_Merkmale"], "KA")
            self.assertEqual(rows[1]["KA"], "0")
            self.assertEqual(rows[2]["Status"], "fehlgeschlagen")
            self.assertEqual(rows[2]["KA"], "")
            with Path(exported["detailCsvPath"]).open(encoding="utf-8-sig", newline="") as handle:
                detail_rows = list(csv.DictReader(handle, delimiter=";"))
            self.assertEqual(len(detail_rows), 3 * len(ACTIVE_CLASS_CODES))
            ka = next(row for row in detail_rows if row["Pruefkoerper_ID"] == "Z-30-01" and row["Merkmalscode"] == "KA")
            self.assertEqual((ka["Vorgeschlagen"], ka["Confidence"], ka["Evidence_Status"]), ("Ja", "unsicher", "vermutet"))

    def test_scientific_parts_retry_persist_and_resume_without_repeating_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen_folder = root / "Z-30-01"
            specimen_folder.mkdir(parents=True)
            image_names = [f"bild_{index:02d}.jpg" for index in range(10)]
            for name in image_names:
                Image.new("RGB", (12, 8), "white").save(specimen_folder / name)
            calls: list[tuple[str, ...]] = []
            first_part_attempts = 0
            second_part_should_fail = True

            def fake_run_analysis(_input_dir: Path, **kwargs: object) -> Path:
                nonlocal first_part_attempts, second_part_should_fail
                images = tuple(kwargs["selected_image_names"])
                calls.append(images)
                if images == tuple(image_names[:8]):
                    first_part_attempts += 1
                    if first_part_attempts == 1:
                        raise TimeoutError("timed out")
                    code = "KA"
                else:
                    if second_part_should_fail:
                        raise ValueError("dauerhafter Mock-Fehler")
                    code = "LSF"
                destination_dir = Path(kwargs["output_dir"])
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{kwargs['run_id']}.json"
                destination.write_text(json.dumps({
                    "schema_version": "4.2", "run_id": kwargs["run_id"],
                    "created_at": "2026-11-04T10:00:00+00:00",
                    "execution": {"provider": "mistral", "actual_model": "ministral-3b-2512", "duration_seconds": 1, "provider_metadata": {}},
                    "classification_basis": {"prompt_version": "5.4", "classification_rules_version": "1.4"},
                    "result": {"status": "KLASSIFIZIERT", "observations_short": [code], "visible_features": [code], "uncertain_points": [], "class_suggestions": [
                        {"suggestion_id": "suggestion-01", "code": code, "name": code, "confidence": .6, "confidence_label": "mittel", "evidence_status": "sicher_sichtbar", "proven_failure_cause": False},
                    ]},
                }), encoding="utf-8")
                return destination

            with patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(bridge_module, "run_analysis", side_effect=fake_run_analysis), \
                    patch.object(bridge_module.time, "sleep") as sleep:
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                run_folder = bridge._project_data() / "Wissenschaftliche_Laeufe/series-1"
                job = {
                    "jobId": "job-1", "runId": "parent-run", "specimen": "Z-30-01",
                    "normalizedSpecimenId": "Z-30-01-01", "provider": "mistral",
                    "model": "ministral-3b-2512", "selectedImages": image_names,
                    "selectedImageRecords": [{"filename": name, "sha256": "mock"} for name in image_names],
                    "selectedImageCount": 10, "providerImageLimit": 8, "partialRequestCount": 2,
                    "status": "running", "stage": "analyzing_images", "mergeStatus": "ausstehend",
                    "parts": [
                        {"partId": "teil-01", "runId": "part-1", "images": image_names[:8], "imageCount": 8, "status": "queued", "attemptCount": 0},
                        {"partId": "teil-02", "runId": "part-2", "images": image_names[8:], "imageCount": 2, "status": "queued", "attemptCount": 0},
                    ],
                }
                batch = {
                    "kind": "scientific_evaluation_series", "batchId": "series-1", "seriesRunId": "series-1",
                    "runFolder": str(run_folder), "projectData": str(bridge._project_data()),
                    "projectPath": str(root), "projectId": "Demo", "evaluationVariant": "1",
                    "classificationRulesVersion": "1.4", "confirmCloud": True,
                    "testSeries": {"seriesId": "test-series", "name": "Mock"}, "jobs": [job],
                }
                first_outcome = bridge._run_scientific_job(batch, job)
                self.assertEqual(first_outcome, "partially_succeeded")
                self.assertEqual(job["parts"][0]["status"], "succeeded")
                self.assertEqual(job["parts"][0]["attemptCount"], 2)
                self.assertTrue(Path(job["parts"][0]["resultPath"]).is_file())
                self.assertTrue(Path(job["partialMergePath"]).is_file())
                sleep.assert_called_once_with(1.0)
                first_part_call_count = calls.count(tuple(image_names[:8]))

                second_part_should_fail = False
                job["parts"][1].update(status="queued", error=None)
                job.update(status="running", stage="analyzing_images")
                second_outcome = bridge._run_scientific_job(batch, job)

            self.assertEqual(second_outcome, "succeeded")
            self.assertEqual(calls.count(tuple(image_names[:8])), first_part_call_count)
            self.assertEqual(job["successfulPartialRequestCount"], 2)
            self.assertEqual(job["mergeStatus"], "vollstaendig")
            self.assertTrue(Path(job["resultPath"]).is_file())
            merged = json.loads(Path(job["resultPath"]).read_text(encoding="utf-8"))
            self.assertEqual({item["code"] for item in merged["result"]["class_suggestions"]}, {"KA", "LSF"})
            self.assertEqual(merged["series_processing"]["partial_request_count"], 2)

    def test_bootstrap_does_not_read_or_rewrite_desktop_reference_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            workspace = Path(temp)
            config = workspace / "Datenbank/Referenzbilder_neu/aktive_hauptversagensarten.json"
            config.parent.mkdir(parents=True)
            original = b'{"sourceVersion":"1.2","selectedCodes":["protected"]}\n'
            config.write_bytes(original)
            original_mtime = config.stat().st_mtime_ns
            path_read_text = Path.read_text

            def deny_only_external_config(path: Path, *args: object, **kwargs: object) -> str:
                if path == config:
                    raise PermissionError(1, "Operation not permitted", str(path))
                return path_read_text(path, *args, **kwargs)

            with patch.object(bridge_module, "WORKSPACE_ROOT", workspace), \
                    patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(Path, "read_text", deny_only_external_config):
                response = DesktopBridge().bootstrap()

            self.assertTrue(response["ok"])
            self.assertEqual(tuple(item["code"] for item in response["classes"]), ACTIVE_CLASS_CODES)
            self.assertEqual(config.read_bytes(), original)
            self.assertEqual(config.stat().st_mtime_ns, original_mtime)

    def test_only_three_supported_provider_model_choices_are_exposed(self) -> None:
        self.assertEqual(
            list(bridge_module.MODEL_CONFIGS),
            [
                ("ollama", "qwen3-vl:4b-instruct"),
                ("openai", "gpt-6.1-sol"),
                ("mistral", "ministral-3b-2512"),
            ],
        )
        self.assertNotIn(("ollama", "llava:latest"), bridge_module.MODEL_CONFIGS)

    def test_safe_name_rejects_empty_names(self) -> None:
        with self.assertRaises(ValueError):
            _safe_name(" / ")

    def test_project_snapshot_counts_number_photos_but_marks_only_break_images_analyzable(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            Image.new("RGB", (20, 10), "white").save(specimen / "Nummer_1.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                result = bridge.open_project_path(str(root))
            self.assertTrue(result["ok"])
            summary = result["project"]["specimens"][0]
            self.assertEqual(summary["imageCount"], 2)
            self.assertEqual(summary["analysisImageCount"], 1)
            self.assertEqual(summary["identificationImageCount"], 1)

    def test_correction_is_stored_separately_from_ai_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            (root / "Z-1").mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                result = bridge.save_correction({"specimen": "Z-1", "primaryCode": "KA",
                                                 "secondaryCodes": ["LSF"], "note": "geprüft"})
                saved = json.loads(Path(result["path"]).read_text())
            self.assertEqual(saved["primaryCode"], "KA")
            self.assertEqual(saved["secondaryCodes"], ["LSF"])
            self.assertTrue(saved["confirmed"])

    def test_image_selection_defaults_to_all_and_persists_per_specimen(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            first = root / "Z-1"
            second = root / "Z-2"
            first.mkdir(parents=True)
            second.mkdir()
            for folder in (first, second):
                for name in ("bruch_1.jpg", "bruch_2.jpg"):
                    Image.new("RGB", (20, 10), "white").save(folder / name)
            original_bytes = {
                path.name: path.read_bytes()
                for path in first.iterdir()
            }

            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                initial = bridge.specimen_detail({"specimen": "Z-1"})
                self.assertEqual(
                    {image["name"] for image in initial["images"] if image["analysisSelected"]},
                    {"bruch_1.jpg", "bruch_2.jpg"},
                )
                self.assertTrue(bridge.save_image_selection({
                    "specimen": "Z-1", "selectedImages": ["bruch_1.jpg"],
                })["ok"])
                second_detail = bridge.specimen_detail({"specimen": "Z-2"})
                self.assertTrue(all(image["analysisSelected"] for image in second_detail["images"]))
                restored = bridge.specimen_detail({"specimen": "Z-1"})
                self.assertEqual(
                    [image["name"] for image in restored["images"] if image["analysisSelected"]],
                    ["bruch_1.jpg"],
                )
                bridge.save_image_selection({
                    "specimen": "Z-1", "selectedImages": ["bruch_1.jpg", "bruch_2.jpg"],
                })
                self.assertTrue(all(
                    image["analysisSelected"]
                    for image in bridge.specimen_detail({"specimen": "Z-1"})["images"]
                ))
                self.assertTrue(bridge.save_image_selection({
                    "specimen": "Z-1", "selectedImages": [],
                })["ok"])
                self.assertFalse(any(
                    image["analysisSelected"]
                    for image in bridge.specimen_detail({"specimen": "Z-1"})["images"]
                ))
                self.assertEqual(
                    {path.name: path.read_bytes() for path in first.iterdir()},
                    original_bytes,
                )

    def test_default_selection_is_limited_to_three_distributed_non_number_photos(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            for index in range(8):
                Image.new("RGB", (20, 10), "white").save(specimen / f"bruch_{index}.jpg")
            Image.new("RGB", (20, 10), "white").save(specimen / "Nummer_1.jpg")

            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                detail = bridge.specimen_detail({"specimen": "Z-1"})
                selected = {
                    image["name"] for image in detail["images"] if image["analysisSelected"]
                }
                self.assertEqual(selected, {"bruch_0.jpg", "bruch_4.jpg", "bruch_7.jpg"})
                number_image = next(
                    image for image in detail["images"] if image["name"] == "Nummer_1.jpg"
                )
                self.assertTrue(number_image["isIdentification"])
                self.assertFalse(number_image["analysisEligible"])
                self.assertFalse(number_image["analysisSelected"])

                legacy_selection = bridge._selection_path("Z-1")
                old_document = json.loads(legacy_selection.read_text(encoding="utf-8"))
                old_document.pop("selectionCustomized")
                old_document["selectedImages"] = old_document["knownImages"]
                legacy_selection.write_text(json.dumps(old_document), encoding="utf-8")
                migrated = bridge.specimen_detail({"specimen": "Z-1"})
                self.assertEqual(
                    {image["name"] for image in migrated["images"] if image["analysisSelected"]},
                    {"bruch_0.jpg", "bruch_4.jpg", "bruch_7.jpg"},
                )

                self.assertTrue(bridge.save_image_selection({
                    "specimen": "Z-1",
                    "selectedImages": [f"bruch_{index}.jpg" for index in range(8)],
                })["ok"])
                manually_selected = bridge.specimen_detail({"specimen": "Z-1"})
                self.assertEqual(
                    sum(image["analysisSelected"] for image in manually_selected["images"]), 8
                )
                rejected = bridge.save_image_selection({
                    "specimen": "Z-1", "selectedImages": ["Nummer_1.jpg"],
                })
                self.assertFalse(rejected["ok"])

    def test_number_photos_remain_visible_for_specimen_with_one_break_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-30-19"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "DSC06541.JPG")
            for name in (
                "Nummer_DSC06539.JPG", "Nummer_DSC06539_2.JPG",
                "Nummer_DSC06539_2_2.JPG", "Nummer_DSC06540.JPG",
                "Nummer_DSC06540_2.JPG",
            ):
                Image.new("RGB", (20, 10), "white").save(specimen / name)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                opened = bridge.open_project_path(str(root))
                detail = bridge.specimen_detail({"specimen": "Z-30-19"})
            self.assertEqual(opened["project"]["specimens"][0]["imageCount"], 6)
            self.assertEqual(len(detail["images"]), 6)
            self.assertEqual(sum(image["analysisSelected"] for image in detail["images"]), 1)
            self.assertEqual(sum(image["isIdentification"] for image in detail["images"]), 5)

    def test_analysis_uses_the_selected_images_for_each_specimen(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            for specimen in ("Z-1", "Z-2"):
                folder = root / specimen
                folder.mkdir(parents=True) if specimen == "Z-1" else None
                if specimen == "Z-2":
                    folder.mkdir()
                Image.new("RGB", (20, 10), "white").save(folder / "a.jpg")
                Image.new("RGB", (20, 10), "black").save(folder / "b.jpg")

            captured: dict[str, list[str]] = {}

            def fake_run_analysis(input_dir: Path, **kwargs: object) -> Path:
                captured[input_dir.name] = list(kwargs["selected_image_names"])
                destination_dir = Path(kwargs["output_dir"])
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{input_dir.name}_test.json"
                destination.write_text(json.dumps({
                    "execution": {"model": "qwen", "duration_seconds": 0},
                    "result": {},
                    "image_selection": {
                        "included": list(kwargs["selected_image_names"]),
                        "excluded": ["b.jpg"] if input_dir.name == "Z-1" else ["a.jpg"],
                    },
                }))
                return destination

            with patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(bridge_module, "run_analysis", side_effect=fake_run_analysis):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                bridge.save_image_selection({"specimen": "Z-1", "selectedImages": ["a.jpg"]})
                bridge.save_image_selection({"specimen": "Z-2", "selectedImages": ["b.jpg"]})
                for specimen in ("Z-1", "Z-2"):
                    started = bridge.start_analysis({"specimen": specimen, "model": "qwen"})
                    self.assertTrue(started["ok"])
                    for _ in range(100):
                        status = bridge.analysis_status({"jobId": started["jobId"]})
                        if status["state"] != "running":
                            break
                        time.sleep(0.01)
                    self.assertEqual(status["state"], "complete")

                self.assertEqual(captured, {"Z-1": ["a.jpg"], "Z-2": ["b.jpg"]})
                exported = bridge.export_specimen({"specimen": "Z-1"})
                self.assertTrue(exported["ok"])
                csv_text = Path(exported["csvPath"]).read_text(encoding="utf-8-sig")
                self.assertIn("Einbezogene_Bilder", csv_text)
                self.assertIn("a.jpg", csv_text)
                self.assertIn("b.jpg", csv_text)
                exported_json = json.loads(Path(exported["jsonPath"]).read_text(encoding="utf-8"))
                self.assertEqual(
                    exported_json["kiOriginal"]["image_selection"],
                    {"included": ["a.jpg"], "excluded": ["b.jpg"]},
                )

    def test_three_selected_break_images_are_persisted_and_sent_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-3"
            specimen.mkdir(parents=True)
            for index in range(1, 6):
                Image.new("RGB", (20, 10), (index * 30, 0, 0)).save(
                    specimen / f"bruch_{index}.jpg"
                )
            Image.new("RGB", (20, 10), "white").save(specimen / "Nummer_Z-3.jpg")
            selected = ["bruch_1.jpg", "bruch_3.jpg", "bruch_5.jpg"]
            captured: dict[str, object] = {}

            def fake_run_analysis(input_dir: Path, **kwargs: object) -> Path:
                captured.update(kwargs)
                destination = Path(kwargs["output_dir"]) / f"{input_dir.name}_three.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({"result": {}}), encoding="utf-8")
                return destination

            with patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(bridge_module, "run_analysis", side_effect=fake_run_analysis):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                saved = bridge.save_image_selection({
                    "specimen": "Z-3", "selectedImages": selected,
                })
                reopened = DesktopBridge()
                reopened.open_project_path(str(root))
                detail = reopened.specimen_detail({"specimen": "Z-3"})
                started = reopened.start_analysis({"specimen": "Z-3", "model": "qwen"})
                for _ in range(100):
                    status = reopened.analysis_status({"jobId": started["jobId"]})
                    if status["state"] != "running":
                        break
                    time.sleep(0.01)

            self.assertTrue(saved["ok"])
            self.assertEqual(
                sorted(image["name"] for image in detail["images"] if image["analysisSelected"]),
                selected,
            )
            self.assertEqual(captured["selected_image_names"], selected)
            self.assertEqual(
                captured["allowed_class_codes"],
                list(ACTIVE_CLASS_CODES),
            )
            self.assertNotIn("Nummer_Z-3.jpg", captured["selected_image_names"])
            self.assertIsNone(captured["evaluation_variant"])
            self.assertEqual(status["state"], "complete")

    def test_variant_one_control_plan_contains_only_images_twelve_features_and_zero_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            for name in ("bruch_1.jpg", "bruch_2.jpg", "bruch_3.jpg"):
                Image.new("RGB", (20, 10), "white").save(specimen / name)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                series = bridge.begin_test_series({"name": "Pilot 1"})["series"]
                result = bridge.evaluation_run_plan({"specimen": "Z-1"})

            self.assertTrue(result["ok"])
            plan = result["plan"]
            self.assertEqual(plan["variant"], "1")
            self.assertEqual(plan["imageCount"], 3)
            self.assertEqual(plan["model"], "qwen3-vl:4b-instruct")
            self.assertEqual(
                [item["code"] for item in plan["classes"]],
                list(ACTIVE_CLASS_CODES),
            )
            self.assertTrue(all(item["description"] for item in plan["classes"]))
            self.assertEqual(plan["referenceImages"], 0)
            self.assertFalse(plan["referenceDatabaseAccessed"])
            self.assertFalse(plan["groundTruthIncluded"])
            self.assertFalse(plan["previousResultsIncluded"])
            self.assertFalse(plan["assessmentsIncluded"])
            self.assertEqual(plan["testSeries"]["seriesId"], series["seriesId"])
            self.assertEqual(plan["testSeries"]["classificationRulesVersion"], "1.4")
            self.assertEqual(plan["testSeries"]["activeClassCount"], 12)
            self.assertTrue(plan["planToken"])

    def test_removed_llava_choice_cannot_start_a_new_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                result = bridge.start_analysis({
                    "specimen": "Z-1", "provider": "ollama", "model": "llava:latest",
                })
            self.assertFalse(result["ok"])
            self.assertIn("Unbekannte Provider-/Modellkombination", result["error"])

    def test_variant_one_qwen_start_passes_documented_codes_and_reference_free_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            captured: dict[str, object] = {}

            def fake_run(input_dir: Path, **kwargs: object) -> Path:
                captured.update(kwargs)
                destination = Path(kwargs["output_dir"]) / "variant1.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({"result": {}}), encoding="utf-8")
                return destination

            with patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(bridge_module, "run_analysis", side_effect=fake_run):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                series = bridge.begin_test_series({"name": "Wissenschaftlicher Pilot"})["series"]
                plan = bridge.evaluation_run_plan({"specimen": "Z-1"})["plan"]
                started = bridge.start_analysis({
                    "specimen": "Z-1", "provider": "ollama", "model": "qwen3-vl:4b-instruct",
                    "evaluationVariant": "1", "evaluationPlanToken": plan["planToken"],
                })
                for _ in range(100):
                    status = bridge.analysis_status({"jobId": started["jobId"]})
                    if status["state"] != "running":
                        break
                    time.sleep(0.01)

            self.assertEqual(status["state"], "complete")
            self.assertEqual(captured["evaluation_variant"], "1")
            self.assertEqual(
                captured["allowed_class_codes"],
                list(ACTIVE_CLASS_CODES),
            )
            self.assertEqual(captured["provider"], "ollama")
            self.assertEqual(captured["test_series"]["series_id"], series["seriesId"])
            self.assertEqual(captured["test_series"]["classification_rules_version"], "1.4")
            self.assertEqual(len(captured["test_series"]["active_class_codes"]), 12)

    def test_variant_one_cannot_start_without_current_control_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                bridge.begin_test_series({"name": "Pilot"})
                result = bridge.start_analysis({
                    "specimen": "Z-1", "provider": "ollama", "model": "qwen3-vl:4b-instruct",
                    "evaluationVariant": "1",
                })
            self.assertFalse(result["ok"])
            self.assertIn("Kontrollübersicht", result["error"])

    def test_unready_evaluation_variant_reports_missing_features_without_starting_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                bridge.begin_test_series({"name": "Pilot"})
                with patch.object(bridge_module, "inspect_variant", return_value={
                    "ready": False,
                    "missingByClass": {"Bu": 1, "AA": 1, "SSA": 1},
                    "errors": [],
                }), patch.object(bridge_module.threading, "Thread") as worker:
                    result = bridge.start_analysis({
                        "specimen": "Z-1", "provider": "ollama",
                        "model": "qwen3-vl:4b-instruct", "evaluationVariant": "2",
                    })
            self.assertFalse(result["ok"])
            self.assertIn("Variante 2 ist noch nicht bereit", result["error"])
            self.assertIn("Bu, AA, SSA", result["error"])
            self.assertIn("kein Modellaufruf", result["error"])
            worker.assert_not_called()

    def test_analysis_cannot_start_without_selected_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                bridge.save_image_selection({"specimen": "Z-1", "selectedImages": []})
                result = bridge.start_analysis({"specimen": "Z-1", "model": "qwen"})
            self.assertFalse(result["ok"])
            self.assertIn("mindestens ein Analysebild", result["error"])

    def test_cancelled_batch_does_not_block_one_image_standard_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-30-19"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            captured: dict[str, object] = {}

            def fake_run_analysis(input_dir: Path, **kwargs: object) -> Path:
                captured.update(kwargs)
                destination = Path(kwargs["output_dir"]) / "result.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({"result": {}}), encoding="utf-8")
                return destination

            with patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(bridge_module, "run_analysis", side_effect=fake_run_analysis):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                bridge.batches["old-batch"] = {
                    "batchId": "old-batch",
                    "jobs": [{"jobId": "old-job", "specimen": "Z-30-19",
                              "provider": "ollama", "model": "qwen3-vl:4b-instruct",
                              "status": "cancelled"}],
                }
                started = bridge.start_analysis({
                    "specimen": "Z-30-19", "provider": "ollama", "model": "qwen3-vl:4b-instruct",
                })
                self.assertTrue(started["ok"])
                for _ in range(100):
                    status = bridge.analysis_status({"jobId": started["jobId"]})
                    if status["state"] != "running":
                        break
                    time.sleep(0.01)

            self.assertEqual(status["state"], "complete")
            self.assertEqual(captured["selected_image_names"], ["bruch.jpg"])
            self.assertIsNone(captured["evaluation_variant"])
            self.assertEqual(bridge.batches["old-batch"]["jobs"][0]["status"], "cancelled")

    def test_running_identical_single_analysis_is_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-30-19"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            entered = threading.Event()
            release = threading.Event()
            calls = 0

            def fake_run_analysis(_input_dir: Path, **kwargs: object) -> Path:
                nonlocal calls
                calls += 1
                entered.set()
                release.wait(2)
                destination = Path(kwargs["output_dir"]) / "result.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({"result": {}}), encoding="utf-8")
                return destination

            with patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(bridge_module, "run_analysis", side_effect=fake_run_analysis):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                first = bridge.start_analysis({"specimen": "Z-30-19", "model": "qwen"})
                self.assertTrue(entered.wait(1))
                duplicate = bridge.start_analysis({"specimen": "Z-30-19", "model": "qwen"})
                release.set()
                for _ in range(100):
                    status = bridge.analysis_status({"jobId": first["jobId"]})
                    if status["state"] != "running":
                        break
                    time.sleep(0.01)

            self.assertTrue(first["ok"])
            self.assertFalse(duplicate["ok"])
            self.assertEqual(duplicate["jobId"], first["jobId"])
            self.assertIn("nicht dupliziert", duplicate["error"])
            self.assertEqual(calls, 1)
            self.assertEqual(status["state"], "complete")

    def test_running_identical_batch_job_blocks_only_the_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-30-19"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                bridge.batches["active-batch"] = {
                    "batchId": "active-batch",
                    "jobs": [{"jobId": "active-job", "specimen": "Z-30-19",
                              "provider": "ollama", "model": "qwen3-vl:4b-instruct",
                              "status": "running"}],
                }
                duplicate = bridge.start_analysis({
                    "specimen": "Z-30-19", "provider": "ollama", "model": "qwen3-vl:4b-instruct",
                })

            self.assertFalse(duplicate["ok"])
            self.assertEqual(duplicate["batchId"], "active-batch")
            self.assertIn("bereits ein Batch-Auftrag", duplicate["error"])
            self.assertFalse(bridge.jobs)

    def test_second_identical_batch_is_rejected_while_first_is_really_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-30-31"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            entered = threading.Event()
            release = threading.Event()

            def fake_run(input_dir: Path, **kwargs: object) -> Path:
                entered.set()
                release.wait(2)
                destination = Path(kwargs["output_dir"]) / f"{input_dir.name}_{kwargs['run_id']}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({"result": {}}), encoding="utf-8")
                return destination

            payload = {
                "specimenIds": ["Z-30-31"],
                "models": [{"provider": "ollama", "model": "qwen3-vl:4b-instruct"}],
            }
            with patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(bridge_module, "run_analysis", side_effect=fake_run):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                first = bridge.start_batch(payload)
                self.assertTrue(entered.wait(1))
                duplicate = bridge.start_batch(payload)
                release.set()
                for _ in range(200):
                    status = bridge.batch_status({"batchId": first["batchId"]})["batch"]
                    if status["status"].startswith("completed"):
                        break
                    time.sleep(0.01)

            self.assertTrue(first["ok"])
            self.assertFalse(duplicate["ok"])
            self.assertEqual(duplicate["conflictStatus"], "running")
            self.assertEqual(duplicate["batchId"], first["batchId"])
            self.assertIn("läuft tatsächlich", duplicate["error"])

    def test_batch_status_marks_persisted_running_job_interrupted_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-30-31"
            specimen.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(specimen / "bruch.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                writer = DesktopBridge()
                writer.open_project_path(str(root))
                project_data = writer._project_data()
                stale = {
                    "batchId": "stale-running", "projectId": "Demo",
                    "projectPath": str(root), "projectData": str(project_data),
                    "status": "running", "sessionId": writer.session_id,
                    "jobs": [{
                        "jobId": "stale-job", "specimen": "Z-30-31",
                        "provider": "ollama", "model": "qwen3-vl:4b-instruct",
                        "status": "running", "stage": "analyzing_images",
                    }],
                }
                writer._write_batch(stale)
                restarted = DesktopBridge()
                restarted.open_project_path(str(root))
                restored = restarted.batch_status({"batchId": "stale-running"})["batch"]

            self.assertEqual(restored["jobs"][0]["status"], "interrupted")
            self.assertEqual(restored["status"], "interrupted")

    def test_new_test_series_keeps_historical_results_and_assessments_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            (root / "Z-1").mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                result_folder = bridge._project_data() / "Ergebnisse"
                result_folder.mkdir(parents=True)
                result_path = result_folder / "Z-1_historical.json"
                result_bytes = json.dumps({
                    "run_id": "old-run", "specimen_id": "Z-1",
                    "created_at": "2026-10-24T00:00:00+00:00",
                    "execution": {"provider": "ollama", "model": "qwen3-vl:4b", "provider_metadata": {}},
                    "classification_basis": {}, "result": {},
                }).encode()
                result_path.write_bytes(result_bytes)
                assessment_path = bridge._project_data() / "Bewertungen/old-run.review.json"
                assessment_path.parent.mkdir(parents=True)
                assessment_bytes = b'{"schemaVersion":"2.0","note":"keep"}'
                assessment_path.write_bytes(assessment_bytes)
                created = bridge.begin_test_series({"name": "Erster wissenschaftlicher Lauf"})
                overview = bridge.project_results()

            self.assertTrue(created["ok"])
            self.assertEqual(created["results"], [])
            self.assertEqual(overview["results"][0]["resultGroup"], "historical")
            self.assertIsNone(overview["results"][0]["seriesId"])
            self.assertEqual(overview["series"]["active"]["seriesId"], created["series"]["seriesId"])
            self.assertEqual(result_path.read_bytes(), result_bytes)
            self.assertEqual(assessment_path.read_bytes(), assessment_bytes)

    def test_history_visibility_persists_and_reset_never_hides_scientific_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                result_folder = bridge._project_data() / "Ergebnisse"
                result_folder.mkdir(parents=True)
                documents = {
                    "historical-run": None,
                    "scientific-run": {"series_id": "series-1", "name": "Pilot"},
                }
                original_bytes: dict[str, bytes] = {}
                for run_id, series in documents.items():
                    path = result_folder / f"Z-1_{run_id}.json"
                    payload = json.dumps({
                        "schema_version": "4.0", "run_id": run_id,
                        "specimen_id": "Z-1", "created_at": f"2026-10-2{4 if series is None else 5}T00:00:00+00:00",
                        "test_series": series,
                        "execution": {"provider": "ollama", "model": "qwen3-vl:4b-instruct", "provider_metadata": {}},
                        "classification_basis": {},
                        "result": {"status": "UNKLAR", "class_suggestions": []},
                    }).encode()
                    path.write_bytes(payload)
                    original_bytes[run_id] = payload

                hidden = bridge.set_result_visibility({"runId": "historical-run", "hidden": True})
                restarted = DesktopBridge()
                restarted.open_project_path(str(root))
                records = {item["runId"]: item for item in restarted.project_results()["results"]}
                restored = restarted.set_result_visibility({"runId": "historical-run", "hidden": False})
                reset = restarted.reset_test_history({"specimen": "Z-1"})
                after_reset = {item["runId"]: item for item in restarted.project_results()["results"]}

            self.assertTrue(hidden["ok"])
            self.assertTrue(records["historical-run"]["hidden"])
            self.assertTrue(restored["ok"])
            self.assertEqual(reset["hiddenCount"], 1)
            self.assertEqual(reset["scientificRunsChanged"], 0)
            self.assertTrue(after_reset["historical-run"]["hidden"])
            self.assertFalse(after_reset["scientific-run"]["hidden"])
            for run_id, payload in original_bytes.items():
                self.assertEqual((result_folder / f"Z-1_{run_id}.json").read_bytes(), payload)

    def test_analysis_run_deletion_is_confirmed_scoped_and_preserves_protected_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-1"
            specimen.mkdir(parents=True)
            original_image = specimen / "bruch.jpg"
            original_image.write_bytes(b"original-image")
            reference = root / "Datenbank" / "Referenzbilder" / "ref.jpg"
            reference.parent.mkdir(parents=True)
            reference.write_bytes(b"reference-image")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                result_folder = bridge._project_data() / "Ergebnisse"
                result_folder.mkdir(parents=True)
                result_paths = []
                assessment_paths = []
                for run_id in ("run-one", "run-two"):
                    path = result_folder / f"Z-1_{run_id}.json"
                    path.write_text(json.dumps({
                        "schema_version": "4.0", "run_id": run_id,
                        "specimen_id": "Z-1", "created_at": "2026-11-02T12:00:00+00:00",
                        "execution": {"provider": "ollama", "model": "qwen3-vl:4b-instruct", "provider_metadata": {}},
                        "classification_basis": {},
                        "result": {"status": "UNKLAR", "class_suggestions": []},
                    }), encoding="utf-8")
                    result_paths.append(path)
                    assessment = bridge._project_data() / "Bewertungen" / f"{run_id}.review.json"
                    assessment.parent.mkdir(parents=True, exist_ok=True)
                    assessment.write_text('{"schemaVersion":"2.0","note":"preserve"}', encoding="utf-8")
                    assessment_paths.append(assessment)
                official = bridge._project_data() / "Offizielle_Historie" / "bestand.json"
                official.parent.mkdir(parents=True)
                official.write_bytes(b"official-history")

                rejected = bridge.delete_analysis_run({"runId": "run-one"})
                deleted = bridge.delete_analysis_run({
                    "runId": "run-one", "confirmDeletion": True,
                })
                deleted_all = bridge.delete_specimen_analysis_runs({
                    "specimen": "Z-1", "confirmDeletion": True,
                })

            self.assertFalse(rejected["ok"])
            self.assertEqual(deleted["deletedRunIds"], ["run-one"])
            self.assertEqual(deleted_all["deletedRunIds"], ["run-two"])
            self.assertFalse(any(path.exists() for path in result_paths))
            self.assertEqual(original_image.read_bytes(), b"original-image")
            self.assertEqual(reference.read_bytes(), b"reference-image")
            self.assertEqual(official.read_bytes(), b"official-history")
            self.assertTrue(all(path.is_file() for path in assessment_paths))

    def test_batch_plan_requires_one_shared_cloud_confirmation_and_runs_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            for specimen in ("Z-1", "Z-2"):
                folder = root / specimen
                folder.mkdir(parents=True)
                Image.new("RGB", (20, 10), "white").save(folder / "bruch.jpg")
            calls: list[tuple[str, str, str, int]] = []
            active = 0
            maximum_active = 0
            guard = threading.Lock()

            def fake_run(input_dir: Path, **kwargs: object) -> Path:
                nonlocal active, maximum_active
                with guard:
                    active += 1
                    maximum_active = max(maximum_active, active)
                calls.append((
                    input_dir.name, str(kwargs["provider"]), str(kwargs["model"]),
                    int(kwargs["ollama_num_ctx"]),
                ))
                time.sleep(0.01)
                destination = Path(kwargs["output_dir"]) / f"{input_dir.name}_{kwargs['run_id']}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({
                    "schema_version": "2.0", "run_id": kwargs["run_id"],
                    "project_id": kwargs["project_id"], "specimen_id": input_dir.name,
                    "created_at": "2026-10-24T00:00:00+00:00",
                    "execution": {"provider": kwargs["provider"], "model": kwargs["model"],
                                  "duration_seconds": 0, "provider_metadata": {}},
                    "classification_basis": {"prompt_hash_sha256": "test"},
                    "result": {"status": "UNKLAR", "primary_feature": None,
                               "secondary_features": []},
                }))
                with guard:
                    active -= 1
                return destination

            models = [
                {"provider": "ollama", "model": "qwen3-vl:4b-instruct"},
                {"provider": "openai", "model": "gpt-6.1-sol"},
            ]
            with patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(bridge_module, "run_analysis", side_effect=fake_run):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                payload = {"specimenIds": ["Z-1", "Z-2"], "models": models}
                plan = bridge.plan_batch(payload)
                self.assertEqual(plan["plan"]["jobCount"], 4)
                self.assertEqual(plan["plan"]["cloudRequestCount"], 2)
                denied = bridge.start_batch(payload)
                self.assertTrue(denied["needsConfirmation"])
                started = bridge.start_batch({**payload, "confirmCloud": True})
                batch_id = started["batchId"]
                for _ in range(500):
                    status = bridge.batch_status({"batchId": batch_id})["batch"]
                    if status["status"].startswith("completed"):
                        break
                    time.sleep(0.01)

            self.assertEqual(status["status"], "completed")
            self.assertEqual(maximum_active, 1)
            self.assertEqual(len(calls), 4)
            qwen_calls = [call for call in calls if call[2] == "qwen3-vl:4b-instruct"]
            self.assertTrue(qwen_calls)
            self.assertTrue(all(call[3] == 24576 for call in qwen_calls))
            self.assertEqual(len({job["runId"] for job in status["jobs"]}), 4)
            self.assertTrue(all(job["status"] == "succeeded" for job in status["jobs"]))
            self.assertTrue(all(job.get("startedAt") and job.get("finishedAt") for job in status["jobs"]))
            self.assertTrue(all(job.get("selectedImages") == ["bruch.jpg"] for job in status["jobs"]))

    def test_batch_can_cancel_jobs_that_have_not_started(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            for specimen in ("Z-1", "Z-2"):
                folder = root / specimen
                folder.mkdir(parents=True)
                Image.new("RGB", (20, 10), "white").save(folder / "bruch.jpg")
            release = threading.Event()

            def fake_run(input_dir: Path, **kwargs: object) -> Path:
                release.wait(2)
                destination = Path(kwargs["output_dir"]) / f"{input_dir.name}_{kwargs['run_id']}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({
                    "run_id": kwargs["run_id"], "specimen_id": input_dir.name,
                    "created_at": "2026-10-24T00:00:00+00:00",
                    "execution": {"provider": "ollama", "model": "qwen3-vl:4b",
                                  "provider_metadata": {}},
                    "classification_basis": {},
                    "result": {"status": "UNKLAR", "primary_feature": None,
                               "secondary_features": []},
                }))
                return destination

            with patch.object(bridge_module, "APP_DATA", Path(data)), \
                    patch.object(bridge_module, "run_analysis", side_effect=fake_run):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                started = bridge.start_batch({
                    "specimenIds": ["Z-1", "Z-2"],
                    "models": [{"provider": "ollama", "model": "qwen3-vl:4b-instruct"}],
                })
                batch_id = started["batchId"]
                for _ in range(100):
                    batch = bridge.batch_status({"batchId": batch_id})["batch"]
                    if any(job["status"] == "running" for job in batch["jobs"]):
                        break
                    time.sleep(0.01)
                cancelled = bridge.cancel_batch_jobs({"batchId": batch_id})["batch"]
                release.set()
                for _ in range(200):
                    batch = bridge.batch_status({"batchId": batch_id})["batch"]
                    if batch["status"] == "completed":
                        break
                    time.sleep(0.01)

            self.assertEqual([job["status"] for job in batch["jobs"]], ["succeeded", "cancelled"])

    def test_terminal_batch_entries_can_be_hidden_without_deleting_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen = root / "Z-30-19"
            specimen.mkdir(parents=True)
            image = specimen / "bruch.jpg"
            image.write_bytes(b"original-image")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                project_data = bridge._project_data()
                result = project_data / "Ergebnisse" / "result.json"
                assessment = project_data / "Bewertungen" / "assessment.json"
                protocol = project_data / "Pruefprotokolle" / "protocol.json"
                for path, content in (
                    (result, b"analysis"),
                    (assessment, b"assessment"),
                    (protocol, b"protocol"),
                ):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                batch = {
                    "batchId": "batch-history",
                    "projectId": "Demo",
                    "projectData": str(project_data),
                    "createdAt": "2026-10-25T10:00:00+00:00",
                    "status": "running",
                    "jobs": [
                        {"jobId": "done", "status": "succeeded", "resultPath": str(result)},
                        {"jobId": "failed", "status": "failed"},
                        {"jobId": "cancelled", "status": "cancelled"},
                        {"jobId": "running", "status": "running"},
                        {"jobId": "queued", "status": "queued"},
                        {"jobId": "interrupted", "status": "interrupted"},
                    ],
                }
                bridge.batches[batch["batchId"]] = batch
                bridge._write_batch(batch)

                rejected = bridge.dismiss_batch_jobs({
                    "batchId": batch["batchId"], "jobIds": ["running"],
                })
                hidden = bridge.dismiss_batch_jobs({
                    "batchId": batch["batchId"], "jobIds": ["cancelled"],
                })
                cleared = bridge.clear_batch_history()
                stored = json.loads(bridge._batch_path(batch["batchId"]).read_text())

            self.assertFalse(rejected["ok"])
            self.assertEqual([job["jobId"] for job in hidden["batch"]["jobs"]],
                             ["done", "failed", "running", "queued", "interrupted"])
            self.assertEqual(cleared["dismissedCount"], 2)
            self.assertEqual(
                {job["jobId"] for job in cleared["batches"][0]["jobs"]},
                {"running", "queued", "interrupted"},
            )
            self.assertEqual(len(stored["jobs"]), 6)
            self.assertEqual(set(stored["dismissedJobIds"]), {"done", "failed", "cancelled"})
            self.assertEqual(result.read_bytes(), b"analysis")
            self.assertEqual(assessment.read_bytes(), b"assessment")
            self.assertEqual(protocol.read_bytes(), b"protocol")
            self.assertEqual(image.read_bytes(), b"original-image")

    def test_batch_list_is_sorted_by_creation_time_instead_of_uuid_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            root.mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                project_data = bridge._project_data()
                for batch_id, created_at in (
                    ("z-old-uuid", "2026-10-20T10:00:00+00:00"),
                    ("a-new-uuid", "2026-10-25T10:00:00+00:00"),
                ):
                    batch = {
                        "batchId": batch_id,
                        "projectId": "Demo",
                        "projectData": str(project_data),
                        "createdAt": created_at,
                        "status": "completed",
                        "jobs": [{"jobId": f"{batch_id}-job", "status": "succeeded"}],
                    }
                    bridge.batches[batch_id] = batch
                    bridge._write_batch(batch)
                listed = bridge.list_batches()

            self.assertEqual(
                [item["batchId"] for item in listed["batches"]],
                ["a-new-uuid", "z-old-uuid"],
            )

    def test_restart_does_not_restore_cancelled_or_orphaned_jobs_as_active(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            root.mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                writer = DesktopBridge()
                writer.open_project_path(str(root))
                project_data = writer._project_data()
                for batch in (
                    {
                        "batchId": "cancelled-old", "projectId": "Demo",
                        "projectData": str(project_data), "status": "running",
                        "jobs": [{"jobId": "cancelled", "status": "cancelled"}],
                    },
                    {
                        "batchId": "orphaned-old", "projectId": "Demo",
                        "projectData": str(project_data), "status": "running",
                        "jobs": [
                            {"jobId": "was-running", "status": "running"},
                            {"jobId": "was-queued", "status": "queued"},
                        ],
                    },
                ):
                    writer._write_batch(batch)

                restarted = DesktopBridge()
                restarted.open_project_path(str(root))
                listed = restarted.list_batches()["batches"]

            by_id = {item["batchId"]: item for item in listed}
            self.assertEqual(by_id["cancelled-old"]["status"], "completed")
            self.assertEqual(
                {job["status"] for job in by_id["orphaned-old"]["jobs"]},
                {"interrupted"},
            )
            self.assertEqual(by_id["orphaned-old"]["status"], "interrupted")

    def test_main_view_does_not_render_persistent_batch_status_panel(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "frontend/src/App.tsx").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('<h2>Batch-Auswertung</h2>', source)
        self.assertNotIn('von {batch.jobs.length} sichtbaren Aufträgen erfolgreich', source)
        self.assertNotIn('>Wartende abbrechen</button>', source)
        self.assertNotIn('>Unterbrochene fortsetzen</button>', source)
        self.assertIn('Batch starten</button>', source)

    def test_reference_folder_open_is_allow_listed_and_read_only(self) -> None:
        bridge = DesktopBridge()
        with patch.object(bridge_module.subprocess, "run") as opened:
            response = bridge.open_evaluation_reference_folder({"variant": "2a", "code": "sp HB"})
        self.assertTrue(response["ok"])
        opened.assert_called_once_with(["open", response["path"]], check=True)
        self.assertFalse(bridge.open_evaluation_reference_folder(
            {"variant": "1", "code": "GSF"}
        )["ok"])
        for variant in ("2", "2a", "3"):
            for code in ("Bu", "AA", "SSA"):
                with self.subTest(variant=variant, code=code), patch.object(
                    bridge_module.subprocess, "run"
                ) as opened_new:
                    response = bridge.open_evaluation_reference_folder(
                        {"variant": variant, "code": code}
                    )
                    self.assertTrue(response["ok"])
                    opened_new.assert_called_once_with(["open", response["path"]], check=True)

    def test_four_tab_view_uses_one_scientific_analysis_path(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "frontend/src/App.tsx").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'const tabs:[TabKey,string][]=[["analysis","Bildanalyse"],'
            '["results","Ergebnisse"],["references_new","Referenzdatenbank"],'
            '["export","Export"]]', source,
        )
        active_view = source.split('{tab==="analysis"', 1)[1].split(
            '{Boolean(false)&&<div className="analysis-layout"', 1
        )[0]
        self.assertIn("1 · PRÜFKÖRPER AUSWÄHLEN", active_view)
        self.assertIn("2 · Bilder", active_view)
        self.assertIn("3 · KI-MODELL", active_view)
        self.assertIn("4 · KLASSIFIKATIONSVARIANTE", active_view)
        self.assertIn("5 · Plan prüfen", active_view)
        self.assertIn("6 · Analyse starten", active_view)
        self.assertIn('api("scientific_series_plan"', source)
        self.assertIn('api("start_scientific_series"', source)

    def test_scientific_plan_accepts_single_multiple_and_all_existing_specimens(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            specimen_ids = ["Probe-01", "Probe-02", "Probe-04"]
            for specimen in specimen_ids:
                folder = root / specimen
                folder.mkdir(parents=True)
                Image.new("RGB", (12, 8), "white").save(folder / "bruch.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                for specimen in specimen_ids:
                    self.assertTrue(bridge.save_image_selection({
                        "specimen": specimen, "selectedImages": ["bruch.jpg"],
                    })["ok"])
                plans = [bridge.scientific_series_plan({
                    "specimenIds": selected, "variant": "1", "provider": "ollama",
                    "model": "qwen3-vl:4b-instruct",
                }) for selected in (["Probe-01"], specimen_ids[:2], specimen_ids)]
            self.assertEqual([item["plan"]["specimenCount"] for item in plans], [1, 2, 3])
            self.assertTrue(all(item["ok"] and item["plan"]["readyToStart"] for item in plans))
            self.assertEqual(
                [item["specimen"] for item in plans[-1]["plan"]["specimens"]], specimen_ids
            )

    def test_plan_reports_unconfirmed_selection_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert" / "Probe-01"
            root.mkdir(parents=True)
            Image.new("RGB", (12, 8), "white").save(root / "bruch.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)), patch.object(
                bridge_module, "run_analysis"
            ) as model_call:
                bridge = DesktopBridge()
                bridge.open_project_path(str(root.parent))
                planned = bridge.scientific_series_plan({
                    "specimenIds": ["Probe-01"], "variant": "1", "provider": "ollama",
                    "model": "qwen3-vl:4b-instruct",
                })
            self.assertTrue(planned["ok"])
            self.assertFalse(planned["plan"]["readyToStart"])
            self.assertEqual(planned["plan"]["missingConfirmations"][0]["specimen"], "Probe-01")
            model_call.assert_not_called()

    def test_export_deletion_is_confined_to_generated_export_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            root.mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                export = bridge._project_data() / "Export"
                export.mkdir(parents=True)
                generated = export / "test.csv"
                generated.write_text("a;b\n", encoding="utf-8")
                protected = bridge._project_data() / "Ergebnisse" / "raw.json"
                protected.parent.mkdir(parents=True)
                protected.write_text("{}", encoding="utf-8")
                listed = bridge.list_export_files()
                refused = bridge.delete_export_files({
                    "exportIds": [listed["files"][0]["exportId"]],
                    "confirmDeletion": False,
                })
                deleted = bridge.delete_export_files({
                    "exportIds": [listed["files"][0]["exportId"]],
                    "confirmDeletion": True,
                })
            self.assertFalse(refused["ok"])
            self.assertTrue(deleted["ok"])
            self.assertFalse(generated.exists())
            self.assertTrue(protected.exists())

    def test_running_scientific_job_accepts_controlled_cancellation_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            root.mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                batch = {
                    "kind": "scientific_evaluation_series", "batchId": "run-1",
                    "seriesRunId": "run-1", "projectData": str(bridge._project_data()),
                    "runFolder": str(bridge._project_data() / "Wissenschaftliche_Laeufe/run-1"),
                    "status": "running", "jobs": [{
                        "jobId": "job-1", "specimen": "Probe-01", "status": "running",
                        "stage": "analyzing_images",
                    }],
                }
                bridge.batches["run-1"] = batch
                bridge._write_batch(batch)
                response = bridge.cancel_batch_jobs({"batchId": "run-1"})
            self.assertTrue(response["ok"])
            self.assertTrue(bridge.batches["run-1"]["cancellationRequested"])
            self.assertTrue(bridge.batches["run-1"]["jobs"][0]["cancellationRequested"])
            self.assertEqual(bridge.batches["run-1"]["jobs"][0]["status"], "running")
    def test_evaluation_start_automatically_creates_current_control_plan(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "frontend/src/App.tsx").read_text(
            encoding="utf-8"
        )
        start = source.split("const startEvaluationRun=async()=>", 1)[1].split(
            "const cancelWaiting", 1
        )[0]
        self.assertLess(start.index('api("evaluation_run_plan"'), start.index('api("start_analysis"'))
        self.assertIn("if(!currentPlan.structureReady)", start)
        self.assertIn("if(!currentPlan.memory?.prepared)", start)

    def test_manual_primary_selection_is_optional_separate_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            (root / "Z-1").mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                result_dir = bridge._project_data() / "Ergebnisse"
                result_dir.mkdir(parents=True)
                source = result_dir / "Z-1_test_run-manual.json"
                source.write_text(json.dumps({
                    "schema_version": "4.0", "run_id": "run-manual", "specimen_id": "Z-1",
                    "result": {"status": "KLASSIFIZIERT", "class_suggestions": [
                        {"suggestion_id": "one", "code": "KA", "confidence": "hoch"},
                        {"suggestion_id": "two", "code": "AA", "confidence": "mittel"},
                    ]},
                }), encoding="utf-8")
                original = source.read_bytes()
                unset = bridge.save_assessment({
                    "runId": "run-manual", "featureReviews": [],
                    "expertPrimaryCode": "", "reviewedBy": "",
                })
                selected = bridge.save_assessment({
                    "runId": "run-manual", "featureReviews": [],
                    "expertPrimaryCode": "KA", "reviewedBy": "Carina Zott",
                })
                missing_name = bridge.save_assessment({
                    "runId": "run-manual", "featureReviews": [],
                    "expertPrimaryCode": "AA", "reviewedBy": "",
                })
            self.assertTrue(unset["ok"])
            self.assertIsNone(unset["assessment"]["manualPrimaryFailure"]["code"])
            self.assertTrue(selected["ok"])
            manual = selected["assessment"]["manualPrimaryFailure"]
            self.assertEqual(manual["code"], "KA")
            self.assertEqual(manual["selectedBy"], "Carina Zott")
            self.assertTrue(manual["selectedAt"])
            self.assertFalse(missing_name["ok"])
            self.assertEqual(source.read_bytes(), original)

    def test_export_and_manual_primary_support_all_twelve_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            (root / "Z-11").mkdir(parents=True)
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                results = bridge._project_data() / "Ergebnisse"
                results.mkdir(parents=True)
                source = results / "Z-11_test_run-eleven.json"
                source.write_text(json.dumps({
                    "schema_version": "4.0", "run_id": "run-eleven", "specimen_id": "Z-11",
                    "execution": {"provider": "mock", "model": "mock"},
                    "result": {"status": "KLASSIFIZIERT", "class_suggestions": [
                        {"suggestion_id": f"s-{index}", "code": code, "confidence": "mittel"}
                        for index, code in enumerate(ACTIVE_CLASS_CODES)
                    ]},
                }), encoding="utf-8")
                original = source.read_bytes()
                for code in ACTIVE_CLASS_CODES:
                    saved = bridge.save_assessment({
                        "runId": "run-eleven", "featureReviews": [],
                        "expertPrimaryCode": code, "reviewedBy": "Testperson",
                    })
                    self.assertTrue(saved["ok"])
                exported = bridge.export_selected_runs({"runIds": ["run-eleven"]})
            document = json.loads(Path(exported["jsonPath"]).read_text(encoding="utf-8"))
            exported_codes = [
                item["code"] for item in document["runs"][0]["analysis"]["result"]["class_suggestions"]
            ]
            self.assertEqual(tuple(exported_codes), ACTIVE_CLASS_CODES)
            self.assertEqual(document["runs"][0]["assessment"]["manualPrimaryFailure"]["code"], "KB")
            self.assertEqual(source.read_bytes(), original)

    def test_historical_llava_result_remains_readable_and_exportable(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as data:
            root = Path(temp) / "Demo" / "Sortiert"
            (root / "Z-1").mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(root / "Z-1" / "bruch.jpg")
            with patch.object(bridge_module, "APP_DATA", Path(data)):
                bridge = DesktopBridge()
                bridge.open_project_path(str(root))
                result_dir = bridge._project_data() / "Ergebnisse"
                result_dir.mkdir(parents=True)
                source = result_dir / "Z-1_ollama_llava_test_run-1.json"
                source.write_text(json.dumps({
                    "schema_version": "2.0", "run_id": "run-1", "project_id": "Demo",
                    "specimen_id": "Z-1", "created_at": "2026-10-25T12:00:00+00:00",
                    "execution": {"provider": "ollama", "model": "llava:latest",
                                  "duration_seconds": 4.2, "provider_metadata": {}},
                    "classification_basis": {"prompt_version": "test", "evaluation_variant": "1",
                                             "references": []},
                    "inputs": [{"filename": "bruch.jpg", "sha256": "test"}],
                    "image_selection": {"included": ["bruch.jpg"], "excluded": []},
                    "result": {"status": "KLASSIFIZIERT",
                               "primary_feature": {"code": "KA", "confidence": 0.7},
                               "secondary_features": [{"code": "LSF"}],
                               "visible_observations": ["Riss am Rand sichtbar."]},
                }), encoding="utf-8")
                assessed = bridge.save_assessment({
                    "runId": "run-1", "verdict": "partially_correct", "primaryCode": "LSF",
                    "secondaryCodes": ["KA"], "note": "Fachlich geprüft.",
                })
                overview = bridge.project_results()
                detail = bridge.result_detail({"runId": "run-1"})
                exported = bridge.export_selected_runs({"runIds": ["run-1"]})
            self.assertTrue(assessed["ok"])
            self.assertEqual(overview["results"][0]["model"], "llava:latest")
            self.assertEqual(overview["results"][0]["primaryCode"], "KA")
            self.assertEqual(overview["results"][0]["assessment"]["primaryCode"], "LSF")
            self.assertEqual([item["name"] for item in detail["runImages"]], ["bruch.jpg"])
            self.assertTrue(detail["runImages"][0]["src"].startswith("data:image/jpeg;base64,"))
            self.assertTrue(exported["ok"])
            document = json.loads(Path(exported["jsonPath"]).read_text(encoding="utf-8"))
            self.assertEqual(document["runs"][0]["analysis"]["result"]["primary_feature"]["code"], "KA")
            self.assertEqual(document["runs"][0]["assessment"]["primaryCode"], "LSF")
            self.assertNotEqual(Path(exported["jsonPath"]).name, source.name)


if __name__ == "__main__":
    unittest.main()
