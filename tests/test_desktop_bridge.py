from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import desktop.bridge as bridge_module
from desktop.bridge import DesktopBridge, _safe_name


class DesktopBridgeTests(unittest.TestCase):
    def test_safe_name_rejects_empty_names(self) -> None:
        with self.assertRaises(ValueError):
            _safe_name(" / ")

    def test_project_snapshot_ignores_number_photos(self) -> None:
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
            self.assertEqual(result["project"]["specimens"][0]["imageCount"], 1)

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
                self.assertNotIn("Nummer_1.jpg", {image["name"] for image in detail["images"]})

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
                {"provider": "ollama", "model": "qwen3-vl:4b"},
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
            qwen_calls = [call for call in calls if call[2] == "qwen3-vl:4b"]
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
                    "models": [{"provider": "ollama", "model": "qwen3-vl:4b"}],
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


if __name__ == "__main__":
    unittest.main()
