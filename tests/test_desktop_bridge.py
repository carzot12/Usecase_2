from __future__ import annotations

import json
import tempfile
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


if __name__ == "__main__":
    unittest.main()
