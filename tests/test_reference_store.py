from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from desktop.reference_store import ReferenceStore


class ReferenceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definitions = {
            code: {
                "name": f"Klasse {code}",
                "definition": f"Definition {code}",
                "feature_type": "fracture_pattern",
                "visual_features": [],
                "sources": [],
                **({"validated_classification_rule": {"klassifikationsziel": f"Ziel {code}"}}
                   if code in {"A", "B", "C", "D", "E", "F"} else {}),
            }
            for code in ("A", "B", "C", "D", "E", "F", "G", "H")
        }

    def test_uses_exactly_the_documented_codes_and_only_shows_active_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = ReferenceStore(Path(temp), self.definitions)
            initial = store.catalog()
            self.assertTrue(initial["configured"])
            self.assertEqual([item["code"] for item in initial["classes"]], list("ABCDEF"))
            self.assertEqual(len(initial["availableClasses"]), 8)
            with self.assertRaisesRegex(ValueError, "klassifikationsregeln.json"):
                store.save_active_codes(["A", "B"])
            selected = ["A", "B", "C", "D", "E", "F"]
            store.save_active_codes(selected)
            catalog = store.catalog()
            self.assertTrue(catalog["configured"])
            self.assertEqual([item["code"] for item in catalog["classes"]], selected)
            self.assertEqual(len(self.definitions), 8)

    def test_multi_import_copies_images_records_paths_and_rejects_hash_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as sources:
            root = Path(temp)
            store = ReferenceStore(root, self.definitions)
            store.save_active_codes(["A", "B", "C", "D", "E", "F"])
            first = Path(sources) / "first.jpg"
            second = Path(sources) / "second.jpg"
            duplicate = Path(sources) / "copy.jpg"
            Image.new("RGB", (20, 10), "red").save(first)
            Image.new("RGB", (20, 10), "blue").save(second)
            duplicate.write_bytes(first.read_bytes())
            original_bytes = {path.name: path.read_bytes() for path in (first, second, duplicate)}

            result = store.import_images(
                code="A",
                paths=[first, second, duplicate],
                specimen_id="Z-1",
                description="fachliche Notiz",
                confirmed=False,
            )

            self.assertEqual(len(result["imported"]), 2)
            self.assertEqual(len(result["duplicates"]), 1)
            self.assertEqual(result["duplicates"][0]["source"], str(duplicate.resolve()))
            for entry in result["imported"]:
                stored = root / entry["storedPath"]
                self.assertTrue(stored.is_file())
                self.assertEqual(entry["specimenId"], "Z-1")
                self.assertEqual(entry["description"], "fachliche Notiz")
                self.assertFalse(entry["confirmed"])
                self.assertEqual(Path(entry["sourcePath"]).parent, Path(sources).resolve())
            self.assertEqual(
                {path.name: path.read_bytes() for path in (first, second, duplicate)},
                original_bytes,
            )
            manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["entries"]), 2)

    def test_existing_image_is_unconfirmed_until_explicitly_registered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "Datenbank/Referenzbilder_neu/A_Klasse_A"
            folder.mkdir(parents=True)
            image = folder / "existing.jpg"
            Image.new("RGB", (20, 10), "green").save(image)
            store = ReferenceStore(root, self.definitions)
            store.save_active_codes(["A", "B", "C", "D", "E", "F"])

            entry = store.catalog()["classes"][0]["references"][0]
            self.assertFalse(entry["registered"])
            self.assertFalse(entry["confirmed"])
            updated = store.update_entry(
                entry["id"], code="A", specimen_id="Z-2",
                description="nachträglich zugeordnet", confirmed=True,
            )["entry"]
            self.assertTrue(updated["confirmed"])
            self.assertEqual(updated["specimenId"], "Z-2")
            self.assertTrue(image.is_file())

    def test_confirmed_entry_requires_explicit_confirmation_before_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as sources:
            store = ReferenceStore(Path(temp), self.definitions)
            store.save_active_codes(["A", "B", "C", "D", "E", "F"])
            source = Path(sources) / "reference.jpg"
            Image.new("RGB", (20, 10), "yellow").save(source)
            entry = store.import_images(
                code="A", paths=[source], specimen_id="Z-3",
                description="bestätigt", confirmed=True,
            )["imported"][0]
            with self.assertRaisesRegex(ValueError, "ausdrücklicher Bestätigung"):
                store.update_entry(
                    entry["id"], code="B", specimen_id="Z-3",
                    description="korrigiert", confirmed=True,
                )
            changed = store.update_entry(
                entry["id"], code="B", specimen_id="Z-3",
                description="korrigiert", confirmed=True,
                confirm_confirmed_change=True,
            )["entry"]
            self.assertEqual(changed["code"], "B")
            self.assertTrue((Path(temp) / changed["storedPath"]).is_file())


if __name__ == "__main__":
    unittest.main()
