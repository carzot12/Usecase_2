from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Tool.ki_bildanalyse import (
    ALLOWED_INPUT_ROOT,
    AnalysisError,
    collect_test_images,
    load_catalog,
    run_analysis,
    select_references,
    validate_and_enrich,
)
from Tool.openai_config import resolve_api_key


TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class OpenAIConfigTests(unittest.TestCase):
    def test_environment_key_has_priority(self) -> None:
        result = resolve_api_key(environment={"OPENAI_API_KEY": "test-only"}, platform="darwin")
        self.assertEqual(result.source, "environment")
        self.assertNotIn("test-only", repr(result))

    def test_keychain_fallback_without_real_keychain(self) -> None:
        result = resolve_api_key(
            environment={},
            platform="darwin",
            runner=lambda *args, **kwargs: SimpleNamespace(stdout="mock-key\n"),
        )
        self.assertEqual(result.source, "macos_keychain")
        self.assertNotIn("mock-key", repr(result))


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()

    def test_catalog_and_confirmed_main_references(self) -> None:
        self.assertEqual(len(self.catalog.codes), 29)
        self.assertEqual(len(self.catalog.references), 26)
        self.assertEqual(len(self.catalog.codes_without_references), 18)
        self.assertTrue(all(reference.path.is_file() for reference in self.catalog.references))
        selected = select_references(self.catalog, 1)
        self.assertEqual(len(selected), 11)
        self.assertEqual(len({reference.code for reference in selected}), 11)

    def test_unknown_class_is_rejected(self) -> None:
        raw = {
            "status": "KLASSIFIZIERT",
            "primary_code": "NICHT_VORHANDEN",
            "primary_evidence": ["sichtbar"],
            "primary_confidence": 0.5,
            "secondary_features": [],
            "visible_observations": [],
            "uncertainty_reasons": [],
            "missing_views": [],
        }
        with self.assertRaises(AnalysisError):
            validate_and_enrich(raw, self.catalog)

    def test_causal_claim_is_rejected(self) -> None:
        raw = {
            "status": "KLASSIFIZIERT",
            "primary_code": "KA",
            "primary_evidence": ["Der Ast verursachte den Bruch."],
            "primary_confidence": 0.7,
            "secondary_features": [],
            "visible_observations": [],
            "uncertainty_reasons": [],
            "missing_views": [],
        }
        with self.assertRaises(AnalysisError):
            validate_and_enrich(raw, self.catalog)

    def test_live_mode_needs_explicit_cost_confirmation(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp:
            (Path(temp) / "test.png").write_bytes(TINY_PNG)
            with self.assertRaisesRegex(AnalysisError, "confirm-paid-request"):
                run_analysis(Path(temp), live=True, confirm_paid_request=False)

    def test_mock_pipeline_writes_validated_json_without_api(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            destination = run_analysis(input_dir, output_dir=Path(output))
            document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["execution"]["mode"], "mock")
        self.assertFalse(document["execution"]["paid_api_request_performed"])
        self.assertEqual(document["result"]["status"], "UNKLAR")
        self.assertEqual(document["classification_basis"]["allowed_class_count"], 29)
        self.assertEqual(document["classification_basis"]["confirmed_main_reference_count_used"], 0)

    def test_reference_copy_is_rejected_as_independent_test(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        source = self.catalog.references[0].path
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp:
            copied = Path(temp) / "umbenannt.jpg"
            copied.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(AnalysisError, "inhaltlich identisch"):
                collect_test_images(Path(temp), self.catalog)

    def test_sorted_specimen_uses_analysis_selection(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp:
            specimen = Path(temp)
            for index in range(4):
                (specimen / f"all_{index}.png").write_bytes(TINY_PNG)
            selection = specimen / "Analyseauswahl"
            selection.mkdir()
            (selection / "view_1.png").write_bytes(TINY_PNG)
            (selection / "view_2.png").write_bytes(TINY_PNG)
            images = collect_test_images(specimen, self.catalog)
        self.assertEqual([path.name for path in images], ["view_1.png", "view_2.png"])


if __name__ == "__main__":
    unittest.main()

