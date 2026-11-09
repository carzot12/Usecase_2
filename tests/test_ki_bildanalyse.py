from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from PIL import Image

from Tool.ki_bildanalyse import (
    ALLOWED_INPUT_ROOT,
    AnalysisError,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_NUM_PREDICT,
    PROMPT_VERSION,
    OllamaOutputError,
    MAX_MISTRAL_IMAGES_PER_REQUEST,
    _parser,
    build_mistral_messages,
    build_ollama_messages,
    build_request_content,
    _parse_ollama_json,
    collect_test_images,
    call_mistral,
    call_ollama,
    direct_variant_references,
    load_catalog,
    merge_partial_classifications,
    response_schema,
    run_analysis,
    select_references,
    validate_and_enrich,
)
from Tool.mistral_config import KEYCHAIN_SERVICE as MISTRAL_KEYCHAIN_SERVICE
from Tool.mistral_config import (
    MistralAPIKeyResolution,
    MistralRequestError,
    resolve_mistral_api_key,
)
from Tool.openai_config import resolve_api_key
from Tool.evaluation_variants import ACTIVE_CLASS_CODES, inspect_variant


TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def direct_reference_report(root: Path, variant_id: str, per_code: int = 1) -> dict[str, object]:
    references = []
    for code in ACTIVE_CLASS_CODES:
        for index in range(per_code):
            path = root / code.replace(" ", "_") / f"ref-{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 12), (index * 30, 40, 60)).save(path)
            references.append({
                "reference_id": f"{code}:{path.relative_to(root).as_posix()}",
                "code": code,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "confirmed": True,
                "specimen": f"REF-{code}-{index}",
                "description": (
                    f"Annotierte Markierung für {code}." if variant_id == "3" else ""
                ),
            })
    return {
        "variantId": variant_id, "ready": True, "referenceRoot": str(root),
        "manifestHash": f"mock-{variant_id}", "references": references,
    }


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


class MistralConfigTests(unittest.TestCase):
    def test_mistral_key_comes_only_from_named_macos_keychain_service(self) -> None:
        captured = {}

        def runner(command, **kwargs):
            captured["command"] = command
            return SimpleNamespace(stdout="mistral-mock-key\n")

        result = resolve_mistral_api_key(platform="darwin", runner=runner)
        self.assertEqual(result.source, "macos_keychain")
        self.assertEqual(result.service, "Usecase_2_Mistral")
        self.assertEqual(MISTRAL_KEYCHAIN_SERVICE, "Usecase_2_Mistral")
        self.assertEqual(captured["command"], [
            "security", "find-generic-password", "-s", "Usecase_2_Mistral", "-w",
        ])
        self.assertNotIn("mistral-mock-key", repr(result))


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.validated_codes = tuple(code for code, definition in sorted(
            cls.catalog.definitions.items(),
            key=lambda item: item[1].get("validated_classification_rule_order", 10**9),
        ) if "validated_classification_rule" in definition)

    def test_catalog_and_confirmed_main_references(self) -> None:
        self.assertEqual(len(self.catalog.codes), 29)
        self.assertEqual(len(self.catalog.references), 26)
        self.assertEqual(len(self.catalog.codes_without_references), 18)
        self.assertTrue(all(reference.path.is_file() for reference in self.catalog.references))
        selected = select_references(self.catalog, 1)
        self.assertEqual(len(selected), 11)
        self.assertEqual(len({reference.code for reference in selected}), 11)

    def test_version_1_4_has_twelve_uniform_rules_and_archives_1_3(self) -> None:
        root = Path(__file__).resolve().parents[1]
        rules = json.loads(
            (root / "Datenbank/klassifikationsregeln.json").read_text(encoding="utf-8")
        )
        required = {
            "code", "name", "klassentyp", "klassifikationsziel",
            "positive_bildmerkmale", "abgrenzung_zu_aehnlichen_klassen",
            "nicht_vergeben_wenn", "unklar_wenn",
            "einschraenkungen_visuelle_erkennung",
            "nicht_zulaessige_ursachenaussage", "fachlicher_pruefstatus",
            "literaturquellen",
        }
        self.assertEqual(rules["meta"]["version"], "1.4")
        self.assertEqual(
            [item["code"] for item in rules["classes"]],
            ["GSF", "LSF", "KA", "SA", "sp HB", "Spl HB", "RiSch", "LR", "Bu", "AA", "SSA", "KB"],
        )
        self.assertTrue(all(set(item) == required for item in rules["classes"]))
        archive = root / "Datenbank/Archiv/klassifikationsregeln_v1.3.json"
        self.assertEqual(
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            "7b7f8ad3fe390e17dc14768215dc2ea893e56af180d538477268e2c0f68ea2a2",
        )

    def test_reference_selection_excludes_same_specimen_and_matching_image_hash(self) -> None:
        first = self.catalog.references[0]
        specimen_id = first.path.parents[1].name
        digest = hashlib.sha256(first.path.read_bytes()).hexdigest()
        selected = select_references(
            self.catalog,
            10,
            excluded_specimen_id=specimen_id,
            excluded_hashes={digest},
        )
        self.assertTrue(selected)
        self.assertTrue(all(item.path.parents[1].name != specimen_id for item in selected))
        self.assertTrue(all(
            hashlib.sha256(item.path.read_bytes()).hexdigest() != digest
            for item in selected
        ))

    def test_variant_one_catalog_does_not_open_reference_manifest(self) -> None:
        allowed = self.validated_codes
        from Tool import ki_bildanalyse as module
        original = module._read_json
        opened: list[str] = []

        def recording_read(path: Path):
            opened.append(path.name)
            return original(path)

        with patch.object(module, "_read_json", side_effect=recording_read):
            catalog = load_catalog(include_references=False, allowed_codes=allowed)
        self.assertEqual(catalog.codes, allowed)
        self.assertFalse(catalog.reference_database_accessed)
        self.assertEqual(catalog.references, ())
        self.assertNotIn("referenzbilder_manifest.json", opened)

    def test_variant_one_rejects_a_class_list_different_from_documented_rules(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp:
            (Path(temp) / "test.png").write_bytes(TINY_PNG)
            with self.assertRaisesRegex(AnalysisError, "Merkmalskatalog"):
                run_analysis(
                    Path(temp), provider="openai", evaluation_variant="1",
                    allowed_class_codes=ACTIVE_CLASS_CODES[:-1],
                )

    def test_variant_one_mock_records_reference_free_execution(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        allowed = ACTIVE_CLASS_CODES
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, \
                tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            destination = run_analysis(
                input_dir,
                provider="openai",
                output_dir=Path(output),
                evaluation_variant="1",
                allowed_class_codes=allowed,
                test_series={
                    "series_id": "scientific-1",
                    "name": "Erste wissenschaftliche Testreihe",
                    "category": "scientific_evaluation",
                    "classification_rules_version": "1.4",
                    "active_class_codes": list(allowed),
                },
            )
            document = json.loads(destination.read_text(encoding="utf-8"))
        basis = document["classification_basis"]
        self.assertEqual(basis["evaluation_variant"], "1")
        self.assertFalse(basis["reference_database_accessed"])
        self.assertEqual(basis["confirmed_main_reference_count_available"], 0)
        self.assertEqual(basis["confirmed_main_reference_count_used"], 0)
        self.assertEqual(basis["allowed_class_codes"], list(allowed))
        self.assertEqual(basis["references"], [])
        self.assertEqual(basis["active_class_set_version"], "klassifikationsregeln/1.4")
        self.assertEqual(len(basis["active_class_descriptions_sha256"]), 64)
        self.assertFalse(basis["ground_truth_transmitted"])
        self.assertFalse(basis["previous_results_transmitted"])
        self.assertFalse(basis["expert_assessments_transmitted"])
        self.assertEqual(document["test_series"]["series_id"], "scientific-1")
        self.assertEqual(document["specimen_id"], input_dir.name)
        self.assertEqual(document["execution"]["provider"], "mock")
        self.assertTrue(document["execution"]["model"])
        self.assertEqual(basis["classification_rules_version"], "1.4")
        self.assertEqual(basis["allowed_class_count"], 12)
        self.assertEqual(len(basis["prompt_hash_sha256"]), 64)
        self.assertEqual(len(document["inputs"][0]["sha256"]), 64)
        self.assertTrue(document["created_at"])
        self.assertIsInstance(document["execution"]["duration_seconds"], float)

    def test_all_twelve_reference_features_are_structured_multilabel_codes(self) -> None:
        catalog = load_catalog(
            include_references=False,
            allowed_codes=ACTIVE_CLASS_CODES,
        )
        self.assertEqual(catalog.active_class_codes, ACTIVE_CLASS_CODES)
        self.assertEqual(catalog.additional_visible_codes, ())
        self.assertEqual(catalog.codes, ACTIVE_CLASS_CODES)
        result = validate_and_enrich({
            "status": "KLASSIFIZIERT",
            "observations_short": ["Kantenast und Riss sichtbar"],
            "visible_features": ["Mehrere Äste nahe beieinander"],
            "class_suggestions": [
                {"code": "KA", "confidence": "hoch", "evidence_status": "sicher_sichtbar"},
                {"code": "RiSch", "confidence": "mittel", "evidence_status": "vermutet"},
                {"code": "AA", "confidence": "mittel", "evidence_status": "sicher_sichtbar"},
            ],
            "uncertain_points": [],
        }, catalog)
        self.assertEqual(
            [item["code"] for item in result["class_suggestions"]],
            ["KA", "RiSch", "AA"],
        )

    def test_variant_one_prompt_contains_twelve_compact_descriptions_and_test_images(self) -> None:
        catalog = load_catalog(
            include_references=False,
            allowed_codes=ACTIVE_CLASS_CODES,
        )
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "bruch.png"
            image.write_bytes(TINY_PNG)
            content = build_request_content([image], catalog, [])
        instructions = json.loads(content[0]["text"])
        self.assertEqual(
            [item["code"] for item in instructions["classes"]],
            list(ACTIVE_CLASS_CODES),
        )
        self.assertEqual(len(instructions["classes"]), 12)
        self.assertEqual(instructions["classes"][-1]["code"], "KB")
        image_items = [item for item in content if item["type"] == "input_image"]
        self.assertEqual(len(image_items), 1)
        self.assertTrue(image_items[0]["image_url"].startswith("data:image/jpeg;base64,"))
        serialized = json.dumps(instructions, ensure_ascii=False)
        self.assertNotIn("confirmed_reference_count", serialized)
        self.assertNotIn("Ground Truth", serialized)
        self.assertNotIn("fachliche Korrektur", serialized)
        self.assertNotIn("fachlicher_pruefstatus", serialized)
        self.assertNotIn("literaturquellen", serialized)
        self.assertNotIn("offene_punkte", serialized)
        self.assertNotIn("http", serialized)
        self.assertTrue(all(
            set(item) == {
                "code", "name", "definition", "positive_image_features",
                "distinguishing_features", "uncertainty_rules",
            }
            for item in instructions["classes"]
        ))
        example = json.loads(
            (Path(__file__).resolve().parents[1]
             / "Datenbank/klassifikationsprompt_v1.4_beispiel.json").read_text(
                 encoding="utf-8"
             )
        )
        self.assertEqual(example["prompt"], instructions)
        self.assertEqual(example["prompt_version"], PROMPT_VERSION)
        self.assertEqual(example["reference_images"], 0)
        self.assertFalse(example["ground_truth_included"])
        archived = json.loads(
            (Path(__file__).resolve().parents[1]
             / "Datenbank/Archiv/klassifikationsprompt_v1.4_prompt_5.4.json").read_text(
                 encoding="utf-8"
             )
        )
        self.assertEqual(
            archived["prompt_version"],
            "5.4-scientific-multilabel-no-ranking",
        )

    def test_prompt_requires_visible_observations_before_candidates(self) -> None:
        content = build_request_content([], self.catalog, [])
        instructions = json.loads(content[0]["text"])
        self.assertIn("zwei strikt getrennten Phasen", instructions["rules"][0])
        self.assertIn("sichtbare Beobachtungen", instructions["rules"][0])
        serialized = json.dumps(instructions, ensure_ascii=False)
        self.assertIn("keine Absätze", serialized)
        self.assertIn("alle ausgewählten Testbilder gemeinsam", serialized)
        self.assertIn("vollständig und gleichberechtigt", serialized)
        self.assertIn("keine Rangfolge", serialized)
        self.assertIn("keine Hauptversagensursache", serialized)
        self.assertIn("charakteristische Struktur tatsächlich sichtbar", serialized)
        self.assertIn("konkrete Bildstelle eindeutig benennbar", serialized)
        self.assertIn("Bloße visuelle Ähnlichkeit mit bereitgestellten Referenzen", serialized)
        self.assertIn("Begleiterscheinung eines anderen Merkmals", serialized)
        self.assertIn("nicht als sicher_sichtbar", serialized)
        self.assertIn("keine Bildkoordinaten", serialized)
        self.assertNotIn("3,5", serialized)
        self.assertNotIn("3.5", serialized)

    def test_compact_response_schema_has_equal_weight_suggestions_without_ranking(self) -> None:
        schema = response_schema(self.catalog.codes)
        self.assertEqual(
            set(schema["required"]),
            {
                "status", "observations_short", "visible_features",
                "class_suggestions", "uncertain_points",
            },
        )
        serialized = json.dumps(schema, ensure_ascii=False)
        self.assertNotIn('"priority"', serialized)
        self.assertNotIn('"primary"', serialized)
        suggestion_schema = schema["properties"]["class_suggestions"]["items"]
        self.assertIn("evidence_status", suggestion_schema["required"])
        self.assertEqual(
            suggestion_schema["properties"]["evidence_status"]["enum"],
            ["sicher_sichtbar", "vermutet"],
        )
        self.assertNotIn("reasoning", serialized)
        self.assertEqual(
            schema["properties"]["observations_short"]["items"]["maxLength"], 120
        )
        self.assertEqual(schema["properties"]["observations_short"]["minItems"], 1)

    def test_compact_output_requires_valid_evidence_status(self) -> None:
        base = {
            "status": "KLASSIFIZIERT",
            "observations_short": ["Kantenast sichtbar"],
            "visible_features": ["Astanschnitt berührt Längskante"],
            "uncertain_points": [],
        }
        for evidence_status in (None, "unklar"):
            with self.subTest(evidence_status=evidence_status), self.assertRaisesRegex(
                AnalysisError, "Evidenzstatus"
            ):
                validate_and_enrich({
                    **base,
                    "class_suggestions": [{
                        "code": "KA", "confidence": "hoch",
                        "evidence_status": evidence_status,
                    }],
                }, self.catalog)

    def test_compact_output_lists_all_suggestions_without_roles(self) -> None:
        result = validate_and_enrich({
            "status": "KLASSIFIZIERT",
            "observations_short": ["Zwei Merkmale sichtbar"],
            "visible_features": ["Astanschnitt", "lokale Faserumlenkung"],
            "class_suggestions": [
                {"code": "KA", "confidence": "hoch", "evidence_status": "sicher_sichtbar"},
                {"code": "LSF", "confidence": "mittel", "evidence_status": "sicher_sichtbar"},
            ],
            "uncertain_points": [],
        }, self.catalog)
        self.assertEqual(result["suggested_feature_codes"], ["KA", "LSF"])
        self.assertNotIn("primary_suggestion_code", result)
        self.assertTrue(all("suggestion_role" not in item for item in result["class_suggestions"]))

    def test_compact_output_is_enriched_without_causal_or_ranking_fields(self) -> None:
        result = validate_and_enrich({
            "status": "KLASSIFIZIERT",
            "observations_short": ["Bruch überwiegend quer", "Bruch überwiegend quer"],
            "visible_features": ["lokale Splitterbildung"],
            "class_suggestions": [
                {"code": "KA", "confidence": "hoch", "evidence_status": "sicher_sichtbar"},
                {"code": "LSF", "confidence": "unsicher", "evidence_status": "vermutet"},
            ],
            "uncertain_points": ["GSF/LSF nicht sicher unterscheidbar"],
        }, self.catalog)
        self.assertEqual(result["observations_short"], ["Bruch überwiegend quer"])
        self.assertEqual(
            [(item["code"], item["confidence_label"]) for item in result["class_suggestions"]],
            [("KA", "hoch"), ("LSF", "unsicher")],
        )
        self.assertEqual(
            [item["confidence"] for item in result["class_suggestions"]],
            [0.85, 0.35],
        )
        self.assertEqual(result["suggested_feature_codes"], ["KA", "LSF"])
        self.assertEqual(
            [item["evidence_status"] for item in result["class_suggestions"]],
            ["sicher_sichtbar", "vermutet"],
        )
        self.assertFalse(any(item["proven_failure_cause"] for item in result["class_suggestions"]))
        self.assertEqual(result["analysis_outcome"], "merkmale_erkannt")

    def test_partial_merge_preserves_all_codes_and_documents_conflicts(self) -> None:
        first = validate_and_enrich({
            "status": "KLASSIFIZIERT", "observations_short": ["Ast an Kante"],
            "visible_features": ["Astanschnitt"], "uncertain_points": [],
            "class_suggestions": [
                {"code": "KA", "confidence": "hoch", "evidence_status": "sicher_sichtbar"},
            ],
        }, self.catalog)
        second = validate_and_enrich({
            "status": "KLASSIFIZIERT", "observations_short": ["Faser lokal schräg"],
            "visible_features": ["Faserumlenkung"], "uncertain_points": [],
            "class_suggestions": [
                {"code": "KA", "confidence": "unsicher", "evidence_status": "vermutet"},
                {"code": "LSF", "confidence": "mittel", "evidence_status": "sicher_sichtbar"},
            ],
        }, self.catalog)
        merged = merge_partial_classifications([first, second], complete=True)
        self.assertEqual([item["code"] for item in merged["class_suggestions"]], ["KA", "LSF"])
        self.assertEqual(merged["merge_status"], "vollstaendig")
        self.assertEqual(merged["partial_request_count"], 2)
        self.assertEqual(merged["partial_evidence_by_code"]["KA"][0]["confidence_label"], "hoch")
        self.assertEqual(merged["merge_conflicts"][0]["type"], "suggestion_disagreement")

    def test_compact_unclear_output_rejects_empty_and_keeps_factual_uncertainty(self) -> None:
        with self.assertRaisesRegex(AnalysisError, "vollständig leere Sichtauswertung"):
            validate_and_enrich({
                "status": "UNKLAR", "observations_short": [], "visible_features": [],
                "class_suggestions": [], "uncertain_points": [],
            }, self.catalog)
        uncertain = validate_and_enrich({
            "status": "UNKLAR", "observations_short": ["Bruchbereich teilweise verdeckt"],
            "visible_features": [], "class_suggestions": [],
            "uncertain_points": ["Zuordnung nicht sicher"],
        }, self.catalog)
        self.assertEqual(uncertain["analysis_outcome"], "fachlich_unklar")

    def test_compact_output_distinguishes_no_features_from_unclear(self) -> None:
        no_features = validate_and_enrich({
            "status": "KEINE_MERKMALE",
            "observations_short": ["Prüfkörper vollständig sichtbar"],
            "visible_features": [],
            "class_suggestions": [],
            "uncertain_points": [],
        }, self.catalog)
        self.assertEqual(no_features["analysis_outcome"], "keine_merkmale_erkennbar")
        with self.assertRaisesRegex(AnalysisError, "Unsicherheitspunkt"):
            validate_and_enrich({
                "status": "UNKLAR",
                "observations_short": ["Bruchbereich nur teilweise sichtbar"],
                "visible_features": [],
                "class_suggestions": [],
                "uncertain_points": [],
            }, self.catalog)

    def test_compact_output_rejects_long_or_causal_observations(self) -> None:
        base = {
            "status": "UNKLAR", "visible_features": [], "class_suggestions": [],
            "uncertain_points": [],
        }
        with self.assertRaisesRegex(AnalysisError, "mehr als 120 Zeichen"):
            validate_and_enrich({**base, "observations_short": ["x" * 121]}, self.catalog)
        with self.assertRaisesRegex(AnalysisError, "Ursachenaussage"):
            validate_and_enrich({
                **base, "observations_short": ["Der Ast verursachte den Bruch"]
            }, self.catalog)

    def test_causal_filter_rejects_claims_but_allows_negated_or_open_statements(self) -> None:
        forbidden = (
            "Der Ast verursachte den Bruch.",
            "Die Bruchursache ist der Kantenast.",
            "Der Ast ist ursächlich für das Versagen.",
            "Der Ast führte zum Bruch.",
            "Der Ast ist nicht nur sichtbar, sondern verursachte den Bruch.",
        )
        allowed = (
            "Bruchursache nicht nachweisbar",
            "Bruchursache nicht beurteilbar",
            "Die Ursache ist aus den Bildern nicht erkennbar",
            "Ein ursächlicher Zusammenhang kann nicht belegt werden",
        )
        base = {
            "status": "UNKLAR", "observations_short": ["Bruchzone sichtbar"],
            "visible_features": [], "class_suggestions": [],
        }
        for sentence in forbidden:
            with self.subTest(sentence=sentence), self.assertRaisesRegex(
                AnalysisError, "Ursachenaussage"
            ):
                validate_and_enrich({
                    **base, "uncertain_points": [sentence],
                }, self.catalog)
        for sentence in allowed:
            with self.subTest(sentence=sentence):
                result = validate_and_enrich({
                    **base, "uncertain_points": [sentence],
                }, self.catalog)
                self.assertEqual(result["uncertain_points"], [sentence])

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
                run_analysis(
                    Path(temp), provider="openai", live=True,
                    confirm_paid_request=False,
                )

    def test_mock_pipeline_writes_validated_json_without_api(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            destination = run_analysis(
                input_dir, provider="openai", output_dir=Path(output)
            )
            document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["execution"]["mode"], "mock")
        self.assertFalse(document["execution"]["paid_api_request_performed"])
        self.assertEqual(document["result"]["status"], "UNKLAR")
        self.assertEqual(document["classification_basis"]["allowed_class_count"], 12)
        self.assertEqual(document["classification_basis"]["confirmed_main_reference_count_used"], 0)
        self.assertEqual(document["execution"]["provider"], "mock")
        self.assertEqual(document["schema_version"], "4.2")
        self.assertEqual(document["result"]["observations_short"], ["Keine reale Bildauswertung"])
        self.assertIn("duration_seconds", document["execution"])

    def test_mock_pipeline_uses_every_explicitly_selected_image_and_records_exclusions(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            names = [f"damage_{index}.jpg" for index in range(5)]
            for index, name in enumerate(names):
                Image.new("RGB", (24, 18), (index * 31, index * 17, index * 9)).save(input_dir / name)
            selected = names[:4]
            destination = run_analysis(
                input_dir,
                provider="openai",
                output_dir=Path(output),
                allowed_input_root=ALLOWED_INPUT_ROOT,
                selected_image_names=selected,
            )
            document = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(document["image_selection"]["included"], selected)
        self.assertEqual(document["image_selection"]["excluded"], names[4:])
        self.assertEqual([item["filename"] for item in document["inputs"]], selected)

    def test_openai_and_ollama_share_the_same_factual_prompt(self) -> None:
        openai_content = build_request_content([], self.catalog, [])
        ollama_messages = build_ollama_messages([], self.catalog, [])
        self.assertEqual(openai_content[0]["text"], ollama_messages[1]["content"])

    def test_variant_2a_builders_send_all_twelve_direct_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            references = direct_variant_references(
                direct_reference_report(Path(temp), "2A")
            )
            self.assertEqual([item.code for item in references], list(ACTIVE_CLASS_CODES))
            self.assertEqual(len(references), 12)

            openai_content = build_request_content([], self.catalog, references)
            ollama_messages = build_ollama_messages([], self.catalog, references)
            self.assertEqual(sum(item["type"] == "input_image" for item in openai_content), 12)
            self.assertEqual(sum("images" in item for item in ollama_messages), 12)
            self.assertFalse(any("reference_memory" in str(item) for item in openai_content))

    def test_variant_2a_mock_records_direct_references_without_model_call(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as references_temp, tempfile.TemporaryDirectory(
            dir=ALLOWED_INPUT_ROOT
        ) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            report = direct_reference_report(Path(references_temp), "2A")
            with patch("Tool.ki_bildanalyse.inspect_variant", return_value=report):
                destination = run_analysis(
                    input_dir, provider="openai", evaluation_variant="2A",
                    output_dir=Path(output), allowed_input_root=ALLOWED_INPUT_ROOT,
                )
            document = json.loads(destination.read_text(encoding="utf-8"))

        settings = document["execution"]["settings"]
        self.assertEqual(settings["reference_delivery_mode"], "direct_images_per_analysis")
        self.assertIsNone(settings["reference_memory_path"])
        self.assertEqual(len(document["classification_basis"]["references"]), 12)
        self.assertEqual(document["execution"]["mode"], "mock")

    def test_variant_2b_mock_sends_every_confirmed_reference_directly(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as references_temp, tempfile.TemporaryDirectory(
            dir=ALLOWED_INPUT_ROOT
        ) as input_temp, tempfile.TemporaryDirectory() as output_temp:
            reference_root = Path(references_temp)
            report = direct_reference_report(reference_root, "2B", per_code=2)
            input_dir = Path(input_temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            with patch("Tool.ki_bildanalyse.inspect_variant", return_value=report):
                destination = run_analysis(
                    input_dir, provider="openai", evaluation_variant="2B",
                    output_dir=Path(output_temp), allowed_input_root=ALLOWED_INPUT_ROOT,
                )
            document = json.loads(destination.read_text(encoding="utf-8"))

        settings = document["execution"]["settings"]
        self.assertEqual(settings["reference_delivery_mode"], "direct_images_per_analysis")
        self.assertIsNone(settings["reference_memory_path"])
        self.assertEqual(len(document["classification_basis"]["references"]), 24)
        self.assertEqual(
            {item["code"] for item in document["classification_basis"]["references"]},
            set(ACTIVE_CLASS_CODES),
        )
        self.assertEqual(document["execution"]["mode"], "mock")

    def test_variant_three_mock_sends_all_annotations_directly(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as references_temp, tempfile.TemporaryDirectory(
            dir=ALLOWED_INPUT_ROOT
        ) as input_temp, tempfile.TemporaryDirectory() as output_temp:
            report = direct_reference_report(Path(references_temp), "3", per_code=2)
            references = direct_variant_references(report)
            content = build_request_content([], self.catalog, references)
            self.assertEqual(sum(item["type"] == "input_image" for item in content), 24)
            self.assertTrue(any("Sichtbare Annotation" in str(item.get("text", "")) for item in content))
            input_dir = Path(input_temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            with patch("Tool.ki_bildanalyse.inspect_variant", return_value=report):
                destination = run_analysis(
                    input_dir, provider="openai", evaluation_variant="3",
                    output_dir=Path(output_temp), allowed_input_root=ALLOWED_INPUT_ROOT,
                )
            document = json.loads(destination.read_text(encoding="utf-8"))

        settings = document["execution"]["settings"]
        self.assertEqual(settings["reference_delivery_mode"], "direct_images_per_analysis")
        self.assertIsNone(settings["reference_memory_path"])
        self.assertEqual(len(document["classification_basis"]["references"]), 24)
        self.assertTrue(all(item["description"] for item in document["classification_basis"]["references"]))

    def test_variant_three_rejects_non_openai_provider_without_transport(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as references_temp, tempfile.TemporaryDirectory(
            dir=ALLOWED_INPUT_ROOT
        ) as input_temp, tempfile.TemporaryDirectory() as output_temp:
            input_dir = Path(input_temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            report = direct_reference_report(Path(references_temp), "3", per_code=2)
            with patch("Tool.ki_bildanalyse.inspect_variant", return_value=report):
                with self.assertRaisesRegex(AnalysisError, "ausschließlich für OpenAI"):
                    run_analysis(
                        input_dir, provider="ollama", model="qwen3-vl:4b-instruct",
                        evaluation_variant="3", output_dir=Path(output_temp),
                        allowed_input_root=ALLOWED_INPUT_ROOT,
                    )

    def test_variant_2a_mistral_is_blocked_without_transport_or_reduction(self) -> None:
        def forbidden(*_args, **_kwargs):
            raise AssertionError("Mistral-Transport darf nicht aufgerufen werden")

        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as references_temp, tempfile.TemporaryDirectory(
            dir=ALLOWED_INPUT_ROOT
        ) as temp:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            report = direct_reference_report(Path(references_temp), "2A")
            with patch("Tool.ki_bildanalyse.inspect_variant", return_value=report):
                with self.assertRaisesRegex(AnalysisError, "12 Referenzbilder"):
                    run_analysis(
                        input_dir, provider="mistral", evaluation_variant="2A",
                        allowed_input_root=ALLOWED_INPUT_ROOT,
                        mistral_opener=forbidden, mistral_key_resolver=forbidden,
                    )

    def test_variant_2a_qwen_request_contains_twelve_references_and_test_images(self) -> None:
        captured: list[dict[str, object]] = []
        raw = json.loads(
            (Path(__file__).resolve().parents[1] / "Tool/mock_response.json").read_text(encoding="utf-8")
        )

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({
                    "message": {"content": json.dumps(raw), "thinking": ""},
                    "done_reason": "stop", "model": "qwen3-vl:4b-instruct",
                }).encode("utf-8")

        def fake_opener(request, timeout):
            captured.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as references_temp, tempfile.TemporaryDirectory(
            dir=ALLOWED_INPUT_ROOT
        ) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            report = direct_reference_report(Path(references_temp), "2A")
            with patch("Tool.ki_bildanalyse.inspect_variant", return_value=report):
                run_analysis(
                    input_dir, provider="ollama", evaluation_variant="2A",
                    output_dir=Path(output), allowed_input_root=ALLOWED_INPUT_ROOT,
                    ollama_opener=fake_opener,
                )

        image_messages = [item for item in captured[0]["messages"] if "images" in item]
        self.assertEqual(len(image_messages), 13)
        self.assertEqual(sum("Referenzbild" in item["content"] for item in image_messages), 12)
        self.assertEqual(sum("Testbild" in item["content"] for item in image_messages), 1)

    def test_variant_2a_openai_request_contains_twelve_references_and_test_images(self) -> None:
        raw = {
            "status": "UNKLAR", "observations_short": ["Mock"],
            "visible_features": [], "class_suggestions": [],
            "uncertain_points": ["Nur Transporttest"],
        }
        captured: dict[str, object] = {}

        class FakeResponses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    output_text=json.dumps(raw), id="mock", model="gpt-6.1-sol",
                    usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as references_temp, tempfile.TemporaryDirectory(
            dir=ALLOWED_INPUT_ROOT
        ) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            report = direct_reference_report(Path(references_temp), "2A")
            with patch("Tool.ki_bildanalyse.inspect_variant", return_value=report):
                run_analysis(
                    input_dir, provider="openai", evaluation_variant="2A", live=True,
                    confirm_paid_request=True, client=SimpleNamespace(responses=FakeResponses()),
                    output_dir=Path(output), allowed_input_root=ALLOWED_INPUT_ROOT,
                )

        content = captured["input"][0]["content"]
        image_items = [item for item in content if item["type"] == "input_image"]
        labels = [item["text"] for item in content if item["type"] == "input_text"]
        self.assertEqual(len(image_items), 13)
        self.assertEqual(sum("Referenzbild" in text for text in labels), 12)
        self.assertEqual(sum(text.startswith("Unbekanntes Testbild") for text in labels), 1)
        self.assertEqual(
            captured["prompt_cache_options"],
            {"mode": "explicit", "ttl": "30m"},
        )
        breakpoint_indexes = [
            index for index, item in enumerate(content)
            if item.get("prompt_cache_breakpoint") == {"mode": "explicit"}
        ]
        self.assertEqual(len(breakpoint_indexes), 1)
        breakpoint_index = breakpoint_indexes[0]
        self.assertEqual(
            content[breakpoint_index]["text"],
            "Ab hier folgen ausschliesslich ausgewaehlte Bilder des unbekannten Testpruefkoerpers.",
        )
        reference_indexes = [
            index for index, item in enumerate(content)
            if item["type"] == "input_text" and "Referenzbild" in item["text"]
        ]
        test_indexes = [
            index for index, item in enumerate(content)
            if item["type"] == "input_text" and item["text"].startswith("Unbekanntes Testbild")
        ]
        self.assertTrue(reference_indexes)
        self.assertTrue(test_indexes)
        self.assertLess(max(reference_indexes), breakpoint_index)
        self.assertGreater(min(test_indexes), breakpoint_index)

    def test_variant_2a_blocks_same_specimen_before_model_call(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as references_temp, tempfile.TemporaryDirectory(
            dir=ALLOWED_INPUT_ROOT
        ) as temp:
            report = direct_reference_report(Path(references_temp), "2A")
            specimen = str(report["references"][0]["specimen"])
            input_dir = Path(temp) / specimen
            input_dir.mkdir()
            (input_dir / "test.png").write_bytes(TINY_PNG)
            with patch("Tool.ki_bildanalyse.inspect_variant", return_value=report):
                with self.assertRaisesRegex(AnalysisError, "gleicher Prüfkörper"):
                    run_analysis(
                        input_dir, provider="openai", evaluation_variant="2A",
                        allowed_input_root=ALLOWED_INPUT_ROOT,
                    )

    def test_mistral_uses_the_same_factual_prompt_and_content_order(self) -> None:
        references = select_references(self.catalog, 1)[:MAX_MISTRAL_IMAGES_PER_REQUEST]
        openai_content = build_request_content([], self.catalog, references)
        mistral_messages = build_mistral_messages([], self.catalog, references)
        mistral_content = mistral_messages[1]["content"]
        self.assertEqual(openai_content[0]["text"], mistral_content[0]["text"])
        self.assertEqual(len(openai_content), len(mistral_content))
        self.assertEqual(
            [item["type"] for item in mistral_content],
            ["text" if item["type"] == "input_text" else "image_url" for item in openai_content],
        )

    def test_mistral_rejects_more_than_eight_images_before_transport(self) -> None:
        references = select_references(self.catalog, 1)[:MAX_MISTRAL_IMAGES_PER_REQUEST]
        with self.assertRaisesRegex(AnalysisError, "maximal 8"):
            build_mistral_messages([Path("test.png")], self.catalog, references)

    def test_mistral_live_mode_needs_explicit_cost_confirmation(self) -> None:
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp:
            (Path(temp) / "test.png").write_bytes(TINY_PNG)
            with self.assertRaisesRegex(AnalysisError, "confirm-paid-request"):
                run_analysis(Path(temp), provider="mistral", live=True)

    def test_mistral_mock_uses_references_without_keychain_or_network(self) -> None:
        def forbidden(*args, **kwargs):
            raise AssertionError("Mock-Test darf weder Schlüsselbund noch Netzwerk verwenden")

        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            destination = run_analysis(
                input_dir,
                provider="mistral",
                output_dir=Path(output),
                mistral_key_resolver=forbidden,
                mistral_opener=forbidden,
            )
            document = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(document["execution"]["provider"], "mistral")
        self.assertEqual(document["execution"]["model"], "ministral-3b-2512")
        self.assertEqual(document["execution"]["mode"], "mock")
        self.assertFalse(document["execution"]["paid_api_request_performed"])
        self.assertFalse(document["execution"]["images_transmitted_to_mistral"])
        self.assertFalse(document["execution"]["images_transmitted_to_cloud"])
        self.assertEqual(document["classification_basis"]["confirmed_main_reference_count_used"], 7)
        self.assertEqual(document["execution"]["provider_metadata"]["cost"]["reported_amount"], 0)

    def test_mistral_live_payload_and_usage_are_recorded_with_fake_transport(self) -> None:
        captured = {}
        raw_result = {
            "status": "UNKLAR", "primary_code": "UNKLAR", "primary_evidence": [],
            "primary_confidence": 0.0, "secondary_features": [],
            "visible_observations": ["Mock-Transport."],
            "uncertainty_reasons": ["Kein echter API-Aufruf."], "missing_views": [],
        }
        response_body = {
            "id": "fake-request", "model": "ministral-3b-2512", "created": 1,
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw_result)}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
        }

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps(response_body).encode("utf-8")

        def fake_opener(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            destination = run_analysis(
                input_dir, provider="mistral", live=True, confirm_paid_request=True,
                output_dir=Path(output), mistral_opener=fake_opener,
                mistral_key_resolver=lambda: MistralAPIKeyResolution("fake-secret"),
            )
            document = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(captured["url"], "https://api.mistral.ai/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "ministral-3b-2512")
        self.assertEqual(captured["payload"]["response_format"]["type"], "json_schema")
        self.assertIn(
            "schema",
            captured["payload"]["response_format"]["json_schema"],
        )
        self.assertNotIn(
            "schema_definition",
            captured["payload"]["response_format"]["json_schema"],
        )
        image_count = sum(
            item["type"] == "image_url"
            for item in captured["payload"]["messages"][1]["content"]
        )
        self.assertEqual(image_count, 8)
        self.assertEqual(
            captured["payload"]["messages"][1]["content"][-2]["type"],
            "image_url",
        )
        self.assertEqual(document["classification_basis"]["confirmed_main_reference_count_used"], 7)
        self.assertEqual(document["execution"]["provider_metadata"]["usage"]["total_tokens"], 168)
        self.assertEqual(
            document["execution"]["provider_metadata"]["request_options"]["image_count"],
            8,
        )
        self.assertTrue(document["execution"]["paid_api_request_performed"])
        self.assertTrue(document["execution"]["images_transmitted_to_mistral"])
        self.assertEqual(
            document["model_output"]["original_response_text"],
            json.dumps(raw_result),
        )
        self.assertEqual(document["model_output"]["parsed_response"], raw_result)
        self.assertNotIn("fake-secret", json.dumps(document))

    def test_openai_raw_response_is_recorded_without_api_key_or_temperature(self) -> None:
        raw_result = {
            "status": "KLASSIFIZIERT",
            "observations_short": ["Astanschnitt sichtbar"],
            "visible_features": ["Kantenast"],
            "class_suggestions": [{
                "code": "KA", "confidence": "mittel",
                "evidence_status": "sicher_sichtbar",
            }],
            "uncertain_points": [],
        }
        original_response_text = json.dumps(raw_result, ensure_ascii=False)
        captured: dict[str, object] = {}

        class FakeResponses:
            def create(self, **kwargs: object) -> SimpleNamespace:
                captured.update(kwargs)
                return SimpleNamespace(
                    output_text=original_response_text,
                    id="fake-openai-response", model="gpt-6.1-sol",
                    usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

        fake_client = SimpleNamespace(responses=FakeResponses(), api_key="fake-secret")
        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, \
                tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            destination = run_analysis(
                input_dir, provider="openai", live=True, confirm_paid_request=True,
                output_dir=Path(output), client=fake_client,
            )
            document = json.loads(destination.read_text(encoding="utf-8"))

        self.assertNotIn("temperature", captured)
        self.assertIs(captured["store"], False)
        self.assertEqual(
            captured["prompt_cache_options"],
            {"mode": "explicit", "ttl": "30m"},
        )
        self.assertEqual(
            document["execution"]["provider_metadata"]["prompt_cache"],
            {
                "mode": "explicit",
                "ttl": "30m",
                "breakpoint": "after_rules_and_variant_references_before_test_images",
            },
        )
        self.assertEqual(
            document["model_output"]["original_response_text"],
            original_response_text,
        )
        self.assertEqual(document["model_output"]["parsed_response"], raw_result)
        self.assertNotIn("fake-secret", json.dumps(document))

    def test_mistral_fake_transport_processes_all_selected_images_in_batches(self) -> None:
        captured_batches = []
        response_body = {
            "id": "fake-request",
            "model": "ministral-3b-2512",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": json.dumps({
                    "status": "UNKLAR", "primary_code": "UNKLAR", "primary_evidence": [],
                    "primary_confidence": 0.0, "secondary_features": [],
                    "visible_observations": [], "uncertainty_reasons": ["Batch unsicher."],
                    "missing_views": [],
                })},
            }],
            "usage": {},
        }

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps(response_body).encode("utf-8")

        def fake_opener(request, timeout):
            payload = json.loads(request.data.decode("utf-8"))
            captured_batches.append(payload)
            return FakeResponse()

        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            names = [f"damage_{index}.jpg" for index in range(10)]
            for index, name in enumerate(names):
                Image.new("RGB", (24, 18), (index * 20, index * 11, index * 7)).save(input_dir / name)
            destination = run_analysis(
                input_dir,
                provider="mistral",
                live=True,
                confirm_paid_request=True,
                output_dir=Path(output),
                max_references_per_class=0,
                selected_image_names=names,
                mistral_opener=fake_opener,
                mistral_key_resolver=lambda: MistralAPIKeyResolution("fake-secret"),
            )
            document = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(len(captured_batches), 2)
        counts = [
            sum(item.get("type") == "image_url" for item in payload["messages"][1]["content"])
            for payload in captured_batches
        ]
        self.assertEqual(counts, [8, 2])
        self.assertEqual(document["image_selection"]["included"], names)
        self.assertEqual(document["execution"]["provider_metadata"]["multi_stage"], True)
        self.assertEqual(
            [name for batch in document["execution"]["provider_metadata"]["image_batches"]
             for name in batch["images"]],
            names,
        )

    def test_mistral_422_reports_validation_body_safely_without_retry(self) -> None:
        calls = 0
        error_body = {
            "object": "error",
            "message": {
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "response_format", "json_schema", "schema"],
                        "msg": "Field required",
                        "input": {
                            "schema_definition": {
                                "image_url": "data:image/jpeg;base64,SECRET_IMAGE_BYTES"
                            }
                        },
                    },
                    {
                        "type": "extra_forbidden",
                        "loc": ["body", "response_format", "json_schema", "schema_definition"],
                        "msg": "Extra inputs are not permitted",
                    },
                ]
            },
        }

        def failing_opener(request, timeout):
            nonlocal calls
            calls += 1
            raise HTTPError(
                request.full_url,
                422,
                "Unprocessable Entity",
                {"x-request-id": "req-local-regression"},
                io.BytesIO(json.dumps(error_body).encode("utf-8")),
            )

        with self.assertRaises(MistralRequestError) as raised:
            call_mistral(
                model="ministral-3b-2512",
                messages=[{"role": "user", "content": "JSON"}],
                catalog=self.catalog,
                api_key="SECRET_API_KEY",
                opener=failing_opener,
            )

        message = str(raised.exception)
        self.assertEqual(calls, 1)
        self.assertIn("HTTP-Status=422", message)
        self.assertIn("req-local-regression", message)
        self.assertIn("Field required", message)
        self.assertIn("schema_definition", message)
        self.assertIn("[geschwärzt]", message)
        self.assertNotIn("SECRET_API_KEY", message)
        self.assertNotIn("SECRET_IMAGE_BYTES", message)

    def test_ollama_json_parser_accepts_markdown_wrapper(self) -> None:
        parsed = _parse_ollama_json('Antwort:\n```json\n{"status":"UNKLAR"}\n```')
        self.assertEqual(parsed, {"status": "UNKLAR"})

    def test_ollama_json_parser_accepts_local_control_characters(self) -> None:
        parsed = _parse_ollama_json('{"visible_observations":["Zeile 1\nZeile 2"]}')
        self.assertEqual(parsed["visible_observations"], ["Zeile 1\nZeile 2"])

    def test_ollama_thinking_only_answer_reports_exhausted_output_budget(self) -> None:
        response_body = {
            "message": {"content": "", "thinking": "interne Analyse"},
            "done_reason": "length",
            "prompt_eval_count": 12000,
            "eval_count": 4096,
        }

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps(response_body).encode("utf-8")

        with self.assertRaises(OllamaOutputError) as raised:
            call_ollama(
                model="qwen3-vl:4b-instruct",
                messages=[{"role": "user", "content": "JSON"}],
                catalog=self.catalog,
                opener=lambda request, timeout: FakeResponse(),
            )
        message = str(raised.exception)
        self.assertIn("ausschließlich internes Thinking", message)
        self.assertIn("Thinking war angefordert: nein", message)
        self.assertIn("Thinking-Zeichen: 15", message)
        self.assertIn("eval_count: 4096", message)
        self.assertIn("done_reason: length", message)
        self.assertEqual(raised.exception.failure_type, "technical_model_output")
        self.assertEqual(raised.exception.diagnostics["reason"], "thinking_only_output_limit")

    def test_ollama_empty_answer_without_thinking_is_a_technical_failure(self) -> None:
        response_body = {"message": {"content": "", "thinking": ""}, "done_reason": "stop"}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps(response_body).encode("utf-8")

        with self.assertRaises(OllamaOutputError) as raised:
            call_ollama(
                model="qwen3-vl:4b-instruct",
                messages=[{"role": "user", "content": "JSON"}],
                catalog=self.catalog,
                opener=lambda request, timeout: FakeResponse(),
            )
        self.assertIn("keinen Text", str(raised.exception))
        self.assertEqual(raised.exception.diagnostics["reason"], "invalid_content")

    def test_ollama_invalid_json_and_plain_unclear_are_not_silent_fallbacks(self) -> None:
        for content in ("{ungueltig", "UNKLAR"):
            response_body = {"message": {"content": content, "thinking": ""}, "done_reason": "stop"}

            class FakeResponse:
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def read(self): return json.dumps(response_body).encode("utf-8")

            with self.subTest(content=content), self.assertRaises(OllamaOutputError) as raised:
                call_ollama(
                    model="qwen3-vl:4b-instruct",
                    messages=[{"role": "user", "content": "JSON"}],
                    catalog=self.catalog,
                    opener=lambda request, timeout: FakeResponse(),
                )
            self.assertEqual(raised.exception.diagnostics["reason"], "invalid_content")
            self.assertEqual(raised.exception.diagnostics["original_response_text"], content)

    def test_ollama_rejects_content_when_done_reason_is_length(self) -> None:
        response_body = {
            "message": {"content": '{"status":"UNKLAR"}', "thinking": ""},
            "done_reason": "length",
            "eval_count": 4096,
        }

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps(response_body).encode("utf-8")

        with self.assertRaises(OllamaOutputError) as raised:
            call_ollama(
                model="qwen3-vl:4b-instruct",
                messages=[{"role": "user", "content": "JSON"}],
                catalog=self.catalog,
                opener=lambda request, timeout: FakeResponse(),
            )
        self.assertIn("Ausgabelimit", str(raised.exception))
        self.assertEqual(raised.exception.diagnostics["reason"], "output_limit")

    def test_ollama_timeout_has_actionable_message(self) -> None:
        with self.assertRaisesRegex(
            AnalysisError, "Zeitüberschreitung.*900 Sekunden.*Modellauslastung"
        ):
            call_ollama(
                model="qwen3-vl:4b-instruct",
                messages=[{"role": "user", "content": "JSON"}],
                catalog=self.catalog,
                opener=lambda request, timeout: (_ for _ in ()).throw(
                    URLError(TimeoutError("timed out"))
                ),
            )

    def test_ollama_pipeline_is_local_and_uses_active_class_references(self) -> None:
        captured = {}
        response_body = {
            "message": {
                "content": json.dumps({
                    "status": "UNKLAR",
                    "primary_code": "UNKLAR",
                    "primary_evidence": [],
                    "primary_confidence": 0.0,
                    "secondary_features": [],
                    "visible_observations": ["Lokaler Test."],
                    "uncertainty_reasons": ["Testantwort."],
                    "missing_views": [],
                })
            },
            "total_duration": 123,
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(response_body).encode("utf-8")

        def fake_opener(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        class ForbiddenOpenAIClient:
            @property
            def responses(self):
                raise AssertionError("OpenAI darf im Ollama-Zweig nicht verwendet werden")

        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            destination = run_analysis(
                input_dir,
                provider="ollama",
                model="qwen3-vl:4b-instruct",
                output_dir=Path(output),
                client=ForbiddenOpenAIClient(),
                ollama_opener=fake_opener,
            )
            document = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(captured["url"], "http://localhost:11434/api/chat")
        self.assertEqual(captured["payload"]["model"], "qwen3-vl:4b-instruct")
        self.assertIs(captured["payload"]["think"], False)
        self.assertEqual(captured["payload"]["options"]["num_predict"], 4096)
        self.assertEqual(captured["payload"]["options"]["num_ctx"], 24576)
        image_count = sum(len(message.get("images", [])) for message in captured["payload"]["messages"])
        self.assertEqual(image_count, 11)  # zehn aktive Referenzen plus ein Testbild
        first_image = next(message["images"][0] for message in captured["payload"]["messages"] if message.get("images"))
        with Image.open(io.BytesIO(base64.b64decode(first_image))) as image:
            self.assertLessEqual(max(image.size), 768)
        self.assertEqual(document["execution"]["provider"], "ollama")
        self.assertEqual(document["execution"]["mode"], "local")
        self.assertFalse(document["execution"]["paid_api_request_performed"])
        self.assertFalse(document["execution"]["images_transmitted_to_openai"])
        self.assertEqual(document["classification_basis"]["confirmed_main_reference_count_used"], 10)
        self.assertEqual(document["execution"]["provider_metadata"]["request_options"]["num_ctx"], 24576)
        self.assertEqual(document["execution"]["provider_metadata"]["request_options"]["image_count"], 11)
        self.assertFalse(document["execution"]["provider_metadata"]["think"])
        self.assertFalse(document["execution"]["provider_metadata"]["stream"])
        self.assertEqual(document["execution"]["settings"]["ollama_num_ctx"], 24576)
        self.assertEqual(document["execution"]["settings"]["ollama_num_predict"], 4096)
        self.assertEqual(document["execution"]["settings"]["ollama_timeout_seconds"], 900)
        self.assertFalse(document["execution"]["settings"]["ollama_think"])
        self.assertFalse(document["execution"]["settings"]["ollama_stream"])
        self.assertEqual(document["execution"]["settings"]["ollama_temperature"], 0)
        self.assertEqual(document["model_output"]["validation_status"], "valid")
        self.assertEqual(
            json.loads(document["model_output"]["original_response_text"]),
            json.loads(response_body["message"]["content"]),
        )
        self.assertEqual(document["model_output"]["parsed_response"], json.loads(response_body["message"]["content"]))
        self.assertEqual(len(document["classification_basis"]["references"]), 10)
        self.assertEqual(document["inputs"][0]["filename"], "test.png")

    def test_ollama_context_cli_parameter_is_flexible(self) -> None:
        args = _parser().parse_args([
            "input", "--provider", "ollama", "--ollama-num-ctx", "24576",
            "--ollama-num-predict", "8192",
        ])
        self.assertEqual(DEFAULT_OLLAMA_NUM_CTX, 24576)
        self.assertEqual(DEFAULT_OLLAMA_NUM_PREDICT, 4096)
        self.assertEqual(args.ollama_num_ctx, 24576)
        self.assertEqual(args.ollama_num_predict, 8192)

    def test_repeated_runs_are_independent_and_log_reproducible_inputs(self) -> None:
        sentinels = ["PREVIOUS_MODEL_RESULT_MUST_NOT_LEAK", "MANUAL_CORRECTION_MUST_NOT_LEAK"]
        captured_payloads = []
        response_body = {
            "message": {"content": json.dumps({
                "status": "UNKLAR", "primary_code": "UNKLAR", "primary_evidence": [],
                "primary_confidence": 0.0, "secondary_features": [],
                "visible_observations": [], "uncertainty_reasons": ["Mock"],
                "missing_views": [],
            })},
            "prompt_eval_count": 10, "eval_count": 5,
        }

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps(response_body).encode("utf-8")

        def fake_opener(request, timeout):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        ALLOWED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ALLOWED_INPUT_ROOT) as temp, \
                tempfile.TemporaryDirectory() as output:
            input_dir = Path(temp)
            output_dir = Path(output)
            (input_dir / "test.png").write_bytes(TINY_PNG)
            (output_dir / "previous.json").write_text(json.dumps({"result": sentinels[0]}))
            correction_dir = output_dir / "Korrekturen"
            correction_dir.mkdir()
            (correction_dir / "manual.json").write_text(json.dumps({"note": sentinels[1]}))
            paths = [
                run_analysis(
                    input_dir, provider="ollama", model="qwen3-vl:4b-instruct",
                    output_dir=output_dir, ollama_opener=fake_opener,
                    run_id=run_id, project_id="Independent-Test",
                )
                for run_id in ("run-a", "run-b")
            ]
            documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

        serialized_requests = json.dumps(captured_payloads, ensure_ascii=False)
        self.assertTrue(all(sentinel not in serialized_requests for sentinel in sentinels))
        self.assertEqual([document["run_id"] for document in documents], ["run-a", "run-b"])
        self.assertEqual(documents[0]["classification_basis"]["prompt_hash_sha256"],
                         documents[1]["classification_basis"]["prompt_hash_sha256"])
        self.assertEqual(len(documents[0]["classification_basis"]["references"]), 10)
        self.assertTrue(all(item["sha256"] for item in documents[0]["classification_basis"]["references"]))
        self.assertEqual(documents[0]["project_id"], "Independent-Test")

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
