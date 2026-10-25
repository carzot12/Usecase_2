from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

from PIL import Image

from Tool.ki_bildanalyse import (
    ALLOWED_INPUT_ROOT,
    AnalysisError,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_NUM_PREDICT,
    MAX_MISTRAL_IMAGES_PER_REQUEST,
    _parser,
    build_mistral_messages,
    build_ollama_messages,
    build_request_content,
    _parse_ollama_json,
    collect_test_images,
    call_mistral,
    call_ollama,
    load_catalog,
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
        self.assertEqual(document["classification_basis"]["allowed_class_count"], 29)
        self.assertEqual(document["classification_basis"]["confirmed_main_reference_count_used"], 0)
        self.assertEqual(document["execution"]["provider"], "mock")
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

    def test_ollama_empty_answer_reports_exhausted_thinking_budget(self) -> None:
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

        with self.assertRaises(AnalysisError) as raised:
            call_ollama(
                model="qwen3-vl:4b",
                messages=[{"role": "user", "content": "JSON"}],
                catalog=self.catalog,
                opener=lambda request, timeout: FakeResponse(),
            )
        message = str(raised.exception)
        self.assertIn("Ollama-Antwort enthaelt keinen Text", message)
        self.assertIn("Thinking-Zeichen: 15", message)
        self.assertIn("eval_count: 4096", message)
        self.assertIn("done_reason: length", message)

    def test_ollama_timeout_has_actionable_message(self) -> None:
        with self.assertRaisesRegex(
            AnalysisError, "Zeitüberschreitung.*900 Sekunden.*Modellauslastung"
        ):
            call_ollama(
                model="qwen3-vl:4b",
                messages=[{"role": "user", "content": "JSON"}],
                catalog=self.catalog,
                opener=lambda request, timeout: (_ for _ in ()).throw(
                    URLError(TimeoutError("timed out"))
                ),
            )

    def test_ollama_pipeline_is_local_and_uses_eleven_references(self) -> None:
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
                model="llava:latest",
                output_dir=Path(output),
                client=ForbiddenOpenAIClient(),
                ollama_opener=fake_opener,
            )
            document = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(captured["url"], "http://localhost:11434/api/chat")
        self.assertEqual(captured["payload"]["model"], "llava:latest")
        self.assertIs(captured["payload"]["think"], False)
        self.assertEqual(captured["payload"]["options"]["num_predict"], 4096)
        self.assertEqual(captured["payload"]["options"]["num_ctx"], 24576)
        image_count = sum(len(message.get("images", [])) for message in captured["payload"]["messages"])
        self.assertEqual(image_count, 12)  # elf Referenzen plus ein Testbild
        first_image = next(message["images"][0] for message in captured["payload"]["messages"] if message.get("images"))
        with Image.open(io.BytesIO(base64.b64decode(first_image))) as image:
            self.assertLessEqual(max(image.size), 768)
        self.assertEqual(document["execution"]["provider"], "ollama")
        self.assertEqual(document["execution"]["mode"], "local")
        self.assertFalse(document["execution"]["paid_api_request_performed"])
        self.assertFalse(document["execution"]["images_transmitted_to_openai"])
        self.assertEqual(document["classification_basis"]["confirmed_main_reference_count_used"], 11)
        self.assertEqual(document["execution"]["provider_metadata"]["request_options"]["num_ctx"], 24576)
        self.assertFalse(document["execution"]["provider_metadata"]["think"])
        self.assertFalse(document["execution"]["provider_metadata"]["stream"])
        self.assertEqual(document["execution"]["settings"]["ollama_num_ctx"], 24576)
        self.assertEqual(document["execution"]["settings"]["ollama_num_predict"], 4096)
        self.assertEqual(document["execution"]["settings"]["ollama_timeout_seconds"], 900)
        self.assertEqual(len(document["classification_basis"]["references"]), 11)
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
                    input_dir, provider="ollama", model="qwen3-vl:4b",
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
        self.assertEqual(len(documents[0]["classification_basis"]["references"]), 11)
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
