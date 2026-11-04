from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from Tool.evaluation import (
    GROUND_TRUTH_PATH,
    VARIANTS,
    build_ground_truth_document,
    build_plan,
    build_feature_extraction_content,
    build_rule_classification_content,
    build_variant_content,
    compute_metrics,
    classify_outcome,
    classify_secure_outcome,
    execute_job,
    load_confirmed_references,
    load_ground_truth,
    run_evaluation,
    select_holdout_references,
)
from Tool.ki_bildanalyse import load_catalog


class EvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.cases = load_ground_truth()
        cls.references = load_confirmed_references()

    def test_ground_truth_contains_only_confirmed_main_references(self) -> None:
        document = build_ground_truth_document()
        self.assertEqual(document["case_count"], 26)
        self.assertEqual(document["covered_class_count"], 11)
        self.assertEqual(len(self.cases), 26)
        self.assertEqual(len({case.specimen_id for case in self.cases}), 26)
        self.assertTrue(all(len(case.image_paths) == 1 for case in self.cases))
        self.assertNotIn("KI", json.dumps(document["ground_truth_policy"]["accepted"]))

    def test_holdout_excludes_same_specimen_and_identical_image(self) -> None:
        for case in self.cases:
            selected = select_holdout_references(case, self.references, 1)
            self.assertTrue(all(ref.specimen_id != case.specimen_id for ref in selected))
            self.assertTrue(all(ref.sha256 not in case.image_sha256 for ref in selected))
            self.assertLessEqual(len(selected), 11)

    def test_plan_is_balanced_across_all_four_variants(self) -> None:
        jobs = build_plan(
            self.cases, self.references, provider="ollama", model="test-model",
            max_references_per_class=1,
        )
        self.assertEqual(len(jobs), 26 * 4)
        for case in self.cases:
            variants = {job["variant"] for job in jobs if job["specimen_id"] == case.specimen_id}
            self.assertEqual(variants, set(VARIANTS))
        self.assertFalse(any(job["same_specimen_reference_present"] for job in jobs))
        self.assertEqual(sum(job["model_request_count"] for job in jobs), 130)

    def test_variant_inputs_are_separated(self) -> None:
        case = self.cases[0]
        references = select_holdout_references(case, self.references, 1)
        contents = {
            variant: build_variant_content(case, variant, self.catalog, references if variant == "C" else ())
            for variant in VARIANTS
        }
        image_counts = {
            variant: sum(item["type"] == "input_image" for item in content)
            for variant, content in contents.items()
        }
        self.assertEqual(image_counts["A"], 1)
        self.assertEqual(image_counts["B"], 1)
        self.assertEqual(image_counts["D"], 1)
        self.assertEqual(image_counts["C"], 1 + len(references))
        first_payloads = {variant: json.loads(content[0]["text"]) for variant, content in contents.items()}
        self.assertNotIn("classes", first_payloads["A"])
        self.assertIn("classes", first_payloads["B"])
        self.assertFalse(any("validated_classification_rule" in item for item in first_payloads["B"]["classes"]))
        self.assertNotIn("classes_and_rules", first_payloads["D"])
        stage_1 = json.loads(build_feature_extraction_content(case, self.catalog)[0]["text"])
        self.assertNotIn("classes", stage_1)
        stage_2 = json.loads(build_rule_classification_content(["sichtbarer Ast"], self.catalog)[0]["text"])
        self.assertIn("classes_and_rules", stage_2)
        self.assertEqual(stage_2["stage_1_visible_observations"], ["sichtbarer Ast"])

    def test_dry_run_creates_plan_without_results_or_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            args = Namespace(
                ground_truth=GROUND_TRUTH_PATH, output_dir=Path(output),
                provider="ollama", model="not-installed-and-never-called",
                max_references_per_class=1, ollama_num_ctx=16384,
                ollama_max_image_edge=768, execute=False, live=False,
                confirm_paid_request=False,
            )
            destination = run_evaluation(args)
            plan = json.loads((destination / "run_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan["jobs"]), 100)
            self.assertFalse((destination / "evaluation_results.json").exists())
            self.assertFalse(plan["run_config"]["paid_api_request_performed"])
            self.assertEqual(plan["run_config"]["planned_model_request_count"], 125)
            self.assertEqual(plan["run_config"]["case_count"], 25)
            self.assertEqual(plan["run_config"]["full_ground_truth_case_count"], 26)
            self.assertEqual(
                plan["run_config"]["active_class_codes"],
                ["GSF", "LSF", "KA", "SA", "sp HB", "Spl HB", "RiSch", "LR", "Bu", "AA", "SSA", "KB"],
            )
            self.assertEqual(plan["run_config"]["ground_truth_classification_rules_version"], "1.0")
            self.assertFalse(plan["run_config"]["ground_truth_rules_version_matches_active"])

    def test_cloud_execution_stays_locked_without_both_live_flags(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            args = Namespace(
                ground_truth=GROUND_TRUTH_PATH, output_dir=Path(output),
                provider="openai", model="never-called",
                max_references_per_class=1, ollama_num_ctx=16384,
                ollama_max_image_edge=768, execute=True, live=False,
                confirm_paid_request=False,
            )
            with self.assertRaisesRegex(Exception, "Cloud-Evaluation gesperrt"):
                run_evaluation(args)

    def test_variant_d_uses_two_separate_model_calls(self) -> None:
        stage_1 = {
            "status": "UNKLAR", "primary_code": "UNKLAR", "primary_evidence": [],
            "primary_confidence": 0.0, "secondary_features": [],
            "visible_observations": ["Sichtbarer Ast an einer Kante."],
            "uncertainty_reasons": [], "missing_views": [],
        }
        stage_2 = {
            "status": "KLASSIFIZIERT", "primary_code": "KA",
            "primary_evidence": ["Sichtbarer Ast an einer Kante."],
            "primary_confidence": 0.8, "secondary_features": [],
            "visible_observations": [], "uncertainty_reasons": [], "missing_views": [],
        }
        replies = [
            (stage_1, {"prompt_eval_count": 10, "eval_count": 5}),
            (stage_2, {"prompt_eval_count": 20, "eval_count": 7}),
        ]
        with patch("Tool.evaluation.call_ollama", side_effect=replies) as mocked:
            record = execute_job(
                case=self.cases[0], variant="D", catalog=self.catalog, references=(),
                provider="ollama", model="fake", ollama_num_ctx=16384,
                ollama_max_image_edge=768,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(record["visible_observations"], stage_1["visible_observations"])
        self.assertEqual(record["total_tokens"], 42)
        self.assertEqual(record["active_class_set_version"], "klassifikationsregeln/1.4")
        self.assertEqual(len(record["active_class_descriptions_sha256"]), 64)
        self.assertEqual(record["outcome"], "correct" if self.cases[0].expected_code == "KA" else "wrong_classification")


class EvaluationMetricTests(unittest.TestCase):
    def test_main_evaluation_includes_suspected_and_secure_view_reports_exclusion(self) -> None:
        suspected = {
            "status": "KLASSIFIZIERT",
            "class_suggestions": [{
                "code": "KA", "evidence_status": "vermutet", "suggestion_role": "primary",
            }],
        }
        self.assertEqual(classify_outcome("KA", suspected), "correct")
        self.assertEqual(
            classify_secure_outcome("KA", suspected), "excluded_no_secure_evidence"
        )
        records = [{
            "expected_primary_code": "KA",
            "evaluation_predicted_code": "KA",
            "secure_evaluation_predicted_code": "UNKLAR",
            "outcome": "correct",
            "secure_outcome": "excluded_no_secure_evidence",
        }]
        main = compute_metrics(records)
        secure = compute_metrics(records, evidence_scope="secure_visible")
        self.assertEqual(main["coverage_rate"], 1.0)
        self.assertEqual(main["accuracy"], 1.0)
        self.assertEqual(secure["coverage_rate"], 0.0)
        self.assertEqual(secure["excluded_record_count"], 1)
        self.assertEqual(secure["effective_correct_rate_all_cases"], 0.0)

    def test_kb_is_reported_but_not_silently_scored(self) -> None:
        kb_only = {"status": "KLASSIFIZIERT", "detected_features": [{"code": "KB"}]}
        with_expected = {"status": "KLASSIFIZIERT", "detected_features": [{"code": "KA"}, {"code": "KB"}]}
        self.assertEqual(classify_outcome("KA", kb_only), "unscored_auxiliary_only")
        self.assertEqual(classify_outcome("KA", with_expected), "correct")
        metrics = compute_metrics([
            {"expected_primary_code": "KA", "predicted_primary_code": "KB",
             "evaluation_predicted_code": "UNSCORED", "outcome": "unscored_auxiliary_only"},
            {"expected_primary_code": "KA", "predicted_primary_code": "KA",
             "evaluation_predicted_code": "KA", "outcome": "correct"},
        ])
        self.assertEqual(metrics["record_count"], 2)
        self.assertEqual(metrics["scored_record_count"], 1)
        self.assertEqual(metrics["unscored_record_count"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_metrics_and_error_categories(self) -> None:
        records = [
            {"expected_primary_code": "KA", "predicted_primary_code": "KA", "outcome": "correct", "variant": "A"},
            {"expected_primary_code": "KA", "predicted_primary_code": "SA", "outcome": "wrong_classification", "variant": "A"},
            {"expected_primary_code": "SA", "predicted_primary_code": "UNKLAR", "outcome": "unclassifiable", "variant": "A"},
            {"expected_primary_code": "SA", "predicted_primary_code": "ERROR", "outcome": "invalid_json", "variant": "A"},
            {"expected_primary_code": "SA", "predicted_primary_code": "ERROR", "outcome": "api_error", "variant": "A"},
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["accuracy"], 0.2)
        self.assertEqual(metrics["outcome_counts"]["invalid_json"], 1)
        self.assertEqual(metrics["outcome_counts"]["api_error"], 1)
        self.assertEqual(metrics["per_class"]["KA"]["support"], 2)
        self.assertEqual(metrics["confusion_matrix"]["values"]["SA"]["ERROR"], 2)


if __name__ == "__main__":
    unittest.main()
