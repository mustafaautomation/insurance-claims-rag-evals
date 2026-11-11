"""Offline metrics, the threshold gate, and the committed report."""

import json

import pytest

from harborline.config import MetricThreshold
from harborline.eval.dataset import load_examples
from harborline.eval.gate import IllustrativeScoresError, apply_gate, require_real_judge_file
from harborline.eval.metrics import non_llm_context_metrics, question_scores
from harborline.eval.report import VariantReport, render_markdown
from harborline.eval.runner import fresh_rows, gate_variant, score_rows
from harborline.eval.snapshots import read_snapshot
from harborline.models import VARIANTS, Row
from harborline.paths import judge_schema_path, repo_root
from harborline.pipeline import Index


def _toy(retrieved: list[str], reference: list[str]) -> Row:
    return Row(
        id="toy",
        variant="improved",
        question="q",
        reference="ref",
        reference_contexts=tuple(reference),
        reference_doc_ids=("doc",),
        expected_behavior="answer",
        required_phrases=(),
        tags=(),
        response="unused",
        retrieved_contexts=tuple(retrieved),
        retrieved_doc_ids=("doc",),
        retrieval_scores=(1.0,),
    )


def test_nonllm_recall_and_precision_match_exact_text() -> None:
    exact = non_llm_context_metrics([_toy(["alpha page"], ["alpha page"])])
    assert exact["non_llm_context_recall"] == pytest.approx(1.0)
    assert exact["non_llm_context_precision"] == pytest.approx(1.0)
    partial = non_llm_context_metrics([_toy(["alpha page"], ["alpha page", "missing page"])])
    assert partial["non_llm_context_recall"] == 0.5
    wrong = non_llm_context_metrics([_toy(["unrelated words"], ["alpha page"])])
    assert wrong["non_llm_context_recall"] == 0.0


def test_gate_fails_when_a_metric_is_below_its_minimum() -> None:
    thresholds = (MetricThreshold(name="hit_rate", minimum=0.9, modes=("offline",)),)
    gate = apply_gate({"hit_rate": 0.2}, thresholds, variant="bad_retriever", mode="offline")
    assert not gate.passed
    assert gate.failures == ("hit_rate",)
    missing = apply_gate({}, thresholds, variant="improved", mode="offline")
    assert not missing.passed


def test_illustrative_judge_file_is_not_a_measurement() -> None:
    payload = json.loads(judge_schema_path().read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["real_run"] is False
    assert payload["metrics"]["faithfulness"] is None
    try:
        require_real_judge_file(payload)
    except IllustrativeScoresError:
        return
    raise AssertionError("illustrative scores must be rejected")


def test_improved_offline_scores_clear_the_gate() -> None:
    gate, scores, _rows = gate_variant("improved")
    assert gate.passed
    for name in (
        "refusal_correctness",
        "hit_rate",
        "mrr",
        "string_presence",
        "non_llm_context_recall",
        "non_llm_context_precision",
    ):
        assert scores[name] == pytest.approx(1.0)


def test_bad_retriever_and_no_guardrail_fail_the_gate() -> None:
    index = Index()
    bad_gate, bad_scores, _bad_rows = gate_variant(
        "bad_retriever", rows=fresh_rows("bad_retriever", index)
    )
    plain_gate, plain_scores, _plain_rows = gate_variant(
        "no_guardrail",
        rows=fresh_rows("no_guardrail", index),
    )
    assert not bad_gate.passed
    assert bad_scores["hit_rate"] == 0.0
    assert bad_scores["non_llm_context_recall"] == 0.0
    assert "hit_rate" in bad_gate.failures
    assert not plain_gate.passed
    assert plain_scores["refusal_correctness"] < 0.95
    assert "refusal_correctness" in plain_gate.failures


def test_snapshots_match_a_fresh_pipeline_run() -> None:
    index = Index()
    examples = load_examples(index.chunks)
    assert len(examples) == 48
    for variant in VARIANTS:
        fresh = fresh_rows(variant, index)
        saved = read_snapshot(variant)
        assert [row.id for row in fresh] == [row.id for row in saved]
        for new, old in zip(fresh, saved, strict=True):
            assert new.response == old.response
            assert new.retrieved_doc_ids == old.retrieved_doc_ids
            assert new.reference_contexts == old.reference_contexts


def test_committed_report_matches_the_scored_snapshots() -> None:
    reports: dict[str, VariantReport] = {}
    for variant in VARIANTS:
        gate, _scores, rows = gate_variant(variant)
        reports[variant] = VariantReport(gate=gate, questions=tuple(question_scores(rows)))
    text = render_markdown(reports, llm_judged=False)
    committed = (repo_root() / "docs" / "eval_report.md").read_text(encoding="utf-8")
    assert committed == text
    assert "1.000" in text
    assert "LLM-judged RAGAS metrics were not run" in text


def test_score_rows_uses_the_recorded_answers() -> None:
    rows = read_snapshot("improved")
    scores = score_rows(rows)
    assert scores["string_presence"] == 1.0
