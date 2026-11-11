"""Run a variant and score it. Offline scoring uses recorded rows or a fresh fake run."""

import asyncio
import json

from harborline.config import MetricThreshold, load_pipeline_config, load_thresholds
from harborline.eval.dataset import load_examples
from harborline.eval.gate import GateResult, apply_gate
from harborline.eval.live import judge_rows, require_openai_key
from harborline.eval.metrics import local_metrics, non_llm_context_metrics
from harborline.eval.snapshots import read_snapshot, write_snapshot
from harborline.models import VARIANTS, Row, Variant
from harborline.paths import data_dir
from harborline.pipeline import Index, run_variant
from harborline.providers import OpenAIProvider


def fresh_rows(variant: Variant, index: Index | None = None) -> list[Row]:
    built = index or Index()
    examples = load_examples(built.chunks)
    return run_variant(examples, variant, index=built)


def score_rows(rows: list[Row]) -> dict[str, float]:
    scores = local_metrics(rows)
    scores.update(non_llm_context_metrics(rows))
    return scores


def gate_variant(
    variant: Variant,
    *,
    mode: str = "offline",
    rows: list[Row] | None = None,
    thresholds: tuple[MetricThreshold, ...] | None = None,
) -> tuple[GateResult, dict[str, float], list[Row]]:
    selected = rows if rows is not None else read_snapshot(variant)
    scores = score_rows(selected)
    gate = apply_gate(
        scores,
        thresholds or load_thresholds(),
        variant=variant,
        mode=mode,
    )
    return gate, scores, selected


def record_variant(variant: Variant, index: Index | None = None) -> list[Row]:
    rows = fresh_rows(variant, index=index)
    write_snapshot(variant, rows)
    return rows


def record_all() -> None:
    index = Index()
    for variant in VARIANTS:
        record_variant(variant, index=index)


def score_live(
    variant: Variant = "improved",
    *,
    record: bool = False,
) -> tuple[GateResult, dict[str, float], list[Row]]:
    """Generate with OpenAI, then score offline metrics and RAGAS judge metrics."""

    require_openai_key()
    config = load_pipeline_config()
    index = Index()
    examples = load_examples(index.chunks)
    rows = run_variant(examples, variant, index=index, provider=OpenAIProvider(config.chat_model))
    scores = score_rows(rows)
    judged = asyncio.run(judge_rows(rows, config))
    scores.update(judged)
    if record:
        _write_live_judge(scores, rows)
    gate = apply_gate(scores, load_thresholds(), variant=variant, mode="live")
    return gate, scores, rows


def _write_live_judge(scores: dict[str, float], rows: list[Row]) -> None:
    target = data_dir() / "snapshots" / "live"
    target.mkdir(parents=True, exist_ok=True)
    write_snapshot("live-responses", rows, path=target / "responses.jsonl")
    payload = {
        "label": "live_run",
        "real_run": True,
        "source": "openai",
        "metrics": {name: scores[name] for name in sorted(scores)},
    }
    (target / "judge_scores.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
