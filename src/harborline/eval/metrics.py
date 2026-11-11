"""Offline metrics: guardrail behavior, retrieval rank, and RAGAS non-LLM scores."""

import asyncio
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from harborline.models import Row
from harborline.prompts import (
    AFFIRM,
    CANNOT_CONFIRM,
    CANNOT_OVERTURN,
    OVERTURN,
    PHONE,
    UNKNOWN_SENTENCE,
)


@dataclass(frozen=True)
class QuestionScore:
    id: str
    question: str
    expected_behavior: str
    behavior_ok: bool
    string_presence: float | None
    hit: bool | None
    reciprocal_rank: float | None
    retrieved_doc_ids: tuple[str, ...]
    response: str


def behavior_ok(row: Row) -> bool:
    text = row.response
    lowered = text.lower()
    if row.expected_behavior == "refuse_approval":
        return CANNOT_CONFIRM in text and AFFIRM not in text
    if row.expected_behavior == "handoff":
        return CANNOT_OVERTURN in text and PHONE in text and OVERTURN not in text
    if row.expected_behavior == "unknown":
        return UNKNOWN_SENTENCE in text
    return not ("i don't know" in lowered or AFFIRM in text or OVERTURN in text)


def retrieval_hit(row: Row) -> bool | None:
    if not row.reference_doc_ids:
        return None
    return any(doc_id in row.reference_doc_ids for doc_id in row.retrieved_doc_ids)


def reciprocal_rank(row: Row) -> float | None:
    if not row.reference_doc_ids:
        return None
    for rank, doc_id in enumerate(row.retrieved_doc_ids, start=1):
        if doc_id in row.reference_doc_ids:
            return 1.0 / rank
    return 0.0


def question_scores(rows: Sequence[Row]) -> list[QuestionScore]:
    presence = _string_presence_by_id(rows)
    scored: list[QuestionScore] = []
    for row in rows:
        scored.append(
            QuestionScore(
                id=row.id,
                question=row.question,
                expected_behavior=row.expected_behavior,
                behavior_ok=behavior_ok(row),
                string_presence=presence.get(row.id),
                hit=retrieval_hit(row),
                reciprocal_rank=reciprocal_rank(row),
                retrieved_doc_ids=row.retrieved_doc_ids,
                response=row.response,
            )
        )
    return scored


def local_metrics(rows: Sequence[Row]) -> dict[str, float]:
    details = question_scores(rows)
    return {
        "refusal_correctness": _mean([1.0 if item.behavior_ok else 0.0 for item in details]),
        "hit_rate": _mean([1.0 if item.hit else 0.0 for item in details if item.hit is not None]),
        "mrr": _mean(
            [item.reciprocal_rank for item in details if item.reciprocal_rank is not None]
        ),
        "string_presence": _mean(
            [item.string_presence for item in details if item.string_presence is not None]
        ),
    }


def non_llm_context_metrics(rows: Sequence[Row]) -> dict[str, float]:
    relevant = [row for row in rows if row.reference_contexts]
    if not relevant:
        raise ValueError("no rows with reference contexts")
    return asyncio.run(_non_llm_context_metrics(relevant))


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric")
    return sum(values) / len(values)


def _string_presence_by_id(rows: Sequence[Row]) -> dict[str, float]:
    jobs: list[tuple[str, str, str]] = []
    for row in rows:
        for phrase in row.required_phrases:
            jobs.append((row.id, phrase, row.response))
    if not jobs:
        return {}
    return asyncio.run(_string_presence(jobs))


async def _string_presence(jobs: list[tuple[str, str, str]]) -> dict[str, float]:
    from ragas.metrics.collections import StringPresence

    metric = StringPresence()
    totals: dict[str, list[float]] = {}
    for example_id, phrase, response in jobs:
        result = await metric.ascore(reference=phrase, response=response)
        totals.setdefault(example_id, []).append(float(result.value))
    return {example_id: sum(values) / len(values) for example_id, values in totals.items()}


async def _non_llm_context_metrics(rows: Sequence[Row]) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Importing NonLLMContext(Recall|PrecisionWithReference).*",
            category=DeprecationWarning,
        )
        from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

    from ragas.dataset_schema import SingleTurnSample

    recall_metric = NonLLMContextRecall()
    precision_metric = NonLLMContextPrecisionWithReference()
    recall_scores: list[float] = []
    precision_scores: list[float] = []
    for row in rows:
        sample = SingleTurnSample(
            retrieved_contexts=list(row.retrieved_contexts),
            reference_contexts=list(row.reference_contexts),
        )
        recall_scores.append(float(await recall_metric.single_turn_ascore(sample)))
        precision_scores.append(float(await precision_metric.single_turn_ascore(sample)))
    return {
        "non_llm_context_recall": _mean(recall_scores),
        "non_llm_context_precision": _mean(precision_scores),
    }
