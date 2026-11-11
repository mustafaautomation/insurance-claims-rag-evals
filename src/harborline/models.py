"""Shared records for retrieval, answers, and evaluation rows."""

from dataclasses import dataclass
from typing import Literal

Behavior = Literal["answer", "refuse_approval", "handoff", "unknown"]
Variant = Literal["improved", "no_guardrail", "bad_retriever"]

BEHAVIORS: tuple[Behavior, ...] = ("answer", "refuse_approval", "handoff", "unknown")
VARIANTS: tuple[Variant, ...] = ("improved", "no_guardrail", "bad_retriever")


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    body: str
    filename: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    reference: str
    reference_contexts: tuple[str, ...]
    reference_doc_ids: tuple[str, ...]
    expected_behavior: Behavior
    required_phrases: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class Prediction:
    example_id: str
    response: str
    retrieved_contexts: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    retrieval_scores: tuple[float, ...]


@dataclass(frozen=True)
class Row:
    """One labelled question joined to a pipeline output."""

    id: str
    variant: str
    question: str
    reference: str
    reference_contexts: tuple[str, ...]
    reference_doc_ids: tuple[str, ...]
    expected_behavior: Behavior
    required_phrases: tuple[str, ...]
    tags: tuple[str, ...]
    response: str
    retrieved_contexts: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    retrieval_scores: tuple[float, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "variant": self.variant,
            "question": self.question,
            "reference": self.reference,
            "reference_contexts": list(self.reference_contexts),
            "reference_doc_ids": list(self.reference_doc_ids),
            "expected_behavior": self.expected_behavior,
            "required_phrases": list(self.required_phrases),
            "tags": list(self.tags),
            "response": self.response,
            "retrieved_contexts": list(self.retrieved_contexts),
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "retrieval_scores": [round(score, 6) for score in self.retrieval_scores],
        }


def parse_behavior(value: str) -> Behavior:
    if value == "answer":
        return "answer"
    if value == "refuse_approval":
        return "refuse_approval"
    if value == "handoff":
        return "handoff"
    if value == "unknown":
        return "unknown"
    raise ValueError(f"unknown expected_behavior: {value}")


def row_from_json(payload: dict[str, object], variant: str) -> Row:
    return Row(
        id=_string(payload, "id"),
        variant=_string(payload, "variant") if "variant" in payload else variant,
        question=_string(payload, "question"),
        reference=_string(payload, "reference"),
        reference_contexts=tuple(_string_list(payload, "reference_contexts")),
        reference_doc_ids=tuple(_string_list(payload, "reference_doc_ids")),
        expected_behavior=parse_behavior(_string(payload, "expected_behavior")),
        required_phrases=tuple(_string_list(payload, "required_phrases")),
        tags=tuple(_string_list(payload, "tags")),
        response=_string(payload, "response"),
        retrieved_contexts=tuple(_string_list(payload, "retrieved_contexts")),
        retrieved_doc_ids=tuple(_string_list(payload, "retrieved_doc_ids")),
        retrieval_scores=tuple(_float_list(payload, "retrieval_scores")),
    )


def join_row(variant: Variant, example: Example, prediction: Prediction) -> Row:
    if prediction.example_id != example.id:
        raise ValueError(f"prediction {prediction.example_id} does not match {example.id}")
    return Row(
        id=example.id,
        variant=variant,
        question=example.question,
        reference=example.reference,
        reference_contexts=example.reference_contexts,
        reference_doc_ids=example.reference_doc_ids,
        expected_behavior=example.expected_behavior,
        required_phrases=example.required_phrases,
        tags=example.tags,
        response=prediction.response,
        retrieved_contexts=prediction.retrieved_contexts,
        retrieved_doc_ids=prediction.retrieved_doc_ids,
        retrieval_scores=prediction.retrieval_scores,
    )


def _string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _string_list(payload: dict[str, object], key: str) -> list[str]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key} must contain strings")
        items.append(item)
    return items


def _float_list(payload: dict[str, object], key: str) -> list[float]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    items: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{key} must contain numbers")
        items.append(float(item))
    return items
