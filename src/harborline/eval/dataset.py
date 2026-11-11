"""Load labelled questions and attach reference contexts from the corpus chunks."""

import json
from pathlib import Path

from harborline.models import Chunk, Example, parse_behavior
from harborline.paths import questions_path


def load_examples(chunks: list[Chunk], path: Path | None = None) -> list[Example]:
    source = path or questions_path()
    loaded: object = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("questions file must be a JSON list")
    by_doc: dict[str, list[str]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk.text)
    examples: list[Example] = []
    seen: set[str] = set()
    for item in loaded:
        if not isinstance(item, dict):
            raise ValueError("each question must be a JSON object")
        payload = {key: value for key, value in item.items()}
        example = _example(payload, by_doc)
        if example.id in seen:
            raise ValueError(f"duplicate question id {example.id}")
        seen.add(example.id)
        examples.append(example)
    return examples


def _example(payload: dict[object, object], by_doc: dict[str, list[str]]) -> Example:
    doc_ids = _string_list(payload, "reference_doc_ids")
    contexts: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in by_doc:
            raise ValueError(f"unknown reference doc {doc_id}")
        contexts.extend(by_doc[doc_id])
    return Example(
        id=_string(payload, "id"),
        question=_string(payload, "question"),
        reference=_string(payload, "reference"),
        reference_contexts=tuple(contexts),
        reference_doc_ids=tuple(doc_ids),
        expected_behavior=parse_behavior(_string(payload, "expected_behavior")),
        required_phrases=tuple(_string_list(payload, "required_phrases")),
        tags=tuple(_string_list(payload, "tags")),
    )


def _string(payload: dict[object, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_list(payload: dict[object, object], key: str) -> list[str]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key} must contain strings")
        items.append(item)
    return items
