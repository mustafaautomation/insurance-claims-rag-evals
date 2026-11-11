"""Unit tests for chunking, embeddings, prompts, and guardrail answers."""

import numpy as np

from harborline.chunking import chunk_text
from harborline.config import load_pipeline_config
from harborline.corpus import load_documents
from harborline.embeddings import HashingEmbedder
from harborline.eval.dataset import load_examples
from harborline.pipeline import Index, run_variant
from harborline.prompts import (
    AFFIRM,
    HANDOFF_RESPONSE,
    PHONE,
    REFUSAL_RESPONSE,
    UNKNOWN_SENTENCE,
    format_user_message,
    guardrails_enabled,
    parse_user_message,
    system_prompt,
)
from harborline.retrieval import PgVectorRetriever


def test_chunk_overlap_covers_the_text() -> None:
    text = "a" * 1000
    chunks = chunk_text(text, size=400, overlap=50)
    assert chunks[0] == "a" * 400
    assert chunks[1][:50] == chunks[0][-50:]
    assert chunks[-1][-1] == "a"
    assert len(chunks) > 1


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("  hello  ", size=100, overlap=10) == ["hello"]
    assert chunk_text("   ", size=100, overlap=10) == []


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    embedder.fit(["alpha beta", "beta gamma"])
    first = embedder.embed_query("alpha beta")
    second = embedder.embed_query("alpha beta")
    assert np.allclose(first, second)
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-5


def test_user_message_round_trip() -> None:
    from harborline.models import Chunk, Hit

    hit = Hit(
        chunk=Chunk(id="hl-about#0", doc_id="hl-about", title="About", text="Body text."),
        score=0.5,
        rank=1,
    )
    message = format_user_message("What is this?", [hit])
    question, pages = parse_user_message(message)
    assert question == "What is this?"
    assert pages[0].doc_id == "hl-about"
    assert pages[0].score == 0.5
    assert pages[0].text == "Body text."


def test_guardrail_prompt_states_the_three_rules() -> None:
    config = load_pipeline_config()
    prompt = system_prompt(guardrails=True, min_score=config.min_retrieval_score)
    assert guardrails_enabled(prompt)
    assert "Never confirm that a claim will be approved" in prompt
    assert "quote the exclusion sentence" in prompt
    assert "human adjuster" in prompt
    assert f"{config.min_retrieval_score:.2f}" in prompt
    plain = system_prompt(guardrails=False, min_score=config.min_retrieval_score)
    assert not guardrails_enabled(plain)
    assert AFFIRM in plain


def test_each_policy_page_is_one_chunk() -> None:
    index = Index()
    assert 15 <= len(load_documents()) <= 30
    assert len(index.chunks) == len({chunk.doc_id for chunk in index.chunks})


def test_required_phrases_occur_in_exactly_one_page() -> None:
    documents = {document.id: document.body for document in load_documents()}
    index = Index()
    examples = load_examples(index.chunks)
    assert 30 <= len(examples) <= 50
    for example in examples:
        for phrase in example.required_phrases:
            hits = [doc_id for doc_id, body in documents.items() if phrase in body]
            assert hits == list(example.reference_doc_ids)


def test_improved_answers_follow_guardrails() -> None:
    index = Index()
    examples = load_examples(index.chunks)
    rows = {row.id: row for row in run_variant(examples, "improved", index=index)}
    assert rows["q33"].response == REFUSAL_RESPONSE
    assert "will be approved and paid" not in rows["q35"].response
    assert rows["q39"].response == HANDOFF_RESPONSE
    assert PHONE in rows["q39"].response
    assert rows["q43"].response == UNKNOWN_SENTENCE
    assert rows["q48"].response == UNKNOWN_SENTENCE
    racing = (
        "Harborline does not cover loss that occurs while the auto is used in a race, "
        "speed contest, or driver-skill event."
    )
    assert racing in rows["q25"].response
    assert rows["q25"].retrieved_doc_ids[0] == "hl-au-racing"


def test_no_guardrail_confirms_approval_and_overturns_disputes() -> None:
    index = Index()
    examples = load_examples(index.chunks)
    rows = {row.id: row for row in run_variant(examples, "no_guardrail", index=index)}
    assert rows["q33"].response == AFFIRM
    assert "overturned" in rows["q39"].response


def test_pgvector_requires_a_url() -> None:
    embedder = HashingEmbedder(dim=8)
    try:
        PgVectorRetriever("", embedder)
    except ValueError as exc:
        assert "database URL" in str(exc)
    else:
        raise AssertionError("expected ValueError")
