"""Ask one policy question from the command line."""

import argparse
import os
import sys

from harborline.config import load_pipeline_config
from harborline.models import VARIANTS, Example, Variant
from harborline.pipeline import Index, answer_example
from harborline.providers import AnthropicProvider, ChatProvider, FakeProvider, OpenAIProvider
from harborline.retrieval import PgVectorRetriever, Retriever, ReversedRetriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harborline-ask")
    parser.add_argument("question")
    parser.add_argument("--provider", choices=["fake", "openai", "anthropic"], default="fake")
    parser.add_argument("--retriever", choices=["faiss", "pgvector"], default="faiss")
    parser.add_argument("--variant", choices=VARIANTS, default="improved")
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args(argv)
    try:
        answer, sources = ask(
            args.question,
            provider_name=args.provider,
            retriever_name=args.retriever,
            variant=args.variant,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(answer)
    if args.show_sources:
        for source in sources:
            print(f"source: {source}")
    return 0


def ask(
    question: str,
    *,
    provider_name: str = "fake",
    retriever_name: str = "faiss",
    variant: Variant = "improved",
) -> tuple[str, tuple[str, ...]]:
    config = load_pipeline_config()
    index = Index(config)
    provider = _provider(provider_name, config.chat_model)
    retriever = _retriever(index, retriever_name, variant)
    example = Example(
        id="ask",
        question=question,
        reference="",
        reference_contexts=(),
        reference_doc_ids=(),
        expected_behavior="answer",
        required_phrases=(),
        tags=(),
    )
    prediction = answer_example(
        example,
        variant=variant,
        index=index,
        provider=provider,
        retriever=retriever,
    )
    return prediction.response, prediction.retrieved_doc_ids


def _provider(name: str, model: str) -> ChatProvider:
    if name == "fake":
        return FakeProvider()
    if name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAIProvider(model)
    if name == "anthropic":
        return AnthropicProvider()
    raise RuntimeError(f"unknown provider {name}")


def _retriever(index: Index, name: str, variant: Variant) -> Retriever | None:
    if name == "faiss":
        return None
    if name != "pgvector":
        raise RuntimeError(f"unknown retriever {name}")
    dsn = os.environ.get("HARBORLINE_DATABASE_URL", "")
    if not dsn:
        raise RuntimeError("Set HARBORLINE_DATABASE_URL to use the pgvector retriever.")
    store = PgVectorRetriever(dsn, index.embedder)
    store.build(index.chunks)
    if variant == "bad_retriever":
        return ReversedRetriever(store, corpus_size=len(index.chunks))
    return store


if __name__ == "__main__":
    raise SystemExit(main())
