"""Retrieve policy chunks and generate an answer with a chat provider."""

from harborline.config import PipelineConfig, load_pipeline_config
from harborline.corpus import chunk_documents, load_documents
from harborline.embeddings import HashingEmbedder
from harborline.models import Example, Prediction, Row, Variant, join_row
from harborline.prompts import format_user_message, system_prompt
from harborline.providers import ChatProvider, FakeProvider
from harborline.retrieval import FaissRetriever, Retriever, ReversedRetriever


class Index:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or load_pipeline_config()
        documents = load_documents()
        self.chunks = chunk_documents(
            documents,
            size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        self.embedder = HashingEmbedder(dim=self.config.embedding_dim)
        self.embedder.fit([chunk.text for chunk in self.chunks])
        self.faiss = FaissRetriever(self.embedder, self.chunks)

    def retriever_for(self, variant: Variant) -> Retriever:
        if variant == "bad_retriever":
            return ReversedRetriever(self.faiss, corpus_size=len(self.chunks))
        return self.faiss

    def uses_guardrails(self, variant: Variant) -> bool:
        return variant != "no_guardrail"


def answer_example(
    example: Example,
    *,
    variant: Variant,
    index: Index,
    provider: ChatProvider | None = None,
    retriever: Retriever | None = None,
) -> Prediction:
    selected = retriever if retriever is not None else index.retriever_for(variant)
    hits = selected.search(example.question, k=index.config.top_k)
    system = system_prompt(
        guardrails=index.uses_guardrails(variant),
        min_score=index.config.min_retrieval_score,
    )
    user = format_user_message(example.question, hits)
    chat = provider if provider is not None else FakeProvider()
    response = chat.complete(system, user)
    return Prediction(
        example_id=example.id,
        response=response,
        retrieved_contexts=tuple(hit.chunk.text for hit in hits),
        retrieved_doc_ids=tuple(hit.chunk.doc_id for hit in hits),
        retrieval_scores=tuple(hit.score for hit in hits),
    )


def run_variant(
    examples: list[Example],
    variant: Variant,
    *,
    index: Index | None = None,
    provider: ChatProvider | None = None,
) -> list[Row]:
    built = index or Index()
    rows: list[Row] = []
    for example in examples:
        prediction = answer_example(example, variant=variant, index=built, provider=provider)
        rows.append(join_row(variant, example, prediction))
    return rows
