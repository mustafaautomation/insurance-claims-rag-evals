"""Live RAGAS judge metrics. This module calls OpenAI only when a key is present."""

import os
from collections.abc import Sequence

from harborline.config import PipelineConfig
from harborline.models import Row


class MissingApiKeyError(RuntimeError):
    pass


def require_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise MissingApiKeyError("OPENAI_API_KEY is not set. Live RAGAS evaluation was not run.")
    return key


async def judge_rows(rows: Sequence[Row], config: PipelineConfig) -> dict[str, float]:
    """Score faithfulness, answer relevancy, context precision, and context recall."""

    require_openai_key()
    from openai import AsyncOpenAI
    from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI()
    llm = llm_factory(config.judge_model, client=client)
    embeddings = embedding_factory("openai", model=config.embedding_model, client=client)
    if not isinstance(embeddings, BaseRagasEmbedding):
        raise TypeError("Answer relevancy needs the modern RAGAS embedding interface.")
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    precision = ContextPrecision(llm=llm)
    recall = ContextRecall(llm=llm)

    totals = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }
    count = 0
    for row in rows:
        contexts = list(row.retrieved_contexts)
        faith = await faithfulness.ascore(
            user_input=row.question,
            response=row.response,
            retrieved_contexts=contexts,
        )
        relevant = await relevancy.ascore(user_input=row.question, response=row.response)
        precise = await precision.ascore(
            user_input=row.question,
            reference=row.reference,
            retrieved_contexts=contexts,
        )
        remembered = await recall.ascore(
            user_input=row.question,
            retrieved_contexts=contexts,
            reference=row.reference,
        )
        totals["faithfulness"] += float(faith.value)
        totals["answer_relevancy"] += float(relevant.value)
        totals["context_precision"] += float(precise.value)
        totals["context_recall"] += float(remembered.value)
        count += 1
    if count == 0:
        raise ValueError("live eval has no rows")
    return {name: value / count for name, value in totals.items()}
