# insurance-claims-rag-evals

[![CI](https://github.com/mustafaautomation/insurance-claims-rag-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/mustafaautomation/insurance-claims-rag-evals/actions/workflows/ci.yml)

Personal open-source demo by Muhammad Mustafa. It is a small retrieval-augmented assistant for a made-up insurer, Harborline Mutual, and an evaluation gate built on [RAGAS](https://docs.ragas.io/) 0.4.3.

The policy pages are synthetic. They were written for this repository. There are no real customers, no real policies, and no real claim files. Harborline Mutual is not a licensed insurer. The phone number `1-800-555-0148` is fictional.

The assistant answers questions about coverage, exclusions, deductibles, and claim steps. It is not allowed to say that a claim will be approved. It has to quote an exclusion in the words of the page. A coverage dispute goes to a human adjuster. A question the pages do not cover gets "I don't know."

## Architecture

```mermaid
flowchart LR
  Q[Question] --> RET[Retriever]
  C[Synthetic policy pages] --> CH[Chunker]
  CH --> EMB[Hashing embeddings]
  EMB --> RET
  RET --> FAISS[FAISS in memory]
  RET --> PG[pgvector in Postgres]
  FAISS --> CTX[Top pages]
  PG --> CTX
  CTX --> GEN[Chat provider]
  Q --> GEN
  GEN --> ANS[Answer]
  ANS --> OFF[Offline metrics]
  ANS --> LIVE[RAGAS judge metrics]
  OFF --> GATE[Threshold gate]
  LIVE --> GATE
  GATE --> REP[Markdown and HTML report]
```

Offline runs use FAISS and a deterministic fake generator. They do not call a model. Live runs use the same retriever, then OpenAI for the answer and for the RAGAS judge metrics. Postgres is optional and is started with Docker Compose.

## Run it locally

Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
uv run harborline-ask "What is the Cedar plan collision deductible?"
uv run harborline-ask --show-sources "Quote the flood exclusion on form HL-HO3 about surface water, waves, or overflow of a body of water."
uv run pytest
uv run ruff check src tests
uv run mypy
uv run harborline-eval --mode offline --variant improved
```

The offline command reads `data/snapshots/improved.jsonl`, scores it, and exits 0 only if every offline metric is at or above `config/thresholds.yml`. These two variants are supposed to fail:

```bash
uv run harborline-eval --mode offline --variant bad_retriever
uv run harborline-eval --mode offline --variant no_guardrail
```

Regenerate the recorded answers and the report after you change the pages, the questions, or the generator:

```bash
uv run harborline-eval --record --compare --report-dir docs
```

`--compare` scores all three variants. Its exit code follows the improved variant. The other two are the before side of the report.

### Postgres and pgvector

```bash
docker compose up -d
export HARBORLINE_DATABASE_URL=postgresql://harborline:harborline@localhost:5432/harborline
uv run harborline-ask --retriever pgvector "What is the Cedar plan collision deductible?"
```

The database password above is only for this local Compose file. Tests and CI use FAISS and do not need Docker.

### Live RAGAS judges

Live mode needs `OPENAI_API_KEY`. It sends every question to the chat model in `config/pipeline.yml`, then scores faithfulness, answer relevancy, context precision, and context recall with RAGAS 0.4.3's collections API (`Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall`).

```bash
export OPENAI_API_KEY=...
uv run harborline-eval --mode live --variant improved
```

`--record` on a live run writes `data/snapshots/live/`, which is gitignored.

This repository was built without an API key. `data/snapshots/judge_scores.illustrative.json` shows the field names and sets every judge score to null. It is not a measurement. The offline gate does not read it. Loading it as if it were a real run raises an error.

GitHub Actions workflow `Live eval` is `workflow_dispatch` only. The eval job runs only when the `OPENAI_API_KEY` secret is set.

## Metrics

Scores run from 0 to 1. Higher is better. The business reading is whether an answer stays on the policy pages and whether the retriever found those pages.

| Metric | What it measures | Mode |
| --- | --- | --- |
| Faithfulness | Whether the answer's claims stick to the retrieved pages. A low score means the answer added something the pages do not say. | live |
| Answer relevancy | Whether the answer addresses the question that was asked. | live |
| Context precision | Whether pages that support the reference answer are ranked above pages that do not. | live |
| Context recall | Whether the retrieved pages cover the claims in the reference answer. | live |
| Non-LLM context recall | RAGAS `NonLLMContextRecall`. String overlap between each labelled source page and the retrieved pages. No judge model. | offline and live |
| Non-LLM context precision | RAGAS `NonLLMContextPrecisionWithReference`. Rank-aware string overlap against the labelled pages. | offline and live |
| String presence | RAGAS `StringPresence`. Whether required phrases, including exclusion sentences, appear verbatim in the answer. | offline and live |
| Refusal correctness | Whether approval questions are refused, disputes are handed to an adjuster, out-of-scope questions say "I don't know", and ordinary questions do not do those things. | offline and live |
| Hit rate | Share of labelled questions whose reference page appears in the top results. Questions with no reference page are left out. | offline and live |
| MRR | Mean reciprocal rank of the first reference page. | offline and live |

RAGAS 0.4.3 still ships `NonLLMContextRecall` and `NonLLMContextPrecisionWithReference` on `ragas.metrics`. The collections package exports `StringPresence` and the four judge metrics, and it does not export the two non-LLM context metrics. The deprecation warning points at collections anyway. This project imports the two context metrics from `ragas.metrics` and filters that warning.

## Thresholds

`config/thresholds.yml` is the gate. A metric is checked only in the modes listed next to it. If a required metric is missing, the gate fails. The offline command exits 1 when any offline metric is below its minimum.

| Metric | Minimum |
| --- | ---: |
| refusal_correctness | 0.950 |
| hit_rate | 0.900 |
| mrr | 0.900 |
| string_presence | 0.900 |
| non_llm_context_recall | 0.900 |
| non_llm_context_precision | 0.900 |
| faithfulness | 0.700 (live only) |
| answer_relevancy | 0.700 (live only) |
| context_precision | 0.700 (live only) |
| context_recall | 0.700 (live only) |

The live minimums are starting points. They have not been measured here, because no judge run was made.

## Sample report

The numbers below are from `uv run harborline-eval --compare` on this commit. They are rounded to three decimal places. LLM-judged metrics are not in the table.

| Metric | Minimum | Improved | No-guardrail prompt | Bad retriever | Improved gate |
| --- | ---: | ---: | ---: | ---: | --- |
| refusal_correctness | 0.950 | 1.000 | 0.667 | 0.333 | pass |
| hit_rate | 0.900 | 1.000 | 1.000 | 0.000 | pass |
| mrr | 0.900 | 1.000 | 1.000 | 0.000 | pass |
| string_presence | 0.900 | 1.000 | 0.762 | 0.238 | pass |
| non_llm_context_recall | 0.900 | 1.000 | 1.000 | 0.000 | pass |
| non_llm_context_precision | 0.900 | 1.000 | 1.000 | 0.000 | pass |

On this 23-page set the improved offline scores sit at 1.000. The labelled questions reuse distinctive wording from the pages, the embedder is a hashing bag-of-words model, and the offline generator quotes the top page instead of paraphrasing. That is why the improved row is at the top of the scale. It is a property of this small synthetic set, not a general claim about other documents.

The no-guardrail prompt drops the approval, dispute, and out-of-scope rules. Refusal correctness falls to 0.667 and string presence to 0.762, so the gate fails. The bad retriever returns the lowest-scoring pages. Hit rate, MRR, and both non-LLM context metrics fall to 0.000, so the gate fails. CI runs the improved gate. Tests run the two failing variants and assert a non-zero exit code.

Full write-up: [docs/eval_report.md](docs/eval_report.md) and [docs/eval_report.html](docs/eval_report.html).

![Harborline evaluation report](docs/eval_report.png)

## What is real in offline mode

`FakeProvider` is not a language model. If the system prompt contains the guardrail block, it refuses approval questions, hands disputes to the adjuster, says "I don't know" for the listed out-of-scope topics and for a weak retrieval score, and otherwise returns the top page unchanged. If that block is absent, approval questions are answered with "Yes, your claim will be approved and paid in full." The recorded JSONL files are outputs of that generator. A test reruns the generator and checks that the files still match.

`OpenAIProvider` and `AnthropicProvider` implement the same `complete(system, user)` interface. Anthropic needs the `anthropic` extra. Embeddings default to the hashing model so CI does not download weights. `OpenAIEmbedder` calls the embeddings API when you construct it yourself.

## Adapt it to another document set

1. Replace `data/corpus/*.md`. Keep the frontmatter `id` and `title`. Say in your README whether the text is synthetic or name the public source and license.
2. Replace `data/eval/questions.json`. Each row needs a question, a reference answer, `reference_doc_ids`, `expected_behavior` (`answer`, `refuse_approval`, `handoff`, or `unknown`), and `required_phrases` for wording that must appear verbatim.
3. Run `uv run harborline-eval --record --compare --report-dir docs`.
4. Set the minimums in `config/thresholds.yml` from a run you actually produced. Do not copy the 1.000 scores above onto a different corpus.
5. Keep a broken retriever or a prompt with the guardrails removed, and keep a test that expects the gate to fail on it.
6. Turn on live judging only with your own API key. Commit judge scores only from a run you made, with `real_run` set to true. Do not fill the illustrative file with invented numbers.

Chunk size, overlap, top-k, the retrieval cutoff, and the model names are in `config/pipeline.yml`.

## Layout

- `src/harborline/` — chunking, hashing embeddings, FAISS, pgvector, prompts, providers, and the ask CLI
- `src/harborline/eval/` — dataset, metrics, gate, report, and the eval CLI
- `data/corpus/` — 23 synthetic policy pages
- `data/eval/questions.json` — 48 labelled questions
- `data/snapshots/` — recorded offline answers, plus the illustrative judge schema
- `config/` — pipeline settings and thresholds
- `docs/eval_report.md` — example report from the offline compare run

CI on pull requests and on `main` runs ruff, mypy, pytest, and the offline gate. None of those jobs use `continue-on-error`.

RAGAS 0.4.3 imports `langchain_community.chat_models.vertexai`, which was removed in langchain-community 0.4. This project pins LangChain to the 0.3 line so that import still works.
