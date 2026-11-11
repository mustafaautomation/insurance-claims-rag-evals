# Harborline Mutual evaluation report

LLM-judged RAGAS metrics were not run for this report. No API key was available, so faithfulness, answer relevancy, context precision, and context recall are not given numbers here. The scores below were produced by the deterministic offline generator, retrieval checks, and RAGAS non-LLM metrics.

The improved row is the one the offline gate accepts or rejects. The no-guardrail prompt drops the approval, dispute, and out-of-scope rules. The bad retriever returns the lowest-scoring policy pages.

## Metric table

| Metric | Minimum | Improved | No-guardrail prompt | Bad retriever | Improved gate |
| --- | ---: | ---: | ---: | ---: | --- |
| refusal_correctness | 0.950 | 1.000 | 0.667 | 0.333 | pass |
| hit_rate | 0.900 | 1.000 | 1.000 | 0.000 | pass |
| mrr | 0.900 | 1.000 | 1.000 | 0.000 | pass |
| string_presence | 0.900 | 1.000 | 0.762 | 0.238 | pass |
| non_llm_context_recall | 0.900 | 1.000 | 1.000 | 0.000 | pass |
| non_llm_context_precision | 0.900 | 1.000 | 1.000 | 0.000 | pass |

## Before and after

| Metric | Improved minus no-guardrail | Improved minus bad retriever |
| --- | ---: | ---: |
| refusal_correctness | +0.333 | +0.667 |
| hit_rate | +0.000 | +1.000 |
| mrr | +0.000 | +1.000 |
| string_presence | +0.238 | +0.762 |
| non_llm_context_recall | +0.000 | +1.000 |
| non_llm_context_precision | +0.000 | +1.000 |

## Improved per-question failures

No per-question failures on the improved variant.

## Baseline failure counts

- no_guardrail: 16 of 48 questions failed a check.
- bad_retriever: 42 of 48 questions failed a check.

## Example failure from the bad retriever

- id: q01
- question: What is the bodily injury liability limit?
- expected behavior: answer
- behavior ok: false
- retrieved docs: hl-out-of-scope, hl-rt-structure, hl-about, hl-claim-home
- response: I don't know. That is outside the Harborline policy documents I can use.
