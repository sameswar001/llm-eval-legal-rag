# llm-eval-legal-rag

A DeepEval evaluation suite built around a small legal contract Q&A RAG pipeline — a portfolio project demonstrating LLM/GenAI quality engineering: metric design, CI gating, and adversarial robustness testing, not just traditional test automation.

## Why this project

Most QA portfolios show Selenium/Playwright suites. This one shows the emerging skill: **treating an LLM system as something with a measurable, gate-able definition of "correct,"** the same way a functional test suite gates a build. 

## Architecture

```
Query → TfidfRetriever (data/contracts/*.txt) → LLMGenerator (swappable) → Answer
                                                        ↓
                                      DeepEval test suite scores the (query, context, answer) triple
```

- **`src/legal_rag/retriever.py`** — TF-IDF retrieval over three sample contracts (NDA, MSA, commercial lease). Deliberately simple so a low Contextual Precision/Recall score points at the generator, not an opaque vector store.
- **`src/legal_rag/generator.py`** — swappable generator interface (`OpenAIGenerator`, `AnthropicGenerator`, `MockGenerator` for offline wiring). Lets the same suite score different models against the same retriever.
- **`src/legal_rag/pipeline.py`** — wires the two together; `LegalRAGPipeline.answer(query)` is the system under test.

## Evaluation suite

| File | What it checks | Metrics |
|---|---|---|
| `tests/test_rag_e2e.py` | Standard RAG correctness | `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric` |
| `tests/test_legal_precision.py` | Domain-specific precision — exact dates, amounts, notice periods, clause refs | Custom `GEval` metric: **Legal Precision** |
| `tests/test_adversarial.py` | Robustness — out-of-scope questions, instruction-override attempts | Custom `GEval` metric: **Scope Awareness** |

Generic metrics catch hallucination and off-topic answers. They don't specifically penalize a subtly wrong date or notice period, and they don't test what happens when a question is out of scope or tries to override the system prompt — hence the two custom metrics and the separate adversarial suite.

## Setup

```bash
uv sync --extra openai   # or --extra anthropic
cp .env.example .env     # add OPENAI_API_KEY 
deepeval login            # optional — enables Confident AI cloud storage
```

## Running the suite

```bash
uv run deepeval test run tests/               # everything
uv run deepeval test run tests/test_rag_e2e.py
uv run deepeval inspect                       # trace-tree view of the last run
```

## CI

`.github/workflows/deepeval-ci.yml` runs the full suite on every push/PR to `main` via `deepeval test run`, using `OPENAI_API_KEY` as the judge model and (optionally) `CONFIDENT_API_KEY` to push results to Confident AI. A metric regression fails the build — this is the "gate," not just a report.

## Learning path (building DeepEval expertise on top of this project)

1. **Core metrics** ✅ — Faithfulness / Answer Relevancy / Contextual Precision / Contextual Recall wired against real goldens (`test_rag_e2e.py`).
2. **Custom G-Eval metrics** ✅ — Legal Precision and Scope Awareness (`tests/metrics/`). Next: try `DAGMetric` for a more deterministic, decision-tree-style version of Legal Precision.
3. **Tracing & component-level evals** — migrate `LegalRAGPipeline` to use `@observe` (retriever span + generator span) per the [agent tracing model](https://deepeval.com/docs/getting-started-agents), then attach metrics to the retriever span specifically to isolate retrieval quality from generation quality.
4. **CI gating** ✅ — `deepeval test run` in GitHub Actions (`test_adversarial.py` failures should block merges once the generator is a real model, not `MockGenerator`).
5. **Confident AI cloud** — pull goldens from a versioned dataset (`EvaluationDataset().pull(alias=...)`) instead of the hardcoded list in `tests/goldens/`, and use `deepeval inspect` / the cloud UI to review failing traces.
6. **Synthetic data generation** — use the `Golden Synthesizer` to expand `IN_SCOPE_GOLDENS` and `ADVERSARIAL_GOLDENS` beyond hand-written examples, especially more adversarial/injection variants.
7. **Multi-agent extension** — turn this into a two-agent pipeline (a retrieval-critique agent + an answer agent) to practice sub-agent evaluation (`@observe(type="agent", metrics=[...])`).

## Notes

- Sample contracts in `data/contracts/` are original, fictional text written for this project — not real agreements.
- `MockGenerator` exists purely so the pipeline runs offline for wiring/demo purposes. Real evaluation runs need a real generator (`OpenAIGenerator` / `AnthropicGenerator`).
