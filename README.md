# llm-eval-legal-rag

A DeepEval evaluation suite — and, layered on top of it, a DeepTeam
red-team suite — built around a small legal contract Q&A RAG pipeline. A
portfolio project demonstrating LLM/GenAI quality engineering: metric
design, CI gating, and adversarial robustness testing at both the
generation and retrieval layers, not just traditional test automation.

## Why this project

Most QA portfolios show Selenium/Playwright suites. This one shows two
emerging skills: **treating an LLM system as something with a measurable,
gate-able definition of "correct,"** the same way a functional test suite
gates a build — and **treating the retrieval layer as its own attack
surface**, not just the model.

## Architecture

**Data flow** — a query passes through retrieval, then generation, then
gets scored:

```
  Query
    │
    ▼
  TfidfRetriever   ──▶  retrieves top-k chunks from data/contracts/*.txt
    │  (context)
    ▼
  LLMGenerator     ──▶  swappable backend: OpenAI / Anthropic / DeepSeek / Mock
    │  (answer)
    ▼
  Answer
    │
    ▼
  DeepEval scores the (query, context, answer) triple
  DeepTeam probes the same pipeline with adversarial queries and
  poisoned corpus documents instead of scoring goldens
```

**Project structure:**

```
llm-eval-legal-rag/
├── pyproject.toml          ← dependency manifest (uv reads this)
├── .env.example
├── data/contracts/         ← fixtures: the "system's" knowledge base
├── src/legal_rag/          ← the system under test
│   ├── retriever.py
│   ├── generator.py
│   └── pipeline.py
├── tests/                  ← the eval suite (this is the actual deliverable)
│   ├── goldens/             ← input questions + expected answers
│   ├── metrics/             ← custom scoring logic, reusable across tests
│   ├── test_rag_e2e.py
│   ├── test_legal_precision.py
│   ├── test_adversarial.py
│   └── red_team/             ← DeepTeam suite — retrieval-layer attacks
│       ├── poisoned_fixtures.py
│       ├── test_indirect_injection_retrieval.py
│       └── test_context_injection_deepteam.py
└── .github/workflows/       ← CI gate
```

**Components:**

- **`src/legal_rag/retriever.py`** — TF-IDF retrieval over three sample contracts (NDA, MSA, commercial lease). Deliberately simple so a low Contextual Precision/Recall score points at the generator, not an opaque vector store.
- **`src/legal_rag/generator.py`** — swappable generator interface (`OpenAIGenerator`, `AnthropicGenerator`, `DeepSeekGenerator`, `MockGenerator` for offline wiring). Lets the same suite score different models against the same retriever. `DeepSeekGenerator` reuses the `openai` SDK pointed at DeepSeek's OpenAI-compatible endpoint rather than a separate SDK.
- **`src/legal_rag/pipeline.py`** — wires the two together; `LegalRAGPipeline.answer(query)` is the system under test.

## Evaluation suite

| File | What it checks | Metrics |
|---|---|---|
| `tests/test_rag_e2e.py` | Standard RAG correctness | `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric` |
| `tests/test_legal_precision.py` | Domain-specific precision — exact dates, amounts, notice periods, clause refs | Custom `GEval` metric: **Legal Precision** |
| `tests/test_adversarial.py` | Robustness — out-of-scope questions, instruction-override attempts | Custom `GEval` metric: **Scope Awareness** |

Generic metrics catch hallucination and off-topic answers. They don't specifically penalize a subtly wrong date or notice period, and they don't test what happens when a question is out of scope or tries to override the system prompt — hence the two custom metrics and the separate adversarial suite.

## Red-teaming suite (DeepTeam)

`tests/test_adversarial.py` above tests robustness at the **generation**
layer — a user directly asking an out-of-scope question or typing
"ignore previous instructions." `tests/red_team/` tests a different,
currently-uncovered surface: the **retrieval** layer. The attacker isn't
the user asking the question — it's whatever ends up in a document the
retriever legitimately pulls back.

| File | What it checks | Framework mapping |
|---|---|---|
| `tests/red_team/poisoned_fixtures.py` | Not a test — three hand-crafted contract clauses with embedded instructions (crude, subtle, and prompt-leakage-oriented), seeded into a throwaway copy of the real corpus | — |
| `tests/red_team/test_indirect_injection_retrieval.py` | Does `TfidfRetriever` genuinely surface a poisoned clause for a natural query (no LLM call, free), and does the generator obey it once retrieved (needs a real generator) | OWASP LLM01:2026 Prompt Injection |
| `tests/red_team/test_context_injection_deepteam.py` | Systematic DeepTeam sweep: `PromptInjection`, `SyntheticContextInjection`, `ContextFlooding` scored against `Robustness`, `PromptLeakage`, `Misinformation`, `Hallucination` | OWASP LLM01:2026, LLM07:2026 Misinformation, LLM08:2026 Hidden Context Exposure |

The system prompt tells the model to answer only from retrieved excerpts
and to admit when they don't cover a question — good scope discipline —
but it doesn't explicitly say to treat excerpt *content* as inert data
rather than instructions. That gap is exactly what this suite probes.

**Cost note:** unlike the DeepEval suite above, `test_context_injection_deepteam.py`
runs a full attack × vulnerability × variation sweep against a live model
on every invocation — noticeably slower and more expensive than a
goldens-based eval run. See [CI](#ci) below for how that's handled.

## Setup

```bash
uv sync --extra openai   # or --extra anthropic / --extra deepseek
cp .env.example .env     # add OPENAI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY
deepeval login            # optional — enables Confident AI cloud storage
```

For the red-team suite, also install DeepTeam (add it to `pyproject.toml`
as its own extra, matching the existing `openai`/`anthropic`/`deepseek`
pattern — e.g. `--extra redteam`):

```bash
uv add deepteam --optional redteam
```

Note: `--extra deepseek` and `--extra openai` resolve to the same `openai` package — DeepSeek's API is OpenAI-compatible, so no separate SDK exists. The judge model DeepEval itself uses for scoring (`FaithfulnessMetric`, `GEval`, etc.) is configured separately and still defaults to OpenAI; see [DeepEval's model docs](https://deepeval.com/docs/metrics-introduction) if you want to swap that too.

## Running the suite

```bash
uv run deepeval test run tests/               # everything DeepEval-based
uv run deepeval test run tests/test_rag_e2e.py
uv run deepeval inspect                       # trace-tree view of the last run

# red-team suite
uv run pytest tests/red_team/test_indirect_injection_retrieval.py::test_poisoned_clause_is_retrieved
# ^ no API key needed — retrieval-only, run this one constantly
uv run pytest tests/red_team/                 # full red-team suite — needs an API key
```

## CI

`.github/workflows/deepeval-ci.yml` runs the full suite on every push/PR to `main` via `deepeval test run`, using `OPENAI_API_KEY` as the judge model and (optionally) `CONFIDENT_API_KEY` to push results to Confident AI. A metric regression fails the build — this is the "gate," not just a report.

The red-team suite is **not** in that blocking gate yet. `test_poisoned_clause_is_retrieved` is cheap enough to add to every PR, but the full DeepTeam sweep's cost and latency make it a poor fit for blocking every push — the plan is a separate, less frequent scheduled workflow (nightly or weekly) rather than adding real API spend to every commit. This mirrors how DeepTeam's own docs recommend running scheduled red-team assessments against a production endpoint rather than treating it as a per-commit gate.

## Learning path (building DeepEval expertise on top of this project)

1. **Core metrics** ✅ — Faithfulness / Answer Relevancy / Contextual Precision / Contextual Recall wired against real goldens (`test_rag_e2e.py`).
2. **Custom G-Eval metrics** ✅ — Legal Precision and Scope Awareness (`tests/metrics/`). Next: try `DAGMetric` for a more deterministic, decision-tree-style version of Legal Precision.
3. **Tracing & component-level evals** — migrate `LegalRAGPipeline` to use `@observe` (retriever span + generator span) per the [agent tracing model](https://deepeval.com/docs/getting-started-agents), then attach metrics to the retriever span specifically to isolate retrieval quality from generation quality.
4. **CI gating** ✅ — `deepeval test run` in GitHub Actions (`test_adversarial.py` failures should block merges once the generator is a real model, not `MockGenerator`).
5. **Confident AI cloud** — pull goldens from a versioned dataset (`EvaluationDataset().pull(alias=...)`) instead of the hardcoded list in `tests/goldens/`, and use `deepeval inspect` / the cloud UI to review failing traces.
6. **Synthetic data generation** — use the `Golden Synthesizer` to expand `IN_SCOPE_GOLDENS` and `ADVERSARIAL_GOLDENS` beyond hand-written examples, especially more adversarial/injection variants.
7. **Multi-agent extension** — turn this into a two-agent pipeline (a retrieval-critique agent + an answer agent) to practice sub-agent evaluation (`@observe(type="agent", metrics=[...])`).
8. **Red-teaming (DeepTeam)** 🚧 — retrieval-layer adversarial suite (`tests/red_team/`) mapped to the OWASP LLM Top 10 2026, layered on top of the same pipeline rather than a separate system. Retrieval-only test running; full DeepTeam sweep and CI scheduling still pending.

## Notes

- Sample contracts in `data/contracts/` are original, fictional text written for this project — not real agreements.
- `MockGenerator` exists purely so the pipeline runs offline for wiring/demo purposes. Real evaluation runs need a real generator (`OpenAIGenerator`). The same applies to red-teaming — see the `MockGenerator` guard in `tests/red_team/test_context_injection_deepteam.py`.
