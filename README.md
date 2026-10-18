# Enterprise Knowledge Assistant

A retrieval-augmented generation (RAG) service that answers employee questions from internal documents, **with an evaluation harness that runs in CI, versioned prompts, and role-based access control**.

Most RAG demos stop at "it answers questions". This one also answers: *how do I know the answers are right, and how do I stop people seeing documents they shouldn't?*

## Features

| Capability | What is implemented | Where |
|---|---|---|
| Document preparation | Load (md, txt, pdf) → clean → tag (department, allowed roles) → section-aware chunking → exact and near-duplicate removal, with an ingest report | `app/ingestion/pipeline.py` |
| Retrieval | FAISS cosine index, retrieval filtered by role **before** generation | `app/vectorstore.py` |
| Generation | LangChain `ChatPromptTemplate`, `ChatOpenAI` and `OpenAIEmbeddings` when `LLM_MODE=openai`; deterministic offline mode otherwise | `app/llm.py`, `app/embeddings.py` |
| Prompt versioning | Three YAML prompt versions, selectable per request and comparable in the eval | `prompts/`, `app/prompts.py` |
| Access control | API key → role (`employee`, `hr`, `finance`, `admin`), constant-time key check, fail-closed tagging, audit log | `app/security.py` |
| Evaluation harness | 49 labelled questions; retrieval hit rate, answer accuracy, faithfulness, hallucination rate, abstention accuracy, access leakage; CI gate | `eval/` |
| REST API + UI | JSON API (`/query`, `/documents`, `/prompts`, `/ingest`, `/health`), small web UI | `app/main.py`, `static/` |
| DevOps | Dockerfile, docker-compose, GitHub Actions (tests, eval gate, report artifact, image build) | `.github/workflows/ci.yml` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

pytest -q                    # 18 tests
python -m eval.run_eval      # evaluation + quality gate
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the UI, or http://localhost:8000/docs for the API docs.

```bash
curl -X POST localhost:8000/query \
  -H "X-API-Key: demo-hr-key" -H "Content-Type: application/json" \
  -d '{"question": "How long does a performance improvement plan last?", "prompt_version": "v2_cited"}'
```

Demo keys (`demo-employee-key`, `demo-hr-key`, `demo-finance-key`, `demo-admin-key`) exist only so the project runs out of the box. Set `API_KEYS` in your environment for anything real.

Docker: `docker compose up --build`.

### Two modes

* **`LLM_MODE=offline` (default)**: no API key. Hashed TF-IDF embeddings plus an *extractive* answerer that quotes the best-matching sentences from retrieved context. Deterministic, so the CI evaluation is reproducible and free.
* **`LLM_MODE=openai`**: set `OPENAI_API_KEY`. Uses OpenAI embeddings and a chat model through LangChain, with the prompt template text. `python -m eval.run_eval --judge` adds an LLM-graded faithfulness score.

## Architecture

```
documents ─► clean ─► tag ─► chunk ─► dedupe ─► embed ─► FAISS index
                                                          │
question + API key ─► role ─► retrieve (role filter) ─► prompt version ─► LLM ─► answer + sources
                                      │
                                 audit log
```

Security design: the role filter is applied inside retrieval. A chunk the caller may not see never reaches the prompt, so it cannot leak through the model's answer. Documents without `allowed_roles` default to admin-only (fail closed). Deduplication never drops a chunk if the surviving copy is visible to fewer roles.

## Evaluation

`eval/dataset.jsonl` has 49 questions over 9 sample documents:

* 40 answerable (employee, HR and finance roles), each with expected keywords and the expected source document
* 5 unanswerable (e.g. pet insurance, cricket scores) where the correct behaviour is to abstain
* 4 restricted (a role asks about a document it may not see) where the correct behaviour is to abstain with no leakage

| Metric | Definition |
|---|---|
| retrieval_hit_rate | Expected document appears in the retrieved sources (answerable questions) |
| answer_accuracy | Answer contains all expected keywords |
| faithfulness | Share of answer sentences supported by the retrieved context (word coverage ≥ 80% and every number present in context) |
| hallucination_rate | Answered responses that are unfaithful, or that answer an unanswerable or restricted question |
| abstention_accuracy | Unanswerable and restricted questions where the system declined |
| false_abstention_rate | Answerable questions where it wrongly declined |
| access_leakage | Restricted documents retrieved, or restricted facts in the answer (must be 0) |

CI runs `python -m eval.run_eval` and fails the build if the default prompt misses a threshold in `eval/thresholds.json`. The report is added to the GitHub Actions job summary.

### Results (offline mode, 49 questions)

| prompt | retrieval hit rate | answer accuracy | faithfulness | hallucination rate | abstention accuracy | false abstention | leakage |
|---|---|---|---|---|---|---|---|
| v1_baseline | 0.975 | 0.900 | 1.0 | 0.0 | 1.0 | 0.075 | 0 |
| **v2_cited** (default) | **1.000** | **0.925** | 1.0 | 0.0 | 1.0 | 0.075 | 0 |
| v3_concise | 0.975 | 0.875 | 1.0 | 0.0 | 1.0 | 0.125 | 0 |

### Findings

1. **Prompt/retrieval settings trade recall for precision.** `v3_concise` (top-2 chunks, higher score threshold) abstains more often (12.5% of answerable questions vs 7.5%) and misses q31 and q19 that `v2_cited` answers. `v2_cited` (top-4) had the best recall without losing precision, so it is the default. In offline mode the versions differ through their parameters (`top_k`, `min_score`, `max_sentences`); with `LLM_MODE=openai` the template wording also takes effect.
2. **Access control held.** 0 leaks across 4 restricted-role questions, and the unit tests check the same for the documents endpoint.
3. **Remaining failures are lexical gaps, not hallucinations.** q09 ("which days… office" vs "Tuesday, Wednesday and Thursday"), q13 ("minimum… length" vs "at least… characters long") and q29 ("unacknowledged page" vs "does not acknowledge") share no words with their answers. The system abstained instead of guessing. Semantic embeddings (`LLM_MODE=openai`) are the obvious fix, and these three questions are the regression test for it.
4. **Retrieval score alone is a weak abstention signal.** Cosine scores for valid and invalid questions overlapped (valid ones as low as 0.04, invalid ones up to 0.19), so the answerer also requires question-word coverage in the chosen sentence. One early failure (q44, "Who is the CEO?") returned an unrelated sentence because only the company name matched; adding company-name tokens to the ignore list fixed it.
5. **Deduplication worked.** The handbook excerpt repeats the annual-leave paragraph; the pipeline dropped the duplicate chunk (37 chunks created, 36 indexed).

### Limitations (read before an interview)

* Faithfulness is a **heuristic**, and the offline answerer is extractive, so faithfulness is near 1.0 by construction. The informative numbers offline are retrieval hit rate, accuracy, abstention and leakage. Run with a real LLM (`LLM_MODE=openai`) and `--judge` to get meaningful faithfulness and hallucination figures.
* The dataset is small and was written alongside the sample documents, and thresholds were tuned against it. Treat the numbers as a regression baseline, not a generalisation claim.
* The vector store scans the full FAISS index and filters afterwards. That is fine for thousands of chunks; at scale push the role filter into the database (e.g. pgvector `WHERE allowed_roles && ARRAY[...]`).
* API keys map to roles in environment config. A production system would use SSO/OIDC and per-user identity.
* The OpenAI path and the Docker image are written but were not exercised in the environment this was built in. The offline path, API, tests and evaluation were run.

## Project layout

```
app/            FastAPI app, RAG service, LLM and embedding backends, security
app/ingestion/  document preparation pipeline
prompts/        versioned prompt templates (YAML)
eval/           dataset, metrics, runner, CI thresholds, latest report
data/sample_docs/  fictional "Northwind Systems" policies (some restricted by role)
tests/          pipeline, API/security and eval-gate tests
static/         web UI
```

## Adding your own documents

Drop `.md`, `.txt` or `.pdf` files in `data/sample_docs/` (or set `DOCS_DIR`). Markdown files can start with:

```yaml
---
title: Travel Policy
department: finance
allowed_roles: [all]      # or [hr, admin], [finance, admin]
---
```

Then `POST /ingest` with an admin key. Add matching questions to `eval/dataset.jsonl` so the CI gate covers them.
