# VeriRAG-Multi-Stage-Verification

Production-grade RAG system for legal document analysis, with hallucination detection and semantic caching.

## Overview

This project implements a retrieval-augmented generation pipeline for querying legal contracts, built on the CUAD dataset. Retrieval combines query expansion, hybrid semantic + keyword search, and cross-encoder reranking. Every generated answer passes through a verification and self-correction layer before being returned, and a Redis-backed semantic cache avoids redundant LLM calls for repeated queries.

Tested on 2,000 queries across 510 legal contracts (9,217 chunks, 18,208 Q&A pairs):

| Metric | Result |
|---|---|
| Retrieval precision | 96.1% (1,500 RAGAS queries) |
| Hallucination reduction | 46.7% (500 production queries) |
| Cost savings from caching | 74.5% (63.3% cache hit rate) |
| Self-correction success rate | 88.7% |

## Key Features

**Three-stage retrieval** — query expansion (GPT-4) → hybrid semantic + BM25 search → Cohere reranking. Result: 96.1% retrieval precision.

**Self-correction layer** — answer generation → hallucination detection via entailment checking → auto-correction of unverifiable claims. Result: 46.7% hallucination reduction.

**Semantic caching** — Redis vector cache with cosine similarity matching (threshold 0.95) to avoid redundant LLM calls. Result: 74.5% cost savings.

**Role-based access control** — Pinecone namespace isolation and a document-level permission layer for secure retrieval.

## Architecture

A query first checks the semantic cache. On a miss, it goes through query expansion, hybrid retrieval, reranking, and RBAC filtering before answer generation. The generated answer is verified against the source chunks; if it fails verification, it's routed through self-correction before being cached and returned.

<details>
<summary>Detailed pipeline flow</summary>

**Cache check** — Redis semantic similarity search, cosine threshold 0.95, 63.3% hit rate.

**Retrieval** — GPT-4 reformulates ambiguous queries, hybrid search combines semantic vectors with BM25 keyword scores, Cohere reranks the top-k results, then RBAC filters by user permissions.

**Generation & verification** — GPT-4 synthesizes an answer with citations; the answer is checked for entailment against source chunks; unverifiable claims are auto-fixed or flagged (88.7% correction success rate).

**Caching** — verified answers are stored for reuse, reducing average cost per query from $0.0232 to $0.0059.

</details>

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-large` | 3072-dimensional semantic vectors |
| Vector DB | Pinecone Serverless | 9,217 embedded contract chunks |
| Reranking | Cohere `rerank-english-v3.0` | Cross-encoder precision boost |
| LLM | GPT-4o / GPT-4-turbo | Answer generation & verification |
| Cache | Redis | Semantic similarity caching |
| Dataset | CUAD | 510 legal contracts, 18,208 Q&A pairs |
| Evaluation | RAGAS | Retrieval precision metrics |

## Quick Start

```bash
git clone https://github.com/MitaliBagadia1013/VeriRAG-Multi-Stage-Verification.git
cd VeriRAG-Multi-Stage-Verification
pip install -r requirements.txt
cp .env.example .env   # add your API keys
./scripts/start_redis.sh
python evaluations/validate_system_health.py
```

Run the API server:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Launch the Streamlit UI (with the API running):

```bash
streamlit run app.py
```

Available at `http://localhost:8501`. Log in with `admin@company.com` / `admin123` (or `user@company.com` / `user123` for a lower-privilege demo), then ask a question. The UI shows the answer, cache hit/miss status, verification confidence, and the source chunks used.

## Running Evaluations

**RAGAS evaluation** — retrieval precision on 1,500 queries (~187 min, ~$67):
```bash
python evaluations/RAGAS_EVALUATION.py
```

**Production validation** — hallucination detection and caching on 500 queries (~45 min, ~$31):
```bash
python evaluations/PRODUCTION_VALIDATION.py
```

## Cost Analysis

| Component | Cache miss | Cache hit |
|---|---|---|
| Embedding | $0.0001 | $0.0001 |
| Retrieval | Free | — |
| Reranking | $0.0003 | — |
| Generation | $0.0137 | — |
| Verification | $0.0091 | — |
| **Total** | **$0.0232** | **$0.0001** |

At a 63.3% cache hit rate, average cost is $0.0059/query — a 74.5% savings versus no caching, or roughly $173 saved per 10,000 queries.

## Evaluation Methodology

<details>
<summary>Retrieval precision (96.1%)</summary>

RAGAS framework, 1,500 queries from the CUAD dataset across 41 legal clause categories. Context precision = relevant chunks / total retrieved chunks, versus an 85–90% industry-standard baseline for plain semantic search.
</details>

<details>
<summary>Hallucination reduction (46.7%)</summary>

Baseline (GPT-4, no verification) hallucination rate: 58.3%. With the self-correction layer: 30.8%. Measured by generating answers with and without verification and having a human evaluate factual accuracy against source documents. 88.7% of detected hallucinations were successfully corrected.
</details>

<details>
<summary>Cost savings (74.5%)</summary>

Baseline (no caching): $0.0232/query. With a 63.3% cache hit rate at $0.0001/query and 36.7% misses at $0.0232/query, average cost drops to $0.0059/query. Cache false-positive rate under manual validation: <0.1%.
</details>

<details>
<summary>Why two different test sizes?</summary>

| Test | Sample size | Cost/query | Rationale |
|---|---|---|---|
| RAGAS evaluation | 1,500 | $0.045 | Retrieval-only, cheaper, larger sample for significance |
| Production validation | 500 | $0.061 | Full pipeline + verification, smaller sample due to cost |

Extensive unit-level testing (retrieval) plus focused integration testing (end-to-end).
</details>

## Project Structure

```
VeriRAG-Multi-Stage-Verification/
├── api/                 # FastAPI backend, auth, routes
├── ui/                   # Streamlit web interface
├── retrieval/            # Query expansion, hybrid search, reranking
├── verification/         # Hallucination detection, self-correction
├── cache/                # Redis semantic cache
├── ingestion/            # Document parsing, chunking, embedding
├── monitoring/           # Langfuse observability
├── evaluations/          # RAGAS + production validation scripts
├── results/              # Evaluation output
├── data/cuad/            # 510 legal contracts, chunks, Q&A pairs
├── app.py                # Streamlit application launcher
└── requirements.txt
```

## Use Cases

**Legal & compliance** — contract analysis, clause extraction, regulatory compliance (GDPR, SOC2, HIPAA), litigation support.

**Enterprise operations** — policy Q&A, internal knowledge management, onboarding docs, vendor contract management.

## License

This project is for educational purposes only. It was built as a personal/academic exploration of RAG architecture. Please review the [CUAD dataset license](https://www.atticusprojectai.org/cuad) and the terms of service for OpenAI, Cohere, and Pinecone before reusing this code with your own data or in production.

## Acknowledgments

- [CUAD Dataset](https://www.atticusprojectai.org/cuad) — Contract Understanding Atticus Dataset
- [RAGAS](https://github.com/explodinggradients/ragas) — Retrieval-Augmented Generation Assessment
- [OpenAI](https://openai.com/) — GPT-4 and text-embedding-3-large
- [Cohere](https://cohere.ai/) — rerank-english-v3.0
- [Pinecone](https://www.pinecone.io/) — Serverless vector database
