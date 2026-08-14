# PRD — Low-Latency Voice-Enabled Dense RAG System

**Document Type:** Product Requirements Document  
**Version:** 1.1  
**Status:** Draft  
**Primary Dataset:** `ai4bharat/MSMARCO-XI`  
**Target End-to-End Latency:** `< 200 ms` for the retrieval path (see §22 — full-pipeline latency including hosted STT/LLM calls is not realistically demonstrable at this bound; see Known Risks below)

---

## 0. Changelog (v1.0 → v1.1)

1. **§22 Latency Requirements** — rewritten. v1.0 implied the full voice-to-answer pipeline would hit <200ms; that's not achievable with hosted Sarvam STT + hosted LLM generation given normal network round-trip time. v1.1 splits the target into a retrieval-path bound (achievable) and a full-pipeline number (measured and reported honestly, not forced to fit).
2. **New: Open Question on scope of the 200ms target.** The task brief's exact wording is ambiguous about whether STT is inside or outside the 200ms clock. Flagged for clarification before final submission — see §22.4.
3. **§25 Benchmark Plan** — added Experiment 0 (Chunking Strategy Comparison). v1.0 committed to one hybrid chunking approach without benchmarking it against alternatives, which is thinner than the treatment given to embeddings, HNSW, Top-K, and quantization elsewhere in the same section.

---

## 1. Product Overview

Build a voice-enabled Retrieval-Augmented Generation (RAG) system that accepts a user's spoken query, converts the speech to text, retrieves relevant information from the AI4Bharat/MSMARCO-XI corpus using dense vector retrieval, and generates a grounded answer.

The system must be designed around an aggressive latency target of less than 200 ms for the retrieval path (see §22 for why the full voice-to-answer number is reported separately rather than forced under the same bound).

### Core Pipeline

```text
User Voice
    ↓
Speech-to-Text (Sarvam)
    ↓
Text Query
    ↓
Multilingual Embedding
    ↓
Qdrant Vector DB + HNSW
    ↓
Top-K Relevant Chunks
    ↓
Context Builder
    ↓
Fast LLM
    ↓
Grounded Final Answer
```

The initial RAG architecture is **Dense Vector RAG**. Hybrid retrieval may be evaluated later as an optimization, but it is not part of the initial architecture.

---

# 2. Problem Statement

Users should be able to ask questions naturally through voice and receive answers based on the provided MSMARCO-XI knowledge corpus.

The challenge is to achieve all of the following simultaneously:

- Accurate semantic retrieval
- Multilingual query handling
- Large-scale vector search
- Grounded answer generation
- Low operational complexity
- Robust error handling
- End-to-end latency below 200 ms

The system must avoid unnecessary online processing because the latency requirement is extremely strict.

---

# 3. Goals

## 3.1 Primary Goals

1. Accept voice input from the user.
2. Convert voice to text using Sarvam STT.
3. Generate multilingual dense embeddings locally.
4. Store and search document/chunk embeddings using Qdrant.
5. Use HNSW for approximate nearest-neighbor retrieval.
6. Retrieve a small number of highly relevant chunks.
7. Generate concise answers using a low-latency LLM.
8. Prevent unsupported or ungrounded answers.
9. Provide structured error handling and retries.
10. Measure P50, P70 and P100 end-to-end latency.
11. Evaluate retrieval quality using appropriate retrieval metrics.

## 3.2 Secondary Goals

- Compare different embedding models.
- Tune HNSW parameters.
- Evaluate vector quantization.
- Benchmark Top-K values.
- Evaluate TOON versus compact JSON for context representation.
- Evaluate Hybrid Retrieval only if Dense Retrieval is insufficient.

---

# 4. Non-Goals

The first version will NOT:

- Use Agentic RAG.
- Use Multi-Query RAG.
- Use Self-RAG.
- Use CRAG.
- Use GraphRAG.
- Use multiple LLM reasoning calls.
- Use a heavyweight cross-encoder reranker.
- Train a new LLM from scratch.
- Re-embed the entire dataset during a user query.
- Make Hybrid RAG part of the baseline without benchmarking.

These approaches introduce additional latency and complexity that are not justified for the initial <200 ms target.

---

# 5. Dataset

## 5.1 Dataset

Primary dataset:

```text
ai4bharat/MSMARCO-XI
```

MSMARCO-XI is based on the MS MARCO retrieval/question-answering dataset and contains multilingual/Indic-language data.

The dataset contains query, answer, passage and relevance-related information that can be used for both retrieval corpus construction and evaluation.

## 5.2 Expected Scale

The project working assumption is approximately:

- 10M training records
- ~2M evaluation/testing records
- ~52 GB total data

The exact split usage must follow the evaluation setup defined for the project.

## 5.3 Important Principle

The dataset is processed **offline**.

The system must NOT perform:

```text
Raw dataset
→ chunk
→ embed
→ index
```

during a user request.

Instead:

```text
Offline:
Dataset → Chunk → Embed → Qdrant Index

Online:
User Query → Embed → Qdrant Search
```

---

# 6. RAG Architecture

## 6.1 Selected RAG Type

### Baseline: Dense Vector RAG

The system will use:

```text
Text
 ↓
Dense Embedding
 ↓
Vector
 ↓
Qdrant
 ↓
HNSW
 ↓
Nearest Neighbors
```

This is preferred as the initial architecture because it minimizes retrieval stages and therefore minimizes latency.

## 6.2 Optional Future Architecture

If Dense Retrieval does not provide sufficient recall:

```text
Dense Retrieval
       +
Sparse/BM25 Retrieval
       ↓
Score Fusion / RRF
       ↓
Top-K
```

This will be evaluated as **Hybrid RAG** only after establishing a strong Dense RAG baseline.

---

# 7. System Architecture

```text
                         ┌─────────────────────┐
                         │     USER VOICE      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Sarvam STT      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              Text Query
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ Input Validation /     │
                       │ Lightweight Guardrail  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Multilingual Embedding │
                       │       Model            │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Qdrant Vector Database │
                       │      + HNSW Index      │
                       └───────────┬────────────┘
                                   │
                                   ▼
                              Top-K Chunks
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Relevance / Threshold  │
                       │       Filtering        │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │    Context Builder     │
                       │    JSON / TOON test    │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │    Fast Generation     │
                       │          LLM           │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Grounding / Response   │
                       │       Validation       │
                       └───────────┬────────────┘
                                   │
                                   ▼
                            Final Response
```

---

# 8. Component Requirements

## 8.1 Speech-to-Text

### Technology

**Sarvam STT**

### Responsibility

Convert user speech into text.

```text
Audio → Sarvam → Text
```

### Requirements

- Support voice input.
- Support the relevant Indic/multilingual languages.
- Prefer streaming/low-latency interaction where supported.
- Measure actual STT latency.
- Handle transcription failure gracefully.

### Output

```json
{
  "text": "user query",
  "language": "detected language"
}
```

---

# 9. Query Processing

The query-processing layer should:

1. Validate the transcription.
2. Detect empty/invalid input.
3. Normalize only when necessary.
4. Preserve important names, IDs and terminology.
5. Forward the query to the embedding model.

Avoid expensive query rewriting because it adds latency.

---

# 10. Embedding Model

## 10.1 Requirement

Use a **small, fast multilingual embedding model**.

The exact model must be selected through benchmarking.

### Selection criteria

- Multilingual/Indic language support
- Retrieval quality
- Low inference latency
- Low memory footprint
- Suitable vector dimensionality
- Local inference support

### Recommended execution

Run the query embedding model locally using an optimized inference runtime such as ONNX Runtime/FastEmbed where supported.

### Offline

```text
Chunk → Embedding → Vector
```

### Online

```text
Query → Same Embedding Model → Query Vector
```

The same embedding model/configuration must be used for corpus and query embeddings.

---

# 11. Chunking Strategy

The project must not rely on one naive fixed-size chunking method.

## Baseline Strategy

Use:

**Sentence-aware + adaptive token-length + metadata-aware chunking**

### Process

```text
Raw Passage
    ↓
Sentence Detection
    ↓
Group Related Sentences
    ↓
Token Length Constraint
    ↓
Optional Small Overlap
    ↓
Chunk + Metadata
```

### Chunk Metadata

Each chunk should retain:

```text
chunk_id
document_id
query_id where applicable
language
source
position
token_count
original passage reference
```

Metadata should allow traceability back to the original dataset record.

---

# 12. Offline Indexing Pipeline

The complete indexing pipeline:

```text
MSMARCO-XI
    ↓
Load / Parse
    ↓
Clean / Normalize
    ↓
Extract Passages
    ↓
Adaptive Chunking
    ↓
Metadata Creation
    ↓
Batch Embedding
    ↓
Qdrant Upload
    ↓
HNSW Index
    ↓
Index Validation
```

This process is executed offline.

---

# 13. Vector Database

## Selected Technology

**Qdrant**

### Responsibility

Qdrant stores:

- Dense vectors
- Chunk identifiers
- Chunk text
- Metadata/payload

Conceptually:

```text
Vector
+
Payload
+
ID
```

### Search

```text
Query Vector
    ↓
Qdrant
    ↓
HNSW
    ↓
Top-K Similar Vectors
```

---

# 14. ANN Index

## Selected Index

**HNSW — Hierarchical Navigable Small World**

### Purpose

HNSW allows approximate nearest-neighbor search without exhaustively comparing the query against every vector.

### Parameters to benchmark

- `ef_search`
- `m`
- construction parameters
- quantization configuration

Do not assume a single parameter setting is optimal.

Benchmark multiple configurations against:

- latency
- recall
- memory

---

# 15. Retrieval

## Baseline

Dense vector similarity search.

### Initial Top-K

Start with:

```text
K = 5
```

Then benchmark:

```text
K = 5
K = 10
```

### Selection Criteria

Choose the smallest K that provides acceptable retrieval quality.

Larger K increases:

- context size
- LLM input tokens
- generation latency
- irrelevant context risk

---

# 16. Reranking

### Initial Decision

**No heavyweight reranker.**

Reason:

The total end-to-end latency target is <200 ms.

The baseline pipeline should be:

```text
Qdrant
 ↓
Top-K
 ↓
LLM
```

If retrieval quality is inadequate, evaluate a lightweight reranker separately.

---

# 17. Context Builder

The context builder converts retrieved chunks into an LLM-ready prompt.

### Responsibilities

- Sort retrieved chunks.
- Remove unnecessary metadata.
- Deduplicate similar content.
- Limit context size.
- Preserve source/chunk identifiers.
- Add grounding instructions.

### Example

```text
SYSTEM:
Answer only using the supplied context.
If the context does not contain enough evidence,
do not fabricate an answer.

CONTEXT:
[Chunk 1]
...

[Chunk 2]
...

QUESTION:
...
```

---

# 18. TOON

TOON is an optional context serialization format.

It is NOT:

- a vector database
- an embedding method
- a RAG method

It is only considered at:

```text
Retrieved Data
    ↓
TOON / JSON
    ↓
LLM
```

## Decision

Start with compact JSON.

Then benchmark:

```text
JSON vs TOON
```

Measure:

- token count
- context construction latency
- LLM latency
- answer quality

Use TOON only if it produces a measurable benefit.

---

# 19. Generation

## Requirement

Use a **small, low-latency instruction-following LLM**.

### Selection criteria

- Very low time-to-first-token
- Fast token generation
- Good instruction following
- Good grounded-answer behavior
- Local or low-latency hosted inference
- Small context requirement

### Generation constraints

- Keep prompts short.
- Retrieve only necessary chunks.
- Limit output length.
- Avoid unnecessary reasoning.
- Stream output where useful.

The exact model must be selected through latency and quality benchmarking.

---

# 20. Guardrails

The system must handle:

1. Off-topic queries
2. Unsafe/inappropriate inputs
3. Insufficient retrieval evidence
4. Hallucinated/ungrounded responses
5. Service failures

## Low-Latency Approach

Avoid a second expensive LLM call solely for guardrails.

Use:

```text
Input rules
+
retrieval relevance
+
structured output constraints
+
post-generation lightweight checks
```

### Insufficient evidence

If retrieval confidence is below a threshold determined experimentally:

```text
Do not generate unsupported answer.
```

Return a controlled fallback:

> "I couldn't find enough relevant information in the knowledge base to answer that."

The similarity threshold must be calibrated experimentally rather than assuming a universal value such as 0.65.

---

# 21. Harness / Orchestrator

The backend orchestrator controls the complete online workflow.

### Responsibilities

- Request validation
- STT invocation
- Query embedding
- Vector retrieval
- Context construction
- LLM invocation
- Guardrails
- Timeout handling
- Retry handling
- Structured input/output
- Logging
- Latency measurement
- Error recovery

### Pipeline

```text
Request
 ↓
Validate
 ↓
STT
 ↓
Embedding
 ↓
Retrieval
 ↓
Context
 ↓
Generation
 ↓
Validation
 ↓
Response
```

Each stage should have a timeout and structured error response.

---

# 22. Latency Requirements

## 22.1 Two Separate Numbers, Not One

The <200ms target is realistic for the **retrieval path** (embedding + Qdrant/HNSW search + context construction) because those stages can run on infrastructure we control (local/private embedding inference, local Qdrant instance). It is **not realistic** for the **full voice-to-answer pipeline**, because that pipeline includes two hosted, network-dependent calls we do not control the latency of:

- **Sarvam STT** — a cloud API call. Audio upload + processing + response typically runs well past 70ms for anything but trivially short clips, even before accounting for network conditions.
- **LLM generation** — even a "fast" hosted model has network round-trip plus time-to-first-token overhead that routinely exceeds the 50–80ms allocated in the original budget.

Committing to <200ms for the full pipeline in a submission and then failing to demonstrate it live is worse than being upfront about which number is bounded and which is measured. So this PRD reports both:

```text
RETRIEVAL-PATH TARGET   (embedding → Qdrant → context build)
Hard requirement: <200 ms
This is the number we engineer against and can credibly guarantee.

FULL-PIPELINE LATENCY   (voice → STT → retrieval → generation → answer)
Not a hard requirement. Measured, reported, and shown honestly
via P50/P70/P100 (see §24.1). Expected realistic range: several
hundred ms to low seconds, dominated by STT + LLM network hops —
not by anything in our control plane.
```

## 22.2 Retrieval-Path Target Budget

These are **engineering targets** for the portion of the pipeline we actually control:

```text
Query Embedding              10–20 ms
Qdrant HNSW Search             5–15 ms
Context + lightweight checks    2–5 ms
------------------------------------
Retrieval-path target         <200 ms   (comfortable headroom)
```

## 22.3 Full-Pipeline Reference Budget (informational, not a hard gate)

```text
STT (Sarvam, hosted)         150–500+ ms   (network + processing, audio-length dependent)
Query Embedding                10–20 ms
Qdrant HNSW Search               5–15 ms
Context + lightweight checks      2–5 ms
LLM generation (hosted)      200–800+ ms   (network + TTFT + decode)
Network / overhead               10–30 ms
------------------------------------------
Realistic full-pipeline total   ~400 ms – 1.5 s
```

These numbers are estimates based on typical hosted-API behavior, not guarantees — actual measurement (§24.1) is what goes in the submission, not this table.

## 22.4 Open Question — Needs Clarification Before Final Submission

The task brief says: *"The full process — chunking + vector DB retrieval + everything through to final output — should complete in under 200ms."* The wording starts the clock at chunking/retrieval, not explicitly at voice input, which is ambiguous:

- **Reading A (strict):** 200ms covers voice-in to answer-out, including STT and generation.
- **Reading B (literal):** 200ms covers only chunking + vector DB retrieval + downstream steps from there, with STT latency reported separately rather than counted against the bound.

This PRD builds for Reading B as the credible target (§22.2) and reports Reading A's number transparently without pretending it clears 200ms. **Recommend confirming with the Hacker House Goa organizers which reading is intended before finalizing the demo narrative** — it changes what you're allowed to claim in the submission video.

---

# 23. Latency Optimization Strategy

## Priority 1 — Reduce network hops

Prefer:

```text
Application
   ↓
Local/private Qdrant
```

rather than unnecessary public-network calls.

## Priority 2 — Local embedding

Avoid external embedding API calls if they threaten the latency budget.

## Priority 3 — Small retrieval set

Start with K=5.

## Priority 4 — Small LLM context

Only send relevant chunks.

## Priority 5 — Fast generation

Use a low-latency model/inference engine.

## Priority 6 — Avoid extra LLM calls

No query rewriting, reranking LLM, CRAG or separate grounding LLM in the baseline.

---

# 24. Evaluation

## 24.1 Latency Metrics

Measure:

- STT latency
- embedding latency
- Qdrant latency
- context construction latency
- generation latency
- total latency

Report:

- **P50**
- **P70**
- **P100**

across a reasonable set of test queries.

## 24.2 Retrieval Metrics

Recommended:

- Recall@K
- MRR
- nDCG

Use the relevance information available in the dataset where applicable.

## 24.3 Generation Metrics

Recommended:

- Grounded-answer rate
- Answer relevance
- Unsupported-answer rate
- Failure rate

---

# 25. Benchmark Plan

The system should be optimized experimentally.

### Experiment 0 — Chunking Strategy Comparison

v1.0 of this PRD committed to a single hybrid chunking approach (sentence-aware + adaptive token-length + metadata-aware, §11) without benchmarking it against alternatives. Every other major component in this section gets an A/B comparison — chunking should too, since the task brief explicitly calls out wanting to see multiple approaches compared, not just a single non-naive strategy asserted as correct.

Compare, on a fixed sample of the corpus:

```text
Strategy A — Fixed-size (baseline control, e.g. 256 tokens, no overlap)
Strategy B — Fixed-size with overlap (e.g. 256 tokens, 20% overlap)
Strategy C — Sentence-aware / semantic splitting
Strategy D — Adaptive + metadata-aware (this PRD's proposed baseline, §11)
```

Measure for each:

```text
Retrieval quality (Recall@K, MRR, nDCG — using dataset relevance labels)
Average chunk count / index size
Indexing time (offline cost)
Retrieval latency impact (if any)
```

### Selection Criteria

Pick the strategy with the best retrieval-quality-to-complexity tradeoff, not automatically the most sophisticated one. If Strategy D doesn't meaningfully beat Strategy B on Recall@K/nDCG, the added complexity isn't justified and a simpler approach should ship instead — this is worth having actual numbers for in the submission, since "we used adaptive metadata-aware chunking" without a comparison is an assertion, not evidence.

### Experiment 1 — Embedding Models

Compare:

```text
Model A
Model B
Model C
```

Measure:

```text
Retrieval quality
Embedding latency
Memory
```

### Experiment 2 — HNSW

Compare:

```text
ef_search = low
ef_search = medium
ef_search = high
```

Measure:

```text
Latency
Recall
```

### Experiment 3 — Top-K

Compare:

```text
K = 5
K = 10
```

### Experiment 4 — Quantization

Compare:

```text
Full precision
vs
Scalar/Binary Quantization
```

Measure:

```text
Memory
Latency
Recall
```

### Experiment 5 — Context Format

Compare:

```text
Compact JSON
vs
TOON
```

### Experiment 6 — Hybrid Retrieval

Only after Dense RAG baseline:

```text
Dense
vs
Dense + BM25
```

Choose Hybrid only if the quality improvement justifies additional latency/complexity.

---

# 26. Technology Stack

## Backend

Recommended:

```text
Python
FastAPI
```

## Dataset Processing

```text
Hugging Face Datasets
Python
```

## Embeddings

```text
Fast multilingual embedding model
ONNX Runtime / FastEmbed where appropriate
```

## Vector DB

```text
Qdrant
```

## ANN

```text
HNSW
```

## Speech

```text
Sarvam STT
```

## Generation

```text
Low-latency instruction LLM
```

## Frontend

A lightweight web interface with:

- microphone input
- transcription display
- answer display
- latency information
- error/fallback display

---

# 27. Project Structure

```text
voice-rag/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── preprocessing/
│   ├── loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   └── metadata.py
│
├── embeddings/
│   ├── model.py
│   └── embed.py
│
├── indexing/
│   ├── qdrant_client.py
│   ├── create_collection.py
│   └── index_dataset.py
│
├── retrieval/
│   ├── retriever.py
│   ├── filters.py
│   └── ranking.py
│
├── speech/
│   └── sarvam_stt.py
│
├── generation/
│   ├── llm.py
│   └── prompt.py
│
├── guardrails/
│   ├── input_guard.py
│   └── grounding.py
│
├── orchestration/
│   └── pipeline.py
│
├── evaluation/
│   ├── latency.py
│   ├── retrieval_metrics.py
│   └── benchmark.py
│
├── api/
│   └── main.py
│
├── frontend/
│
├── configs/
│   └── config.yaml
│
└── README.md
```

---

# 28. End-to-End Request Flow

### Step 1

User speaks.

```text
"What is the best way to improve sleep?"
```

### Step 2

Sarvam converts speech to text.

### Step 3

The multilingual embedding model converts the query into a dense vector.

### Step 4

Qdrant searches its HNSW index.

### Step 5

Top 5–10 relevant chunks are returned.

### Step 6

The context builder prepares a compact LLM prompt.

### Step 7

The fast LLM generates an answer using the retrieved evidence.

### Step 8

Lightweight grounding/validation checks are applied.

### Step 9

The final answer is returned.

---

# 29. Failure Handling

## STT Failure

Return:

```text
Unable to understand the audio. Please try again.
```

## Empty Query

Return:

```text
Please provide a question.
```

## Qdrant Failure

Return:

```text
The knowledge retrieval service is temporarily unavailable.
```

## No Relevant Context

Return:

```text
I couldn't find enough relevant information in the knowledge base.
```

## LLM Failure

Retry once if the retry fits the configured timeout budget; otherwise return a controlled error.

All failures must be logged.

---

# 30. Security and Reliability

The system should:

- validate input size
- limit audio duration
- limit query length
- sanitize metadata
- prevent prompt injection from retrieved content
- avoid exposing internal errors
- enforce generation limits
- log failures without storing unnecessary sensitive user data

---

# 31. Success Criteria

The project is considered successful when:

### Functional

- Voice input works.
- Speech is converted to text.
- Relevant passages are retrieved.
- Answers are generated from retrieved context.
- Unsupported queries receive controlled fallbacks.
- System handles failures gracefully.

### Retrieval

- Dense retrieval achieves acceptable Recall@K/MRR/nDCG.
- Retrieval quality is validated using available relevance information.

### Performance

- Retrieval-path latency is demonstrated against the **<200 ms target** (§22.2).
- Full-pipeline latency is measured and reported honestly (§22.3), without claiming a bound the hosted STT/LLM calls can't credibly meet.
- P50, P70 and P100 are reported.
- Individual component latency is reported.

### Engineering

- Offline indexing is reproducible.
- Online pipeline is modular.
- Harness provides structured orchestration.
- Errors and retries are handled.
- Guardrails are implemented.

---

# 32. Final Recommended Architecture

```text
                         OFFLINE
                            │
                  AI4Bharat/MSMARCO-XI
                            │
                            ▼
                    Adaptive Chunking
                            │
                            ▼
                  Multilingual Embedding
                            │
                            ▼
                         Qdrant
                            │
                            ▼
                           HNSW
                            │
                            ▼
                     SEARCH INDEX
                            │
════════════════════════════╪════════════════════════════
                            │
                          ONLINE
                            │
                       User Voice
                            │
                            ▼
                       Sarvam STT
                            │
                            ▼
                       Text Query
                            │
                            ▼
                 Input Validation
                            │
                            ▼
                Multilingual Embedding
                            │
                            ▼
                    Qdrant + HNSW
                            │
                            ▼
                         Top-K
                            │
                            ▼
                  Relevance Filtering
                            │
                            ▼
                    Context Builder
                            │
                            ▼
                     Fast LLM
                            │
                            ▼
                  Grounding Validation
                            │
                            ▼
                     Final Answer
```

---

# 33. Final Technology Decision

| Component | Final Initial Choice |
|---|---|
| Dataset | `ai4bharat/MSMARCO-XI` |
| Speech-to-Text | **Sarvam STT** |
| RAG | **Dense Vector RAG** |
| Vector DB | **Qdrant** |
| ANN Index | **HNSW** |
| Embedding | **Fast multilingual local model — benchmark before finalizing** |
| Chunking | **Adaptive sentence/semantic + metadata-aware** |
| Retrieval | **Top-K, start with 5** |
| Reranker | **None initially** |
| Context | **Compact JSON initially** |
| TOON | **Optional benchmark** |
| Generation | **Small/fast instruction model — benchmark before finalizing** |
| Guardrails | **Lightweight deterministic + retrieval-based** |
| Orchestration | **FastAPI/Python backend harness** |
| Evaluation | **P50/P70/P100 + retrieval + grounding metrics** |
| Hybrid RAG | **Future experiment, not baseline** |

---

# 34. Key Design Principle

The system should follow this rule:

> **Do expensive work offline. Keep the online path extremely short.**

### Offline

```text
52 GB
 ↓
Chunk
 ↓
Embed
 ↓
Index
```

### Online

```text
Voice
 ↓
STT
 ↓
Embed
 ↓
Search
 ↓
Generate
```

Every additional online model call, retrieval stage, network hop or reranking stage must justify its latency cost.

**Baseline architecture: Dense Vector RAG + Qdrant + HNSW.**

Hybrid retrieval, reranking, TOON, quantization and alternative databases are optimization experiments—not assumptions that should be added before benchmarking.
