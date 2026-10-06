# RAG Failure Diagnosis System

## Problem
RAG systems can fail for different reasons: the right evidence may not be retrieved, ranking may be poor, the context may be irrelevant, or the generator may produce unsupported claims. A single answer-quality score does not identify the cause.

## Goal
Build an evaluation layer that turns an observed bad RAG response into a measurable diagnosis.

## Failure taxonomy
1. Retrieval miss — relevant evidence was not in the top-k.
2. Retrieval irrelevance — retrieved passages do not match the query.
3. Ranking failure — relevant evidence exists but is poorly ranked.
4. Grounding failure — generated answer contains unsupported content.
5. Answer relevance failure — answer does not address the query.
6. Latency regression — one pipeline stage dominates response time.
7. Unsupported claim — individual claims lack evidence.

## Core experiment
Compare retrieval configurations:
- TF-IDF lexical
- semantic proxy
- hybrid RRF
- hybrid + neural reranking (optional dependency)

Record Recall@K, Precision@K, MRR, NDCG, context relevance, faithfulness, claim support, latency.

## Important limitation
The offline implementation intentionally uses transparent lexical heuristics and a deterministic generator so it can run without API keys. These are teaching/evaluation baselines, not replacements for a production semantic judge. The advanced path adds Sentence Transformers and a cross-encoder when those packages/models are available.
