# Interview Guide — RAG Failure Diagnosis System

## 1. What problem does the project solve?
It diagnoses *why* a RAG system produced a poor answer rather than only assigning a quality score. It separates retrieval, ranking, grounding, answer relevance and latency failures.

## 2. Why hybrid retrieval?
Lexical retrieval is strong for exact terminology; dense retrieval is stronger for semantic similarity. Combining their rankings can improve recall across different query types.

## 3. What is RRF?
Reciprocal Rank Fusion combines ranked lists using a score proportional to 1/(constant + rank). It avoids requiring the component retrievers to have calibrated scores.

## 4. Why rerank after retrieval?
Initial retrieval should optimize recall cheaply. A reranker can spend more computation on a smaller candidate set and optimize ordering for the query-document pair.

## 5. Recall@K vs MRR vs NDCG?
Recall@K asks whether relevant items were retrieved. MRR rewards how early the first relevant item appears. NDCG supports graded ranking quality and discounts lower positions.

## 6. What is faithfulness?
Whether the generated answer is supported by the supplied context. In this project the offline score is a transparent token-support heuristic; a production system should use stronger semantic or LLM-based verification and calibration.

## 7. How do you diagnose a failure?
Use evidence from metrics and traces. Low Recall@K suggests retrieval miss; low context relevance suggests irrelevant retrieval; low faithfulness with good retrieval suggests a generation/grounding problem; high latency points to the slowest trace span.

## 8. Why not use accuracy?
RAG evaluation is not a simple classification problem. Different dimensions need different metrics and human/LLM judgment.

## 9. What would you improve?
Use real sentence-transformer embeddings, a cross-encoder reranker, pgvector, LLM-as-a-judge with calibration, OpenTelemetry, statistical significance tests, and agent/tool-call evaluation.
