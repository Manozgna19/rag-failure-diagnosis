# Advanced Build Roadmap

## Level 1 — current
- lexical retrieval
- semantic proxy retrieval
- RRF
- reranking
- retrieval metrics
- answer metrics
- traces
- root-cause analysis

## Level 2 — real embeddings
Replace `SemanticProxyRetriever` with Sentence Transformers. Compare retrieval Recall@K and MRR before/after.

## Level 3 — cross-encoder reranking
Use a cross-encoder to score `(query, document)` pairs. Measure whether reranking improves MRR/NDCG.

## Level 4 — real LLM
Replace the deterministic generator with an LLM. Keep the trace schema and compare model versions.

## Level 5 — LLM judge + human calibration
Evaluate faithfulness/relevance with an LLM judge, then calibrate against a human-labeled sample. Report inter-rater agreement and threshold trade-offs.

## Level 6 — production observability
Persist traces to PostgreSQL, add dashboards, cost accounting, OpenTelemetry and alerts.

## Level 7 — agent reliability
Add tool-call traces, tool success rate, invalid-tool-call rate, planning loops, and task-completion evaluation.

## Level 8 — safety
Add PII leakage tests, prompt-injection tests, harmful-output checks and a red-team benchmark.

## Level 9 — experiment science
Run controlled experiments over retrieval strategy, reranker, prompt and model. Report mean, standard deviation, confidence intervals and significance tests.
