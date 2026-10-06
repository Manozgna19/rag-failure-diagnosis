# Architecture decisions

### Retrieval
Two independent signals are used: lexical TF-IDF and a semantic-proxy retriever. RRF merges rankings so that a document supported by both channels receives stronger rank evidence.

### Reranking
The reranker adds query-term overlap to the fused score. It is intentionally transparent so the candidate can explain every part of the scoring formula.

### Evaluation
Metrics are decomposed rather than collapsed into one score. This makes it possible to distinguish retrieval failure from generation/grounding failure.

### Observability
Every major stage is a trace span. A production implementation should replace the in-memory tracer with OpenTelemetry or another distributed tracing system.

### Root-cause analysis
The analyzer maps low metrics to likely failure categories and suggests targeted remediation. This is a rule-based diagnostic layer; future work can learn the mapping from historical traces.
