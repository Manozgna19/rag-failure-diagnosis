# RAG Failure Diagnosis System

**An AI evaluation and root-cause analysis platform for diagnosing failures in Retrieval-Augmented Generation (RAG) systems.**

This is an independently implemented research/learning project. It is inspired by the general LLM evaluation and observability problem space, not a copy of any proprietary or third-party implementation.

## Why this project is different
A normal RAG demo asks: **"Can the model answer?"**
This project asks: **"If the answer is bad, why did it fail?"**

It measures retrieval quality, ranking quality, grounding, claim support, answer relevance and latency, then maps those signals to likely root causes and recommended fixes.

## Architecture
```text
Query
  |
  +--> Lexical Retrieval ----+
  |                          |
  +--> Semantic Retrieval ---+--> RRF Fusion --> Reranking --> Context
  |                          |
  +--> Dense Retrieval ------+       (optional advanced model)
                                      |
                                      v
                                  Generator
                                      |
                    +-----------------+------------------+
                    |                 |                  |
               Retrieval         Generation          Tracing
               evaluation        evaluation           spans
                    |                 |                  |
                    +-----------------+------------------+
                                      |
                              Failure Diagnosis
                                      |
                         Root cause + recommendation
```

## Metrics
- Recall@K
- Precision@K
- MRR
- NDCG@K
- Context relevance
- Answer relevance
- Faithfulness
- Claim support
- Hallucination risk
- End-to-end latency

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.airlab.demo
pytest -q
```

Optional API:
```bash
uvicorn app.api:app --reload
```

Optional dashboard:
```bash
streamlit run app/dashboard.py
```

## Advanced neural mode
Install the optional packages:
```bash
pip install sentence-transformers
```
The pipeline will use a Sentence Transformer and cross-encoder reranker when models are available. The offline baseline remains available without them.

## Project docs
- `docs/PROJECT_SPEC.md` — problem definition and failure taxonomy
- `docs/INTERVIEW_GUIDE.md` — interview concepts and limitations
- `docs/RESUME.md` — resume wording
- `docs/BUILD_ROADMAP.md` — future production upgrades

## Ethics / attribution
See `NOTICE.md`. Do not present third-party source code or an upstream repository as your original work. This implementation is independently written.
