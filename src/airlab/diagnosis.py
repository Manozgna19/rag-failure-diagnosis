"""Root-cause diagnosis for RAG failures."""
from dataclasses import dataclass, asdict

@dataclass
class Diagnosis:
    category: str
    severity: str
    confidence: float
    evidence: list[str]
    recommendation: str

    def to_dict(self): return asdict(self)


def diagnose(metrics, claims=None):
    d=[]
    recall=metrics.get("recall_at_5", 0)
    context=metrics.get("context_relevance", 0)
    faith=metrics.get("faithfulness", 0)
    answer=metrics.get("answer_relevance", 0)
    latency=metrics.get("latency_ms", 0)

    if recall < 0.5 and metrics.get("relevant_ids_available", False):
        d.append(Diagnosis("retrieval_miss", "high", round(1-recall,2),
            [f"Recall@5={recall:.2f}"],
            "Improve query rewriting, chunking, dense retrieval, hybrid fusion, or reranking."))
    if context < 0.35:
        d.append(Diagnosis("retrieval_irrelevance", "high", round(1-context,2),
            [f"Context relevance={context:.2f}"],
            "Inspect retrieved chunks and add dense retrieval or a stronger reranker."))
    if faith < 0.60:
        d.append(Diagnosis("grounding_failure", "high", round(1-faith,2),
            [f"Faithfulness={faith:.2f}"],
            "Require evidence-backed generation and verify claims before returning the answer."))
    if answer < 0.35:
        d.append(Diagnosis("answer_relevance", "medium", round(1-answer,2),
            [f"Answer relevance={answer:.2f}"],
            "Improve prompt structure, query decomposition, and answer-length control."))
    if latency > 1500:
        d.append(Diagnosis("latency", "medium", min(1, latency/5000),
            [f"Latency={latency:.0f}ms"],
            "Profile retrieval/reranking/generation separately and cache embeddings or results."))
    if claims:
        unsupported=sum(not c["supported"] for c in claims)
        if unsupported:
            d.append(Diagnosis("unsupported_claim", "high", min(1, unsupported/len(claims)),
                [f"{unsupported}/{len(claims)} claims below support threshold"],
                "Add claim-level evidence verification and reject or qualify unsupported claims."))
    return d or [Diagnosis("healthy", "low", 0.95, ["No configured failure threshold was crossed"], "Continue monitoring on a larger evaluation set.")]
