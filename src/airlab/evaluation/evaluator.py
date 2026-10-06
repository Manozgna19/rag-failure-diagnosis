from .metrics import context_relevance, answer_relevance, faithfulness, recall_at_k, reciprocal_rank, hallucination_risk
from .advanced_metrics import precision_at_k, ndcg_at_k, claim_support
from ..models import EvaluationResult

def evaluate(query, answer, retrieved, relevant_ids, trace):
    contexts=[r.document.text for r in retrieved]; ids=[r.document.doc_id for r in retrieved]
    metrics={
      "context_relevance":round(context_relevance(query,contexts),4),
      "answer_relevance":round(answer_relevance(query,answer),4),
      "faithfulness":round(faithfulness(answer,contexts),4),
      "hallucination_risk":hallucination_risk(answer,contexts),
      "recall_at_5":round(recall_at_k(ids,relevant_ids,5),4),
      "precision_at_5":round(precision_at_k(ids,relevant_ids,5),4),
      "mrr":round(reciprocal_rank(ids,relevant_ids),4),
      "ndcg_at_5":round(ndcg_at_k(ids,relevant_ids,5),4),
      "latency_ms":round(sum(s.latency_ms for s in trace.spans),2),
      "relevant_ids_available":bool(relevant_ids),
    }
    score, claims=claim_support(answer,contexts)
    metrics["claim_support"]=round(score,4)
    failures=[]; explanations=[]
    if metrics["faithfulness"]<.60: failures.append("grounding_failure"); explanations.append("Answer contains weakly supported content.")
    if metrics["context_relevance"]<.35: failures.append("retrieval_failure"); explanations.append("Retrieved context has weak query overlap.")
    if metrics["answer_relevance"]<.35: failures.append("answer_relevance_failure"); explanations.append("Answer weakly addresses the query.")
    if claims and any(not c["supported"] for c in claims): failures.append("unsupported_claim")
    return EvaluationResult(metrics, failures, explanations, trace), claims
