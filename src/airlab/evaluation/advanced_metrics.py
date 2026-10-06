"""Evaluation metrics for retrieval and grounded generation."""
import math
import re
from collections import Counter


def normalize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def precision_at_k(retrieved_ids, relevant_ids, k=5):
    rel = set(relevant_ids)
    if k <= 0: return 0.0
    return sum(x in rel for x in retrieved_ids[:k]) / k


def ndcg_at_k(retrieved_ids, relevant_ids, k=5):
    rel = set(relevant_ids)
    gains = [1 if x in rel else 0 for x in retrieved_ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def claim_support(answer, contexts):
    """Transparent claim-level support heuristic; not an LLM judge."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    ctx = set(normalize(" ".join(contexts)))
    if not sentences: return 0.0, []
    details=[]
    for s in sentences:
        toks=normalize(s)
        if not toks:
            continue
        overlap=sum(t in ctx for t in toks)/len(toks)
        details.append({"claim":s,"support":round(overlap,4),"supported":overlap>=0.55})
    score=sum(x["support"] for x in details)/len(details) if details else 0.0
    return score, details


def evidence_coverage(answer, contexts):
    a=set(normalize(answer)); c=set(normalize(" ".join(contexts)))
    return len(a & c)/len(a) if a else 0.0


def semantic_proxy_similarity(a, b):
    av=Counter(normalize(a)); bv=Counter(normalize(b))
    vocab=set(av)|set(bv)
    if not vocab: return 0.0
    dot=sum(av[x]*bv[x] for x in vocab)
    na=math.sqrt(sum(v*v for v in av.values())); nb=math.sqrt(sum(v*v for v in bv.values()))
    return dot/(na*nb) if na and nb else 0.0
