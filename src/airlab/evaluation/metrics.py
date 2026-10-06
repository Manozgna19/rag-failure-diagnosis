import re

def _tokens(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

def context_relevance(query, contexts):
    q = _tokens(query)
    if not q or not contexts: return 0.0
    vals = []
    for c in contexts:
        vals.append(len(q & _tokens(c)) / len(q))
    return min(1.0, sum(vals) / len(vals))

def answer_relevance(query, answer):
    q = _tokens(query); a = _tokens(answer)
    return 0.0 if not q else min(1.0, len(q & a) / len(q))

def faithfulness(answer, contexts):
    a = _tokens(answer)
    c = set().union(*[_tokens(x) for x in contexts]) if contexts else set()
    if not a: return 0.0
    return len(a & c) / len(a)

def recall_at_k(retrieved_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)
    if not relevant: return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)

def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant: return 1.0 / i
    return 0.0

def hallucination_risk(answer, contexts):
    return round(1.0 - faithfulness(answer, contexts), 4)
