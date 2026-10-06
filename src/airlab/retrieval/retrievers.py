from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..models import Document, RetrievedDocument

class LexicalRetriever:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def search(self, query: str, k: int = 5):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        idx = np.argsort(-scores)[:k]
        return [RetrievedDocument(self.docs[i], float(scores[i]), r+1, "lexical") for r, i in enumerate(idx)]

class SemanticProxyRetriever:
    """Teaching substitute for a dense encoder. Replace with SentenceTransformer in production."""
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english", analyzer="word", ngram_range=(1, 3))
        self.matrix = self.vectorizer.fit_transform([d.title + " " + d.text for d in docs])

    def search(self, query: str, k: int = 5):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        idx = np.argsort(-scores)[:k]
        return [RetrievedDocument(self.docs[i], float(scores[i]), r+1, "semantic_proxy") for r, i in enumerate(idx)]

def reciprocal_rank_fusion(lists: list[list[RetrievedDocument]], k: int = 60):
    scores = defaultdict(float)
    by_id = {}
    for results in lists:
        for item in results:
            scores[item.document.doc_id] += 1.0 / (k + item.rank)
            by_id[item.document.doc_id] = item.document
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [RetrievedDocument(by_id[doc_id], score, rank+1, "rrf") for rank, (doc_id, score) in enumerate(ranked)]

def rerank(query: str, results: list[RetrievedDocument], top_k: int = 5):
    q_terms = set(query.lower().split())
    rescored = []
    for item in results:
        words = set(item.document.text.lower().split())
        overlap = len(q_terms & words) / max(1, len(q_terms))
        score = 0.7 * item.score + 0.3 * overlap
        rescored.append((score, item.document))
    rescored.sort(key=lambda x: x[0], reverse=True)
    return [RetrievedDocument(doc, float(score), i+1, "reranked") for i, (score, doc) in enumerate(rescored[:top_k])]
