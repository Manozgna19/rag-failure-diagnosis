"""Optional production-style retrieval components.

The project runs without heavyweight models. If sentence-transformers / cross-encoder
are installed, these classes provide true dense retrieval and neural reranking.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:  # optional dependency
    SentenceTransformer = None
    CrossEncoder = None

@dataclass
class RetrievalConfig:
    top_k: int = 5
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class DenseRetriever:
    def __init__(self, documents, model_name=None):
        if SentenceTransformer is None:
            raise RuntimeError("Install sentence-transformers to use DenseRetriever")
        self.documents = documents
        self.model = SentenceTransformer(model_name or RetrievalConfig.dense_model)
        self.embeddings = self.model.encode([d.title + " " + d.text for d in documents], normalize_embeddings=True)

    def search(self, query, k=5):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.asarray(self.embeddings) @ q
        idx = np.argsort(-scores)[:k]
        from ..models import RetrievedDocument
        return [RetrievedDocument(self.documents[i], float(scores[i]), r+1, "dense") for r, i in enumerate(idx)]

class CrossEncoderReranker:
    def __init__(self, model_name=None):
        if CrossEncoder is None:
            raise RuntimeError("Install sentence-transformers to use CrossEncoderReranker")
        self.model = CrossEncoder(model_name or RetrievalConfig.reranker_model)

    def rerank(self, query, results, top_k=5):
        pairs = [(query, r.document.title + " " + r.document.text) for r in results]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, results), key=lambda x: float(x[0]), reverse=True)[:top_k]
        from ..models import RetrievedDocument
        return [RetrievedDocument(r.document, float(score), i+1, "cross_encoder") for i, (score, r) in enumerate(ranked)]
