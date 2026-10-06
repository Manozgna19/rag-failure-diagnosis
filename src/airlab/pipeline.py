from .tracing import Tracer
from .retrieval.retrievers import LexicalRetriever, SemanticProxyRetriever, reciprocal_rank_fusion, rerank
from .retrieval.advanced import DenseRetriever, CrossEncoderReranker
from .generator import grounded_demo_answer
from .evaluation.evaluator import evaluate
from .diagnosis import diagnose

class ReliabilityPipeline:
    def __init__(self, documents, advanced=False):
        self.documents=documents; self.advanced=advanced
        self.lexical=LexicalRetriever(documents); self.semantic=SemanticProxyRetriever(documents)
        self.dense=None; self.cross=None
        if advanced:
            try: self.dense=DenseRetriever(documents)
            except Exception: self.dense=None
            try: self.cross=CrossEncoderReranker()
            except Exception: self.cross=None

    def run(self, query, relevant_ids=None, top_k=5):
        tracer=Tracer(); relevant_ids=relevant_ids or []
        with tracer.span("retrieval.lexical", k=top_k): lexical=self.lexical.search(query,top_k)
        with tracer.span("retrieval.semantic_proxy", k=top_k): semantic=self.semantic.search(query,top_k)
        lists=[lexical,semantic]
        if self.dense:
            with tracer.span("retrieval.dense", k=top_k, model="all-MiniLM-L6-v2"): dense=self.dense.search(query,top_k)
            lists.append(dense)
        with tracer.span("retrieval.rrf_fusion", channels=len(lists)): fused=reciprocal_rank_fusion(lists)
        if self.cross:
            with tracer.span("retrieval.cross_encoder_rerank", model="ms-marco-MiniLM-L-6-v2"): retrieved=self.cross.rerank(query,fused,top_k)
        else:
            with tracer.span("retrieval.lexical_rerank"): retrieved=rerank(query,fused,top_k)
        with tracer.span("generation", model="deterministic-demo"): answer=grounded_demo_answer(query,retrieved)
        tracer.annotate(query=query,retrieved_ids=[r.document.doc_id for r in retrieved])
        evaluation,claims=evaluate(query,answer,retrieved,relevant_ids,tracer.trace)
        causes=diagnose(evaluation.metrics,claims)
        return {"answer":answer,"retrieved":retrieved,"evaluation":evaluation,"claims":claims,"root_causes":causes}
