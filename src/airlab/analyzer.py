from dataclasses import dataclass
from .models import EvaluationResult

@dataclass
class RootCause:
    severity: str
    category: str
    cause: str
    recommendation: str

def analyze(result: EvaluationResult):
    out = []
    m = result.metrics
    if "retrieval_failure" in result.failures:
        out.append(RootCause("high", "retrieval", "Relevant evidence was not retrieved strongly enough.", "Improve chunking, query expansion, dense retrieval, or reranking."))
    if "grounding_failure" in result.failures:
        out.append(RootCause("high", "generation", "The answer contains unsupported content.", "Increase grounding constraints and add evidence-aware generation or verification."))
    if "answer_relevance_failure" in result.failures:
        out.append(RootCause("medium", "generation", "The response is weakly aligned with the user query.", "Improve prompt structure, query decomposition, and answer relevance evaluation."))
    if not out:
        out.append(RootCause("low", "none", "No major reliability failure was detected by the configured heuristics.", "Continue monitoring with a larger evaluation set."))
    return out
