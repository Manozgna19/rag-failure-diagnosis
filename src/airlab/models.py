from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievedDocument:
    document: Document
    score: float
    rank: int
    channel: str

@dataclass
class TraceSpan:
    name: str
    start_ms: float
    end_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return self.end_ms - self.start_ms

@dataclass
class Trace:
    trace_id: str
    spans: list[TraceSpan] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    failures: list[str]
    explanations: list[str]
    trace: Trace
