from src.airlab.demo import load_docs
from src.airlab.pipeline import ReliabilityPipeline

def test_pipeline_returns_metrics_and_trace():
    out=ReliabilityPipeline(load_docs()).run("What is faithfulness in RAG?", ["d2"])
    assert "faithfulness" in out["evaluation"].metrics
    assert len(out["evaluation"].trace.spans) >= 4
    assert out["retrieved"]
