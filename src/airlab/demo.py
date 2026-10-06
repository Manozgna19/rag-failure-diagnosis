import json
from pathlib import Path
from .models import Document
from .pipeline import ReliabilityPipeline
from .analyzer import analyze

def load_docs():
    p=Path(__file__).resolve().parents[2]/"data/knowledge.json"
    return [Document(**x) for x in json.loads(p.read_text())]

def main():
    pipe=ReliabilityPipeline(load_docs())
    q="How can we measure whether a RAG answer is supported by evidence?"
    out=pipe.run(q,["d2"],5)
    print("\n=== AI Reliability Lab ===")
    print("QUERY:",q)
    print("\nANSWER:\n",out["answer"])
    print("\nRETRIEVAL:")
    for r in out["retrieved"]: print(f"  {r.rank}. {r.document.doc_id} {r.score:.4f} [{r.channel}]")
    print("\nMETRICS:")
    for k,v in out["evaluation"].metrics.items(): print(f"  {k}: {v}")
    print("\nROOT CAUSES:")
    for c in out["root_causes"]: print(f"  [{c.severity}] {c.category}: {c.category}")
    print("\nTRACE:")
    for s in out["evaluation"].trace.spans: print(f"  {s.name}: {s.latency_ms:.2f} ms")

if __name__=="__main__": main()
