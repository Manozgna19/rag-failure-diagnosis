import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.airlab.models import Document
from src.airlab.pipeline import ReliabilityPipeline

app=FastAPI(title="RAG Failure Diagnosis API",version="1.0.0")
DATA=Path(__file__).resolve().parents[1]/"data/knowledge.json"
docs=[Document(**x) for x in json.loads(DATA.read_text())]
pipeline=ReliabilityPipeline(docs, advanced=False)

class Query(BaseModel):
    query:str=Field(min_length=3)
    relevant_ids:list[str]=[]
    top_k:int=Field(default=5, ge=1, le=20)

@app.get("/health")
def health(): return {"status":"ok","documents":len(docs),"mode":"offline-deterministic"}

@app.post("/diagnose")
def diagnose_query(q:Query):
    out=pipeline.run(q.query,q.relevant_ids,q.top_k); ev=out["evaluation"]
    return {"answer":out["answer"],"retrieved":[{"id":r.document.doc_id,"title":r.document.title,"score":r.score,"channel":r.channel} for r in out["retrieved"]],"metrics":ev.metrics,"failures":ev.failures,"claims":out["claims"],"root_causes":[c.to_dict() for c in out["root_causes"]],"trace":[{"name":s.name,"latency_ms":s.latency_ms,"attributes":s.attributes} for s in ev.trace.spans]}
