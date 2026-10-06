from dataclasses import dataclass
from statistics import mean
from .pipeline import ReliabilityPipeline

@dataclass
class ExperimentRun:
    name: str
    metrics: dict[str, float]

def compare(pipeline, cases, top_k=5):
    runs=[]
    for name, variant in cases:
        vals=[]
        for case in variant:
            out=pipeline.run(case["query"], case.get("relevant_ids", []), top_k)
            vals.append(out["evaluation"].metrics)
        keys=vals[0].keys() if vals else []
        runs.append(ExperimentRun(name, {k: round(mean(x[k] for x in vals),4) for k in keys}))
    return runs
