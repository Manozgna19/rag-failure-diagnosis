import streamlit as st

from src.airlab.demo import load_docs
from src.airlab.pipeline import ReliabilityPipeline


st.set_page_config(
    page_title="RAG Failure Diagnosis",
    layout="wide"
)

st.title("RAG Failure Diagnosis System")
st.caption(
    "Evaluate a RAG pipeline, trace its stages, and diagnose likely failure causes."
)

pipe = ReliabilityPipeline(load_docs())

q = st.text_input(
    "Query",
    "How can we evaluate whether a RAG answer is supported by evidence?"
)

if st.button("Diagnose"):

    out = pipe.run(q)

    # -------------------------
    # Evaluation Metrics
    # -------------------------
    st.subheader("Evaluation Metrics")

    c1, c2, c3, c4 = st.columns(4)

    m = out["evaluation"].metrics

    c1.metric(
        "Faithfulness",
        f"{m['faithfulness']:.2f}"
    )

    c2.metric(
        "Context Relevance",
        f"{m['context_relevance']:.2f}"
    )

    c3.metric(
        "Answer Relevance",
        f"{m['answer_relevance']:.2f}"
    )

    c4.metric(
        "Recall@5",
        f"{m['recall_at_5']:.2f}"
    )

    # -------------------------
    # Diagnosis
    # -------------------------
    st.subheader("Diagnosis")

    for x in out["root_causes"]:

        message = (
            f"{x.severity.upper()} · "
            f"{x.category} · "
            f"confidence {x.confidence:.2f}"
        )

        if x.severity == "high":
            st.error(message)
        else:
            st.warning(message)

        st.write(
            "**Evidence:**",
            "; ".join(x.evidence)
        )

        st.write(
            "**Recommendation:**",
            x.recommendation
        )

    # -------------------------
    # Answer
    # -------------------------
    st.subheader("Answer")
    st.write(out["answer"])

    # -------------------------
    # Claim-level Evidence
    # -------------------------
    st.subheader("Claim-level Evidence")
    st.dataframe(
        out["claims"],
        use_container_width=True
    )

    # -------------------------
    # Retrieved Evidence
    # -------------------------
    st.subheader("Retrieved Evidence")

    for r in out["retrieved"]:

        st.write(
            f"**{r.rank}. {r.document.title}** · "
            f"{r.score:.4f} · "
            f"{r.channel}"
        )

        st.caption(r.document.text)

    # -------------------------
    # Trace
    # -------------------------
    st.subheader("Pipeline Trace")

    trace_data = [
        {
            "span": s.name,
            "latency_ms": round(s.latency_ms, 2),
            "attributes": s.attributes
        }
        for s in out["evaluation"].trace.spans
    ]

    st.json(trace_data)