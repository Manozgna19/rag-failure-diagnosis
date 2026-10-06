from .models import RetrievedDocument

def grounded_demo_answer(query: str, retrieved: list[RetrievedDocument]) -> str:
    if not retrieved:
        return "I do not have enough evidence in the knowledge base to answer this question."
    # Deterministic teaching generator: extracts a concise evidence-backed response.
    lead = retrieved[0].document
    return f"Based on the retrieved evidence, {lead.title}: {lead.text}"
