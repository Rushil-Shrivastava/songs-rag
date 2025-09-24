from langfuse import observe
from typing import Dict, Any
import re

from tools import load_df

@observe()
def evaluate_response(user_query: str, response: str, retrieved_docs: list) -> Dict[str, Any]:
    """
    Evaluate chatbot response along 4 axes:
    - Tone
    - Factuality
    - Hallucination detection
    - RAG metrics (retrieval relevance)
    """

    df = load_df()
    all_titles = set(df["track_name"].str.lower())
    all_artists = set(df["track_artist"].str.lower())

    # --- Tone (simple heuristic, could be LLM-based too)
    tone = "neutral"
    if "sorry" in response.lower():
        tone = "apologetic"
    elif "!" in response:
        tone = "friendly"

    # --- Factuality & Hallucination
    hallucination = False
    factuality = "correct"

    # If bot gives info about song not in dataset → hallucination
    found_titles = [t for t in all_titles if t in response.lower()]
    found_artists = [a for a in all_artists if a in response.lower()]

    if not found_titles and not found_artists:
        hallucination = True
        factuality = "incorrect"

    # --- RAG metrics (very simplified relevance score)
    retrieved_texts = " ".join([d.get("metadata", {}).get("track_name", "").lower() for d in retrieved_docs])
    precision = len([t for t in found_titles if t in retrieved_texts]) / (len(found_titles) + 1e-6)
    recall = len([t for t in found_titles if t in retrieved_texts]) / (len(retrieved_docs) + 1e-6)

    return {
        "tone": tone,
        "factuality": factuality,
        "hallucination": hallucination,
        "rag_precision": round(precision, 2),
        "rag_recall": round(recall, 2),
    }
