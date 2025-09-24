import json
from typing import Dict, Any
import openai
from config import PERPLEXITY_API_KEY, PERPLEXITY_API_URL, LANGFUSE_SECRET_API_KEY, LANGFUSE_PUBLIC_API_KEY
from tools import load_df, recommend_by_mood, retrieve, find_by_name, songs_by_genre, build_index, compute_from_dataset
from db import add_favorite, list_favorites, favorites_exists
from evaluation import evaluate_response
from openai import OpenAI
import re

client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url=PERPLEXITY_API_URL)

from langfuse import Langfuse
from langfuse import observe

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_API_KEY,
    secret_key=LANGFUSE_SECRET_API_KEY,
    host="https://cloud.langfuse.com" 
)

TOOLS = [
    "search",
    "compute_from_dataset",
    "details",
    "favorites_add",
    "favorites_list",
    "recommend_genre",
    "recommend_mood",
    "upsert_index"
]

INTENT_PROMPT = """
You are an intent parser. Convert the user's query into a JSON object selecting one tool.
You MUST output only valid JSON. Do not include explanations or extra text.

Rules:
- Default limit = 5 if not specified.
- Aggregations allowed: ["max","min","mean","median","sum","top","bottom"].
- Always use compute_from_dataset for queries about "most", "least", "happiest", "saddest", "top N".
- Use search only if the query is free-form (like "tell me about Taylor Swift" or "find Bohemian Rhapsody").

Examples:
User: "Tell me the most popular song"
Output: {"tool": "compute_from_dataset", "params": {"column": "track_popularity", "agg": "max", "limit": 1}}

User: "Which is the happiest song?"
Output: {"tool": "compute_from_dataset", "params": {"column": "valence", "agg": "max", "limit": 1}}

User: "Give me the saddest song"
Output: {"tool": "compute_from_dataset", "params": {"column": "valence", "agg": "min", "limit": 1}}

User: "Top 5 energetic songs"
Output: {"tool": "compute_from_dataset", "params": {"column": "energy", "agg": "top", "limit": 5}}

User: "Top 10 most popular songs"
Output: {"tool": "compute_from_dataset", "params": {"column": "track_popularity", "agg": "top", "limit": 10}}

User: "Tell me about a good song"
Output: {"tool": "search", "params": {"q": "Tell me about a good song", "top_k": 4}}

Available tools:
- search: params { "q": string, "top_k": int }
- compute_from_dataset: params { "column": string, "agg": string, "limit": int, "filter": {...} }
- details: params { "name": string, "artist": string? }
- favorites_add: params { "by_name": string?, "by_genre": string? }
- favorites_list: params {}
- recommend_genre: params { "genre": string, "limit": int }
- recommend_mood: params { "mood": string, "limit": int }
- upsert_index: no params
- stats: no params
- random_song: no params
"""

def parse_intent(user_query: str) -> Dict[str, Any]:
    print(f"[parse_intent] User query: {user_query}")

    resp = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "system", "content": "You output only valid JSON intents for tools."},
            {"role": "user", "content": INTENT_PROMPT + "\nUser query:\n" + user_query}
        ],
        temperature=0,
        max_tokens=200,
    )

    text = resp.choices[0].message.content.strip()
    print(f"[parse_intent] Extracted text (raw): {text}")

    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip())
        text = text.strip()
        print(f"[parse_intent] Cleaned text: {text}")

    try:
        obj = json.loads(text)
        print(f"[parse_intent] Parsed JSON: {obj}")

        if obj.get("tool") not in TOOLS:
            print("[parse_intent] Tool not in TOOLS → defaulting to search")
            return {"tool": "search", "params": {"q": user_query, "top_k": 4}}

        return obj
    except Exception as e:
        print(f"[parse_intent] JSON parsing failed: {e}")
        return {"tool": "search", "params": {"q": user_query, "top_k": 4}}

@observe()
def execute_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    tool = intent.get("tool")
    params = intent.get("params", {}) or {}

    if tool == "search":
        return {
            "tool": "search",
            "results": retrieve(params.get("q", ""), params.get("top_k", 4)),
        }

    if tool == "compute_from_dataset":
        return {
            "tool": "compute_from_dataset",
            "result": compute_from_dataset(
                column=params.get("column"),
                agg=params.get("agg", "max"),
                limit=params.get("limit", 5),
                filters=params.get("filter", {}),
            ),
        }

    if tool == "details":
        return {
            "tool": "details",
            "result": find_by_name(params.get("name", ""), artist=params.get("artist")),
        }

    if tool == "favorites_add":
        added = []
        if "by_name" in params:
            song = find_by_name(params["by_name"])
            if song:
                if not favorites_exists(song["track_id"]):
                    add_favorite(song["track_id"])
                    added.append({"track_id": song["track_id"], "track_name": song["track_name"], "track_artist": song["track_artist"]})
                else:
                    added.append({"track_id": song["track_id"], "track_name": song["track_name"], "track_artist": song["track_artist"]})
        if "by_genre" in params:
            for s in songs_by_genre(params["by_genre"], 50):
                if not favorites_exists(s["track_id"]):
                    add_favorite(s["track_id"])
                    added.append(s["track_id"])
        return {"tool": "favorites_add", "added": added}

    if tool == "favorites_list":
        fav_ids = list_favorites()
        df = load_df()
        favs = df[df["track_id"].isin([f["track_id"] for f in fav_ids])][["track_id", "track_name", "track_artist"]]
        return {
            "tool": "favorites_list",
            "favorites": favs.to_dict(orient="records"),
        }

    if tool == "recommend_genre":
        return {
            "tool": "recommend_genre",
            "recommendations": songs_by_genre(
                params.get("genre", ""), params.get("limit", 5)
            ),
        }

    if tool == "recommend_mood":
        return {
            "tool": "recommend_mood",
            "recommendations": recommend_by_mood(
                params.get("mood", ""), 
                params.get("limit", 5)
            ),
        }

    if tool == "upsert_index":
        return {"tool": "upsert_index", "count": build_index()}

    if tool == "stats":
        df = load_df()
        return {"tool": "stats", "rows": len(df), "columns": list(df.columns)}

    if tool == "random_song":
        df = load_df()
        row = df.sample(1).iloc[0].to_dict()
        return {"tool": "random_song", "result": row}

    return {"tool": tool, "error": "unknown"}

@observe()
def generate_answer(tool_result: Dict[str, Any], user_query: str = "") -> str:
    """
    Use LLM to generate a natural, formatted answer from tool results.
    """
    t = tool_result.get("tool")

    context = ""
    if t == "search":
        context = f"Search results: {tool_result.get('results', [])}"
    elif t == "compute_from_dataset":
        context = f"Computation result: {tool_result.get('result', {})}"
    elif t == "details":
        context = f"Song details: {tool_result.get('result', {})}"
    elif t == "favorites_add":
        context = f"Favorites added: {tool_result.get('added', [])}"
    elif t == "favorites_list":
        context = f"Favorites list: {tool_result.get('favorites', [])}"
    elif t.startswith("recommend"):
        context = f"Recommendations: {tool_result.get('recommendations', [])}"
    elif t == "upsert_index":
        context = f"Indexed {tool_result.get('count', 0)} songs."
    elif t == "stats":
        context = f"Dataset stats: {tool_result}"
    elif t == "random_song":
        context = f"Random song: {tool_result.get('result', {})}"
    else:
        context = str(tool_result)

    prompt = f"""
You are a helpful music assistant. 
The user asked: "{user_query}"

Here are raw tool results (JSON):
{context}

Instructions:
- ONLY use the data provided above.
- If the results list is empty or irrelevant, respond with exactly: "I don't have any info about it."
- Do not invent or hallucinate songs outside this dataset.
- Format nicely (use lists or tables if multiple songs).
- Highlight key attributes (name, artist, popularity, mood).
- Keep it concise and user-friendly.
"""

    resp = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "system", "content": "You are a music assistant that formats results beautifully."},
            {"role": "user", "content": prompt}
        ],
        extra_body={"disable_search": True},
        temperature=0.7,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()

@observe()
def handle_query(user_query: str) -> Dict[str, Any]:
    intent = parse_intent(user_query)
    result = execute_intent(intent)
    answer = generate_answer(result, user_query)

    eval_result = evaluate_response(
        user_query=user_query,
        response=answer,
        retrieved_docs=result.get("results", []) 
    )

    return {
        "answer": answer,
        "intent": intent,
        "tool_result": result,
        "evaluation": eval_result
    }