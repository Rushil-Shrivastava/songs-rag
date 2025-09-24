import pandas as pd
from typing import List, Dict, Optional
from rapidfuzz import process
import numpy as np
import os

from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import Settings

Settings.llm = None

from config import SONGS_CSV, COHERE_API_KEY, TOP_K
from db import list_favorites

# ----------------- Features & Aliases -----------------
FEATURE_COLUMNS = ["valence", "energy", "danceability", "tempo"]

COLUMN_ALIASES = {
    "popularity": "track_popularity",
    "track popularity": "track_popularity",
    "energy": "energy",
    "valence": "valence",
    "happiness": "valence",
    "danceability": "danceability",
    "acousticness": "acousticness",
    "loudness": "loudness",
    "tempo": "tempo",
    "duration": "duration_ms",
    "duration_ms": "duration_ms",
}

# ----------------- Index Setup -----------------
STORAGE_DIR = "./storage"

embed_model = CohereEmbedding(api_key=COHERE_API_KEY, model_name="embed-english-v3.0")

def build_index():
    """
    Build a fresh LlamaIndex from songs.csv and persist it.
    """
    if not SONGS_CSV.exists():
        raise FileNotFoundError("songs.csv not found")

    df = pd.read_csv(SONGS_CSV, dtype=str).fillna("")
    documents = []
    for _, row in df.iterrows():
        text = f"""
        Track: {row['track_name']}
        Artist: {row['track_artist']}
        Genre: {row['playlist_genre']}
        Album: {row['track_album_name']}
        Year: {row['track_album_release_date']}
        Popularity: {row['track_popularity']}
        Valence: {row['valence']}
        Energy: {row['energy']}
        """
        documents.append(Document(text=text, metadata={"track_id": row["track_id"]}))

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, llm=None)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index

def load_or_build_index():
    """
    Load index from storage if exists, else build it.
    """
    if os.path.exists(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context, embed_model=embed_model, llm=None)
    return build_index()

INDEX = load_or_build_index()

# ----------------- Retrieval -----------------
def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    try:
        query_engine = INDEX.as_query_engine(similarity_top_k=top_k, llm=None)
        response = query_engine.query(query)

        results = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                results.append({
                    "score": node.score,
                    "metadata": node.node.metadata,
                    "text": node.node.text
                })

        return results or [{"error": "No results found"}]
    except Exception as e:
        print(f"[retrieve] Error: {e}")
        return [{"error": str(e)}]

# ----------------- Vector Similarity Helpers -----------------
def song_vector(song):
    return np.array([song.get(col, 0.0) for col in FEATURE_COLUMNS])

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------- Dataset Helpers -----------------
def load_df():
    if not SONGS_CSV.exists():
        raise FileNotFoundError("songs.csv not found")
    df = pd.read_csv(SONGS_CSV, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    for col in ("track_popularity", "valence", "energy", "acousticness", "danceability", "tempo", "duration_ms"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

def list_favorites_detailed():
    df = load_df()
    track_ids = [f["track_id"] for f in list_favorites()]
    favs = df[df["track_id"].isin(track_ids)]
    return favs.to_dict(orient="records")

def resolve_column(col: str) -> str:
    if not col:
        return col
    col_norm = col.strip().lower()
    choices = list(COLUMN_ALIASES.keys())
    best_match, score, _ = process.extractOne(col_norm, choices)
    if score > 80:
        return COLUMN_ALIASES[best_match]
    return COLUMN_ALIASES.get(col_norm, col)

def find_by_name(name: str) -> Optional[Dict]:
    df = load_df()
    choices = df["track_name"].tolist()
    best, score, idx = process.extractOne(name, choices, score_cutoff=70)
    if best:
        return df.iloc[idx].to_dict()
    return None

def songs_by_genre(genre: str, limit: int = 20):
    df = load_df()
    mask = df["playlist_genre"].str.lower() == genre.lower()
    return df[mask].head(limit).to_dict(orient="records")

def recommend_by_mood(mood: str, limit: int = 5):
    df = load_df()
    favorites = list_favorites_detailed()

    if mood.lower() in ("happy", "happiest"):
        df = df.sort_values("valence", ascending=False)
    elif mood.lower() in ("sad", "saddest"):
        df = df.sort_values("valence", ascending=True)
    elif mood.lower() in ("calm", "chill"):
        df = df.sort_values("energy", ascending=True)
    elif mood.lower() in ("energetic", "hype"):
        df = df.sort_values("energy", ascending=False)

    fav_ids = [f["track_id"] for f in favorites]
    df = df[~df["track_id"].isin(fav_ids)]

    if not favorites:
        return df.head(limit).to_dict(orient="records")

    fav_vectors = [song_vector(f) for f in favorites]
    candidate_rows = df.head(200)
    candidates = []
    for _, row in candidate_rows.iterrows():
        row_vec = song_vector(row)
        sim_scores = [cosine_similarity(row_vec, fv) for fv in fav_vectors]
        avg_sim = np.mean(sim_scores) if sim_scores else 0.0
        candidates.append((avg_sim, row.to_dict()))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in candidates[:limit]]

def compute_from_dataset(column: str, agg: str = "max", limit: int = 5, filters: dict = None) -> dict:
    df = load_df()
    col_resolved = resolve_column(column)
    if col_resolved not in df.columns:
        return {"error": f"Column '{column}' not found."}

    if filters:
        if "artist" in filters:
            df = df[df["track_artist"].str.contains(filters["artist"], case=False, na=False)]
        if "year_range" in filters and "track_album_release_date" in df.columns:
            start, end = filters["year_range"]
            df["year"] = pd.to_datetime(df["track_album_release_date"], errors="coerce").dt.year
            df = df[(df["year"] >= start) & (df["year"] <= end)]

    if df.empty:
        return {"error": "No rows match the given filters."}

    results = None
    if agg in ["max", "min"]:
        ascending = agg == "min"
        row = df.sort_values(col_resolved, ascending=ascending).iloc[0]
        results = [row.to_dict()]
    elif agg == "mean":
        results = [{"value": df[col_resolved].mean()}]
    elif agg == "median":
        results = [{"value": df[col_resolved].median()}]
    elif agg == "sum":
        results = [{"value": df[col_resolved].sum()}]
    elif agg == "top":
        rows = df.sort_values(col_resolved, ascending=False).head(limit)
        results = rows.to_dict(orient="records")
    elif agg == "bottom":
        rows = df.sort_values(col_resolved, ascending=True).head(limit)
        results = rows.to_dict(orient="records")
    else:
        return {"error": f"Unsupported aggregation '{agg}'."}

    return {"column": col_resolved, "agg": agg, "results": results}
