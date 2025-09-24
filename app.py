from fastapi import FastAPI
from pydantic import BaseModel
from agent import handle_query
from db import init_db
from tools import build_index, find_by_name
from db import add_favorite, list_favorites

init_db()

app = FastAPI(title="Dynamic Songs RAG Chatbot")

class ChatReq(BaseModel):
    query: str

class FavoriteReq(BaseModel):
    song_name: str

@app.post("/chat")
def chat(req: ChatReq):
    return handle_query(req.query)

@app.post("/index")
def index():
    count = build_index()
    return {"indexed": count}

@app.post("/favorites")
def add_to_favorites(fav: FavoriteReq):
    song = find_by_name(fav.song_name)
    if not song:
        return {"error": f"No match found for '{fav.song_name}'"}
    fav_id = add_favorite(song["track_id"])
    return {
        "message": "Added to favorites",
        "favorite_id": fav_id,
        "song": {
            "track_id": song["track_id"],
            "track_name": song["track_name"],
            "track_artist": song["track_artist"],
        },
    }

@app.get("/favorites")
def get_favorites():
    favs = list_favorites()
    return {"favorites": favs}
