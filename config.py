from pathlib import Path
import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

ROOT = Path(__file__).resolve().parent
SONGS_CSV = ROOT / "songs.csv"

# Pinecone 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "rag-songs-index")

# Perplexity (exposed via OpenAI-compatible endpoint)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_URL = os.getenv("PERPLEXITY_API_URL") 

# Langfuse
LANGFUSE_SECRET_API_KEY = os.getenv("LANGFUSE_SECRET_API_KEY")
LANGFUSE_PUBLIC_API_KEY = os.getenv("LANGFUSE_PUBLIC_API_KEY")
LANGFUSE_PROJECT = os.getenv("LANGFUSE_PROJECT", "rag-chatbot")

# COHERE
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Favorites DB
DB_FILE = ROOT / "favorites.db"

# RAG
TOP_K = int(os.getenv("TOP_K", "4"))