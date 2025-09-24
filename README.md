# 🎵 Songs RAG Chatbot with Evaluation & Observability

A **Retrieval-Augmented Generation (RAG) chatbot** that answers user questions about songs **only from a local dataset**.  
The bot prevents hallucinations, logs **evaluations** (tone, factuality, hallucinations, RAG metrics), supports **favorites & recommendations**, and provides **observability with Langfuse**.

---

## 🚀 Features
- **RAG pipeline** → Answers are always grounded in `songs.csv`.  
- **Hallucination prevention** → Replies *“Sorry, I don’t have that information.”* if query is outside dataset.  
- **Favorites API** → Add/remove songs from favorites.  
- **Mood-based recommendations** → Suggest songs based on moods (happy, sad, chill, energetic).  
- **Evaluation**:  
  - Tone (**neutral**, **apologetic**, **friendly**)  
  - Factuality (**correct vs dataset**)  
  - Hallucination detection  
  - **RAG precision & recall**  
- **Observability with Langfuse** → Logs user queries, retrievals, responses, and evaluations.  

---

## 📂 Project Structure
```bash
songs-rag/
├── app.py              # FastAPI app (chat + favorites APIs)
├── agent.py            # Intent parsing, execution, answer generation
├── tools.py            # RAG utilities (retrieval, search, mood recs, dataset ops)
├── db.py               # SQLite for storing favorites
├── evaluation.py       # Automatic evaluation (tone, factuality, hallucination, metrics)
├── songs.csv           # Dataset (20–30 songs: title, artist, genre, mood)
├── chat_ui.py          # Streamlit frontend (chatbot UI)
├── config.py           # Config loader for env vars
├── requirements.txt    # Dependencies
└── README.md           # Project docs

---```

## ⚙️ Setup

### **1. Clone & Create Virtual Environment**
```bash
git clone https://github.com/<USERNAME>/songs-rag-chatbot.git
cd songs-rag-chatbot

python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or: .venv\Scripts\activate  # Windows
```

### **2. Install Dependencies
```bash
pip install -r requirements.txt
```

### **3. Set Environment Variables
Create a .env file in the project root:
```bash
# Embeddings (Cohere free tier)
COHERE_API_KEY=your_cohere_api_key

# Perplexity LLM (if used)
PERPLEXITY_API_KEY=your_perplexity_key
PERPLEXITY_API_URL=https://api.perplexity.ai

# Langfuse (observability)
LANGFUSE_PUBLIC_API_KEY=your_public_key
LANGFUSE_SECRET_API_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# Dataset
SONGS_CSV=songs.csv
```


### Run the App
Backend (FastAPI)
`uvicorn app:app --reload`

Runs the backend at http://127.0.0.1:8000

Frontend (Streamlit)
`streamlit run chat_ui.py`

Opens chatbot UI in your browser.