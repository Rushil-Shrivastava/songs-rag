import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Songs RAG Chatbot", page_icon="🎵")
st.title("🎵 Songs RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me about songs..."):
    # Show user msg
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    try:
        resp = requests.post(API_URL, json={"query": prompt, "user_id": "demo"})
        data = resp.json()
        answer = data.get("answer", "Error: no answer")
    except Exception as e:
        answer = f"Error contacting API: {e}"

    # Show bot msg
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
