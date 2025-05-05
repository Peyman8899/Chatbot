# streamlit_chatbot.py
import streamlit as st
import os
import uuid
from datetime import datetime
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import time
import hashlib
from dotenv import load_dotenv
from document_processor import process_and_embed_document, build_prompt_with_context

load_dotenv()

# --- CONFIG ---
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = "https://aigateway-dev.ms.com/openai/v1/"
embedding_model = "text-embedding-ada-002"

# --- DEFINE LLM ---
llm = OpenAI(model="gpt-3.5-turbo", api_key=openai.api_key, base_url=openai_base_url)

# --- INIT CHROMA + EMBEDDING FUNC ---
embedding_func = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=embedding_model)
chroma_client = chromadb.PersistentClient(path="./chroma_repo")
collection = chroma_client.get_or_create_collection(name="chat_sessions", embedding_function=embedding_func)

# --- AUTH ---
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def auth_user():
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email and password:
            st.session_state.user_id = hash_password(email.lower())
            st.session_state.email = email
            st.rerun()

if "user_id" not in st.session_state:
    auth_user()
    st.stop()

# --- SESSION HANDLER ---
def list_sessions():
    return collection.get(where={"user_id": st.session_state.user_id})

def save_session(session_id, messages, title=None):
    conversation = "\n".join([f"{m.role.upper()}: {m.content}" for m in messages])
    try:
        collection.delete(ids=[session_id])
    except Exception:
        pass
    collection.add(
        documents=[conversation],
        ids=[session_id],
        metadatas=[{
            "user_id": st.session_state.user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "title": title or messages[1].content[:40] if len(messages) > 1 else "New Session"
        }]
    )

def load_session(session_id):
    result = collection.get(ids=[session_id])
    lines = result["documents"][0].split("\n")
    history = []
    for line in lines:
        if line.startswith("USER: "):
            history.append(ChatMessage(role="user", content=line[6:]))
        elif line.startswith("ASSISTANT: "):
            history.append(ChatMessage(role="assistant", content=line[11:]))
    return history

def group_sessions_by_date(sessions):
    grouped = {}
    for sid, meta in zip(sessions['ids'], sessions['metadatas']):
        date_key = meta['timestamp'].split("T")[0]
        if date_key not in grouped:
            grouped[date_key] = []
        grouped[date_key].append((sid, meta.get("title", "Untitled")))
    return grouped

# --- TOKEN TRIMMER ---
def truncate_messages(messages, max_tokens=7000):
    truncated = []
    total = 0
    for msg in reversed(messages):
        token_count = len(msg.content.split()) * 1.5
        if total + token_count > max_tokens:
            break
        truncated.insert(0, msg)
        total += token_count
    return truncated

# --- UI ---
st.set_page_config(page_title="Memory Chatbot", layout="wide")
st.sidebar.title("📁 Sessions")
session_list = list_sessions()
session_ids = [doc_id for doc_id in session_list['ids']]

if "chat_id" not in st.session_state:
    st.session_state.chat_id = "session_" + str(uuid.uuid4())
    st.session_state.chat_history = [ChatMessage(role="system", content="You are a helpful AI Assistant.")]
    st.session_state.loaded_sessions = {}
    st.session_state.session_titles = {}
    st.session_state.renaming_session_id = None
    st.session_state.renaming_value = ""
    st.session_state.all_sessions = session_list
    st.session_state.uploaded_doc = False
    # Optionally clear any previous document chunks with same chat_id if reused
    try:
        doc_collection = chroma_client.get_or_create_collection(name="doc_qa", embedding_function=embedding_func)
        doc_collection.delete(where={"chat_id": st.session_state.chat_id})
    except Exception as e:
        print("Doc cleanup error:", e)

else:
    session_list = list_sessions()
    st.session_state.all_sessions = session_list

grouped_sessions = group_sessions_by_date(st.session_state.all_sessions)
st.sidebar.markdown("### ➕ New Chat")
if st.sidebar.button("Start new chat"):
    st.session_state.chat_id = "session_" + str(uuid.uuid4())
    st.session_state.chat_history = [ChatMessage(role="system", content="You are a helpful AI Assistant.")]
    st.session_state.uploaded_doc = False

st.sidebar.markdown("---")
for date, sessions in grouped_sessions.items():
    st.sidebar.markdown(f"**{date}**")
    for sid, title in sessions:
        with st.sidebar.expander(title, expanded=False):
            if st.button("🗨️ Open", key=f"open_{sid}"):
                if sid not in st.session_state.loaded_sessions:
                    history = load_session(sid)
                    st.session_state.loaded_sessions[sid] = history
                st.session_state.chat_id = sid
                st.session_state.chat_history = st.session_state.loaded_sessions[sid]
            if st.button("✏️ Rename", key=f"rename_btn_{sid}"):
                st.session_state.renaming_session_id = sid
                st.session_state.renaming_value = title
            if st.button("🗑️ Delete", key=f"delete_btn_{sid}"):
                collection.delete(ids=[sid])
                time.sleep(1)
                st.session_state.renaming_session_id = None
                st.rerun()

        if st.session_state.renaming_session_id == sid:
            new_title = st.text_input("Rename session:", value=st.session_state.renaming_value, key=f"rename_input_{sid}")
            if st.button("✅ Save Name", key=f"confirm_rename_{sid}"):
                save_session(sid, st.session_state.loaded_sessions.get(sid, []), title=new_title)
                st.session_state.renaming_session_id = None
                time.sleep(1)
                st.rerun()

st.title("Chat with Memory")
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("🔍 Gap Analysis"):
        st.info("Gap Analysis triggered (logic not yet implemented).")

# --- File Upload & Document Processing ---
uploaded_file = st.file_uploader("Upload PDF or DOCX to ask questions about it", type=["pdf", "docx"])
if uploaded_file is not None:
    process_and_embed_document(uploaded_file, st.session_state.chat_id)
    st.session_state.uploaded_doc = True
    st.success("Document uploaded and processed. You can now ask questions about it.")

# --- Chat ---
user_input = st.chat_input("Say something...")
if user_input:
    st.session_state.chat_history.append(ChatMessage(role="user", content=user_input))
    trimmed_history = truncate_messages(st.session_state.chat_history)
    full_prompt = build_prompt_with_context(user_input, st.session_state.chat_id, trimmed_history, st.session_state.uploaded_doc)
    response = llm.chat(full_prompt).message.content
    st.session_state.chat_history.append(ChatMessage(role="assistant", content=response))
    save_session(st.session_state.chat_id, st.session_state.chat_history)
    st.session_state.loaded_sessions[st.session_state.chat_id] = st.session_state.chat_history

# --- Display ---
for msg in st.session_state.chat_history:
    with st.container():
        if msg.role == "user":
            st.markdown(f"<div style='text-align:right; background-color:#f0f0f0; padding:8px; border-radius:8px; margin:5px 0;'>{msg.content}</div>", unsafe_allow_html=True)
        elif msg.role == "assistant":
            st.markdown(f"<div style='text-align:left; background-color:#e6f7ff; padding:8px; border-radius:8px; margin:5px 0;'>{msg.content}</div>", unsafe_allow_html=True)
