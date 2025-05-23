# chatbot_improved.py - Improved version based on chatbot2.py
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
from document_processor_improved import process_and_embed_document, build_prompt_with_context

load_dotenv()

# --- CONFIG ---
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
embedding_model = "text-embedding-3-large"  # Updated to the latest embedding model with 3072 dimensions

# --- DEFINE LLM ---
# Configure LLM with higher max_tokens to allow longer responses
llm = OpenAI(
    model="gpt-4o", 
    api_key=openai.api_key, 
    base_url=openai_base_url, 
    streaming=True,
    max_tokens=4000  # Reduced to stay within OpenAI's rate limits
)

# --- INIT CHROMA + EMBEDDING FUNC ---
# Create embedding function with the new model
try:
    embedding_func = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=embedding_model)
    chroma_client = chromadb.PersistentClient(path="./chroma_repo")
    # Use a new collection name for the new embedding model
    collection = chroma_client.get_or_create_collection(
        name="chat_sessions_embedding3",  # New collection name for the new embedding model
        embedding_function=embedding_func
    )
except Exception as e:
    print(f"Error creating collection with new embedding model: {e}")
    # Fall back to the old embedding model if there's an issue
    embedding_model_fallback = "text-embedding-ada-002"
    embedding_func = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=embedding_model_fallback)
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
        elif line.startswith("SYSTEM: "):
            history.append(ChatMessage(role="system", content=line[8:]))
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
def truncate_messages(messages, max_tokens=25000):  # Balanced to stay within OpenAI's rate limits
    truncated = []
    total = 0
    # Always keep the system message if it exists
    if messages and messages[0].role == "system":
        truncated.append(messages[0])
        messages = messages[1:]
    
    # Add messages from newest to oldest until we hit the token limit
    for msg in reversed(messages):
        token_count = len(msg.content.split()) * 1.5  # Simple estimation
        if total + token_count > max_tokens:
            break
        truncated.insert(0, msg)
        total += token_count
    
    # If we have a system message, make sure it's at the beginning
    if truncated and truncated[0].role != "system" and len(truncated) > 1 and truncated[-1].role == "system":
        system_msg = truncated.pop(-1)
        truncated.insert(0, system_msg)
        
    return truncated

# --- UI SETUP ---
st.set_page_config(page_title="Improved Document Chatbot", layout="wide")

# --- AUTHENTICATION ---
if "user_id" not in st.session_state:
    auth_user()
    st.stop()

# --- SESSION INITIALIZATION ---
if "chat_id" not in st.session_state:
    st.session_state.chat_id = "session_" + str(uuid.uuid4())
    st.session_state.chat_history = [ChatMessage(role="system", content="You are a helpful AI Assistant.")]
    st.session_state.loaded_sessions = {}
    st.session_state.session_titles = {}
    st.session_state.renaming_session_id = None
    st.session_state.renaming_value = ""
    st.session_state.all_sessions = list_sessions()
    st.session_state.uploaded_doc = False
    # Initialize uploader_key for file uploader reset between sessions
    st.session_state.uploader_key = 0
    # Optionally clear any previous document chunks with same chat_id if reused
    try:
        doc_collection = chroma_client.get_or_create_collection(name="doc_qa", embedding_function=embedding_func)
        doc_collection.delete(where={"chat_id": st.session_state.chat_id})
    except Exception as e:
        print("Doc cleanup error:", e)
else:
    st.session_state.all_sessions = list_sessions()

# --- SIDEBAR: SESSIONS MANAGEMENT ---
st.sidebar.title("📁 Sessions")

# Sort dates in reverse order (newest first)
grouped_sessions = group_sessions_by_date(st.session_state.all_sessions)
sorted_dates = sorted(grouped_sessions.keys(), reverse=True)

# New chat button
st.sidebar.markdown("### ➕ New Chat")
if st.sidebar.button("Start new chat"):
    st.session_state.chat_id = "session_" + str(uuid.uuid4())
    st.session_state.chat_history = [ChatMessage(role="system", content="You are a helpful AI Assistant.")]
    # Reset document state
    st.session_state.uploaded_doc = False
    # Clear file uploader state by using session state key
    st.session_state.uploader_key += 1
    st.rerun()

# Display existing sessions
st.sidebar.markdown("---")
for date in sorted_dates:
    st.sidebar.markdown(f"**{date}**")
    # Sort sessions within each date by timestamp (newest first)
    sessions = sorted(grouped_sessions[date], key=lambda x: st.session_state.all_sessions['metadatas'][st.session_state.all_sessions['ids'].index(x[0])].get('timestamp', ''), reverse=True)
    for sid, title in sessions:
        with st.sidebar.expander(title, expanded=False):
            if st.button("🗨️ Open", key=f"open_{sid}"):
                if sid not in st.session_state.loaded_sessions:
                    history = load_session(sid)
                    st.session_state.loaded_sessions[sid] = history
                st.session_state.chat_id = sid
                st.session_state.chat_history = st.session_state.loaded_sessions[sid]
                st.rerun()
            if st.button("✏️ Rename", key=f"rename_btn_{sid}"):
                st.session_state.renaming_session_id = sid
                st.session_state.renaming_value = title
                st.rerun()
            if st.button("🗑️ Delete", key=f"delete_btn_{sid}"):
                collection.delete(ids=[sid])
                if sid in st.session_state.loaded_sessions:
                    del st.session_state.loaded_sessions[sid]
                st.rerun()

        if st.session_state.renaming_session_id == sid:
            new_title = st.text_input("Rename session:", value=st.session_state.renaming_value, key=f"rename_input_{sid}")
            if st.button("✅ Save Name", key=f"confirm_rename_{sid}"):
                save_session(sid, st.session_state.loaded_sessions.get(sid, []), title=new_title)
                st.session_state.renaming_session_id = None
                st.rerun()

# --- MAIN CONTENT ---
st.title("Improved Document Chatbot")

# --- File Upload & Document Processing ---
#with st.expander("📄 Document Upload", expanded=not st.session_state.uploaded_doc):
uploaded_file = st.file_uploader(
    "Upload a document", 
    type=["pdf", "docx", "txt", "csv", "tsv", "json", "xml", "html", "htm", "md", "markdown", "xls", "xlsx"],
    key=f"file_uploader_{st.session_state.uploader_key}"
)
if uploaded_file is not None:
    process_and_embed_document(uploaded_file, st.session_state.chat_id)
    st.session_state.uploaded_doc = True
    st.success("Document uploaded and processed. You can now ask questions about it.")

# --- Display current chat history ---
st.markdown("### Conversation")
for msg in st.session_state.chat_history:
    if msg.role == "system":
        continue  # Skip system messages in the display
    
    with st.container():
        if msg.role == "user":
            st.markdown(
                f"<div style='text-align:right; background-color:#f0f0f0; padding:12px; "
                f"border-radius:10px; margin:8px 0; box-shadow: 2px 2px 5px #ddd;'>{msg.content}</div>",
                unsafe_allow_html=True,
            )
        elif msg.role == "assistant":
            st.markdown(
                f"<div style='text-align:left; background-color:#e6f7ff; padding:12px; "
                f"border-radius:10px; margin:8px 0; box-shadow: 2px 2px 5px #ddd;'>{msg.content}</div>",
                unsafe_allow_html=True,
            )

# --- Chat input ---
user_input = st.chat_input("Ask a question...")
if user_input:
    # 1. Add user message to chat history
    st.session_state.chat_history.append(ChatMessage(role="user", content=user_input))

    # 2. Show user message immediately
    with st.container():
        st.markdown(
            f"<div style='text-align:right; background-color:#f0f0f0; padding:12px; border-radius:10px; margin:8px 0; box-shadow: 2px 2px 5px #ddd;'>{user_input}</div>",
            unsafe_allow_html=True
        )

    # 3. Prepare prompt using history and document context if available
    trimmed_history = truncate_messages(st.session_state.chat_history)
    
    # Add a system message to remind the model about the document if one is uploaded
    if st.session_state.uploaded_doc:
        # Add a reminder about the document to the system message
        if trimmed_history and trimmed_history[0].role == "system":
            trimmed_history[0] = ChatMessage(role="system", content="You are a helpful AI Assistant. The user has uploaded a document and is asking questions about it. Answer based on the document content only.")
        else:
            trimmed_history.insert(0, ChatMessage(role="system", content="You are a helpful AI Assistant. The user has uploaded a document and is asking questions about it. Answer based on the document content only."))
    
    # 4. Build prompt with context if document is uploaded
    full_prompt = build_prompt_with_context(
        user_input,
        st.session_state.chat_id,
        trimmed_history,
        st.session_state.uploaded_doc
    )

    # 5. Stream assistant response with improved handling for longer responses
    with st.spinner("Assistant is thinking..."):
        response_stream = llm.stream_chat(full_prompt)

        full_response = ""
        message_placeholder = st.empty()
        update_interval = 10  # Update display every 10 tokens for better performance
        token_count = 0
        
        for stream_resp in response_stream:
            if isinstance(stream_resp, ChatMessage):
                token = stream_resp.content
            else:
                token = stream_resp.delta or ""
                
            full_response += token
            token_count += 1
            
            # Only update UI periodically for better performance with long responses
            if token_count % update_interval == 0 or token == "" or len(token) > 10:
                message_placeholder.markdown(
                    f"<div style='text-align:left; background-color:#e6f7ff; padding:12px; border-radius:10px; margin:8px 0; box-shadow: 2px 2px 5px #ddd; overflow-wrap: break-word; word-wrap: break-word;'>{full_response}</div>",
                    unsafe_allow_html=True
                )

    # 6. Add assistant message to history and save session
    st.session_state.chat_history.append(ChatMessage(role="assistant", content=full_response))
    save_session(st.session_state.chat_id, st.session_state.chat_history)
    st.session_state.loaded_sessions[st.session_state.chat_id] = st.session_state.chat_history

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Improved Document Chatbot powered by LlamaIndex and OpenAI</div>", unsafe_allow_html=True)
