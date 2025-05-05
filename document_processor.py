# document_processor.py
import os
import tempfile
import fitz  # PyMuPDF
import docx2txt
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_func = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-ada-002")
chroma_client = chromadb.PersistentClient(path="./chroma_repo")
collection = chroma_client.get_or_create_collection(name="doc_qa", embedding_function=embedding_func)

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        doc = fitz.open(tmp_path)
        text = "\n".join(page.get_text() for page in doc)
        os.remove(tmp_path)
        return text

    elif uploaded_file.name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        text = docx2txt.process(tmp_path)
        os.remove(tmp_path)
        return text
    else:
        return ""

def chunk_text(text, chunk_size=500, overlap=100):
    paragraphs = text.split("\n")
    chunks = []
    current = []
    current_length = 0
    for para in paragraphs:
        if not para.strip():
            continue
        if current_length + len(para) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-(overlap // 10):]  # keep some overlap
            current_length = sum(len(p) for p in current)
        current.append(para)
        current_length += len(para)
    if current:
        chunks.append(" ".join(current))
    return chunks

def process_and_embed_document(uploaded_file, chat_id):
    text = extract_text_from_file(uploaded_file)
    chunks = chunk_text(text)
    ids = [f"{chat_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chat_id": chat_id, "chunk_index": i} for i in range(len(chunks))]

    batch_size = 100  # to avoid hitting token limits
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        collection.add(documents=batch_chunks, ids=batch_ids, metadatas=batch_metadatas)

def build_prompt_with_context(user_input, chat_id, history, has_doc):
    if not has_doc:
        return history
    query_embedding = embedding_func([user_input])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3, where={"chat_id": chat_id})
    relevant_chunks = results["documents"][0]
    context = "\n---\n".join(relevant_chunks)
    system_msg = ChatMessage(role="system", content=f"Use the following document context to answer the user's query.\n\n{context}")
    return [system_msg] + history + [ChatMessage(role="user", content=user_input)]
