# document_processor_improved.py
import os
import tempfile
import fitz  # PyMuPDF
import docx2txt
import csv
import json
import pandas as pd
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
# Create embedding function with the new model
embedding_func = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-large")  # 3072 dimensions

# Create a new collection with a different name for the new embedding model
chroma_client = chromadb.PersistentClient(path="./chroma_repo")

try:
    # Try to get the collection with the new embedding model
    collection = chroma_client.get_or_create_collection(
        name="doc_qa_embedding3",  # New collection name for the new embedding model
        embedding_function=embedding_func
    )
except Exception as e:
    print(f"Error creating collection with new embedding model: {e}")
    # Fall back to the old embedding model if there's an issue
    embedding_func = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-ada-002")
    collection = chroma_client.get_or_create_collection(name="doc_qa", embedding_function=embedding_func)

# Cache for processed documents to avoid redundant processing
processed_documents = {}

def extract_text_from_file(uploaded_file):
    """Extract text from various file formats with improved error handling."""
    try:
        file_extension = os.path.splitext(uploaded_file.name.lower())[1]
        
        # Create a temporary file to work with
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
            
        # Process based on file extension
        if file_extension == ".pdf":
            # PDF files
            doc = fitz.open(tmp_path)
            text = "\n".join(page.get_text() for page in doc)
            
        elif file_extension == ".docx":
            # Word documents
            text = docx2txt.process(tmp_path)
            
        elif file_extension == ".txt":
            # Plain text files
            with open(tmp_path, 'r', encoding='utf-8', errors='replace') as file:
                text = file.read()
                
        elif file_extension == ".csv":
            # CSV files
            df = pd.read_csv(tmp_path)
            # Convert DataFrame to a readable text format
            text = f"CSV File Contents:\n\n{df.to_string()}"
            
        elif file_extension == ".tsv":
            # TSV files
            df = pd.read_csv(tmp_path, sep='\t')
            text = f"TSV File Contents:\n\n{df.to_string()}"
            
        elif file_extension == ".json":
            # JSON files
            with open(tmp_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            # Pretty print JSON with indentation
            text = f"JSON File Contents:\n\n{json.dumps(data, indent=2)}"
            
        elif file_extension == ".xml":
            # XML files
            tree = ET.parse(tmp_path)
            root = tree.getroot()
            # Convert XML to a readable text format
            lines = []
            
            def process_element(element, depth=0):
                indent = "  " * depth
                lines.append(f"{indent}<{element.tag}>")
                if element.text and element.text.strip():
                    lines.append(f"{indent}  {element.text.strip()}")
                for child in element:
                    process_element(child, depth + 1)
                lines.append(f"{indent}</{element.tag}>")
                
            process_element(root)
            text = "XML File Contents:\n\n" + "\n".join(lines)
            
        elif file_extension in [".html", ".htm"]:
            # HTML files
            with open(tmp_path, 'r', encoding='utf-8', errors='replace') as file:
                html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            # Extract text content, removing script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator="\n")
            # Clean up excessive whitespace
            text = re.sub(r'\n+', '\n', text)
            text = "HTML File Contents:\n\n" + text
            
        elif file_extension == ".md" or file_extension == ".markdown":
            # Markdown files
            with open(tmp_path, 'r', encoding='utf-8', errors='replace') as file:
                text = file.read()
            text = "Markdown File Contents:\n\n" + text
            
        elif file_extension in [".xls", ".xlsx"]:
            # Excel files
            df_dict = pd.read_excel(tmp_path, sheet_name=None)
            sheets_text = []
            
            for sheet_name, df in df_dict.items():
                sheet_text = f"Sheet: {sheet_name}\n{df.to_string()}"
                sheets_text.append(sheet_text)
                
            text = "Excel File Contents:\n\n" + "\n\n".join(sheets_text)
            
        else:
            # For unsupported file types, try to read as text
            try:
                with open(tmp_path, 'r', encoding='utf-8', errors='replace') as file:
                    text = file.read()
                text = f"File Contents ({file_extension}):\n\n{text}"
            except:
                text = f"Unable to extract text from {uploaded_file.name}. File format {file_extension} is not supported."
        
        # Clean up the temporary file
        os.remove(tmp_path)
        return text
        
    except Exception as e:
        print(f"Error extracting text from {uploaded_file.name}: {e}")
        return f"Error processing {uploaded_file.name}: {str(e)}"

def chunk_text(text, chunk_size=2500, overlap=300):
    """Simple and reliable chunking with larger chunks and significant overlap."""
    # Use a much simpler chunking approach with larger chunks and more overlap
    paragraphs = text.split("\n")
    chunks = []
    current = []
    current_length = 0
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        # If adding this paragraph would exceed chunk size and we have content
        if current_length + len(para) > chunk_size and current:
            # Save current chunk
            chunks.append("\n".join(current))  # Preserve newlines for better list formatting
            # Keep substantial overlap for better context continuity
            overlap_size = min(len(current), overlap // 5)
            current = current[-overlap_size:]
            current_length = sum(len(p) for p in current)
            
        current.append(para)
        current_length += len(para)
    
    # Add the last chunk if there's anything left
    if current:
        chunks.append("\n".join(current))  # Preserve newlines
        
    return chunks

def process_and_embed_document(uploaded_file, chat_id):
    """Process and embed document with caching to avoid redundant processing."""
    try:
        # Check if this document has already been processed
        doc_key = f"{chat_id}_{uploaded_file.name}_{uploaded_file.size}"
        if doc_key in processed_documents:
            print(f"Document already processed: {uploaded_file.name}")
            return True
            
        # Extract text from the document
        text = extract_text_from_file(uploaded_file)
        if not text.strip():
            print(f"No text could be extracted from {uploaded_file.name}")
            return False
            
        # Create chunks with balanced size and overlap to stay within rate limits
        chunks = chunk_text(text, chunk_size=2500, overlap=300)
        
        # Generate IDs and metadata for each chunk
        ids = [f"{chat_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "chat_id": chat_id, 
            "chunk_index": i,
            "filename": uploaded_file.name,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]

        # Process in batches to avoid token limits
        batch_size = 50  # Smaller batch size for better reliability
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            collection.add(
                documents=batch_chunks, 
                ids=batch_ids, 
                metadatas=batch_metadatas
            )
            
        # Cache the document to avoid reprocessing
        processed_documents[doc_key] = True
        return True
        
    except Exception as e:
        print(f"Error processing document: {e}")
        return False

def build_prompt_with_context(user_input, chat_id, history, has_doc):
    """Build prompt with enhanced context retrieval for better responses."""
    if not has_doc:
        return history
        
    try:
        # Balance between comprehensive context and rate limits
        n_results = 15  # Reduced to stay within OpenAI's rate limits
        
        # Get relevant chunks with adjusted parameters
        query_embedding = embedding_func([user_input])[0]
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results,
            where={"chat_id": chat_id}
        )
        
        relevant_chunks = results["documents"][0]
        
        # Add metadata about the chunks
        chunk_metadata = []
        for i, metadata in enumerate(results["metadatas"][0]):
            chunk_index = metadata.get("chunk_index", "unknown")
            total_chunks = metadata.get("total_chunks", "unknown")
            chunk_metadata.append(f"Chunk {chunk_index+1}/{total_chunks}")
            
        # Combine chunks with their metadata
        context_with_metadata = []
        for i, (chunk, meta) in enumerate(zip(relevant_chunks, chunk_metadata)):
            context_with_metadata.append(f"{meta}:\n{chunk}")
            
        # Join all context with clear separators
        context = "\n---\n".join(context_with_metadata)
        
        # Create a simple, direct system message focused on comprehensive answers
        system_prompt = """You are a helpful AI Assistant analyzing a document.

Use ONLY the following document excerpts to answer the user's query. If the information isn't in these excerpts, acknowledge that limitation.

Document Context:
{context}

IMPORTANT INSTRUCTIONS:
1. Answer thoroughly and completely based ONLY on the provided context.
2. Include ALL relevant information from the context in your response.
3. Maintain original formatting, numbering, and structure from the document.
4. Do not summarize or abbreviate lists - provide the complete information.
5. Do not make up or infer information not present in the context.
"""
            
        # Create the system message
        system_msg = ChatMessage(
            role="system", 
            content=system_prompt.format(context=context)
        )
        
        # Add system message at the beginning
        return [system_msg] + history
        
    except Exception as e:
        print(f"Error building prompt with context: {e}")
        return history
