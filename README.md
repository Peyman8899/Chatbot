
Memory Chatbot with Document QA Support

A multi-session AI chatbot built with **Streamlit**, powered by **OpenAI GPT 4o**, with persistent memory using **ChromaDB** and document-aware responses through PDF/DOCX upload support.

---

## 🚀 Features

- Multi-session chat memory stored with ChromaDB
- PDF and DOCX document upload and Q&A
- Paragraph-based, overlapping chunking for semantic preservation
- Contextual embedding using OpenAI's `text-embedding-ada-002`
- Session rename, delete, and auto-saving
- Auth via email + password (hashed, local use only)
- Gap Analysis button (placeholder)
- Persistent storage across restarts

---

##Project Structure

```
Chat_bot/
├── streamlit_chatbot.py        # Main Streamlit app
├── document_processor.py       # Upload, chunk, and embed logic
├── chroma_repo/                # Persistent vector store (auto-generated)
├── requirements.txt            # Dependencies
└── .env                        # OpenAI API key storage (optional)
```

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd Chat_bot
```

### 2. Create & activate virtual environment

```bash
python -m venv myenv
source myenv/bin/activate   # Windows: myenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

Either in a `.env` file:

```
OPENAI_API_KEY=sk-xxxx
```

Or hardcoded in the scripts.

---

## ▶️ Run the App

```bash
streamlit run streamlit_chatbot.py
```

---

## 📚 How It Works

- **Authentication**: Email + password (hashed); sessions tied to user_id
- **Chat Sessions**: Stored with metadata (user_id, title, timestamp)
- **Document Upload**: Extracts text (fitz for PDFs, docx2txt for DOCX), then chunked with overlap
- **Embedding**: Each chunk embedded via OpenAI and stored in ChromaDB
- **Retrieval**: Relevant chunks retrieved and added to LLM prompt for question answering

---

## 🧪 Future Enhancements

- Async background processing for large documents
- CSV, TXT support
- Admin dashboard for session analytics
- Deployment on cloud (Streamlit Sharing, AWS, etc.)

---

## 🧑‍💻 Author

Built by [Your Name]. Contact: [your.email@example.com]
