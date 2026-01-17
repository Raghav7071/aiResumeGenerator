# DocuMind Knowledge Engine 🧠

DocuMind is an intelligent RAG (Retrieval-Augmented Generation) application designed to turn static academic papers and technical documents into conversational knowledge. It leverages **ChromaDB** for high-performance vector search and **Llama 3.1** (via Groq) for lightning-fast, cited answers.

---

## ✨ Features

*   **Semantic Search**: Finds relevant information based on meaning, not just keywords.
*   **Page-Level Citations**: Every answer includes the exact file and page number used as a source.
*   **Modern React UI**: A premium, dark-mode-first interface built with Tailwind CSS and Lucide.
*   **FastAPI Backend**: High-performance streaming responses for a real-time chat experience.
*   **ChromaDB Persistence**: Your knowledge base survives restarts using local vector storage.

---

## 🏛️ Architecture

```
User (React UI)  <─── Served from Root (/) ───  FastAPI (Backend)
     │                                            │
     ├── Upload PDF  ──────────→  /upload  ───────┤
     │                              │             │
     │                              └── Ingestion Pipeline
     │                                      ├── Extract (PyPDF)
     │                                      ├── Chunk (Recursive)
     │                                      ├── Embed (MiniLM)
     │                                      └── Index (ChromaDB)
     │
     └── Ask Question ─────────→  /chat  ─────────┤
                                    │             │
                                    └── Retrieval Pipeline
                                            ├── Embed Query
                                            ├── Neural Search
                                            └── GenAI (Groq)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React (Vanilla JS), Tailwind CSS, Lucide Icons |
| **Backend** | FastAPI (Python 3.11/3.12) |
| **Vector DB** | ChromaDB |
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) |
| **LLM** | Llama 3.1 8B/70B (via Groq) |
| **Deployment** | Render |

---

## 🚀 Deployment (Custom UI)

### Option 1: Render (Recommended)
Render is the easiest way to host the DocuMind FastAPI backend and Custom UI.

1.  **Create a New Web Service**: Log in to [Render](https://render.com) and connect your GitHub repository.
2.  **Configuration**:
    *   **Runtime**: Python
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3.  **Environment Variables**:
    *   Add `GROQ_API_KEY`: (Your Key from console.groq.com)
    *   Add `PYTHON_VERSION`: `3.11.9`

> [!IMPORTANT]
> On Render's Free tier, documents are deleted when the server restarts (ephemeral storage). For persistent storage, a paid "Persistent Disk" is required.

---

## 💻 Local Setup

1.  **Clone & Install**:
    ```bash
    git clone https://github.com/Adarsh-code169/DocuMind.git
    cd DocuMind
    python3 -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

2.  **Env Config**:
    Create a `.env` file and add:
    ```
    GROQ_API_KEY=your_key_here
    ```

3.  **Run**:
    ```bash
    uvicorn backend.main:app --reload --port 9000
    ```
    Open **`http://localhost:9000`** to see your app!

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py            # FastAPI Entry Point (Custom UI)
│   ├── models.py          # Pydantic Schemas
│   ├── chroma_store.py    # ChromaDB Wrapper
│   ├── pipeline/          # RAG Pipelines (Ingest/Retrieve)
│   └── static/            # Frontend (index.html, JS, CSS)
├── render.yaml            # Render Blueprint for easy deployment
├── requirements.txt       # Unified Dependencies
└── streamlit_interface_backup.py  # Original Streamlit version (Backup)
```

---
