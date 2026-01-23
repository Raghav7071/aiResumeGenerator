import os
from dotenv import load_dotenv
load_dotenv()

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.models import ChatRequest, ChatResponse
from backend.chroma_store import ChromaVectorStore
from backend.pipeline.ingestion import process_and_store_document
from backend.pipeline.retrieval import RetrievalPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

vector_store = None
retrieval_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ChromaDB and RAG pipeline on application startup."""
    global vector_store, retrieval_pipeline

    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        logger.warning("GROQ_API_KEY not set — chat won't work")

    # Use a persistent ChromaDB store
    persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    vector_store = ChromaVectorStore(persist_directory=persist_dir)
    retrieval_pipeline = RetrievalPipeline(api_key=groq_key, vector_store=vector_store)
    
    logger.info("App ready")
    yield


app = FastAPI(title="Chroma RAG API", version="1.0.0", lifespan=lifespan)

# Configure CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from backend.pipeline.ingestion import process_and_store_document_stream
from backend.pipeline.retrieval import retrieve_and_generate_answer_stream

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    content = await file.read()
    return StreamingResponse(process_and_store_document_stream(content, file.filename, vector_store), media_type="text/event-stream")

@app.post("/chat")
async def chat(request: ChatRequest):
    groq_key = os.getenv("GROQ_API_KEY")
    return StreamingResponse(retrieve_and_generate_answer_stream(request.message, vector_store, groq_key), media_type="text/event-stream")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static/index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "vector_store": "chromadb"}

@app.get("/status")
async def status():
    return {"status": "online", "vector_store": "ready" if vector_store else "initializing"}

@app.get("/documents")
async def get_documents():
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return {"documents": vector_store.list_documents()}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    success = vector_store.delete_document(filename)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete {filename}")
    return {"status": "deleted", "filename": filename}

@app.post("/reset")
async def reset_database():
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    success = vector_store.reset_collection()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset database")
    return {"status": "reset", "message": "All documents cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
