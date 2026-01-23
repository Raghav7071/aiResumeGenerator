import io
import logging
from pypdf import PdfReader
from typing import List, Dict

# Use the existing embeddings model wrapper
from backend.embeddings import embedding_model

logger = logging.getLogger(__name__)

# Configure chunking for the text data
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text_from_pdf(file_content: bytes, filename: str) -> List[Dict]:
    """Extract page text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_content))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page_number": i + 1,
                "filename": filename
            })
        else:
            logger.warning(f"Skipping page {i+1} of '{filename}' — no text found")

    logger.info(f"Extracted {len(pages)}/{len(reader.pages)} pages from '{filename}'")
    return pages


def chunk_text(pages: List[Dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Chunk text documents into overlapping segments."""
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        start = 0

        # Use a sliding window to create overlapping text segments
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append({
                "id": f"{page['filename']}_p{page['page_number']}_{chunk_id}",
                "text": text[start:end],
                "metadata": {
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "chunk_id": chunk_id
                }
            })
            chunk_id += 1
            start += (chunk_size - overlap)

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

def process_and_store_document_stream(file_content: bytes, filename: str, vector_store):
    """
    Execute the document ingestion pipeline with status streaming.
    Yields JSON status updates for each completed stage.
    """
    import json

    yield json.dumps({"status": "Extracting text from PDF..."}) + "\n"
    pages = extract_text_from_pdf(file_content, filename)
    if not pages:
        yield json.dumps({"error": "No text content found in PDF."}) + "\n"
        return
    
    yield json.dumps({"status": f"Successfully extracted {len(pages)} pages. Chunking..."}) + "\n"
    chunks = chunk_text(pages)
    if not chunks:
        yield json.dumps({"error": "Failed to create document chunks."}) + "\n"
        return

    yield json.dumps({"status": f"Created {len(chunks)} chunks. Generating embeddings..."}) + "\n"
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.generate_batch(texts)

    yield json.dumps({"status": "Embeddings generated. Syncing to ChromaDB..."}) + "\n"
    success = vector_store.insert(embeddings, chunks)
    if not success:
        yield json.dumps({"error": "Database synchronization failed."}) + "\n"
        return

    yield json.dumps({"status": "Ingestion complete!", "chunks": len(chunks)}) + "\n"

def process_and_store_document(file_content: bytes, filename: str, vector_store) -> int:
    """Synchronous version of the ingestion pipeline for legacy compatibility."""
    pages = extract_text_from_pdf(file_content, filename)
    if not pages: return 0
    chunks = chunk_text(pages)
    if not chunks: return 0
    embeddings = embedding_model.generate_batch([c["text"] for c in chunks])
    if vector_store.insert(embeddings, chunks):
        return len(chunks)
    return 0
