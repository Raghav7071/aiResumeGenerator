import logging
import chromadb
from typing import List, Dict

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Client to interface with the ChromaDB vector store."""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "research_papers"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._ensure_collection()

    def _ensure_collection(self):
        """Create the collection if it does not exist, otherwise retrieve it."""
        try:
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' ready")
            return collection
        except Exception as e:
            logger.error(f"Failed to create/get collection: {e}")
            raise e

    def insert(self, vectors: List[List[float]], documents: List[Dict]) -> bool:
        """Insert vector embeddings, documents, and associated metadata into the database."""
        if not vectors or not documents:
            return False

        ids = []
        embeddings = []
        metadatas = []

        for vec, doc in zip(vectors, documents):
            doc_id = doc["id"]
            text = doc["text"]
            # Include the document text in the metadata for future retrieval
            meta = {**doc["metadata"], "text": text}
            ids.append(doc_id)
            embeddings.append(vec)
            metadatas.append(meta)

        try:
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Inserted {len(ids)} vectors into ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return False

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """Search the database for vectors similar to the provided query."""
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )

            parsed = []
            if not results["ids"] or not results["ids"][0]:
                return parsed

            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if "distances" in results and results["distances"] else []

            for i in range(len(ids)):
                meta = metadatas[i] if i < len(metadatas) else {}
                text = meta.pop("text", "")
                
                # Convert the cosine distance to a similarity score
                dist = distances[i] if i < len(distances) else 0.0
                score = 1.0 - dist

                parsed.append({
                    "text": text,
                    "metadata": meta,
                    "score": score
                })

            return parsed
        except Exception as e:
            logger.error(f"Search request failed: {e}")
            return []

    def list_documents(self) -> List[Dict]:
        """List unique documents stored in the database."""
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results["metadatas"]
            
            docs = {}
            for meta in metadatas:
                fname = meta.get("filename")
                if not fname: continue
                if fname not in docs:
                    docs[fname] = {
                        "filename": fname,
                        "chunk_count": 0,
                        "page_count": 0
                    }
                docs[fname]["chunk_count"] += 1
                docs[fname]["page_count"] = max(docs[fname]["page_count"], meta.get("page_number", 0))
            
            return list(docs.values())
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def delete_document(self, filename: str) -> bool:
        """Delete all vectors associated with a specific document."""
        try:
            self.collection.delete(where={"filename": filename})
            logger.info(f"Deleted document '{filename}' from collection")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document '{filename}': {e}")
            return False

    def reset_collection(self) -> bool:
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._ensure_collection()
            logger.info("Collection reset complete")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
