import os
import logging
from groq import Groq
from typing import List, Dict
from backend.embeddings import embedding_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.3
MAX_TOKENS = 1024


class RetrievalPipeline:
    """Pipeline for handling querying, retrieval from the vector database, and answer generation."""

    def __init__(self, api_key: str, vector_store, model: str = None):
        self.client = Groq(api_key=api_key)
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.vector_store = vector_store
        logger.info(f"Retrieval pipeline using model: {self.model}")

    def run_stream(self, query: str, history: List[Dict] = None, top_k: int = 5):
        """Execute the retrieval pipeline with status streaming."""
        import json
        
        yield json.dumps({"status": "Vectorizing your query..."}) + "\n"
        query_vec = embedding_model.generate(query)
        if not query_vec:
            yield json.dumps({"error": "Failed to generate query embeddings."}) + "\n"
            return
            
        yield json.dumps({"status": "Searching vector database for relevant context..."}) + "\n"
        results = self.vector_store.search(query_vec, top_k=top_k)
        
        if not results:
            yield json.dumps({"status": "No relevant documents found. Generating response..."}) + "\n"
        else:
            yield json.dumps({"status": f"Found {len(results)} relevant segments. Synthesizing answer..."}) + "\n"

        # Generate the final answer
        try:
            answer_data = self.generate_answer(query, results)
            yield json.dumps(answer_data) + "\n"
        except Exception as e:
            yield json.dumps({"error": f"Generation failed: {str(e)}"}) + "\n"

    def run(self, query: str, top_k: int = 5) -> Dict:
        """Execute the retrieval pipeline."""
        
        # Generate embeddings for the query
        query_vec = embedding_model.generate(query)
        if not query_vec:
            raise ValueError("Couldn't generate embeddings for the query.")
            
        # Retrieve document chunks relevant to the query
        results = self.vector_store.search(query_vec, top_k=top_k)
        
        # Generate the response based on retrieved chunks
        return self.generate_answer(query, results)

    def generate_answer(self, query: str, context: List[Dict]) -> Dict:
        """Generate an answer using the provided context."""

        # Avoid calling the LLM if no context was found in the database
        if not context:
            return {
                "answer": "I couldn't find relevant info in the uploaded documents. Try uploading a PDF on this topic first.",
                "citations": []
            }

        # Format the context to include source references for citations
        context_str = "\n\n".join([
            f"Source [{i+1}]: {item['text']} (File: {item['metadata'].get('filename')}, Page: {item['metadata'].get('page_number')})"
            for i, item in enumerate(context)
        ])

        system_prompt = (
            "You are a research assistant using RAG. "
            "Answer based only on the provided context. "
            "If the answer isn't in the context, say so. "
            "Cite sources like [Source 1]."
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=1
            )
            answer = completion.choices[0].message.content
            return {"answer": answer, "citations": [c["metadata"] for c in context]}
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"answer": "Sorry, something went wrong while generating the answer. Please try again.", "citations": []}

def retrieve_and_generate_answer_stream(query: str, vector_store, api_key: str, history: List[Dict] = None, top_k: int = 5):
    """Helper for streaming retrieval and generation."""
    pipeline = RetrievalPipeline(api_key=api_key, vector_store=vector_store)
    return pipeline.run_stream(query, history=history, top_k=top_k)

def retrieve_and_generate_answer(query: str, vector_store, api_key: str, top_k: int = 5) -> Dict:
    """Helper function to execute the retrieval and generation pipeline."""
    pipeline = RetrievalPipeline(api_key=api_key, vector_store=vector_store)
    return pipeline.run(query, top_k=top_k)
