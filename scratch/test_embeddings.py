import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from backend.embeddings import embedding_model

def test_api_config():
    print(f"Model Name: {MODEL_NAME if 'MODEL_NAME' in globals() else 'Not found'}")
    print(f"HF API URL: {HF_API_URL if 'HF_API_URL' in globals() else 'Not found'}")
    print(f"Headers configured: {'Authorization' in embedding_model.headers}")
    
    # We can't actually call the API without a real key, 
    # but we can verify the class structure and interface.
    print("EmbeddingModel interface checks:")
    print(f"Has generate: {hasattr(embedding_model, 'generate')}")
    print(f"Has generate_batch: {hasattr(embedding_model, 'generate_batch')}")

if __name__ == "__main__":
    from backend.embeddings import MODEL_NAME, HF_API_URL
    test_api_config()
