import os
import httpx
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Let's test the standard model URL
TEST_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

def test_url():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with httpx.Client() as client:
        response = client.post(
            TEST_URL,
            headers=headers,
            json={"inputs": "Test sentence", "options": {"wait_for_model": True}}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Success! First vector element:", response.json()[0][0])
        else:
            print("Error:", response.text)

if __name__ == "__main__":
    test_url()
