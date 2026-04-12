
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"Using API Key: {api_key[:10]}...")

try:
    client = genai.Client(api_key=api_key)
    
    print("\n--- Testing Model Listing ---")
    for m in client.models.list():
        print(f"Model ID: {m.name}")
    
    print("\n--- Testing Generation (gemini-1.5-flash) ---")
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents='Hello, this is a test.'
    )
    print(f"Generation Result: {response.text}")
    
    print("\n--- Testing Embedding (text-embedding-004) ---")
    emb_res = client.models.embed_content(
        model='text-embedding-004',
        contents='Hello world'
    )
    print(f"Embedding Success: Vector length {len(emb_res.embeddings[0].values)}")

except Exception as e:
    print(f"\nCaught Exception: {e}")
