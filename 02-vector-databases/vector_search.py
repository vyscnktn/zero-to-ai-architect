import requests
from qdrant_client import QdrantClient

def get_local_embedding(text, model_name="nomic-embed-text"):
    """Generates a high-dimensional vector from text using the local Ollama matrix."""
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": model_name, 
        "input": text
    }
    response = requests.post(url, json=payload, timeout=30)
    return response.json()["embeddings"][0]

if __name__ == "__main__":
    print("\n=== Local AI Architecture Laboratory: Semantic Search ===")
    
    COLLECTION_NAME = "defense_tech_knowledge"
    user_query = "Tell me about military network security."
    print(f"[*] User Query: '{user_query}'\n")
    
    # Compute query coordinates
    query_vector = get_local_embedding(user_query)
    
    # Connect to the local running container port
    client = QdrantClient(url="http://localhost:6333")
    
    # FIX: Use query_points with explicit parameter mapping
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1
    )
    
    print("[+] Top Mathematical Match Found in Qdrant:")
    # Iterate through the returned points
    for hit in search_result.points:
        print(f" -> Text: {hit.payload['text']}")
        print(f" -> Cosine Similarity Score: {hit.score:.4f}")
