import requests

def get_local_embedding(text, model_name="qwen2.5-coder:7b"):
    """Generates a vector for incoming user queries."""
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": model_name, 
        "input": text
    }
    response = requests.post(url, json=payload, timeout=30)
    return response.json()["embeddings"][0]


def search_vector_db(collection_name, query_text, top_k=1):
    """Converts a query to a vector and searches Qdrant using Cosine Similarity."""
    # 1. Convert the search query into the exact same high-dimensional space
    query_vector = get_local_embedding(query_text)
    
    # 2. Query Qdrant's search endpoint
    qdrant_url = f"http://localhost:6333/collections/{collection_name}/points/search"
    
    payload = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True # Instruct Qdrant to return the underlying raw text
    }
    
    response = requests.post(qdrant_url, json=payload)
    if response.status_code == 200:
        return response.json()["result"]
    else:
        print(f"[-] Search failed: {response.text}")
        return []

if __name__ == "__main__":
    print("\n=== Local AI Architecture Laboratory: Semantic Search ===")
    COLLECTION_NAME = "defense_tech_knowledge"
    
    # Test a concept that is not an exact word match but a semantic match
    user_query = "Tell me about military network security."
    print(f"[*] User Query: '{user_query}'")
    
    results = search_vector_db(COLLECTION_NAME, user_query, top_k=1)
    
    print("\n[+] Top Mathematical Match Found in Qdrant:")
    for match in results:
        print(f" -> Score (Cosine Similarity): {match['score']:.4f}")
        print(f" -> Document ID: {match['id']}")
        print(f" -> Content: {match['payload']['text']}")
