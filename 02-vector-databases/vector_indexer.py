import requests

def get_local_embedding(text, model_name="qwen2.5-coder:7b"):
    """Generates a high-dimensional vector from text using the local Ollama matrix."""
    
    url = "http://localhost:11434/api/embed"
    
    
    payload = {
        "model": model_name, 
        "input": text
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
           
            return response.json()["embeddings"][0]
        else:
            raise Exception(f"Ollama Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to Ollama. Ensure the service is active.")


def init_qdrant_collection(collection_name, vector_size):
    """Initializes a clean collection in the local Qdrant database using Cosine Similarity."""
    qdrant_url = f"http://localhost:6333/collections/{collection_name}"
    
    # Configuration schema for high-dimensional geometric search
    schema = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine"  # Using Cosine Similarity for semantic matching
        }
    }
    
    # Create or overwrite the collection
    response = requests.put(qdrant_url, json=schema)
    if response.status_code in [200, 201]:
        print(f"[+] Qdrant collection '{collection_name}' initialized successfully with size {vector_size}.")
    else:
        print(f"[-] Failed to initialize collection: {response.text}")

def upsert_vectors(collection_name, records):
    """Pushes vector embeddings along with their raw text payload into Qdrant."""
    qdrant_url = f"http://localhost:6333/collections/{collection_name}/points"
    
    payload = {"points": records}
    response = requests.post(qdrant_url, json=payload)
    
    if response.status_code == 200:
        print(f"[+] Successfully indexed {len(records)} points into Qdrant.")
    else:
        print(f"[-] Upsert operation failed: {response.text}")

if __name__ == "__main__":
    print("=== Local AI Architecture Laboratory: Milestone 2 ===")
    
    COLLECTION_NAME = "defense_tech_knowledge"
    
    # Sample structured knowledge base entries
    raw_documents = [
        {"id": 1, "text": "Autonomous drone navigation systems rely heavily on edge-computing AI models."},
        {"id": 2, "text": "Cybersecurity protocols in modern defense infrastructure enforce strict zero-trust networks."},
        {"id": 3, "text": "Baking perfect sourdough bread requires precise ambient humidity and hydration calculations."}
    ]
    
    # Get vector size by testing the first document embedding
    print("[*] Testing local embedding dimensions...")
    sample_vector = get_local_embedding(raw_documents[0]["text"])
    VECTOR_SIZE = len(sample_vector)
    
    # Initialize the isolated Vector database collection
    init_qdrant_collection(COLLECTION_NAME, VECTOR_SIZE)
    
    # Generate embeddings and construct Qdrant storage packets (points)
    qdrant_records = []
    for doc in raw_documents:
        print(f"[*] Vectorizing Document ID {doc['id']}...")
        vector = get_local_embedding(doc["text"])
        
        # Structuring the record points matching Qdrant schema specifications
        record = {
            "id": doc["id"],
            "vector": vector,
            "payload": {"text": doc["text"]} # Storing raw text alongside the vector
        }
        qdrant_records.append(record)
        
    # Push records live into the containerized database
    upsert_vectors(COLLECTION_NAME, qdrant_records)
