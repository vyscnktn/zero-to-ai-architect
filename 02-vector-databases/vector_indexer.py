import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def get_local_embedding(text, model_name="nomic-embed-text"):
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

if __name__ == "__main__":
    print("=== Local AI Architecture Laboratory: Milestone 2 ===")
    
    COLLECTION_NAME = "defense_tech_knowledge"
    
    raw_documents = [
        {"id": 1, "text": "Autonomous drone navigation systems rely heavily on edge-computing AI models."},
        {"id": 2, "text": "Cybersecurity protocols in modern defense infrastructure enforce strict zero-trust networks."},
        {"id": 3, "text": "Baking perfect sourdough bread requires precise ambient humidity and hydration calculations."}
    ]
    
    print("[*] Testing local embedding dimensions...")
    sample_vector = get_local_embedding(raw_documents[0]["text"])
    VECTOR_SIZE = len(sample_vector)
    
    # Initialize the official Qdrant SDK Client targeting your local container port
    client = QdrantClient(url="http://localhost:6333")
    
    # Drop existing collection cleanly to reset state
    print("[*] Reinitializing collection...")
    client.delete_collection(collection_name=COLLECTION_NAME)
    
    # Create fresh collection using explicit SDK Models (Bypasses manual JSON errors!)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[+] Qdrant collection '{COLLECTION_NAME}' initialized successfully with size {VECTOR_SIZE}.")
    
    # Process and build official Point structures
    points = []
    for doc in raw_documents:
        print(f"[*] Vectorizing Document ID {doc['id']}...")
        vector = get_local_embedding(doc["text"])
        
        # Wrap directly using the SDK's explicit PointStruct object
        points.append(
            PointStruct(
                id=doc["id"],
                vector=vector,
                payload={"text": doc["text"]}
            )
        )
        
    # Push live to your container via the SDK's native upsert endpoint
    print("[*] Uploading vector points to Qdrant cluster...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points
    )
    print(f"[+] Successfully indexed {len(points)} points into Qdrant using native SDK layers.")
