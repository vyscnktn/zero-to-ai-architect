import requests
import numpy as np

def get_embedding(text, model_name="nomic-embed-text"):
    """Fetches raw 768-dimensional float coordinates from local Ollama container."""
    url = "http://localhost:11434/api/embed"
    payload = {"model": model_name, "input": text}
    response = requests.post(url, json=payload)
    return response.json()["embeddings"][0]

def calculate_cosine_similarity(vector_a, vector_b):
    """Calculates the exact geometric angle alignment between two arrays."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

if __name__ == "__main__":
    print("=== Milestone 1 & 2 Laboratory: Embedding Mechanics ===")
    
    # Baseline Reference Document (What we store in our 'database')
    base_document = "Cybersecurity protocols in modern defense infrastructure enforce strict zero-trust networks."
    print(f"\n[Base Document]: '{base_document}'")
    
    # Test Sentences designed to challenge the vector grid
    test_cases = {
        "Conceptually Close (Different Words)": 
            "Military digital systems safeguard critical government assets using highly restricted access barriers.",
        
        "Context Flipped (Same Keywords, Reversed Meaning)": 
            "Cybersecurity protocols in modern defense infrastructure are completely broken, creating zero secure networks.",
        
        "Irrelevant Content (No Relationship)": 
            "Baking perfect sourdough bread requires precise  ambient humidity and hydration calculations."
    }
    
    # Compute vector for the base document
    base_vector = get_embedding(base_document)
    
    print("\n[*] Running Geometric Similarity Analysis...\n")
    print("-" * 80)
    
    for label, text in test_cases.items():
        # Compute vector for test case
        test_vector = get_embedding(text)
        
        # Calculate the mathematical alignment score
        score = calculate_cosine_similarity(base_vector, test_vector)
        
        print(f"🔬 [Test Category]: {label}")
        print(f"   -> String: \"{text}\"")
        print(f"   -> 📐 Cosine Similarity Score: {score:.4f}")
        print("-" * 80)
