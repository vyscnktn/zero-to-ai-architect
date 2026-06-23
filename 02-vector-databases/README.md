# Milestone 2: Containerized Vector Databases & Semantic Search Laboratory

In this laboratory environment, we deployed **Qdrant (Vector Database)** and **Ollama (Nomic-Embed-Text)** inside isolated Docker containers to dissect how text embedding models map human language into geometric spaces.

## 🔬 High-Dimensional Spaces & Cosine Similarity Lab (`test_math.py`)

Using our `test_math.py` script, we analyzed how the **`nomic-embed-text`** embedding engine translates sentences into a **768-dimensional coordinate system** and calculates their alignment angle using **Cosine Similarity**.

### 📊 Experimental Results & Key Takeaways

Using a cybersecurity base document as a benchmark reference, our system revealed critical architectural insights under pressure:

1. **The Token Injection Test (Sourdough Bread Modification):**
   * Pushing a completely irrelevant sentence about baking sourdough yielded a base similarity score of **0.4008**.
   * Simply injecting the single word **"secure"** into that exact baking sentence forced the score up to **0.4377**.
   * *Takeaway:* A single highly weighted token can physically warp the entire coordinate array, pulling the sentence geometrically closer to the cybersecurity neighborhood.

2. **The Great Vector Trap (Context Flipping):**
   * **Conceptually Close (Different Words):** `0.6088` (Describes actual asset protection protocols using zero vocabulary overlap).
   * **Context Flipped (Same Keywords, Reversed Logic):** **`0.8951`** (Uses identical terms but states that the infrastructure is completely broken with zero security).
   * *Takeaway:* **Embedding models evaluate topical similarity (what the text is about), not logical agreement or sentiment truth values.** Because both sentences occupy the exact same niche defense domain, their mathematical vector arrays point in almost the exact same directional angle, triggering an incredibly high score.

### 🎯 Core Systems Architect Takeaway
Vector databases like Qdrant are highly efficient at finding the correct "context folder" or topic neighborhood using embedding models like **`nomic-embed-text`**. However, they lack the capacity for logical reasoning. To build a robust system, we must take these retrieved contexts and feed them into the attention mechanism of a separate text generation LLM like **Ollama (`qwen2.5-coder:7b`)** to compute the actual logic, facts, and underlying constraints.
