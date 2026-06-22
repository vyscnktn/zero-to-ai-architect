# Zero to AI & Cloud Systems Architect Journey 🚀

Welcome to my personal engineering ledger. This repository documents my step-by-step evolution from core Machine Learning/Deep Learning foundations into a full-fledged **AI & Cloud Systems Architect**. 

Instead of relying on third-party cloud APIs out of the box, the entire system is built, tested, and audited locally inside an isolated Linux environment (Fedora, Docker, Ollama) before being scaled globally using **AWS** and **Terraform**.

---

## 🗺️ Architectural Roadmap

### 📍 Milestone 1: Local LLM Orchestration & Automation (Completed)
* **Core Concepts:** Subword Tokenization, High-Dimensional Weights, Local Inference Control, Structural Programmatic Payloads.
* **Stack:** Python 3, Ollama, Qwen 2.5 Coder (7B).
* **Implementation:** Built a local Python orchestration layer that hooks directly into a native AI backend. Programmatically ingested raw source files to create an automated **Static Application Security Testing (SAST)** tool that audits C++ code for memory vulnerabilities (e.g., Buffer Overflows).

### 📍 Milestone 2: Containerized Vector Databases & Semantic Search (Completed)
* **Core Concepts:** High-Dimensional Vector Spaces, HNSW (Hierarchical Navigable Small World) Graph Indexing, Cosine Similarity, Multi-Container Orchestration, Environmental Portability.
* **Stack:** Docker, Docker Compose, Qdrant Vector DB, Ollama, Python 3, Qwen 2.5 Coder (7B).
* **Architectural Pivot:** Initially designed to run on native Linux systemd daemons, the orchestration layer was intentionally refactored into a containerized multi-service topology. This architectural shift bypasses host-specific systemd API limitations regarding embedding compilation, isolates the development sandboxes, and guarantees the entire pipeline can be spun up seamlessly on any host machine using a single configuration manifest.
* **Implementation:** 1. Configured a declarative `docker-compose.yml` manifest managing a Qdrant storage node and an isolated Ollama engine booted with environmental embedding matrix acceleration enabled (`OLLAMA_EMBEDDINGS=1`).
  2. Developed `vector_indexer.py` to programmatically ingest raw unstructured English data, compute high-dimensional spatial coordinate embeddings locally via the containerized model, and upsert them as `Points` into Qdrant.
  3. Developed `vector_search.py` to convert incoming ad-hoc semantic user queries into vectors and search the HNSW index using mathematical Cosine Similarity scoring.

### ⏳ Milestone 3: Production-Grade Local RAG (Retrieval-Augmented Generation) [Up Next]
* **Core Concepts:** Context-Window Optimization, Information Retrieval, Document Chunking Strategies, Hallucination Suppression.
* **Stack:** LangChain, Python.
* **Objective:** Ingest system documentation PDFs, chunk them semantically, and force the local LLM to answer queries based strictly on the retrieved local data matrix.

### ⏳ Milestone 4: Managed MLOps & Infrastructure as Code (Cloud Scaling)
* **Core Concepts:** Cloud Native AI Architectures, Managed Vector Indices, Enterprise Security.
* **Stack:** AWS (S3, Glue, Bedrock, SageMaker), Terraform.
* **Objective:** Translate the verified local pipeline into a scalable, secure AWS infrastructure matrix using declarative configuration files.

---

## 📂 Project Directory Structure

```text
zero-to-ai-architect/
│
├── README.md                 # Project showcase and architectural roadmap
│
├── 01-local-llm-core/        # Milestone 1: Local Core Orchestration
│   ├── sast_engine.py        # Python SAST controller script
│   └── target_code.cpp       # Vulnerable target C++ source file
│
└── 02-vector-databases/      # Milestone 2: Multi-Container Storage & Search
    ├── docker-compose.yml    # Infrastructure-as-Code service definition
    ├── vector_indexer.py     # Embedding ingestion and indexing engine
    └── vector_search.py      # Mathematical search and retrieval layer
