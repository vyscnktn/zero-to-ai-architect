# Zero to AI & Cloud Architect Journey 🚀

Welcome to my personal engineering ledger. This repository documents my step-by-step evolution from core Machine Learning/Deep Learning concepts to a full-fledged **AI & Cloud Systems Architect**. 

Instead of relying on expensive cloud APIs from day one, this entire pipeline is built, tested, and broken locally inside a Linux labor environment (Fedora, Docker, Ollama) before being scaled globally using **AWS** and **Terraform**.

---

## 🗺️ Architectural Roadmap

### 📍 Milestone 1: Local LLM Orchestration & Automation (Current)
* **Core Concepts:** Subword Tokenization, Embedding Matrices, Local Inference, API-less Orchestration.
* **Stack:** Python 3, Ollama, Qwen 2.5 Coder (7B).
* **Implementation:** Moved beyond static terminal chats (`ollama run`). Built a production-ready Python bridge to turn a local LLM into an automated **DevSecOps static code analysis tool** that scans C++ source code for memory vulnerabilities (e.g., Buffer Overflows).

### ⏳ Milestone 2: Vector Databases & Geometric Similarity Search (Up Next)
* **Core Concepts:** High-Dimensional Vector Spaces, Cosine Similarity, HNSW (Hierarchical Navigable Small World) Graphs.
* **Stack:** Docker, Qdrant Vector DB.
* **Objective:** Spin up an isolated Vector DB instance locally via Docker Compose and stream mathematically structured semantic embeddings into memory.

### ⏳ Milestone 3: Production-Grade RAG (Retrieval-Augmented Generation)
* **Core Concepts:** Context-Window Management, Information Retrieval, Prompt Augmentation, Hallucination Control.
* **Stack:** LangChain / LlamaIndex, Python.
* **Objective:** Build an autonomous pipeline that reads system documentation PDFs, chunks them semantically, and forces the local LLM to answer queries based strictly on the retrieved local data.

### ⏳ Milestone 4: Infrastructure as Code (Cloud Scaling)
* **Core Concepts:** Cloud Native AI Architecture, Managed MLOps, Infrastructure as Code (IaC).
* **Stack:** AWS (S3, Glue, Bedrock, SageMaker), Terraform.
* **Objective:** Translate the entire local architecture into a cloud-native AWS matrix using declarative Terraform configuration files.

---

## 🛠️ Project Structure

```text
zero-to-ai-architect/
│
├── README.md                 # Project showcase and architectural ledger
│
└── 01-local-llm-core/        # Milestone 1: Core Orchestration
    ├── qwen_requests.py      # Local Python orchestration layer
    └── target_code.cpp       # Vulnerable source file used for the DevSecOps test
