# ChromaDB RAG Pipeline: Core Concepts & Implementation

This repository contains a clean, end-to-end implementation of a Retrieval-Augmented Generation (RAG) pipeline utilizing **ChromaDB** as a local vector store, **LangChain** for orchestration, and **OpenAI** for high-dimensional text embeddings and LLM generation.

---

## 📌 Theoretical Architecture & Notes

### 1. Relational Databases vs. Vector Databases
* **Traditional Relational Databases (e.g., MySQL, PostgreSQL):** Ideal for structured tables, CSVs, or exact value matching. They rely on rigid schemas and struggle with unstructured data because they cannot natively infer contextual meaning. Finding similar text or image items requires intensive manual tagging.
* **Vector Databases (e.g., ChromaDB):** Built specifically to handle unstructured data like PDFs, text corpora, audio, or images. Instead of scanning for exact character-matching text strings, they execute mathematical **semantic similarity searches** to look up data based on actual intent and meaning.

### 2. The Core Pipeline Mechanics
1. **Data Ingestion:** Raw unstructured text files are read directly into the application memory.
2. **Text Chunking:** The larger document corpus is divided into shorter, manageable sections based on `chunk_size` and `chunk_overlap`. This ensures the text cleanly fits within the strict input context windows of Large Language Models.
3. **Vector Embeddings:** An embedding model converts the text fragments into a high-dimensional vector space. Contextually similar concepts naturally align close to one another within this mathematical field.
4. **Indexing & Retrieval:** ChromaDB uses specialized vector indexes to instantly pre-cluster and isolate relevant text coordinates, avoiding the need to compute geometric distances across the entire database point-by-point during a query.

### 3. Key Pipeline Hyperparameters
* **`chunk_size`**: The absolute maximum character limit allowed per isolated chunk.
* **`chunk_overlap`**: The boundary characters shared between consecutive text splits to prevent contextual loss at edge cuts.
* **`k` (`search_kwargs`)**: The parameter that determines the number of matching document chunks the database engine returns.
* **`chain_type="stuff"`**: Instructs LangChain to bundle ("stuff") all the retrieved text fragments into a single prompt payload for the LLM.

---

## 🛠️ Complete Implementation

This script automates downloading the document dataset, fragmenting the text corpus, generating high-dimensional embeddings, setting up local disk storage persistence, and querying the final LLM chain interface.

### Prerequisites

Install the necessary dependencies via pip:
```bash
pip install langchain langchain-community langchain-openai chromadb ticktoken
