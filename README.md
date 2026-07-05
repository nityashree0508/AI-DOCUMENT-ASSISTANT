# AI Document Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system that lets users upload documents and ask natural-language questions, with answers grounded in the document content via semantic retrieval and cross-encoder reranking.

🔗 **Live Repo:** [github.com/nityashree0508/AI-Document-Assistant](https://github.com/nityashree0508/AI-Document-Assistant)

---

## Overview

Traditional keyword search fails when users don't know the exact terms used in a document. This project solves that by combining dense semantic retrieval with LLM-based generation, orchestrated through a structured LangGraph workflow — allowing users to query PDFs conversationally and get accurate, context-grounded answers instead of hallucinated ones.

## Features

- 📄 **PDF Ingestion Pipeline** — upload and parse documents automatically
- ✂️ **Recursive Document Chunking** — splits text intelligently to preserve context across chunk boundaries
- 🧠 **BGE Embedding Model** — converts text chunks into dense vector representations
- ⚡ **FAISS Vector Store** — fast similarity search over embedded chunks
- 🎯 **Cross-Encoder Reranking** — re-scores retrieved chunks for higher precision before generation
- 🔁 **LangGraph Orchestration** — structured, stateful RAG workflow instead of a single linear chain
- 💬 **Gemini-based Answer Generation** — grounded, context-aware responses using custom prompt templates
- 🖥️ **FastAPI Backend + Streamlit UI** — clean API layer with an interactive frontend for real-time Q&A

## Architecture

```
User Query
    │
    ▼
PDF Ingestion → Recursive Chunking → BGE Embeddings → FAISS Index
                                                            │
                                                            ▼
                                              Semantic Retrieval (top-k chunks)
                                                            │
                                                            ▼
                                              Cross-Encoder Reranking
                                                            │
                                                            ▼
                                        LangGraph Workflow → Gemini (Answer Generation)
                                                            │
                                                            ▼
                                                  Streamlit UI (Response)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| Orchestration | LangGraph |
| Embeddings | BGE |
| Vector Store | FAISS |
| Reranking | Cross-Encoder |
| LLM | Google Gemini |
| Prompting | ChatPromptTemplate (LangChain) |

## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/nityashree0508/AI-Document-Assistant.git
cd AI-Document-Assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your API keys
# Create a .env file with:
# GEMINI_API_KEY=your_key_here

# Run the FastAPI backend
uvicorn main:app --reload

# In a separate terminal, run the Streamlit frontend
streamlit run app.py
```

## Usage

1. Launch the app and upload a PDF document
2. Ask a question related to the document's content
3. The system retrieves the most relevant chunks, reranks them, and generates a grounded answer

## Future Improvements

- Multi-document support with source citation
- Conversation memory for follow-up questions
- Support for additional file formats (DOCX, TXT)
- Deployment via Docker

## Author

**Nitya Shree**
B.E. Electrical and Electronics Engineering, SSN College of Engineering
