# AI Document Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system for research paper question answering — combining hybrid retrieval, cross-encoder reranking, and guardrails to deliver accurate, grounded answers instead of hallucinated ones.

🔗 **Live Repo:** [github.com/nityashree0508/AI-Document-Assistant](https://github.com/nityashree0508/AI-Document-Assistant)

---

## Overview

Standard RAG pipelines relying on pure semantic search often miss relevant chunks when queries use different vocabulary than the source document, or fail to reject irrelevant/adversarial queries. This project addresses that with a hybrid retrieval strategy (dense + keyword search fused via RRF), cross-encoder reranking for precision, automatic query rewriting, and guardrails against prompt injection and off-topic queries — orchestrated through a structured LangGraph workflow.

## ✨ Features

- 📄 Research paper question answering
- 🔍 Hybrid Retrieval (FAISS + BM25 + RRF)
- 🎯 Cross-Encoder reranking for improved relevance
- 🧠 LangGraph workflow orchestration
- ✍️ Automatic query rewriting
- 🛡️ Prompt injection & off-topic query guardrails
- 🤖 Google Gemini 2.5 Flash integration
- ⚡ FastAPI REST backend
- 🌐 Streamlit web interface
- 📚 Semantic document search using BGE embeddings

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| LLM | Google Gemini 2.5 Flash |
| Frameworks | LangChain, LangGraph |
| Embeddings | BAAI BGE Small v1.5 |
| Vector Database | FAISS |
| Keyword Search | BM25 |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | MS MARCO MiniLM Cross-Encoder |
| Evaluation | RAGAS (integration in progress) |

## 🏗️ Architecture

```
PDF
 │
 ▼
Document Loader
 │
 ▼
Recursive Text Chunking
 │
 ▼
BGE Embeddings
 │
 ▼
FAISS Vector Store
 │
 ▼
Hybrid Retrieval (FAISS + BM25)
 │
 ▼
Reciprocal Rank Fusion
 │
 ▼
Cross-Encoder Reranker
 │
 ▼
Guardrail
 │
 ▼
Query Rewriter
 │
 ▼
Gemini 2.5 Flash
 │
 ▼
Answer
```

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

1. Launch the app and upload a research paper (PDF)
2. Ask a question related to the document's content
3. The system retrieves relevant chunks via hybrid search, fuses and reranks them, rewrites the query if needed, applies guardrails, and generates a grounded answer via Gemini 2.5 Flash

## Future Improvements

- Full RAGAS evaluation integration
- Multi-document support with source citation
- Conversation memory for follow-up questions
- Docker deployment

## Author

**Nitya Shree**
B.E. Electrical and Electronics Engineering, SSN College of Engineering
