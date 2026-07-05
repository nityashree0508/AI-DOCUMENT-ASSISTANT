from fastapi import FastAPI
from pydantic import BaseModel

from config import DATA_DIR

from ingestion.loader import PDFLoader
from ingestion.chunking import Chunker
from ingestion.embeddings import EmbeddingModel

from retrieval.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever

from graph.llm import GeminiLLM
from graph.workflow import RAGWorkflow


app = FastAPI(
    title="AI Document Assistant"
)


class ChatRequest(BaseModel):
    question: str


print("Loading RAG pipeline...")

loader = PDFLoader(DATA_DIR)

docs = loader.load_documents()

chunker = Chunker()

chunks = chunker.split(docs)

embedding_model = EmbeddingModel().get_model()

vector_store = VectorStore(embedding_model)

vector_db = vector_store.create(chunks)

retriever = HybridRetriever(
    chunks,
    vector_db
)

llm = GeminiLLM().get_llm()

workflow = RAGWorkflow(
    retriever,
    llm
)

print("RAG pipeline loaded.")


@app.post("/chat")
def chat(request: ChatRequest):

    result = workflow.graph.invoke(
        {
            "question": request.question
        }
    )

    return {
        "answer": result["answer"]
    }