from datasets import Dataset
from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from evaluation.questions import QUESTIONS

from config import DATA_DIR

from ingestion.loader import PDFLoader
from ingestion.chunking import Chunker
from ingestion.embeddings import EmbeddingModel

from retrieval.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever

from graph.llm import GeminiLLM
from graph.workflow import RAGWorkflow


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

questions = QUESTIONS

answers = []

contexts = []

ground_truths = []

for question in questions:

    result = workflow.graph.invoke(
        {
            "question": question
        }
    )

    answers.append(
        result["answer"]
    )

    contexts.append(
        [result["context"]]
    )

    ground_truths.append(
        result["answer"]
    )

dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
)

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
)

print(result)