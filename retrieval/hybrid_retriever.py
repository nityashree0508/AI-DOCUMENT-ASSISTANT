from retrieval.retriever import Retriever
from retrieval.fusion import ReciprocalRankFusion
from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, chunks, vector_db):

        self.faiss = Retriever(vector_db)

        self.chunks = chunks

        tokenized = [
            doc.page_content.lower().split()
            for doc in chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

        self.rrf = ReciprocalRankFusion()

    def retrieve(self, query, k=5):

        faiss_docs = self.faiss.retrieve(query)

        tokenized_query = query.lower().split()

        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_docs = sorted(
            zip(self.chunks, bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )

        bm25_docs = [
            doc
            for doc, _ in bm25_docs[:k]
        ]

        return self.rrf.fuse(
            faiss_docs,
            bm25_docs,
            k
        )