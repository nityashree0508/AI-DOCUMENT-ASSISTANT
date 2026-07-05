from collections import defaultdict


class ReciprocalRankFusion:

    def fuse(
        self,
        faiss_docs,
        bm25_docs,
        k=60,
        top_k=10
    ):

        scores = defaultdict(float)

        doc_lookup = {}

        for rank, doc in enumerate(faiss_docs, start=1):

            key = doc.page_content

            doc_lookup[key] = doc

            scores[key] += 1 / (k + rank)

        for rank, doc in enumerate(bm25_docs, start=1):

            key = doc.page_content

            doc_lookup[key] = doc

            scores[key] += 1 / (k + rank)

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc_lookup[key]
            for key, _ in ranked[:top_k]
        ]