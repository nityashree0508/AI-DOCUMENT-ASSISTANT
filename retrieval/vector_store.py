from langchain_community.vectorstores import FAISS

from config import VECTOR_DB_PATH


class VectorStore:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def create(self, chunks):

        vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_model
        )

        vector_db.save_local(VECTOR_DB_PATH)

        return vector_db

    def load(self):

        return FAISS.load_local(
            VECTOR_DB_PATH,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
