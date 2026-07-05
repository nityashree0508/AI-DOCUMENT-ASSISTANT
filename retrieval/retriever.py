from config import TOP_K


class Retriever:

    def __init__(self, vector_db):

        self.retriever = vector_db.as_retriever(

            search_type="similarity",

            search_kwargs={

                "k": TOP_K

            }

        )

    def retrieve(self, query):

        return self.retriever.invoke(query)