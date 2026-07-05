from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph import START, END

from graph.prompts import RAG_PROMPT, REWRITE_PROMPT
from reranker.cross_encoder import Reranker
from guardrails.guardrail import Guardrail


class GraphState(TypedDict):
    question: str
    rewritten_question: str
    docs: list
    context: str
    answer: str
    allowed: bool


class RAGWorkflow:

    def __init__(self, retriever, llm):

        self.retriever = retriever
        self.llm = llm
        self.reranker = Reranker()
        self.guardrail = Guardrail(llm)

        self.graph = self.build_graph()

    def rewrite(self, state):

        result = self.guardrail.check(
            state["question"]
        )

        if result != "ALLOW":

            return {
                "rewritten_question": state["question"],
                "answer": "Sorry, I can only answer questions related to the uploaded document."
            }

        messages = REWRITE_PROMPT.invoke(
            {
                "history": "",
                "question": state["question"]
            }
        )

        response = self.llm.invoke(messages)

        return {
            "rewritten_question": response.content.strip()
        }

    def retrieve(self, state):

        if state.get("answer"):

            return {
                "docs": []
            }

        docs = self.retriever.retrieve(
            state["rewritten_question"]
        )

        return {
            "docs": docs
        }

    def rerank(self, state):

        if state.get("answer"):

            return {
                "context": ""
            }

        docs = self.reranker.rerank(
            state["rewritten_question"],
            state["docs"]
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        return {
            "context": context
        }

    def generate(self, state):

        if state.get("answer"):

            return {
                "answer": state["answer"]
            }

        messages = RAG_PROMPT.invoke(
            {
                "context": state["context"],
                "question": state["question"]
            }
        )

        response = self.llm.invoke(messages)

        return {
            "answer": response.content
        }

    def build_graph(self):

        workflow = StateGraph(GraphState)

        workflow.add_node(
            "rewrite",
            self.rewrite
        )

        workflow.add_node(
            "retrieve",
            self.retrieve
        )

        workflow.add_node(
            "rerank",
            self.rerank
        )

        workflow.add_node(
            "generate",
            self.generate
        )

        workflow.add_edge(
            START,
            "rewrite"
        )

        workflow.add_edge(
            "rewrite",
            "retrieve"
        )

        workflow.add_edge(
            "retrieve",
            "rerank"
        )

        workflow.add_edge(
            "rerank",
            "generate"
        )

        workflow.add_edge(
            "generate",
            END
        )

        return workflow.compile()