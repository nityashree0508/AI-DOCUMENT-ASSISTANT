from graph.prompts import REWRITE_PROMPT


class QueryRewriter:

    def __init__(self, llm):

        self.llm = llm

    def rewrite(
        self,
        question,
        history=""
    ):

        messages = REWRITE_PROMPT.invoke(
            {
                "history": history,
                "question": question
            }
        )

        response = self.llm.invoke(messages)

        return response.content.strip()