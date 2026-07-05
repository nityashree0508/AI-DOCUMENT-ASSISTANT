from graph.prompts import GUARDRAIL_PROMPT


class Guardrail:

    def __init__(self, llm):

        self.llm = llm

    def check(self, question):

        messages = GUARDRAIL_PROMPT.invoke(
            {
                "question": question
            }
        )

        response = self.llm.invoke(messages)

        return response.content.strip().upper()