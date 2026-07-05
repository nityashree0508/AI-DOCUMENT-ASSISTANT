from langchain.memory import ConversationBufferMemory


class ChatMemory:

    def __init__(self):

        self.memory = ConversationBufferMemory(
            return_messages=True
        )

    def load(self):

        return self.memory.load_memory_variables(
            {}
        )

    def save(

        self,

        question,

        answer

    ):

        self.memory.save_context(

            {

                "input": question

            },

            {

                "output": answer

            }

        )