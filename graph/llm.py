from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY


class GeminiLLM:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(

            model="gemini-2.5-flash",

            google_api_key=GOOGLE_API_KEY,

            temperature=0.2

        )

    def get_llm(self):

        return self.llm