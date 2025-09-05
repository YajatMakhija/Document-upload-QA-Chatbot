import os
import torch
from transformers import  AutoModelForCausalLM
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from search import Search

# Query = ""
# vectorstore = None


class LLM:
    def __init__(self, Query: str, vectorstore: str):
        self.Query = Query
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",   # Gemini embedding model
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
        self.vectorstore = vectorstore

    def Embed_query_and_generate_response(self):
        query_embedding = self.embeddings.embed_query(self.Query)
        self.sr = Search(self.Query, self.vectorstore)
        response = self.sr.search()
        # retrieved_chunks = response 
        retrieved_chunks = [doc.page_content for doc in response]
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""
        Answer the following question using only the context provided.
        If the answer is not in the context, say "I don't know."

        Context:     {context}

        Question:   {self.Query}"""
        final_answer = self.model.invoke(prompt)
        return final_answer.content



