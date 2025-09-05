import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Search:
    def __init__(self, vectorstores: dict, descriptions: dict, model="gemini-2.5-flash"):
        """
        Full RAG pipeline with agentic routing:
        - Multiple FAISS vectorstores
        - Gemini LLM for routing and answering
        - Conversation memory (last 6 turns)
        """
        self.vectorstores = vectorstores
        self.descriptions = descriptions

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, k=6, output_key="answer"
        )

        self.retrievers = {
            name: vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5}
            )
            for name, vs in vectorstores.items()
        }

        self.router_template = """You are an expert at routing a user question to the most relevant document.
Available documents:
{datasource_list}
Based on the question: "{question}", return ONLY the filename of the most relevant document or 'none' if no document is relevant. Do not explain."""
        self.router_prompt = ChatPromptTemplate.from_template(self.router_template)
        self.router_chain = self.router_prompt | self.llm | StrOutputParser()

        self.rag_template = """Answer the following question using only the context provided.
If the answer is not in the context, say "I don't know."
Context: {context}
Question: {question}"""
        self.rag_prompt = ChatPromptTemplate.from_template(self.rag_template)
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()

    def ask(self, query: str):
        """Ask a question, route to the appropriate vectorstore, get answer + sources + chat history"""
        datasource_list = "\n".join([f"- {name}: {desc}" for name, desc in self.descriptions.items()])
        
        selected_datasource = self.router_chain.invoke({
            "question": query,
            "datasource_list": datasource_list
        }).strip()
        logger.info(f"Query: {query}, Selected document: {selected_datasource}")

        if selected_datasource == "none" or selected_datasource not in self.vectorstores:
            answer = "I don't know. No relevant document was found."
            sources = []
        else:
            documents = self.retrievers[selected_datasource].invoke(query)
            context = "\n\n".join([doc.page_content for doc in documents])
            answer = self.rag_chain.invoke({"context": context, "question": query})
            sources = [
                f"{doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})"
                for doc in documents
            ]

        self.memory.save_context({"question": query}, {"answer": answer})

        return {
            "answer": answer,
            "sources": list(set(sources)),  
            "chat_history": [
                {"role": m.type, "content": m.content}
                for m in self.memory.chat_memory.messages
            ]
        }