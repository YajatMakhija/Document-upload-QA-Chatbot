from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import os
import pickle

load_dotenv()

# QUERY ="" 


class Search:
    def __init__(self, QUERY, vectorstore):
        self.QUERY = QUERY
        self.vectorstore = vectorstore

    def search(self):
        # Directly pass the query text; FAISS handles embedding internally
        results = self.vectorstore.similarity_search(self.QUERY, k=5)
        return results

    
