import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


class PDFIngestion:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.pkl_path = "pdf_documents.pkl"
        self.index_path = "vectorstore_index"
        self.vectorstore = None

    def load_file(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()

        if len(documents) == 0:
            raise ValueError("No documents found")

        if len(documents) > int(os.getenv("MAX_PAGES")):
            raise ValueError(f"Too many documents found: {len(documents)}")

        # Save documents temporarily (optional)
        with open(self.pkl_path, "wb") as f:
            pickle.dump(documents, f)

        return documents

    def create_embeddings(self, documents, vectorstore):
        # Chunk documents
        self.vectorstore = vectorstore


        count = 0
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",   # Gemini embedding model
                google_api_key=os.getenv("GEMINI_API_KEY")
                )
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save vectorstore locally
        vectorstore.save_local(self.index_path)
        count+=1

        return vectorstore

    def delete_temp_files(self):
        # Delete pickle file
        if os.path.exists(self.pkl_path):
            os.remove(self.pkl_path)
            print("Temporary pickle file deleted.")

        # Optionally delete original PDF
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print("Original PDF deleted.")

