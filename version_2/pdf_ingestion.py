import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

load_dotenv()

class PDFIngestion:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.pkl_path = "pdf_documents.pkl"

    def load_file(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()

        if len(documents) == 0:
            raise ValueError("No documents found")

        if len(documents) > int(os.getenv("MAX_PAGES")):
            raise ValueError(f"Too many documents found: {len(documents)}")

        with open(self.pkl_path, "wb") as f:
            pickle.dump(documents, f)

        return documents

    def create_embeddings(self, documents, filename, descriptions: dict = {}):
        index_path = f"vectorstore_{filename}_index"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        if filename in descriptions:
            pass
        else:
            descriptions[filename] = self.create_description(filename, documents, descriptions)
            print(f"Description created: {descriptions[filename]}")
        embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",   # Gemini embedding model
                google_api_key=os.getenv("GEMINI_API_KEY")
                )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print(descriptions[filename])
        vectorstore.save_local(index_path)

        return vectorstore

    def create_description(self, filename, documents, descriptions: dict):
        
        if filename in descriptions:
            return "The Description of file already exists"
        else:
            summary_prompt = "Summarize this document in one sentence: " + " ".join([doc.page_content for doc in documents[:3]])
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0
            )

            summary = llm.invoke(summary_prompt).content
            descriptions[filename] = summary
            return summary

    def delete_temp_files(self, file_path):
        # Delete pickle file
        if os.path.exists(self.pkl_path):
            os.remove(self.pkl_path)
            print("Temporary pickle file deleted.")

        # Optionally delete original PDF
        if os.path.exists(file_path):
            os.remove(file_path)
            print("Original PDF deleted.")

