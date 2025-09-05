from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import shutil
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from search import Search 
from pdf_ingestion import PDFIngestion
from dotenv import load_dotenv
import logging

app = FastAPI()
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

uri = os.getenv("URI")
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["Document-Metadata"]
collection = db["mycollection"]

vectorstores = {}  
descriptions = {}  
file_paths = {}  


# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    index_path = os.path.join("static", "index.html")
    return FileResponse(index_path)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        pdf = PDFIngestion(file_path)
        docs = pdf.load_file()
        
        # Generate vectorstore and description
        vectorstore = pdf.create_embeddings(docs, file.filename, descriptions=descriptions)
        if isinstance(vectorstore, str) and "Error" in vectorstore:
            raise HTTPException(status_code=500, detail=vectorstore)
        
        vectorstores[file.filename] = vectorstore
        file_paths[file.filename] = file_path
        
        document = {
            "file_name": file.filename,
            "size": os.path.getsize(file_path),
            "pages": len(docs),
            "description": descriptions.get(file.filename, "No description generated")
        }
        result = collection.insert_one(document)
        print(f"Inserted document with ID: {result.inserted_id}, Description: {descriptions.get(file.filename)}")

        return {
            "file_name": file.filename,
            "size": os.path.getsize(file_path),
            "pages": len(docs),
            "description": descriptions.get(file.filename, "No description generated"),
            "message": "Ingestion complete ✅"
        }
    except Exception as e:
        print(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    model: str = "gemini-2.5-flash"  

@app.post("/query")
async def query(req: QueryRequest):
    # Validate model name

    if not vectorstores:
        logger.warning("Query attempted with no vectorstores initialized.")
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload first.")
    
    try:
        logger.info(f"Processing query with model: {req.model}")
        sr = Search(vectorstores=vectorstores, descriptions=descriptions)
        response = sr.ask(req.query)
        logger.info(f"Query processed: {req.query}, Answer: {response['answer']}, Selected: {response.get('selected_document', 'N/A')}")
        return {
            "answer": response["answer"],
            "sources": response["sources"],
            "chat_history": response["chat_history"],
            "selected_document": response.get("selected_document", "none")
        }
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/files")
async def list_files():
    return {"files": list(vectorstores.keys())}

@app.post("/delete")
async def delete():
    try:
        logger.info(f"Starting deletion of all files. Current vectorstores: {list(vectorstores.keys())}")
        # Delete all files and vectorstores
        for file_name in list(file_paths.keys()):
            file_path = file_paths.get(file_name)
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            else:
                logger.warning(f"File path not found or does not exist: {file_path}")

            index_path = f"vectorstore_{file_name}_index"
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
                logger.info(f"Vectorstore for {file_name} deleted.")
            else:
                logger.warning(f"Vectorstore path not found: {index_path}")

        # Clear global dictionaries
        vectorstores.clear()
        file_paths.clear()
        logger.info("All vectorstores and file paths cleared.")

        return {"message": "All files and vectorstores deleted successfully."}
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)