from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os

from pdf_ingestion import PDFIngestion
from llm import LLM

app = FastAPI()

# Store vectorstore globally
vectorstore = None

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    index_path = os.path.join("static", "index.html")
    return FileResponse(index_path)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global vectorstore

    # Save file
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Ingest + create embeddings
    pdf = PDFIngestion(file_path)
    docs = pdf.load_file()
    vectorstore = pdf.create_embeddings(docs, vectorstore=vectorstore)

    return {
        "file_name": file.filename,
        "size": os.path.getsize(file_path),
        "pages": len(docs),
        "message": "Ingestion complete ✅"
    }


class QueryRequest(BaseModel):
    query: str
    model: str = "gemini-pro"


@app.post("/query")
async def query(req: QueryRequest):
    global vectorstore
    if vectorstore is None:
        return {"error": "No document uploaded yet. Please upload first."}

    rag_llm = LLM(req.query, vectorstore=vectorstore)
    answer = rag_llm.Embed_query_and_generate_response()
    return {"answer": answer}





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
