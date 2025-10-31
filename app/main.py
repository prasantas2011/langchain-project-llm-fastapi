from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from app.services.rag_service import RAGService

app = FastAPI(title="Chat with Your PDF")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

rag_service = RAGService(index_dir="faiss_index")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Chat with Your PDF API!"}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    msg = rag_service.build_index(file_path)
    return JSONResponse({"message": msg})

@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    try:
        answer = rag_service.ask_question(query)
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.post("/ask_with_fallback/")
async def ask_question_with_fallback(query: str = Form(...)):
    try:
        answer = rag_service.ask_question_with_fallback(query)
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.post("/ask_with_fallback_new/")
async def ask_question_with_fallback_new(query: str = Form(...)):
    try:
        answer = rag_service.ask_question_with_fallback_new(query)
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)})
