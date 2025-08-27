from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import uuid

from src.services import pdf_parser, rag_pipeline, embeddings

router = APIRouter()

# --- healthcheck para o Docker ---
@router.get("/health")
def health():
    return {"status": "ok"}

# --- inicializa nova sessão (coleção vazia) ---
@router.post("/start_chat")
def start_chat():
    session_id = str(uuid.uuid4())
    # cria/garante uma coleção vazia para esta sessão
    embeddings.reset_session(session_id)
    return {"session_id": session_id}

# --- upload/indexação de PDFs para a sessão ---
@router.post("/documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")
    if not session_id:
        raise HTTPException(status_code=400, detail="Faltou session_id.")

    all_chunks = []
    for file in files:
        # salva o PDF (opcional; útil só pra debug/inspeção)
        save_path = os.path.join("RAG-Challenge/data/uploaded_pdfs", file.filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # extrai texto + chunk
        chunks = pdf_parser.extract_text_and_chunk(save_path)
        all_chunks.extend(chunks)

    # grava embeddings no vetor DB da sessão
    indexed_points = embeddings.add_chunks_to_index(chunks=all_chunks, session_id=session_id)

    return JSONResponse(
        content={
            "message": "Documents processed successfully",
            "documents_indexed": len(files),
            "total_chunks": len(all_chunks),
            "indexed_points": indexed_points,
        }
    )

# --- pergunta com RAG, por sessão ---
class QuestionBody(BaseModel):
    question: str
    session_id: str

@router.post("/question")
def ask_question(payload: QuestionBody):
    if not payload.question:
        raise HTTPException(status_code=400, detail="Missing 'question'.")
    if not payload.session_id:
        raise HTTPException(status_code=400, detail="Missing 'session_id'.")

    response = rag_pipeline.answer_question(
        question=payload.question,
        session_id=payload.session_id
    )
    return response
