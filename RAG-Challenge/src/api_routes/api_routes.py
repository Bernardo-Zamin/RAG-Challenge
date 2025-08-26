"""API routes for document upload and question answering.

in the RAG-Challenge application.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from src.services import pdf_parser, rag_pipeline, embeddings
import os

router = APIRouter()


@router.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload PDF documents.

    Extract and chunk their text, add the chunks to the embeddings index, and
    return a summary.
    """
    all_chunks = []

    for file in files:
        file_path = os.path.join(
            "RAG-Challenge/data/uploaded_pdfs",
            file.filename
        )
        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = pdf_parser.extract_text_and_chunk(file_path)
        embeddings.add_chunks_to_index(chunks)
        all_chunks.extend(chunks)

    return JSONResponse(
        content={
            "message": "Documents processed successfully",
            "documents_indexed": len(files),
            "total_chunks": len(all_chunks),
        }
    )


@router.post("/question")
async def ask_question(payload: dict):
    """
    Answer a question using the RAG pipeline.

    Expects a JSON payload with a 'question' field.
    """
    question = payload.get("question")
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Missing 'question' field in request."
        )

    response = rag_pipeline.answer_question(question)
    return response
