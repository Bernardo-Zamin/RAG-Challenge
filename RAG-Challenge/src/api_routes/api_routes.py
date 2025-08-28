"""API routes for document upload, chat session, and question answering."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List
from src.services import embeddings, rag_pipeline
from src.models.models import AIResponse, UploadResponse, QuestionRequest
import os
import uuid


router = APIRouter()


@router.post("/start_chat", response_model=dict)
def start_chat():
    """Create a new chat session."""
    return {"session_id": str(uuid.uuid4())}


@router.post("/documents", response_model=UploadResponse)
async def upload_documents(
    session_id: str = Form(...), files: List[UploadFile] = File(...)
) -> UploadResponse:
    """
    Upload PDF documents, save them to the server, index them for the given session,
    and return statistics about the indexing process.

    Args:
        session_id (str): The unique identifier for the chat session.
        files (List[UploadFile]): List of PDF files to upload and index.

    Returns:
        UploadResponse: Contains a message, number of documents indexed,
                        total chunks processed, and indexed points.
    """
    all_chunks = 0
    indexed = 0
    for file in files:
        path = os.path.join("RAG-Challenge/data/uploaded_pdfs", file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        info = embeddings.index_pdf(path, session_id)
        all_chunks += info["total_chunks"]
        indexed += info["indexed_points"]
    return UploadResponse(
        message="Documents processed successfully",
        documents_indexed=len(files),
        total_chunks=all_chunks,
        indexed_points=indexed,
    )


@router.post("/question", response_model=AIResponse)
async def ask_question(payload: QuestionRequest) -> AIResponse:
    """
    Answer a question for a given chat session using the RAG pipeline.

    Args:
        payload (QuestionRequest): Contains the question and session_id.

    Returns:
        AIResponse: The answer and references from the RAG pipeline.
    """
    question = payload.question
    session_id = payload.session_id
    if not question or not session_id:
        raise HTTPException(
            status_code=400, detail="Missing 'question' or 'session_id'."
        )

    answer_data = rag_pipeline.answer_question(question, session_id)
    return AIResponse(
        answer=answer_data["answer"],
        references=answer_data["references"],
    )
