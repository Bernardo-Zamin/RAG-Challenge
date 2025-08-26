"""Pydantic models for question answering and document upload responses."""

from pydantic import BaseModel
from typing import List


class QuestionRequest(BaseModel):
    """Request model for submitting a question."""

    question: str


class AnswerResponse(BaseModel):
    """Response model containing the answer and references."""

    answer: str
    references: List[str]


class UploadResponse(BaseModel):
    """Response model for document upload results."""

    message: str
    documents_indexed: int
    total_chunks: int
