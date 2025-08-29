"""Main entry point for the RAG Challenge FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api_routes.api_routes import router as api_router

app = FastAPI(
    title="RAG Challenge API",
    description=("API to upload PDFs and questions via RAG with local LLM (Ollama)"),
    version="2.0.0",
)

# Allow access to Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)
