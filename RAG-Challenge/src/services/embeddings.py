from sentence_transformers import SentenceTransformer
import numpy as np
from . import pdf_parser
from ..vector_database.qdrant_store import QdrantStore
from concurrent.futures import ThreadPoolExecutor

model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384


def ensure_store(session_id: str) -> QdrantStore:
    """
    Ensure a QdrantStore instance exists for the given session ID.

    Args:
        session_id (str): The session identifier.

    Returns:
        QdrantStore: An instance of QdrantStore for the session.
    """
    return QdrantStore(collection=f"session_{session_id}", dim=DIM)


def encode_texts(texts: list) -> np.ndarray:
    """
    Encode a list of texts into embeddings using the SentenceTransformer model.

    Args:
        texts (list): List of text strings to encode.

    Returns:
        np.ndarray: Array of embeddings as float32.
    """
    embs = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embs.astype(np.float32)


def index_pdf(path: str, session_id: str) -> dict:
    """
    Extracts text from a PDF, encodes the text chunks, and indexes them in the vector store for the given session.

    Args:
        path (str): The file path to the PDF document.
        session_id (str): The session identifier.

    Returns:
        dict: A dictionary containing the total number of chunks and indexed points.
    """
    store = ensure_store(session_id)
    chunks = pdf_parser.extract_text_and_chunk(path)

    # Parallelize text encoding
    with ThreadPoolExecutor() as executor:
        embs = list(
            executor.map(
                lambda c: encode_texts([c["text"]])[0],
                chunks
            )
        )

    store.upsert(embs, chunks)
    return {"total_chunks": len(chunks), "indexed_points": len(chunks)}


def search(
    question: str,
    session_id: str,
    top_k: int = 3,
    source: str | None = None
):
    """
    Search for relevant chunks in the vector store based on the input question.

    Args:
        question (str): The query string to search for.
        session_id (str): The session identifier.
        top_k (int, optional): Number of top results to return. Defaults to 3.
        source (str | None, optional): Optional source filter. Defaults to None.

    Returns:
        list: Search results from the vector store.
    """
    store = ensure_store(session_id)
    q = encode_texts([question])[0]
    return store.search(q, top_k=top_k, source_filter=source)
