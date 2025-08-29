from sentence_transformers import SentenceTransformer
import numpy as np
from . import pdf_parser
from ..vector_database.qdrant_store import QdrantStore
from concurrent.futures import ThreadPoolExecutor

model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384

# Each session uses its own collection (e.g., "session_<uuid>")
def ensure_store(session_id: str) -> QdrantStore:
    return QdrantStore(collection=f"session_{session_id}", dim=DIM)

def encode_texts(texts: list) -> np.ndarray:
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    return embs.astype(np.float32)

def index_pdf(path: str, session_id: str) -> dict:
    store = ensure_store(session_id)
    chunks = pdf_parser.extract_text_and_chunk(path)

    # Parallelize text encoding
    with ThreadPoolExecutor() as executor:
        embs = list(executor.map(lambda c: encode_texts([c["text"]])[0], chunks))

    store.upsert(embs, chunks)
    return {"total_chunks": len(chunks), "indexed_points": len(chunks)}

def search(question: str, session_id: str, top_k: int = 3, source: str | None = None):
    store = ensure_store(session_id)
    q = encode_texts([question])[0]
    return store.search(q, top_k=top_k, source_filter=source)
