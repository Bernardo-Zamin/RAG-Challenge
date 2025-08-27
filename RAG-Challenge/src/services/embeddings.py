import os
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMB_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384  # all-MiniLM-L6-v2

_model = SentenceTransformer(EMB_MODEL)
_qdrant = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

def _collection_name(session_id: str) -> str:
    return f"session_{session_id}"

def reset_session(session_id: str):
    name = _collection_name(session_id)
    # drop se existir; criar de novo
    if name in [c.name for c in _qdrant.get_collections().collections]:
        _qdrant.delete_collection(name)
    _qdrant.recreate_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=DIMENSION, distance=qm.Distance.COSINE),
    )

def add_chunks_to_index(chunks: list, session_id: str) -> int:
    """
    Indexa 'chunks' na coleção da sessão.
    Retorna a quantidade de vetores inseridos.
    """
    if not chunks:
        return 0
    name = _collection_name(session_id)
    # garante coleção (caso /start_chat não tenha sido chamado)
    if name not in [c.name for c in _qdrant.get_collections().collections]:
        reset_session(session_id)

    embs = _model.encode(chunks, convert_to_numpy=True).astype(np.float32)
    points = [
        qm.PointStruct(
            id=i,
            vector=emb.tolist(),
            payload={"text": chunks[i]}
        ) for i, emb in enumerate(embs)
    ]
    _qdrant.upsert(collection_name=name, points=points, wait=True)
    return len(points)

def search_similar_chunks(question: str, session_id: str, top_k: int = 5):
    name = _collection_name(session_id)
    if name not in [c.name for c in _qdrant.get_collections().collections]:
        # sessão vazia
        return ["[ERROR] No documents processed yet."]

    q_emb = _model.encode([question], convert_to_numpy=True).astype(np.float32)[0]
    res = _qdrant.search(
        collection_name=name,
        query_vector=q_emb.tolist(),
        limit=top_k,
        with_payload=True
    )
    if not res:
        return ["No documents processed yet."]
    return [hit.payload.get("text", "") for hit in res]
