"""
Qdrant vector database store utilities for managing session-based collections.

and embeddings.
"""

import os
import uuid
from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DIMENSION = 384  # same as all-MiniLM-L6-v2

_client = QdrantClient(url=QDRANT_URL)


def collection_name(session_id: str) -> str:
    """Generate a collection name based on the session ID."""
    return f"session_{session_id}"


def create_or_reset_collection(session_id: str) -> None:
    """
    Create a new collection or reset an existing one for the given session ID.
    """
    name = collection_name(session_id)
    # recreate collection
    _client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
    )


def upsert_embeddings(
    session_id: str,
    embeddings: List[List[float]],
    texts: List[str]
) -> int:
    """
    Upsert embeddings and their corresponding texts into
    the session's collection.

    Args:
        session_id (str): The session identifier.
        embeddings (List[List[float]]): List of embedding vectors.
        texts (List[str]): List of texts corresponding to the embeddings.

    Returns:
        int: The number of points upserted.
    """

    name = collection_name(session_id)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": txt},
        )
        for vec, txt in zip(embeddings, texts)
    ]
    _client.upsert(collection_name=name, points=points)
    return len(points)


def search(
    session_id: str,
    query_vector: List[float],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Search for the most similar embeddings in the session's collection.

    Args:
        session_id (str): The session identifier.
        query_vector (List[float]): The embedding vector to search for.
        top_k (int, optional): The number of top results to return.
        Defaults to 5.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the text and its
        similarity score.
    """
    name = collection_name(session_id)
    results = _client.search(
        collection_name=name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [(r.payload.get("text", ""), float(r.score)) for r in results]


def delete_collection(session_id: str) -> None:
    """
    Delete the collection associated with the given session ID.

    Args:
        session_id (str): The session identifier.
    """
    name = collection_name(session_id)
    _client.delete_collection(collection_name=name)
