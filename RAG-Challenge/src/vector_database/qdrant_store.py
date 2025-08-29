"""Qdrant vector database store for upserting, searching, and managing collections."""

from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

import os
import uuid

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


class QdrantStore:
    """A store for managing Qdrant vector database collections, upserting, and searching vectors."""  

    def __init__(self, collection: str, dim: int = 384):
        """
        Initialize the QdrantStore with a collection name and vector dimension.

        Args:
            collection (str): The name of the Qdrant collection to use.
            dim (int, optional): The dimension of the vectors. Defaults to 384.
        """
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def delete_all_collections(self):
        """Deleta todas as collections do Qdrant."""
        collections = self.client.get_collections().collections
        for collection in collections:
            self.client.delete_collection(collection.name)
        print("All collections deleted from Qdrant.")

    def _ensure_collection(self):
        if self.collection not in [
            c.name for c in self.client.get_collections().collections
        ]:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.DOT),
            )

    def reset(self):
        """Reset the current collection by recreating it with the specified vector parameters."""  
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.DOT),
        )

    def upsert(self, embeddings: List[List[float]], chunks: List[Dict]):
        """
        Upsert (insert or update) vectors and their associated metadata into the Qdrant collection. 

        Args:
            embeddings (List[List[float]]): List of vector embeddings to upsert.
            chunks (List[Dict]): List of metadata dictionaries corresponding to each embedding.
        """
        points = []
        for emb, ch in zip(embeddings, chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={
                        "text": ch["text"],
                        "source": ch["meta"]["source"],
                        "page": ch["meta"]["page"],
                        "order": ch["meta"]["order"],
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k=8, source_filter: str | None = None):
        """
        Search for the most similar vectors in the collection.

        Args:
            query_vector (List[float]): The query vector to search for similar vectors.
            top_k (int, optional): The number of top results to return. Defaults to 8.
            source_filter (str, optional): If provided, filters results by the given source.

        Returns:
            List[Dict]: A list of dictionaries containing the matched text, score, source, page, and order.
        """
        flt = None
        if source_filter:
            flt = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter),
                    )
                ]
            )
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k * 2,  # pega sobra pra deduplicar
            query_filter=flt,
        )

        hits = []
        seen = set()
        for r in results:
            txt = r.payload["text"]
            key = (
                r.payload.get("source"),
                r.payload.get("page"),
                r.payload.get("order"),
            )
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                {
                    "text": txt,
                    "score": r.score,
                    "source": r.payload.get("source"),
                    "page": r.payload.get("page"),
                    "order": r.payload.get("order"),
                }
            )
            if len(hits) >= top_k:
                break
        hits.sort(key=lambda x: (x["page"], x["order"]))
        return hits
