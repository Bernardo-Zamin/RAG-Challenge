from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

import os
import uuid

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

class QdrantStore:
    def __init__(self, collection: str, dim: int = 384):
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        if self.collection not in [c.name for c in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def reset(self):
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def upsert(self, embeddings: List[List[float]], chunks: List[Dict]):
        points = []
        for emb, ch in zip(embeddings, chunks):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text": ch["text"],
                    "source": ch["meta"]["source"],
                    "page": ch["meta"]["page"],
                    "order": ch["meta"]["order"]
                }
            ))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k=8, source_filter: str | None = None):
        flt = None
        if source_filter:
            flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_filter))])
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k * 2,  # pega sobra pra deduplicar
            query_filter=flt
        )
        # Ordena por score DESC, mas preserva ordem crescente quando empates pela “order”
        hits = []
        seen = set()
        for r in results:
            txt = r.payload["text"]
            key = (r.payload.get("source"), r.payload.get("page"), r.payload.get("order"))
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "text": txt,
                "score": r.score,
                "source": r.payload.get("source"),
                "page": r.payload.get("page"),
                "order": r.payload.get("order")
            })
            if len(hits) >= top_k:
                break
        hits.sort(key=lambda x: (x["page"], x["order"]))  # sequência de leitura
        return hits
