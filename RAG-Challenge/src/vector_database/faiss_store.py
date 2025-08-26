"""
This module provides functions to manage a FAISS vector index and associated
text chunks for the RAG-Challenge project.
"""

import os
import faiss
import pickle

INDEX_PATH = "RAG-Challenge/data/vector_store/faiss.index"
CHUNKS_PATH = "RAG-Challenge/data/vector_store/chunks.pkl"
DIMENSION = 384  # Dimension of the embeddings from the all-MiniLM-L6-v2 model


def load_faiss_index():
    """
    Load the FAISS index from disk if it exists, otherwise create a new index.
    """
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(DIMENSION)
    return index


def save_faiss_index(index):
    """
    Save the FAISS index to disk at the specified INDEX_PATH.
    """
    faiss.write_index(index, INDEX_PATH)


def load_chunks():
    """
    Load the text chunks from disk if they exist,
    otherwise return an empty list.
    """
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            return pickle.load(f)
    return []


def save_chunks(chunks):
    """
    Save the text chunks to disk at the specified CHUNKS_PATH.
    """
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
