"""
This module provides functions for embedding text chunks and searching for
similar chunks using a FAISS index.
"""

from sentence_transformers import SentenceTransformer
from src.vector_database import faiss_store
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss_store.load_faiss_index()
chunks_db = faiss_store.load_chunks()


def build_index_from_pdfs(pdf_dir: str = "RAG-Challenge/data/uploaded_pdfs"):
    """
    Parseia os PDFs, gera embeddings dos textos extraídos e salva o índice FAISS e os chunks.

    Args:
        pdf_dir (str): Caminho para o diretório contendo os PDFs.
    """
    from src.services import pdf_parser  # Import interno para evitar import circular

    print("[INFO] Extraindo texto dos PDFs...")
    texts = pdf_parser.extract_text_from_pdfs(pdf_dir)

    if not texts:
        print("[WARNING] Nenhum texto encontrado nos PDFs.")
        return

    print(f"[INFO] {len(texts)} chunks extraídos. Gerando embeddings...")
    add_chunks_to_index(texts)
    print("[SUCCESS] Índice FAISS atualizado com sucesso.")


def add_chunks_to_index(chunks: list):
    """
    Add a list of text chunks to the FAISS index and update the chunks
    database.

    Args:
        chunks (list): List of text chunks to be embedded and added.
    """
    global chunks_db, index
    embeddings = model.encode(chunks)
    index.add(np.array(embeddings, dtype=np.float32))
    chunks_db.extend(chunks)

    faiss_store.save_faiss_index(index)
    faiss_store.save_chunks(chunks_db)


def search_similar_chunks(question: str, top_k: int = 5):
    """
    Search for the most similar text chunks to the given question using,
    the FAISS index.

    Args:
        question (str): The input question to search for similar chunks.
        top_k (int, optional): The number of top similar chunks to return.
        Defaults to 5.

    Returns:
        list: A list of the most similar text chunks.
    """
    if not chunks_db:
        return ["[ERROR] No documents processed yet."]
    q_embed = model.encode([question])
    D, I = index.search(np.array(q_embed, dtype=np.float32), top_k)
    return [chunks_db[i] for i in I[0]]
