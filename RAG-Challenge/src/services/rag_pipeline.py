from src.services.embeddings import search_similar_chunks
from src.services.ollama_client import query_ollama

def answer_question(question: str, session_id: str, top_k: int = 5):
    # busca contexto na coleção da sessão
    similar_chunks = search_similar_chunks(question, session_id=session_id, top_k=top_k)
    if similar_chunks and isinstance(similar_chunks, list):
        context = "\n\n".join(similar_chunks)
    else:
        context = ""

    prompt = (
        "You are an assistant that answers based on the CONTEXT below. "
        "If there is not enough information, just say what you know but don't elaborate.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    answer = query_ollama(prompt)
    return {"answer": answer, "references": similar_chunks or []}
