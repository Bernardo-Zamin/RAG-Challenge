from .embeddings import search
from .ollama_client import query_ollama
from ..vector_database.qdrant_store import QdrantStore

MAX_REFS_UI = 3


def new_chat():
    """Start a new chat and clear the vector database."""
    store = QdrantStore(collection="temp", dim=384)
    store.delete_all_collections()
    print("New chat started. All collections cleared.")


def _build_prompt(question: str, contexts: list[dict]) -> str:
    ctx = "\n\n".join([f"[p.{c['page']} #{c['order']}] {c['text']}" for c in contexts])
    return (
        "You are a concise assistant. Prefer using the provided *Context* to answer the *Question*. "
        "If the *Context* (PDF) doesn’t contain the answer or doesn’t exist, answer from your own knowledge. "
        "Do not fabricate details from the Context; if unsure, say you don't know briefly. "
        "Reply in the same language as the Question.\n\n"
        f"*Context*:\n{ctx}\n\n"
        f"*Question*: {question}\n"
        "Answer:"
    )


def answer_question(question: str, session_id: str, source: str | None = None):
    """
    Answers a question using retrieved context and a language model.

    Args:
        question (str): The question to answer.
        session_id (str): The session identifier for context retrieval.
        source (str | None, optional): The source to filter context. Defaults to None.

    Returns:
        dict: A dictionary with the answer and a list of reference snippets.
    """
    ctx = search(question, session_id, top_k=6, source=source)
    prompt = _build_prompt(question, ctx)
    answer = query_ollama(prompt)

    ui_refs = []
    seen = set()
    for c in ctx:
        key = (c["source"], c["page"], c["order"])
        if key in seen:
            continue
        seen.add(key)
        snippet = c["text"].strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        ui_refs.append(f"p.{c['page']} • {snippet}")
        if len(ui_refs) >= MAX_REFS_UI:
            break

    return {"answer": answer, "references": ui_refs}
