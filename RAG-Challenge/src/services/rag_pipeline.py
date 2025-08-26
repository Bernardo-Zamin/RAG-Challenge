"""
This module provides the main pipeline for answering questions using
retrieval-augmented generation.

"""

from src.services.embeddings import search_similar_chunks
from src.services.ollama_client import query_ollama
from src.models.models import AnswerResponse


def answer_question(question: str) -> AnswerResponse:
    """
    Answers a question using retrieval-augmented generation by searching for
    similar context chunks and generating a response using the Ollama client.

    Args:
        question (str): The question to answer.

    Returns:
        AnswerResponse: The generated answer and the context references.
    """

    context_chunks = search_similar_chunks(question, top_k=5)
    context = "\n\n".join(context_chunks)

    prompt = f"""Context:
{context}

Question: {question}
Answer:"""

    answer = query_ollama(prompt)
    return AnswerResponse(answer=answer.strip(), references=context_chunks)
