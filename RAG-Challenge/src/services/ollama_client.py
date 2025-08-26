"""
Client for interacting with the Ollama API.
Generates responses using a specified model.
"""

import requests

OLLAMA_URL = "http://ollama:11434/api/chat"
OLLAMA_MODEL = "tinyllama"

def query_ollama(prompt: str) -> str:
    """
    Sends a prompt to the Ollama API and returns the generated response.

    Args:
        prompt (str): The input prompt to send to the model.

    Returns:
        str: The generated response from the model.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
