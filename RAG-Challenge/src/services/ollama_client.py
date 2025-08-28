import os
import time
import requests

# Use /api/chat (OK for Qwen Instruct)
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434") + "/api/chat"
# Change the default to Qwen (or let it come from docker-compose)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")

# Qwen may take more time on CPU
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))
BACKOFF = int(os.getenv("OLLAMA_BACKOFF", "5"))

def query_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
            "num_ctx": 4096,   # Qwen accepts; adjust if you want higher/lower
        },
    }
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            # If error, show the body to facilitate debugging
            if r.status_code >= 400:
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.text}", response=r)
            data = r.json()
            # /api/chat returns in data["message"]["content"]
            return data["message"]["content"]
        except requests.RequestException as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(BACKOFF * (attempt + 1))
            else:
                raise
    raise last_err
