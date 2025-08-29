import math
from typing import List, Dict
import fitz  # PyMuPDF

import tiktoken

# Configuração do tokenizer e limites
TOKENIZER = tiktoken.get_encoding("cl100k_base")  # Escolha o tokenizer apropriado
CHUNK_SIZE = 300  # Número máximo de tokens por chunk
CHUNK_OVERLAP = 50  # Sobreposição em tokens entre chunks consecutivos


def _split_text(text: str, chunk_size: int, overlap: int):
    tokens = TOKENIZER.encode(text)  # Tokeniza o texto
    chunks = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(n, start + chunk_size)
        chunk_tokens = tokens[start:end]
        chunks.append(
            TOKENIZER.decode(chunk_tokens)
        )  # Decodifica os tokens de volta para texto
        start = max(end - overlap, 0)  # Aplica a sobreposição

    return chunks


def extract_text_and_chunk(path: str) -> List[Dict]:
    out = []
    doc = fitz.open(path)
    order = 0

    for pno in range(len(doc)):
        page = doc[pno]
        # Extrai o texto da página inteira de forma simples
        page_text = page.get_text("text").strip()
        if page_text:  # Apenas adiciona se houver texto na página
            out.append(
                {
                    "id": f"{path}::p{pno+1}::o{order}",
                    "text": page_text,
                    "meta": {"page": pno + 1, "source": path, "order": order},
                }
            )
            order += 1

    doc.close()
    print(f"Generated {len(out)} chunks for {path}")
    return out
