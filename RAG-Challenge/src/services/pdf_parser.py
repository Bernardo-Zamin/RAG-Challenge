"""PDF parsing and text chunking utilities."""

from typing import List, Dict
import fitz

import tiktoken

# Tokenizer configuration and limits
TOKENIZER = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def _split_text(text: str, chunk_size: int, overlap: int):
    tokens = TOKENIZER.encode(text)
    chunks = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(n, start + chunk_size)
        chunk_tokens = tokens[start:end]
        chunks.append(TOKENIZER.decode(chunk_tokens))
        start = max(end - overlap, 0)

    return chunks


def extract_text_and_chunk(path: str) -> List[Dict]:
    """
    Extracts text from each page of a PDF file and returns a list of dictionaries containing the text and metadata.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        List[Dict]: A list of dictionaries, each containing the extracted text and associated metadata for a page.
    """
    out = []
    doc = fitz.open(path)
    order = 0

    for pno in range(len(doc)):
        page = doc[pno]
        page_text = page.get_text("text").strip()
        if page_text:
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
