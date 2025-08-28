import math
from typing import List, Dict
import fitz  # PyMuPDF

# tamanhos “seguros” pro MiniLM (char-based) — ajuste se preferir
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def _split_text(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        # tenta cortar no último espaço pra evitar quebrar palavras
        cut = text.rfind(" ", start, end)
        if cut == -1 or cut <= start + chunk_size * 0.5:
            cut = end
        chunks.append(text[start:cut].strip())
        start = max(cut - overlap, 0)
        if cut == n:
            break
    return [c for c in chunks if c]

def extract_text_and_chunk(path: str) -> List[Dict]:
    """
    Lê o PDF em ordem de páginas e retorna lista de dicts:
    { "id": str, "text": str, "meta": {"page": int, "source": str, "order": int} }
    """
    out = []
    doc = fitz.open(path)
    order = 0
    for pno in range(len(doc)):
        page = doc[pno]
        # get_text("blocks") mantém a ordem de layout (x0,y0,x1,y1, text, block_no, ...)
        blocks = page.get_text("blocks")
        # ordena por posição vertical e depois horizontal
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        page_text = "\n".join([b[4] for b in blocks if b[4].strip()])
        for chunk in _split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
            out.append({
                "id": f"{path}::p{pno+1}::o{order}",
                "text": chunk,
                "meta": {"page": pno + 1, "source": path, "order": order}
            })
            order += 1
    doc.close()
    return out
