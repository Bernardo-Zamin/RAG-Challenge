"""PDF parsing utilities for extracting and chunking text from PDF files."""

import fitz
from typing import List


def extract_text_and_chunk(
    file_path: str, max_chunk_size: int = 500
) -> List[str]:
    """
    Extract text from a PDF file and split it into chunks of a specified
    maximum size.

    Args:
        file_path (str): The path to the PDF file.
        max_chunk_size (int, optional): The maximum size of each text chunk.
            Defaults to 500.

    Returns:
        List[str]: A list of text chunks extracted from the PDF.
    """

    doc = fitz.open(file_path)
    chunks = []

    for page in doc:
        text = page.get_text()
        paragraphs = text.split("\n")

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += " " + para
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para
        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks
