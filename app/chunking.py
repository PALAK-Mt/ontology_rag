import re
from pathlib import Path
from typing import List


def load_raw_text(file_path: Path) -> str:
    """
    Load raw text from a .txt file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """
    Remove Project Gutenberg boilerplate and clean whitespace.
    """
    # Remove Gutenberg header/footer
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*", text)
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*", text)

    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    
    # Normalize whitespace
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive newlines
    text = text.strip()

    return text


def chunk_text(text: str, max_tokens: int = 350) -> List[str]:
    """
    Chunk cleaned text into segments based on approximate token count.
    Approximate 1 token ≈ 0.75 words for English.
    """
    words = text.split()
    chunks = []
    chunk_size = max_tokens * 0.75  # Rough token-to-word ratio
    chunk_size = int(chunk_size)

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def preprocess_book(file_path: Path, max_tokens: int = 350) -> List[str]:
    """
    Load, clean, and chunk a book into manageable text segments.
    """
    raw = load_raw_text(file_path)
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned, max_tokens=max_tokens)
    return chunks

