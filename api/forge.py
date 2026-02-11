"""
Forge -- create multimodal cartridges from uploaded documents.

Extracted from vector_plus_studio_v83.py lines 1067-1095.
"""

import os
from .engine import engine, TextRegionEncoder
from .cartridge_io import save_cartridge_multimodal, DATA_DIR

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None


def extract_text_from_file(filename: str, content: bytes) -> str | None:
    """Extract text from uploaded file bytes."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        return content.decode('utf-8', errors='replace')

    elif ext == ".pdf" and PyPDF2:
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)

    elif ext == ".docx" and docx:
        import io
        d = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in d.paragraphs)

    return None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def forge_cartridge(name: str, files: list[tuple[str, bytes]],
                    chunk_size: int = 500) -> dict:
    """
    Create a multimodal cartridge from uploaded files.

    Args:
        name: Cartridge name
        files: List of (filename, content_bytes) tuples
        chunk_size: Words per chunk (0 = no chunking)

    Returns:
        dict with success, pattern_count, path, message
    """
    all_texts = []

    for filename, content in files:
        text = extract_text_from_file(filename, content)
        if text and text.strip():
            if chunk_size > 0:
                chunks = chunk_text(text, chunk_size=chunk_size)
                all_texts.extend(chunks)
            else:
                all_texts.append(text)

    if not all_texts:
        return {"success": False, "message": "No text extracted from files"}

    # Embed
    embeddings = engine.embed_documents(all_texts)

    # Compute compressed lengths
    text_encoder = TextRegionEncoder()
    compressed_lens = [len(text_encoder.compress_text(t)) for t in all_texts]

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.pkl")
    save_cartridge_multimodal(path, embeddings, all_texts, compressed_lens)

    return {
        "success": True,
        "pattern_count": len(all_texts),
        "path": path,
        "message": f"Forged {name}.pkl with {len(all_texts)} patterns",
    }
