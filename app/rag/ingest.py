"""Document ingestion: text, markdown, PDF, and images.

Text-like files are read directly. PDFs use the embedded text layer when it
exists; scanned PDFs and images go through Gemini multimodal extraction,
which turns tables, forms and screenshots into structured markdown before
chunking, so they become searchable like any other document.
"""

from __future__ import annotations

import io
import mimetypes
from pathlib import Path

from app.rag.chunker import Chunk, chunk_text

TEXT_EXT = {".md", ".txt", ".rst", ".csv", ".json"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}

EXTRACTION_PROMPT = (
    "Transcribe this document into clean GitHub-flavoured markdown. Keep headings, "
    "convert tables to markdown tables, and describe charts or diagrams in one or two "
    "factual sentences. Do not add information that is not in the document."
)


def extract_with_gemini(data: bytes, mime_type: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=data, mime_type=mime_type), EXTRACTION_PROMPT],
    )
    return resp.text or ""


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_bytes(name: str, data: bytes, model: str, multimodal: bool = True) -> str:
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXT:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        text = _pdf_text(data)
        # A PDF with almost no text layer is a scan: fall back to vision.
        if len(text.strip()) < 40 and multimodal:
            return extract_with_gemini(data, "application/pdf", model)
        return text
    if ext in IMAGE_EXT:
        if not multimodal:
            raise ValueError(f"{name}: image ingestion needs multimodal extraction enabled")
        mime = mimetypes.guess_type(name)[0] or "image/png"
        return extract_with_gemini(data, mime, model)
    raise ValueError(f"unsupported file type: {ext}")


def ingest_bytes(name: str, data: bytes, model: str, multimodal: bool = True) -> list[Chunk]:
    return chunk_text(load_bytes(name, data, model, multimodal), source=name)


def ingest_dir(directory: str | Path, model: str, multimodal: bool = True) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXT | IMAGE_EXT | {".pdf"}:
            if path.suffix.lower() in IMAGE_EXT and not multimodal:
                continue
            chunks.extend(ingest_bytes(path.name, path.read_bytes(), model, multimodal))
    return chunks
