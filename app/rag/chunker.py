"""Structure-aware text chunking.

Splits on markdown headings first so each chunk keeps its section title
(useful for citations), then packs paragraphs into windows of roughly
``max_chars`` with a character overlap between consecutive windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    section: str
    text: str


def _split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(2).strip(), text[m.end() : end]))
    return sections


def _windows(paragraphs: list[str], max_chars: int, overlap: int) -> list[str]:
    out: list[str] = []
    buf = ""
    for para in paragraphs:
        # Hard-split paragraphs that are bigger than a whole window.
        while len(para) > max_chars:
            head, para = para[:max_chars], para[max_chars - overlap :]
            if buf:
                out.append(buf)
                buf = ""
            out.append(head)
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            out.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n\n{para}" if tail else para
    if buf.strip():
        out.append(buf)
    return out


def chunk_text(
    text: str, source: str, max_chars: int = 900, overlap: int = 120
) -> list[Chunk]:
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    chunks: list[Chunk] = []
    for section, body in _split_sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for window in _windows(paragraphs, max_chars, overlap):
            chunks.append(
                Chunk(
                    id=f"{source}#{len(chunks)}",
                    source=source,
                    section=section,
                    text=window.strip(),
                )
            )
    return chunks
