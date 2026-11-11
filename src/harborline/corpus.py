"""Load the synthetic policy corpus and cut it into chunks."""

from pathlib import Path

import yaml

from harborline.chunking import chunk_text
from harborline.models import Chunk, Document
from harborline.paths import corpus_dir


def load_documents(directory: Path | None = None) -> list[Document]:
    root = directory or corpus_dir()
    documents: list[Document] = []
    for path in sorted(root.glob("*.md")):
        documents.append(_parse_document(path))
    if not documents:
        raise ValueError(f"no policy documents in {root}")
    ids = [document.id for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("document ids must be unique")
    return documents


def chunk_documents(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        parts = chunk_text(document.body, size=size, overlap=overlap)
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    id=f"{document.id}#{index}",
                    doc_id=document.id,
                    title=document.title,
                    text=part,
                )
            )
    return chunks


def _parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path.name} is missing frontmatter")
    end = raw.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{path.name} frontmatter is not closed")
    loaded: object = yaml.safe_load(raw[4:end])
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} frontmatter must be a mapping")
    doc_id = loaded.get("id")
    title = loaded.get("title")
    if not isinstance(doc_id, str) or not isinstance(title, str):
        raise ValueError(f"{path.name} needs string id and title")
    body = raw[end + 5 :].strip()
    if not body:
        raise ValueError(f"{path.name} has an empty body")
    return Document(id=doc_id, title=title, body=body, filename=path.name)
