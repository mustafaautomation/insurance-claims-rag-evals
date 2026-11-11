"""FAISS inner-product search and a pgvector backend with the same hit shape."""

from typing import Protocol

import numpy as np

from harborline.embeddings import Embedder
from harborline.models import Chunk, Hit


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


class FaissRetriever:
    def __init__(self, embedder: Embedder, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        import faiss

        self.embedder = embedder
        self.chunks = list(chunks)
        matrix = embedder.embed_documents([chunk.text for chunk in self.chunks])
        index = faiss.IndexFlatIP(embedder.dim)
        index.add(matrix)
        self._index = index

    def search(self, query: str, k: int) -> list[Hit]:
        if k <= 0:
            return []
        query_vector = np.ascontiguousarray(self.embedder.embed_query(query).reshape(1, -1))
        limit = min(k, len(self.chunks))
        scores, indices = self._index.search(query_vector, limit)
        found: list[Hit] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            position = int(index)
            if position < 0:
                continue
            chunk = self.chunks[position]
            found.append(Hit(chunk=chunk, score=round(float(score), 6), rank=0))
        found.sort(key=lambda hit: (-hit.score, hit.chunk.doc_id, hit.chunk.id))
        return [
            Hit(chunk=hit.chunk, score=hit.score, rank=rank)
            for rank, hit in enumerate(found, start=1)
        ]


class ReversedRetriever:
    """Returns the lowest-scoring chunks. This is the deliberately bad retriever."""

    def __init__(self, base: Retriever, corpus_size: int) -> None:
        if corpus_size <= 0:
            raise ValueError("corpus_size must be positive")
        self.base = base
        self.corpus_size = corpus_size

    def search(self, query: str, k: int) -> list[Hit]:
        if k <= 0:
            return []
        ranked = self.base.search(query, k=self.corpus_size)
        worst = list(reversed(ranked))[:k]
        return [
            Hit(chunk=hit.chunk, score=hit.score, rank=rank)
            for rank, hit in enumerate(worst, start=1)
        ]


class PgVectorRetriever:
    """Postgres + pgvector. Build and search use the same embedder as the FAISS index."""

    def __init__(self, dsn: str, embedder: Embedder) -> None:
        if not dsn:
            raise ValueError("pgvector requires a database URL")
        if embedder.dim <= 0 or embedder.dim > 4096:
            raise ValueError("unsupported embedding dimension")
        self.dsn = dsn
        self.embedder = embedder

    def build(self, chunks: list[Chunk]) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        with psycopg.connect(self.dsn, connect_timeout=5) as conn:
            register_vector(conn)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("DROP TABLE IF EXISTS chunks")
            conn.execute(
                "CREATE TABLE chunks ("
                "id text PRIMARY KEY, "
                "doc_id text NOT NULL, "
                "title text NOT NULL, "
                "text text NOT NULL, "
                f"embedding vector({self.embedder.dim})"
                ")"
            )
            for chunk, vector in zip(chunks, vectors, strict=True):
                conn.execute(
                    "INSERT INTO chunks (id, doc_id, title, text, embedding) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (chunk.id, chunk.doc_id, chunk.title, chunk.text, vector.tolist()),
                )

    def search(self, query: str, k: int) -> list[Hit]:
        import psycopg
        from pgvector.psycopg import register_vector

        if k <= 0:
            return []
        query_vector = self.embedder.embed_query(query).tolist()
        with psycopg.connect(self.dsn, connect_timeout=5) as conn:
            register_vector(conn)
            cursor = conn.execute(
                "SELECT id, doc_id, title, text, 1 - (embedding <=> %s) AS score "
                "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (query_vector, query_vector, k),
            )
            rows = cursor.fetchall()
        hits: list[Hit] = []
        for rank, row in enumerate(rows, start=1):
            chunk = Chunk(id=str(row[0]), doc_id=str(row[1]), title=str(row[2]), text=str(row[3]))
            hits.append(Hit(chunk=chunk, score=round(float(row[4]), 6), rank=rank))
        return hits
