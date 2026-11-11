"""Deterministic hashing embeddings, plus an optional OpenAI embedder."""

import hashlib
import itertools
import math
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "for",
        "in",
        "on",
        "is",
        "are",
        "be",
        "by",
        "with",
        "that",
        "this",
        "it",
        "as",
        "at",
        "from",
        "if",
        "your",
        "you",
        "my",
        "we",
        "our",
        "these",
        "those",
        "was",
        "were",
        "can",
        "when",
        "what",
        "which",
        "who",
        "how",
        "does",
        "do",
        "please",
    }
)

Vector = np.ndarray[tuple[int], np.dtype[np.float32]]
Matrix = np.ndarray[tuple[int, int], np.dtype[np.float32]]


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


class Embedder(Protocol):
    dim: int

    def fit(self, texts: list[str]) -> None: ...

    def embed_documents(self, texts: list[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...


class HashingEmbedder:
    """Bag-of-words hashing embedder with IDF weights. No network and no model download."""

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self.dim = dim
        self._idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        docs = [set(tokenize(text)) for text in texts]
        document_count = len(docs)
        frequencies: dict[str, int] = {}
        for tokens in docs:
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
        self._idf = {
            token: math.log((document_count + 1) / (count + 1)) + 1.0
            for token, count in frequencies.items()
        }

    def embed_documents(self, texts: list[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = [self._embed_one(text) for text in texts]
        stacked = np.vstack(rows)
        return np.ascontiguousarray(stacked, dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> Vector:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        for token in tokens:
            self._add(vector, token, self._weight(token))
        for left, right in itertools.pairwise(tokens):
            weight = 0.5 * (self._weight(left) + self._weight(right))
            self._add(vector, f"{left}_{right}", weight)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= np.float32(norm)
        return vector

    def _weight(self, token: str) -> float:
        # Tokens that never appear in the corpus cannot match a page. Giving them
        # a large default weight only adds hash collisions.
        return self._idf.get(token, 0.0)

    def _add(self, vector: Vector, feature: str, weight: float) -> None:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = np.float32(1.0 if digest[4] % 2 == 0 else -1.0)
        vector[index] += sign * np.float32(weight)


class OpenAIEmbedder:
    """OpenAI embeddings. Used only when a caller selects the openai embedding backend."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.dim = 1536
        self.model = model

    def fit(self, texts: list[str]) -> None:
        del texts

    def embed_documents(self, texts: list[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = [self.embed_query(text) for text in texts]
        return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        from openai import OpenAI

        client = OpenAI()
        response = client.embeddings.create(model=self.model, input=text)
        values = response.data[0].embedding
        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"expected dimension {self.dim}, got {vector.shape}")
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= np.float32(norm)
        return np.ascontiguousarray(vector, dtype=np.float32)
