"""FAISS vector store with role-based filtering.

The filter runs inside the store, *before* anything reaches the LLM prompt. Chunks a user is
not allowed to see are never retrieved, so they cannot leak through the generated answer.
(With pgvector you would push the same filter into SQL: WHERE allowed_roles && ARRAY[:role].)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app import config
from app.embeddings import get_embedder_class
from app.ingestion.pipeline import Chunk


@dataclass
class Hit:
    chunk: dict
    score: float


def can_see(role: str, allowed_roles: list) -> bool:
    if role == "admin":
        return True
    allowed = {r.lower() for r in allowed_roles}
    return "all" in allowed or role in allowed


class VectorStore:
    def __init__(self, index, chunks: list[dict], embedder):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorStore":
        texts = [c.text for c in chunks]
        embedder = get_embedder_class()().fit(texts)
        vecs = embedder.embed(texts)
        index = faiss.IndexFlatIP(vecs.shape[1])   # cosine, vectors are L2-normalised
        index.add(vecs)
        return cls(index, [c.to_dict() for c in chunks], embedder)

    def search(self, query: str, role: str, k: int = 4) -> list[Hit]:
        if self.index.ntotal == 0:
            return []
        qv = self.embedder.embed([query])
        # Index is small enough to scan fully; this guarantees the filter never starves results.
        n = self.index.ntotal
        scores, ids = self.index.search(qv, n)
        hits = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            ch = self.chunks[i]
            if can_see(role, ch["allowed_roles"]):
                hits.append(Hit(ch, float(s)))
            if len(hits) >= k:
                break
        return hits

    def save(self, directory: Path | None = None) -> None:
        d = Path(directory or config.INDEX_DIR)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        (d / "chunks.json").write_text(json.dumps(self.chunks, indent=1))
        self.embedder.save(d)

    @classmethod
    def load(cls, directory: Path | None = None) -> "VectorStore":
        d = Path(directory or config.INDEX_DIR)
        index = faiss.read_index(str(d / "faiss.index"))
        chunks = json.loads((d / "chunks.json").read_text())
        return cls(index, chunks, get_embedder_class().load(d))

    @classmethod
    def exists(cls, directory: Path | None = None) -> bool:
        return (Path(directory or config.INDEX_DIR) / "faiss.index").exists()
