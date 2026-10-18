"""Embedding backends.

TfidfHashEmbedder  - offline, deterministic, no API key. Hashed uni+bigram TF-IDF vectors.
OpenAIEmbedder     - real semantic embeddings via LangChain (LLM_MODE=openai).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from app import config
from app.textutils import tokenize

DIM = 4096


def _features(text: str) -> list[str]:
    toks = tokenize(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def _bucket(feature: str) -> int:
    return zlib.crc32(feature.encode()) % DIM


class TfidfHashEmbedder:
    name = "tfidf-hash"

    def __init__(self, idf: np.ndarray | None = None):
        self.idf = idf if idf is not None else np.ones(DIM, dtype="float32")

    def fit(self, texts: list[str]) -> "TfidfHashEmbedder":
        df = np.zeros(DIM, dtype="float32")
        for t in texts:
            df[list({_bucket(f) for f in _features(t)})] += 1
        n = max(len(texts), 1)
        self.idf = (np.log((1 + n) / (1 + df)) + 1.0).astype("float32")
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype="float32")
        for i, t in enumerate(texts):
            for f in _features(t):
                out[i, _bucket(f)] += 1.0
        out = np.log1p(out) * self.idf          # sublinear tf * idf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    def save(self, directory: Path) -> None:
        np.save(directory / "idf.npy", self.idf)

    @classmethod
    def load(cls, directory: Path) -> "TfidfHashEmbedder":
        return cls(np.load(directory / "idf.npy"))


class OpenAIEmbedder:
    name = "openai"

    def __init__(self):
        from langchain_openai import OpenAIEmbeddings

        self._emb = OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL)

    def fit(self, texts):  # nothing to fit
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(self._emb.embed_documents(texts), dtype="float32")
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)

    def save(self, directory: Path) -> None:
        pass

    @classmethod
    def load(cls, directory: Path) -> "OpenAIEmbedder":
        return cls()


def get_embedder_class():
    return OpenAIEmbedder if config.LLM_MODE == "openai" else TfidfHashEmbedder
