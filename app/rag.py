"""RAG service: retrieve (role-filtered) -> generate."""
from __future__ import annotations

import time

from app import config, prompts
from app.ingestion.pipeline import run_pipeline
from app.llm import ABSTAIN, get_llm, is_abstention
from app.vectorstore import VectorStore


class RAGService:
    def __init__(self, store: VectorStore | None = None):
        self.store = store
        self.llm = get_llm()

    # -- index management --
    def ingest(self, docs_dir=None, persist: bool = True) -> dict:
        chunks, report = run_pipeline(docs_dir)
        self.store = VectorStore.build(chunks)
        if persist:
            self.store.save()
        return report.to_dict()

    def ensure_index(self) -> None:
        if self.store is None:
            if VectorStore.exists():
                self.store = VectorStore.load()
            else:
                self.ingest()

    # -- query --
    def ask(self, question: str, role: str, prompt_version: str | None = None) -> dict:
        self.ensure_index()
        prompt = prompts.get(prompt_version)
        t0 = time.perf_counter()
        hits = self.store.search(question, role=role, k=prompt.top_k)
        answer = self.llm.generate(question, hits, prompt)
        return {
            "answer": answer,
            "abstained": is_abstention(answer),
            "prompt_version": prompt.name,
            "sources": [
                {
                    "source": h.chunk["source"],
                    "title": h.chunk["title"],
                    "section": h.chunk["section"],
                    "score": round(h.score, 4),
                    "text": h.chunk["text"],
                }
                for h in hits
            ],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
