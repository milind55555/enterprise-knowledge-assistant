"""Metric definitions. Each is a small function so it is easy to read, test and defend in an interview.

faithfulness (heuristic): an answer sentence counts as supported when >=80% of its content words
appear in the retrieved context AND every number in it also appears in the context. Citations like
[file.md] are stripped first. With LLM_MODE=openai you can add --judge to also get an LLM-graded score.
"""
from __future__ import annotations

import re

from app.llm import is_abstention
from app.textutils import split_sentences, tokenize

_CITATION = re.compile(r"\[[^\]]+\]")
_NUMBER = re.compile(r"\d[\d,.]*")


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text).strip()


def faithfulness(answer: str, context: str) -> float:
    if is_abstention(answer):
        return 1.0
    ctx_tokens = set(tokenize(context))
    ctx_numbers = {n.strip(".,") for n in _NUMBER.findall(context)}
    sentences = split_sentences(strip_citations(answer))
    if not sentences:
        return 0.0
    supported = 0
    for s in sentences:
        toks = tokenize(s)
        if not toks:
            supported += 1
            continue
        coverage = sum(t in ctx_tokens for t in toks) / len(toks)
        numbers_ok = all(n.strip(".,") in ctx_numbers for n in _NUMBER.findall(s))
        supported += int(coverage >= 0.8 and numbers_ok)
    return supported / len(sentences)


def retrieval_hit(sources: list[str], expected_docs: list[str]) -> bool:
    return any(s in expected_docs for s in sources)


def answer_correct(answer: str, keywords: list[str]) -> bool:
    if is_abstention(answer):
        return False
    a = answer.lower()
    return all(k.lower() in a for k in keywords)


def leaked(sources: list[str], answer: str, forbidden_docs: list[str], forbidden_keywords: list[str]) -> bool:
    if any(s in forbidden_docs for s in sources):
        return True
    return (not is_abstention(answer)) and any(k.lower() in answer.lower() for k in forbidden_keywords)
