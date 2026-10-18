"""Answer generators.

ExtractiveLLM - offline and deterministic. Selects the best-matching sentences from the retrieved
                context, so it can only say things that are in the documents. It uses the prompt's
                params (top_k, min_score, max_sentences); the template text is for the real LLM.
OpenAILLM     - LangChain ChatPromptTemplate | ChatOpenAI | StrOutputParser, using the template.
"""
from __future__ import annotations

from app import config
from app.prompts import PromptVersion
from app.textutils import split_sentences, tokenize

ABSTAIN = "I don't have enough information in the knowledge base to answer that."

# Words that appear in nearly every policy document and say nothing about the topic asked.
GENERIC = {"policy", "company", "northwind", "employee", "work", "get", "much", "many", "long", "need", "use", "system"}


def is_abstention(answer: str) -> bool:
    return "don't have enough information" in answer.lower()


def format_context(hits) -> str:
    return "\n\n".join(
        f"[{h.chunk['source']} | {h.chunk['section']}]\n{h.chunk['text']}" for h in hits
    )


class ExtractiveLLM:
    name = "extractive"

    def generate(self, question: str, hits, prompt: PromptVersion) -> str:
        if not hits or hits[0].score < prompt.min_score:
            return ABSTAIN
        q = {t for t in tokenize(question) if t not in GENERIC}
        if not q:
            return ABSTAIN
        scored = []
        for rank, h in enumerate(hits):
            for sent in split_sentences(h.chunk["text"]):
                st = set(tokenize(sent))
                overlap = len(q & st) / len(q)
                scored.append((overlap + 0.05 * h.score - 0.01 * rank, overlap, sent, h.chunk["source"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_overlap = scored[0][1] if scored else 0
        if best_overlap < 0.34:
            return ABSTAIN
        # a single matching word out of a multi-word question is too weak to answer from
        if len(q) >= 3 and round(best_overlap * len(q)) < 2:
            return ABSTAIN
        picked, seen = [], set()
        for total, overlap, sent, src in scored:
            if overlap < max(0.34, best_overlap * 0.6) or sent in seen:
                continue
            seen.add(sent)
            picked.append(f"{sent} [{src}]")
            if len(picked) >= prompt.max_sentences:
                break
        return " ".join(picked) if picked else ABSTAIN


class OpenAILLM:
    name = "openai"

    def __init__(self):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(model=config.OPENAI_CHAT_MODEL, temperature=0)
        self._ChatPromptTemplate = ChatPromptTemplate
        self._parser = StrOutputParser()

    def generate(self, question: str, hits, prompt: PromptVersion) -> str:
        if not hits:
            return ABSTAIN
        chain = self._ChatPromptTemplate.from_template(prompt.template) | self._llm | self._parser
        return chain.invoke({"context": format_context(hits), "question": question}).strip()


def get_llm():
    return OpenAILLM() if config.LLM_MODE == "openai" else ExtractiveLLM()
