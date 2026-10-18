"""Document preparation pipeline: load -> clean -> tag -> chunk -> deduplicate.

Every stage is a small pure function so it can be unit tested on its own.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

SUPPORTED = {".md", ".txt", ".pdf"}
_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class RawDoc:
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    title: str
    department: str
    allowed_roles: list
    section: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestReport:
    documents: int = 0
    chunks_created: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    chunks_indexed: int = 0
    skipped_files: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 1. load ----------
def parse_front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    return meta, text[m.end():]


def load_documents(docs_dir: Path, report: IngestReport | None = None) -> list[RawDoc]:
    docs = []
    for path in sorted(Path(docs_dir).glob("**/*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED:
            if report is not None:
                report.skipped_files.append(path.name)
            continue
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
            meta = {}
        else:
            meta, text = parse_front_matter(path.read_text(encoding="utf-8"))
        docs.append(RawDoc(source=path.name, text=text, metadata=meta))
    return docs


# ---------- 2. clean ----------
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)           # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse blank lines
    text = re.sub(r"(?m)^\s*Page \d+( of \d+)?\s*$", "", text)  # PDF page footers
    return text.strip()


# ---------- 3. tag ----------
def tag_document(doc: RawDoc) -> dict:
    """Return the metadata every chunk of this document inherits.

    Access control is *fail-closed*: a document with no allowed_roles is visible to admins only.
    """
    meta = doc.metadata
    roles = meta.get("allowed_roles") or ["admin"]
    if isinstance(roles, str):
        roles = [roles]
    return {
        "title": meta.get("title") or Path(doc.source).stem.replace("_", " ").title(),
        "department": meta.get("department", "general"),
        "allowed_roles": [str(r).lower() for r in roles],
    }


# ---------- 4. chunk ----------
def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown by headings so each chunk knows its section."""
    parts, current, buf = [], "", []
    for line in text.split("\n"):
        if line.startswith("#"):
            if buf and "".join(buf).strip():
                parts.append((current, "\n".join(buf).strip()))
            current, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        parts.append((current, "\n".join(buf).strip()))
    return parts


def chunk_document(doc: RawDoc, meta: dict) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = []
    for section, body in _split_sections(clean_text(doc.text)):
        for i, piece in enumerate(splitter.split_text(body)):
            cid = hashlib.sha1(f"{doc.source}|{section}|{i}|{piece}".encode()).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=piece,
                    source=doc.source,
                    section=section,
                    **meta,
                )
            )
    return chunks


# ---------- 5. deduplicate ----------
def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _shingles(text: str, n: int = 3) -> set:
    words = _normalise(text).split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(chunks: list[Chunk], threshold: float | None = None,
                report: IngestReport | None = None) -> list[Chunk]:
    """Drop exact duplicates (hash) and near duplicates (Jaccard over word 3-shingles).

    A duplicate is only dropped when the earlier chunk is visible to *at least* the same roles,
    otherwise we could silently hide content from people who are allowed to see it.
    """
    threshold = config.DEDUPE_JACCARD if threshold is None else threshold
    kept: list[Chunk] = []
    seen_hash: dict[str, Chunk] = {}
    kept_shingles: list[set] = []
    for ch in chunks:
        h = hashlib.sha1(_normalise(ch.text).encode()).hexdigest()
        prior = seen_hash.get(h)
        if prior is not None and set(ch.allowed_roles) <= set(prior.allowed_roles):
            if report is not None:
                report.exact_duplicates_removed += 1
            continue
        sh = _shingles(ch.text)
        near = False
        for k, ksh in zip(kept, kept_shingles):
            if set(ch.allowed_roles) <= set(k.allowed_roles) and jaccard(sh, ksh) >= threshold:
                near = True
                break
        if near:
            if report is not None:
                report.near_duplicates_removed += 1
            continue
        seen_hash.setdefault(h, ch)
        kept.append(ch)
        kept_shingles.append(sh)
    return kept


# ---------- orchestration ----------
def run_pipeline(docs_dir: Path | None = None) -> tuple[list[Chunk], IngestReport]:
    report = IngestReport()
    docs = load_documents(docs_dir or config.DOCS_DIR, report)
    report.documents = len(docs)
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, tag_document(doc)))
    report.chunks_created = len(all_chunks)
    final = deduplicate(all_chunks, report=report)
    report.chunks_indexed = len(final)
    return final, report
