"""Central configuration, read from environment variables."""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "sample_docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "data" / "index"))
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", BASE_DIR / "prompts"))
AUDIT_LOG = Path(os.getenv("AUDIT_LOG", BASE_DIR / "data" / "audit.jsonl"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
DEDUPE_JACCARD = float(os.getenv("DEDUPE_JACCARD", "0.85"))

# "offline" = deterministic TF-IDF embeddings + extractive answers (no API key needed)
# "openai"  = OpenAI embeddings + chat model through LangChain
LLM_MODE = os.getenv("LLM_MODE", "offline").lower()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

DEFAULT_PROMPT_VERSION = os.getenv("DEFAULT_PROMPT_VERSION", "v2_cited")

ROLES = ("employee", "hr", "finance", "admin")

# Demo keys so the project runs out of the box. Override with API_KEYS in any real use:
#   API_KEYS='{"<long-random-key>": "hr", "<another>": "employee"}'
DEMO_KEYS = {
    "demo-employee-key": "employee",
    "demo-hr-key": "hr",
    "demo-finance-key": "finance",
    "demo-admin-key": "admin",
}


def load_api_keys() -> dict:
    raw = os.getenv("API_KEYS")
    if raw:
        keys = json.loads(raw)
        bad = {r for r in keys.values() if r not in ROLES}
        if bad:
            raise ValueError(f"Unknown roles in API_KEYS: {bad}")
        return keys
    return dict(DEMO_KEYS)
