"""Prompt versioning: each prompt is a YAML file in prompts/ with its own params.

Compare versions with:  python -m eval.run_eval --compare
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from app import config


@dataclass(frozen=True)
class PromptVersion:
    name: str
    description: str
    template: str
    top_k: int
    min_score: float
    max_sentences: int


@lru_cache(maxsize=1)
def load_all() -> dict[str, PromptVersion]:
    versions = {}
    for path in sorted(config.PROMPTS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        p = data.get("params", {})
        versions[data["name"]] = PromptVersion(
            name=data["name"],
            description=data.get("description", ""),
            template=data["template"],
            top_k=int(p.get("top_k", 4)),
            min_score=float(p.get("min_score", 0.15)),
            max_sentences=int(p.get("max_sentences", 3)),
        )
    return versions


def get(name: str | None = None) -> PromptVersion:
    versions = load_all()
    name = name or config.DEFAULT_PROMPT_VERSION
    if name not in versions:
        raise KeyError(f"Unknown prompt version '{name}'. Available: {sorted(versions)}")
    return versions[name]
