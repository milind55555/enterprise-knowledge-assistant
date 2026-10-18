"""Evaluation harness. Runs the dataset against every prompt version and gates CI.

Usage:
    python -m eval.run_eval                # run all prompt versions, gate the default one
    python -m eval.run_eval --prompt v3_concise
    python -m eval.run_eval --judge        # (LLM_MODE=openai) add an LLM-as-judge faithfulness score

Exit code is 1 when the gated prompt version misses any threshold in eval/thresholds.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import config, prompts
from app.llm import is_abstention
from app.rag import RAGService
from eval import metrics

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "dataset.jsonl"
THRESHOLDS = EVAL_DIR / "thresholds.json"
RESULTS = EVAL_DIR / "results"


def load_dataset() -> list[dict]:
    return [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]


def run_version(service: RAGService, dataset: list[dict], version: str) -> dict:
    rows = []
    for item in dataset:
        out = service.ask(item["question"], role=item["role"], prompt_version=version)
        sources = [s["source"] for s in out["sources"]]
        context = "\n".join(s["text"] for s in out["sources"])
        answer = out["answer"]
        row = {
            "id": item["id"], "type": item["type"], "role": item["role"],
            "question": item["question"], "answer": answer, "sources": sources,
            "abstained": is_abstention(answer),
            "faithfulness": metrics.faithfulness(answer, context),
        }
        if item["type"] == "answerable":
            row["retrieval_hit"] = metrics.retrieval_hit(sources, item["expected_docs"])
            row["correct"] = metrics.answer_correct(answer, item["keywords"])
        else:
            row["leaked"] = metrics.leaked(sources, answer, item.get("forbidden_docs", []),
                                           item.get("forbidden_keywords", []))
        rows.append(row)
    return {"version": version, "rows": rows, "summary": summarise(rows)}


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] == "answerable"]
    no_answer = [r for r in rows if r["type"] != "answerable"]
    answered = [r for r in rows if not r["abstained"]]
    hallucinated = [
        r for r in answered
        if r["faithfulness"] < 1.0 or r["type"] != "answerable"
    ]
    return {
        "questions": len(rows),
        "retrieval_hit_rate": _mean(r["retrieval_hit"] for r in ans),
        "answer_accuracy": _mean(r["correct"] for r in ans),
        "faithfulness": _mean(r["faithfulness"] for r in answered),
        "hallucination_rate": round(len(hallucinated) / len(answered), 4) if answered else 0.0,
        "abstention_accuracy": _mean(r["abstained"] for r in no_answer),
        "false_abstention_rate": _mean(r["abstained"] for r in ans),
        "access_leakage": sum(1 for r in rows if r.get("leaked")),
    }


def check(summary: dict) -> list[str]:
    thresholds = json.loads(THRESHOLDS.read_text())
    failures = []
    for metric, rule in thresholds.items():
        value = summary.get(metric)
        if value is None:
            continue
        if "min" in rule and value < rule["min"]:
            failures.append(f"{metric} = {value} (needs >= {rule['min']})")
        if "max" in rule and value > rule["max"]:
            failures.append(f"{metric} = {value} (needs <= {rule['max']})")
    return failures


def write_report(results: list[dict], gated: str, failures: list[str]) -> str:
    cols = ["retrieval_hit_rate", "answer_accuracy", "faithfulness", "hallucination_rate",
            "abstention_accuracy", "false_abstention_rate", "access_leakage"]
    lines = [f"# Evaluation report", "",
             f"Mode: `{config.LLM_MODE}` | Questions: {results[0]['summary']['questions']} | Gated prompt: `{gated}`", "",
             "| prompt | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r in results:
        s = r["summary"]
        lines.append(f"| {r['version']} | " + " | ".join(str(s[c]) for c in cols) + " |")
    lines += ["", "## Gate", ""]
    lines += ["PASS" if not failures else "FAIL"] + [f"- {f}" for f in failures]
    for r in results:
        bad = [x for x in r["rows"]
               if x.get("correct") is False or x.get("retrieval_hit") is False
               or x.get("leaked") or (x["type"] != "answerable" and not x["abstained"])]
        lines += ["", f"## Failures: {r['version']} ({len(bad)})", ""]
        for x in bad:
            lines.append(f"- **{x['id']}** ({x['type']}, role={x['role']}) {x['question']}  \n  answer: {x['answer'][:160]}  \n  retrieved: {x['sources']}")
    return "\n".join(lines) + "\n"


def llm_judge(service, rows) -> float | None:  # pragma: no cover - needs an API key
    if config.LLM_MODE != "openai":
        return None
    from langchain_openai import ChatOpenAI

    judge = ChatOpenAI(model=config.OPENAI_CHAT_MODEL, temperature=0)
    scores = []
    for r in rows:
        if r["abstained"]:
            continue
        ctx = "\n".join(s["text"] for s in service.ask(r["question"], r["role"])["sources"])
        verdict = judge.invoke(
            f"Context:\n{ctx}\n\nAnswer:\n{r['answer']}\n\n"
            "Is every claim in the answer supported by the context? Reply only YES or NO."
        ).content.strip().upper()
        scores.append(1.0 if verdict.startswith("YES") else 0.0)
    return _mean(scores)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="gate only this prompt version (default: configured default)")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args(argv)

    service = RAGService()
    service.ingest(persist=False)
    dataset = load_dataset()
    gated = args.prompt or config.DEFAULT_PROMPT_VERSION
    results = [run_version(service, dataset, v) for v in prompts.load_all()]
    if args.judge:
        for r in results:
            r["summary"]["llm_judge_faithfulness"] = llm_judge(service, r["rows"])

    gated_result = next(r for r in results if r["version"] == gated)
    failures = check(gated_result["summary"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(results, indent=1))
    report = write_report(results, gated, failures)
    (RESULTS / "report.md").write_text(report)
    print(report.split("## Failures")[0])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
