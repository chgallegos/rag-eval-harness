"""Run the golden set against a bot and produce a scored report.

The report is plain JSON so it can be diffed, archived per-release, posted
to a PR, or graphed over time. `regression.py` compares two of them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .adapter import BotAdapter
from .scorers import (
    score_accuracy,
    score_consistency,
    score_grounding,
    score_relevance,
    score_safety,
)

CONSISTENCY_RUNS = 5


def load_golden(path: str | Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["cases"]


def run_case(bot: BotAdapter, case: dict) -> dict:
    resp = bot.answer(case["query"])
    result: dict = {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "answer": resp.answer,
        "scores": {},
    }

    must_refuse = case.get("must_refuse", False)
    result["scores"]["safety"] = score_safety(resp, must_refuse)

    if not must_refuse:
        result["scores"]["accuracy"] = score_accuracy(
            resp, case.get("expected_facts", []))
        result["scores"]["grounding"] = score_grounding(resp)
        if "expected_doc" in case:
            result["scores"]["relevance"] = score_relevance(
                resp, case["expected_doc"])
        answers = [bot.answer(case["query"]).answer
                   for _ in range(CONSISTENCY_RUNS)]
        result["scores"]["consistency"] = score_consistency(answers)

    return result


def aggregate(case_results: list[dict]) -> dict:
    dims: dict[str, list[float]] = {}
    for r in case_results:
        for dim, s in r["scores"].items():
            dims.setdefault(dim, []).append(s["score"])
    return {dim: round(sum(v) / len(v), 4) for dim, v in dims.items()}


def run_suite(bot: BotAdapter, golden_path: str | Path,
              label: str = "run") -> dict:
    cases = load_golden(golden_path)
    results = [run_case(bot, c) for c in cases]
    report = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(results),
        "aggregate": aggregate(results),
        "cases": results,
    }
    return report


def save_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
