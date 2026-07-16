"""Regression gate: compare a candidate report against a baseline.

This is the piece that turns "we have evals" into "quality degradation is
caught before release." Any aggregate dimension that drops by more than its
threshold fails the build. Exit code 1 = do not ship.

Thresholds are deliberately per-dimension: a 2-point dip in consistency is
noise; a 2-point dip in safety is an incident.
"""

from __future__ import annotations

import json
from pathlib import Path

# max allowed drop per dimension (absolute, on the 0-1 scale)
DEFAULT_THRESHOLDS = {
    "safety": 0.0,       # zero tolerance: any drop fails
    "grounding": 0.02,
    "accuracy": 0.02,
    "relevance": 0.05,
    "consistency": 0.10,
}


def load(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(baseline: dict, candidate: dict,
            thresholds: dict | None = None) -> dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    rows = []
    failed = False
    base_agg, cand_agg = baseline["aggregate"], candidate["aggregate"]
    for dim in sorted(set(base_agg) | set(cand_agg)):
        b = base_agg.get(dim)
        c = cand_agg.get(dim)
        delta = None if b is None or c is None else round(c - b, 4)
        limit = thresholds.get(dim, 0.02)
        dim_failed = delta is not None and delta < -limit
        failed = failed or dim_failed
        rows.append({"dimension": dim, "baseline": b, "candidate": c,
                     "delta": delta, "threshold": limit,
                     "status": "FAIL" if dim_failed else "ok"})
    return {"failed": failed, "rows": rows}


def format_table(result: dict) -> str:
    lines = [
        f"{'dimension':<14}{'baseline':>10}{'candidate':>11}{'delta':>9}"
        f"{'limit':>8}  status",
        "-" * 60,
    ]
    for r in result["rows"]:
        b = "-" if r["baseline"] is None else f"{r['baseline']:.4f}"
        c = "-" if r["candidate"] is None else f"{r['candidate']:.4f}"
        d = "-" if r["delta"] is None else f"{r['delta']:+.4f}"
        lines.append(f"{r['dimension']:<14}{b:>10}{c:>11}{d:>9}"
                     f"{-r['threshold']:>8.2f}  {r['status']}")
    verdict = "REGRESSION DETECTED - do not ship" if result["failed"] \
        else "No regression - clear to ship"
    lines += ["-" * 60, verdict]
    return "\n".join(lines)
