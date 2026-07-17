"""Dashboard for the RAG evaluation harness.

One Flask app, three endpoints:

  GET  /            the UI
  POST /api/run     run the golden set against a bot config, return the report
  POST /api/compare diff two reports with the regression gate

The dashboard is a window onto the same code paths CI uses -- runner.py and
regression.py -- not a reimplementation. If the numbers here and in CI ever
disagreed, that would itself be a bug worth catching.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, jsonify, render_template, request

from rag_eval.adapter import DemoBot
from rag_eval.regression import compare
from rag_eval.runner import run_suite

app = Flask(__name__)

FAILURE_MODES = ["none", "hallucinate", "bad_retrieval", "no_refusal"]

# reports live in memory per process; the dashboard is a demo surface,
# not a datastore. CI's artifacts are the durable record.
REPORTS: dict[str, dict] = {}


@app.get("/")
def index():
    return render_template("index.html", failure_modes=FAILURE_MODES)


@app.post("/api/run")
def api_run():
    body = request.get_json(force=True) or {}
    mode = body.get("failure_mode", "none")
    if mode not in FAILURE_MODES:
        return jsonify({"error": f"unknown failure mode: {mode}"}), 400
    label = body.get("label") or mode
    bot = DemoBot(ROOT / "corpus", seed=7,
                  failure_mode="" if mode == "none" else mode)
    report = run_suite(bot, ROOT / "golden" / "golden_set.yaml", label=label)
    REPORTS[label] = report
    return jsonify(report)


@app.post("/api/compare")
def api_compare():
    body = request.get_json(force=True) or {}
    base, cand = body.get("baseline"), body.get("candidate")
    if base not in REPORTS or cand not in REPORTS:
        return jsonify({"error": "run both configurations first"}), 400
    result = compare(REPORTS[base], REPORTS[cand])
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
