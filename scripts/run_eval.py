#!/usr/bin/env python3
"""Run the golden set and write a scored report.

Usage:
  python scripts/run_eval.py                          # clean run -> reports/report.json
  python scripts/run_eval.py --label candidate --out reports/candidate.json
  RAG_EVAL_FAILURE=hallucinate python scripts/run_eval.py --label broken --out reports/broken.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_eval.adapter import DemoBot
from rag_eval.runner import run_suite, save_report

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--label", default="baseline")
parser.add_argument("--out", default="reports/report.json")
parser.add_argument("--seed", type=int, default=7)
args = parser.parse_args()

bot = DemoBot(ROOT / "corpus", seed=args.seed)
report = run_suite(bot, ROOT / "golden" / "golden_set.yaml", label=args.label)
save_report(report, ROOT / args.out)

print(f"label: {report['label']}   cases: {report['n_cases']}")
print(json.dumps(report["aggregate"], indent=2))
print(f"\nreport written to {args.out}")
