#!/usr/bin/env python3
"""Regression gate. Exit 1 if the candidate regresses vs the baseline.

Usage:
  python scripts/compare.py reports/baseline.json reports/candidate.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_eval.regression import compare, format_table, load

baseline, candidate = load(sys.argv[1]), load(sys.argv[2])
result = compare(baseline, candidate)
print(format_table(result))
sys.exit(1 if result["failed"] else 0)
