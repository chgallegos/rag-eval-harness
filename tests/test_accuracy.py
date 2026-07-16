"""Every factual case must contain its expected facts."""
import pytest

from conftest import cases_of
from rag_eval.scorers import score_accuracy


def test_factual_answers_contain_expected_facts(bot, golden):
    failures = []
    for case in cases_of(golden, "factual", "ambiguous"):
        resp = bot.answer(case["query"])
        s = score_accuracy(resp, case["expected_facts"])
        if s["missing"]:
            failures.append(f"{case['id']}: missing {s['missing']} "
                            f"in answer: {resp.answer!r}")
    assert not failures, "\n".join(failures)
