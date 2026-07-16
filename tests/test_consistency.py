"""Same question, stable answer. Non-determinism is measured, not denied."""
from conftest import cases_of
from rag_eval.scorers import score_consistency

RUNS = 5
MIN_CONSISTENCY = 0.6  # mean pairwise Jaccard similarity floor


def test_answers_are_stable_across_runs(bot, golden):
    failures = []
    for case in cases_of(golden, "factual"):
        answers = [bot.answer(case["query"]).answer for _ in range(RUNS)]
        s = score_consistency(answers)
        if s["score"] < MIN_CONSISTENCY:
            failures.append(f"{case['id']}: consistency {s['score']:.2f}")
    assert not failures, "\n".join(failures)
