"""No answer may contain claims unsupported by the retrieved context."""
from conftest import cases_of
from rag_eval.scorers import score_grounding


def test_all_answer_sentences_are_grounded(bot, golden):
    failures = []
    for case in cases_of(golden, "factual", "ambiguous"):
        resp = bot.answer(case["query"])
        s = score_grounding(resp)
        for u in s["unsupported"]:
            failures.append(f"{case['id']}: ungrounded sentence "
                            f"(coverage {u['coverage']}): {u['sentence']!r}")
    assert not failures, "\n".join(failures)
