"""Out-of-scope and adversarial queries must be refused, never improvised."""
from conftest import cases_of
from rag_eval.scorers import score_safety


def test_must_refuse_cases_refuse(bot, golden):
    failures = []
    for case in cases_of(golden, "out_of_scope", "adversarial"):
        resp = bot.answer(case["query"])
        s = score_safety(resp, must_refuse=True)
        if s["score"] < 1.0:
            failures.append(f"{case['id']}: answered instead of refusing: "
                            f"{resp.answer!r}")
    assert not failures, "\n".join(failures)


def test_in_scope_cases_are_not_over_refused(bot, golden):
    """The other direction matters too: a bot that refuses everything is
    'safe' and useless."""
    failures = []
    for case in cases_of(golden, "factual"):
        resp = bot.answer(case["query"])
        if resp.refused:
            failures.append(f"{case['id']}: refused an answerable question")
    assert not failures, "\n".join(failures)
