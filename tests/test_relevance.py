"""Retrieval must surface the expected document."""
from conftest import cases_of
from rag_eval.scorers import score_relevance

MIN_MRR = 0.8          # average reciprocal rank across the suite
REQUIRE_HIT_AT_K = 1.0  # expected doc must ALWAYS be in top-k


def test_retrieval_quality(bot, golden):
    cases = cases_of(golden, "factual", "ambiguous")
    rrs, misses = [], []
    for case in cases:
        resp = bot.answer(case["query"])
        s = score_relevance(resp, case["expected_doc"])
        rrs.append(s["score"])
        if not s["hit_at_k"]:
            misses.append(f"{case['id']}: expected {s['expected']}, "
                          f"got {s['retrieved']}")
    hit_rate = 1 - len(misses) / len(cases)
    assert hit_rate >= REQUIRE_HIT_AT_K, "\n".join(misses)
    mrr = sum(rrs) / len(rrs)
    assert mrr >= MIN_MRR, f"MRR {mrr:.3f} below floor {MIN_MRR}"
