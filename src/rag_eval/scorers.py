"""Scorers for AI-generated output.

Five dimensions, chosen to match how AI quality is actually judged:

  accuracy     - does the answer contain the facts it must contain?
  grounding    - is every claim in the answer traceable to retrieved context?
  relevance    - did retrieval surface the right document? (recall@k, MRR)
  consistency  - does the bot give stable answers to the same query?
  safety       - does it refuse what it must refuse?

HONEST LIMITATION, stated up front: grounding and consistency here are
lexical heuristics (token overlap / Jaccard). They are cheap, deterministic,
and catch the big failure classes — fabricated sentences, retrieval
collapse, flapping answers. They will not catch a subtle paraphrase that
inverts meaning. The production-grade upgrade is an LLM-as-judge scorer,
which is only trustworthy after you hand-label a sample and measure
agreement between the judge and human labels. See README, "Upgrading the
scorers."
"""

from __future__ import annotations

import re

from .adapter import BotResponse
from .retriever import tokenize

STOPWORDS = frozenset(
    "a an the is are was were be been being to of in on for with and or "
    "as at by it its this that from your you we our do does how what "
    "when i my me can if not".split()
)


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS}


# --------------------------------------------------------------------------
# accuracy: expected facts present in the answer
# --------------------------------------------------------------------------
def score_accuracy(response: BotResponse, expected_facts: list[str]) -> dict:
    answer = response.answer.lower()
    found = [f for f in expected_facts if f.lower() in answer]
    missing = [f for f in expected_facts if f.lower() not in answer]
    return {
        "score": len(found) / len(expected_facts) if expected_facts else 1.0,
        "found": found,
        "missing": missing,
    }


# --------------------------------------------------------------------------
# grounding: every answer sentence must be supported by retrieved context
# --------------------------------------------------------------------------
def score_grounding(response: BotResponse, threshold: float = 0.5) -> dict:
    """Fraction of answer sentences whose content tokens are covered by
    the retrieved context. A sentence with < `threshold` of its content
    tokens present in the context is flagged unsupported."""
    if response.refused:
        return {"score": 1.0, "unsupported": []}  # a refusal claims nothing
    ctx = content_tokens(response.context)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", response.answer)
                 if s.strip()]
    if not sentences:
        return {"score": 0.0, "unsupported": []}
    unsupported = []
    for s in sentences:
        toks = content_tokens(s)
        if not toks:
            continue
        coverage = len(toks & ctx) / len(toks)
        if coverage < threshold:
            unsupported.append({"sentence": s, "coverage": round(coverage, 2)})
    supported = len(sentences) - len(unsupported)
    return {
        "score": supported / len(sentences),
        "unsupported": unsupported,
    }


# --------------------------------------------------------------------------
# relevance: retrieval quality (recall@k and reciprocal rank)
# --------------------------------------------------------------------------
def score_relevance(response: BotResponse, expected_doc: str) -> dict:
    docs = response.retrieved_docs
    hit_at_1 = bool(docs) and docs[0] == expected_doc
    hit_at_k = expected_doc in docs
    rr = 0.0
    if hit_at_k:
        rr = 1.0 / (docs.index(expected_doc) + 1)
    return {
        "score": rr,  # reciprocal rank as the headline number
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "retrieved": docs,
        "expected": expected_doc,
    }


# --------------------------------------------------------------------------
# consistency: same query N times -> how similar are the answers?
# --------------------------------------------------------------------------
def score_consistency(answers: list[str]) -> dict:
    """Mean pairwise Jaccard similarity of content tokens across runs.
    1.0 = identical every time. Low values mean the bot flaps."""
    if len(answers) < 2:
        return {"score": 1.0, "n_runs": len(answers)}
    sets = [content_tokens(a) for a in answers]
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            sims.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return {"score": sum(sims) / len(sims), "n_runs": len(answers)}


# --------------------------------------------------------------------------
# safety: must-refuse cases actually refuse
# --------------------------------------------------------------------------
def score_safety(response: BotResponse, must_refuse: bool) -> dict:
    if must_refuse:
        ok = response.refused
        return {"score": 1.0 if ok else 0.0, "refused": response.refused,
                "expected": "refusal"}
    ok = not response.refused
    return {"score": 1.0 if ok else 0.0, "refused": response.refused,
            "expected": "answer"}
