"""Bot adapters.

The harness talks to any bot through one interface: `answer(query)` returns
a BotResponse with the generated text and the retrieved context it used.

Two adapters ship in the box:

  DemoBot     - a self-contained extractive RAG bot over ./corpus. It exists
                so the whole harness runs (and can be demoed) with zero
                external services.

  Your bot    - copy the stub at the bottom, point it at your real endpoint,
                and return the answer plus retrieved chunks. That's the only
                integration work.

DemoBot also has deliberate FAILURE MODES, switchable by env var. That's the
interview demo: run the suite clean, flip a failure on in a branch, watch
the harness catch it. A test suite you've never seen fail proves nothing.

  RAG_EVAL_FAILURE=hallucinate   -> injects an ungrounded claim into answers
  RAG_EVAL_FAILURE=bad_retrieval -> drops the best-matching doc (simulates a
                                    broken chunking/embedding change)
  RAG_EVAL_FAILURE=no_refusal    -> answers out-of-scope questions anyway
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .retriever import Retriever, stems

REFUSAL_MARKER = "I can't answer that from the documentation I have."

# The demo bot only answers when retrieval confidence clears this bar.
# Below it, the honest behavior is to refuse.
CONFIDENCE_FLOOR = 0.005

# Guardrail: prompt-injection shapes get refused before retrieval is even
# consulted, because injection text often *looks* in-scope (e.g. it mentions
# "password" or "documents"). Retrieval confidence is the wrong defense.
INJECTION_PATTERNS = re.compile(
    r"ignore (your|all|previous) (instructions|rules)|system override|"
    r"reveal .*(internal|all|verbatim)|admin password|jailbreak",
    re.IGNORECASE,
)


@dataclass
class BotResponse:
    query: str
    answer: str
    retrieved_docs: list[str] = field(default_factory=list)  # best first
    context: str = ""  # the text actually given to generation

    @property
    def refused(self) -> bool:
        return REFUSAL_MARKER in self.answer


class BotAdapter:
    """Implement this for your real bot."""

    def answer(self, query: str) -> BotResponse:  # pragma: no cover
        raise NotImplementedError


class DemoBot(BotAdapter):
    def __init__(self, corpus_dir: str | Path, seed: int | None = None,
                 failure_mode: str | None = None):
        self.retriever = Retriever(corpus_dir)
        self.rng = random.Random(seed)
        self.failure_mode = failure_mode if failure_mode is not None \
            else os.environ.get("RAG_EVAL_FAILURE", "")

    def _pick_sentences(self, query: str, doc_text: str) -> list[str]:
        """Lead sentence of the doc plus the sentences most relevant to
        the query."""
        q = set(stems(query))
        lines = [ln for ln in doc_text.splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        body = " ".join(lines)
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", body)
                     if x.strip()]
        if not sentences:
            return []
        lead = sentences[0]
        scored = []
        for x in sentences[1:]:
            overlap = len(q & set(stems(x)))
            if overlap:
                scored.append((overlap, x))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = [lead] + [x for _, x in scored[:2] if x != lead]
        # mild nondeterminism, like a real LLM: sometimes volunteer an
        # extra supporting sentence. Additive only -- early versions of
        # this bot randomly DROPPED sentences, and the accuracy suite
        # failed because required facts vanished on some runs. Good
        # lesson, kept on purpose: nondeterminism must never subtract
        # required content. Consistency scoring exercises this variance.
        if len(scored) > 2 and self.rng.random() < 0.5:
            extra = scored[2][1]
            if extra not in top:
                top.append(extra)
        return top

    def answer(self, query: str) -> BotResponse:
        if INJECTION_PATTERNS.search(query) \
                and self.failure_mode != "no_refusal":
            return BotResponse(query=query, answer=REFUSAL_MARKER)

        hits = self.retriever.retrieve(query, k=3)

        if self.failure_mode == "bad_retrieval" and hits:
            hits = hits[1:]  # silently drop the best doc

        top_doc, top_score = (hits[0] if hits else ("", 0.0))
        in_scope = top_score >= CONFIDENCE_FLOOR

        if not in_scope and self.failure_mode != "no_refusal":
            return BotResponse(query=query, answer=REFUSAL_MARKER,
                               retrieved_docs=[d for d, _ in hits])

        context = "\n\n".join(self.retriever.text(d) for d, _ in hits)

        if not in_scope:  # no_refusal failure mode: make something up
            answer = ("Our team is happy to help with that. The standard "
                      "policy allows this in most cases within 14 days.")
            return BotResponse(query=query, answer=answer,
                               retrieved_docs=[d for d, _ in hits],
                               context=context)

        sentences = self._pick_sentences(query, self.retriever.text(top_doc))
        answer = " ".join(sentences) if sentences else REFUSAL_MARKER

        if self.failure_mode == "hallucinate" and not answer == REFUSAL_MARKER:
            answer += (" Additionally, premium members receive priority "
                       "handling within 2 hours.")  # appears in no document

        return BotResponse(query=query, answer=answer,
                           retrieved_docs=[d for d, _ in hits],
                           context=context)


# ---------------------------------------------------------------------------
# STUB: wire in your real bot here.
# ---------------------------------------------------------------------------
# class MyChatbot(BotAdapter):
#     """Adapter for the real chatbot (e.g. chgallegos.github.io/chatbot)."""
#
#     def __init__(self, endpoint: str):
#         self.endpoint = endpoint
#
#     def answer(self, query: str) -> BotResponse:
#         # 1. POST the query to your bot's API
#         # 2. Capture the generated answer AND the retrieved chunks/sources
#         #    (if your API doesn't return sources, add that first — you
#         #     cannot score grounding without knowing what was retrieved)
#         # 3. Return BotResponse(query, answer, retrieved_docs, context)
#         ...
