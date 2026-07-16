"""A small TF-IDF retriever over the markdown corpus.

Pure python on purpose: no vector DB, no embeddings service, nothing to
install or mock. The point of the harness is to test the *pattern*
(retrieve -> generate -> score). Swap this class for your real retriever
when you wire in a production bot; the scorers don't care where the
context came from.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())


def stem(token: str) -> str:
    """Very light suffix stripping so 'supported' matches 'support' and
    'limits' matches 'limit'. Not linguistics, just enough for matching."""
    for suffix in ("ing", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


STOPWORDS = frozenset(
    "a an the is are was were be been being to of in on for with and or "
    "as at by it its this that from your you we our do does how what "
    "when i my me can if not about all after which per must us".split()
)


def stems(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text)
            if t not in STOPWORDS and stem(t) not in STOPWORDS]


class Retriever:
    def __init__(self, corpus_dir: str | Path):
        self.docs: dict[str, str] = {}
        self.doc_tokens: dict[str, Counter] = {}
        for path in sorted(Path(corpus_dir).glob("*.md")):
            text = path.read_text()
            self.docs[path.name] = text
            self.doc_tokens[path.name] = Counter(stems(text))
        self.n_docs = len(self.docs)
        # document frequency per term
        self.df: Counter = Counter()
        for toks in self.doc_tokens.values():
            for term in toks:
                self.df[term] += 1

    def _score(self, query_tokens: list[str], doc: str) -> float:
        toks = self.doc_tokens[doc]
        length = sum(toks.values()) or 1
        score = 0.0
        for term in query_tokens:
            if term not in toks:
                continue
            tf = toks[term] / length
            idf = math.log((self.n_docs + 1) / (self.df[term] + 1)) + 1
            score += tf * idf
        return score

    def retrieve(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Return [(doc_name, score)] for the top-k docs, best first."""
        q = stems(query)
        ranked = sorted(
            ((d, self._score(q, d)) for d in self.docs),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(d, s) for d, s in ranked[:k]]

    def text(self, doc_name: str) -> str:
        return self.docs[doc_name]
