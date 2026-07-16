import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_eval.adapter import DemoBot
from rag_eval.runner import load_golden


@pytest.fixture(scope="session")
def bot():
    """Bot under test, fixed seed. Respects RAG_EVAL_FAILURE so the seeded
    failure modes can be demonstrated straight from pytest:

        RAG_EVAL_FAILURE=hallucinate pytest    # grounding tests catch it
        RAG_EVAL_FAILURE=bad_retrieval pytest  # relevance tests catch it
        RAG_EVAL_FAILURE=no_refusal pytest     # safety tests catch it
    """
    return DemoBot(ROOT / "corpus", seed=7)


@pytest.fixture(scope="session")
def golden():
    return load_golden(ROOT / "golden" / "golden_set.yaml")


def cases_of(golden, *categories):
    return [c for c in golden if c["category"] in categories]
