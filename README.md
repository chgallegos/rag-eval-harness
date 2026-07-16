# RAG Evaluation Harness

An automated evaluation and regression-testing harness for RAG chatbots.
It scores AI-generated answers on five dimensions, catches quality
degradation before release, and blocks the ship when a change makes the
bot worse.

Built by Chris Gallegos. The design comes from two years as the SME on an
enterprise generative search product, where I found these failure modes in
production, after customers had already seen them. This harness moves that
work upstream.

## Why this exists

You cannot unit-test an LLM answer with `assertEquals`. The output is
non-deterministic, "correct" has multiple dimensions, and the scariest
failures are the confident ones. What you can do:

1. Maintain a **golden test set**: real queries with known-good expectations.
2. Score answers on the dimensions that matter, not string equality.
3. Benchmark every change against a baseline and **fail the build** when
   quality drops.

That is the whole harness. Three moving parts, plain JSON in between.

## The five dimensions

| Dimension   | Question it answers                                  | Catches                        |
|-------------|------------------------------------------------------|--------------------------------|
| accuracy    | Does the answer contain the facts it must contain?   | Wrong or missing information   |
| grounding   | Is every claim traceable to the retrieved context?   | Hallucination                  |
| relevance   | Did retrieval surface the right document?            | Broken chunking / embeddings   |
| consistency | Does the same question get a stable answer?          | Flapping, prompt fragility     |
| safety      | Does it refuse what it must refuse (and only that)?  | Injection, scope creep, over-refusal |

Safety is scored in both directions on purpose. A bot that refuses
everything scores perfectly on refusal and is useless; the suite fails
over-refusal too.

## Quickstart

```bash
pip install -r requirements.txt
pytest                                   # 6 tests, all green
python scripts/run_eval.py               # full scored report -> reports/report.json
```

## The demo: watch it catch a regression

A test suite you have never seen fail proves nothing. The built-in demo bot
has three seeded failure modes, each simulating a real production failure
class:

```bash
# a model/prompt change starts fabricating claims
RAG_EVAL_FAILURE=hallucinate pytest        # grounding tests FAIL

# a chunking/embedding change silently breaks retrieval
RAG_EVAL_FAILURE=bad_retrieval pytest      # relevance/over-refusal tests FAIL

# a guardrail gets disabled
RAG_EVAL_FAILURE=no_refusal pytest         # safety tests FAIL
```

And the release-gate version of the same story:

```bash
python scripts/run_eval.py --label baseline --out reports/baseline.json
RAG_EVAL_FAILURE=hallucinate python scripts/run_eval.py --label candidate --out reports/candidate.json
python scripts/compare.py reports/baseline.json reports/candidate.json
```

```
dimension       baseline  candidate    delta   limit  status
------------------------------------------------------------
accuracy          1.0000     1.0000  +0.0000   -0.02  ok
consistency       0.9634     0.9741  +0.0107   -0.10  ok
grounding         1.0000     0.7244  -0.2756   -0.02  FAIL
relevance         1.0000     1.0000  +0.0000   -0.05  ok
safety            1.0000     1.0000  +0.0000   -0.00  ok
------------------------------------------------------------
REGRESSION DETECTED - do not ship
```

Exit code 1. In CI (`.github/workflows/eval.yml`) that blocks the merge.

Thresholds are per-dimension because tolerance is per-dimension: a 2-point
dip in consistency is noise, a 2-point dip in safety is an incident. Safety
has zero tolerance.

## Project layout

```
golden/golden_set.yaml    the golden test set (queries + expectations)
corpus/                   mock knowledge base the demo bot answers from
src/rag_eval/
  adapter.py              BotAdapter interface, DemoBot, failure modes
  retriever.py            small TF-IDF retriever (swappable)
  scorers.py              the five scorers
  runner.py               runs the suite, writes report JSON
  regression.py           compares reports, gates the release
scripts/run_eval.py       CLI: run the suite
scripts/compare.py        CLI: regression gate (exit 1 on regression)
tests/                    pytest suite, one file per dimension
ui-tests/                 Playwright examples for the UI layer
.github/workflows/        CI: pytest -> eval -> regression gate -> artifact
```

## Wiring in a real bot

The harness only needs one method. Implement `BotAdapter.answer(query)`
(stub at the bottom of `src/rag_eval/adapter.py`) returning the generated
answer **and the retrieved context it used**. If your bot's API does not
expose what it retrieved, add that first: grounding cannot be scored
without knowing what the model was given.

Then replace the corpus and golden set with your real content and real
queries. The best source of golden cases is production: every defect you
find in the wild becomes a case, so it can never regress silently again.

## Honest limitations (read before the interview)

- **The scorers are lexical heuristics.** Grounding and consistency use
  token overlap and Jaccard similarity. Cheap, deterministic, and they
  catch the big failure classes: fabricated sentences, retrieval collapse,
  flapping answers. They will not catch a fluent paraphrase that inverts
  meaning. The upgrade path is an LLM-as-judge scorer, and a judge is only
  trustworthy after you hand-label a sample of outputs and measure
  judge-vs-human agreement. Deploying an unvalidated judge just moves the
  hallucination problem into the test suite.
- **The demo bot is extractive**, so its grounding is nearly perfect by
  construction. That is intentional: it makes the clean baseline
  trustworthy, so any drop is signal. Real generative bots start lower.
- **The golden set is 20 cases.** Enough to prove the pattern; a production
  suite wants hundreds, weighted toward past incidents.
- **Consistency uses set similarity**, so it measures content stability,
  not ordering or tone.

## Design decisions worth defending

- **Failure modes ship in the product code.** The demo is "watch it catch
  the bug", not "watch it pass". A green suite that has never seen red is
  untested itself.
- **Nondeterminism must be additive.** An early version of the demo bot
  randomly dropped sentences to simulate LLM variance, and the accuracy
  suite failed intermittently because required facts vanished on some runs.
  That is exactly the class of flake that erodes trust in AI evals, so the
  fix is a rule: variance may add supporting content, never remove required
  content. The consistency scorer measures what remains.
- **Reports are plain JSON.** Diffable, archivable per release, postable to
  a PR. The regression gate is just a diff with opinions.
- **Zero-tolerance safety threshold.** Every other dimension gets a noise
  allowance. Safety regressions ship as incidents, so they do not ship.
