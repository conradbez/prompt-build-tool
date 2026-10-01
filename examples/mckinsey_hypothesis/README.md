# Hypothesis-driven business research

Picks a random business idea and works it the way a strategy consultancy would:
frame the question, state an answer up front, break it into a MECE hypothesis
tree, test each branch, then roll the results back up.

Each model is one hypothesis in a map (see [`motivation.md`](../../motivation.md)).
Lower-order hypotheses (market, competition, economics) feed the governing one,
so when the answer looks wrong you fix the branch that drifted instead of
rewriting the whole prompt.

```
business_idea ─► situation ─► issue_tree ─┬─► branch_market ──────┐
  (python)        (SCQ)      (governing    ├─► branch_competition ─┼─► synthesis ─► memo
                              hypothesis)  └─► branch_economics ───┘   (pyramid)   (template)
```

| Model | What it does |
|---|---|
| `business_idea` | Python model: picks a random idea from a list |
| `situation` | Situation, Complication, Key Question; success criteria |
| `issue_tree` | Governing hypothesis plus three MECE branch hypotheses |
| `branch_*` | Predicts what it expects to see, names what would disprove it, then weighs evidence |
| `synthesis` | Answer first, three supporting arguments, whether the hypothesis survived |
| `memo` | Template model: assembles the client-ready memo, no LLM call |

## Run

```bash
cd examples/mckinsey_hypothesis
export GEMINI_API_KEY=...
pbt run                                  # writes outputs/memo.md
pbt run --promptdata seed=$RANDOM        # a different random idea
pbt run --promptdata idea="Drone-delivered pharmacy for rural Ireland"
pbt test                                 # check the landmarks
```

`business_idea` is cached on its rendered code, so plain `pbt run` gives the
same idea every time. Pass a new `seed` to reroll.

## Tests: the landmarks

The tests check the reasoning, not the prose:

- `issue_tree_is_mece`: the three branches don't overlap and together answer the key question
- `hypotheses_are_falsifiable`: every branch names a concrete result that would disprove it
- `synthesis_follows_evidence`: the recommendation matches the branch verdicts
