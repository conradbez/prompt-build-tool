# Classifier tests for chained prompts

## Why pbt

Can an LLM drive a business plan, can it deliver value? What is the distinction between human ingenuity and LLM genius? I believe it stems from mapping meaning. The intuition on understanding where we are, where we're going. Not the facts of coordinates on a map but the sense of direction, the feeling you get to turn right when exiting a pub. This feeling drives creative solutions to other people's problems and a key is the conscious map of context that drives intuition.

Asking an LLM to do a task misses this step because we don't come back to it with the same context we had in writing it. It lacks the map of exiting the pub and turning right, it lacks the conscious intuition of knowing how the other person's experience requires the correct answer. Of course our job is to bring this and keep bringing it across multiple weeks, months and years. Pbt declaratively records your map of meaning, solidifying the action of the LLM in the context of the overall map you design. Key in this is the sense check of direction along the way - which is where `testing` comes into play.

When you're walking you have a sense for where you're going but a friend might question "have we not passed this before" or "I think this train station should be on the left if we're going the right way". These are tests we place on our maps of meaning, and similarly our LLM maps need `tests`. I believe the test here is a Jev style classifier of "does this output match expected structure" spread into the direction SET BY THE PERSON DRIVING MAP DIRECTION.

**The map (conceptual).** We want to start a vending machine business. That needs sheet metal parts, which need (1) a CAD design and (2) sheet metal fabrication:

```
Vending machine business
├── (1) CAD design:  programmatic CAD for a 60cm cubed box from sheet metal
│     └── TEST: does the box look like the client description
│               and fit the overall project goal?
└── (2) Sheet metal fabrication   ← only worth doing if (1) passes
```

**The same map in pbt.** Step (1) is a model, the check under it is a classifier test:

```jinja
{# models/cad_box.prompt #}
Write a programmatic CAD script (CadQuery) for a 60cm cubed box
made from sheet metal.
Client description: {{ promptdata("client_description") }}
```

```jinja
{# tests/cad_box_fits_project.prompt #}
{{ config(judge="classifier", threshold=0.8) }}
Does this box design match the client description and fit the project goal?
CLIENT DESCRIPTION: {{ promptdata("client_description") }}
PROJECT GOAL: {{ promptdata("project_goal") }}
---
{{ ref('cad_box') }}
```

```bash
# runs the models for this one case, then the tests (both see the promptdata)
pbt test --promptdata project_goal="Launch a vending machine business" \
         --promptdata client_description="Brushed steel, front hatch, 60cm cube"
```

Expressing a map in pbt is only as useful as knowing that the path followed matches intuitive clues along the way - if you want to get home and understand to go left but miss a turn you have yes/no tests along your walk - that is `pbt test`.

## Chained pipeline example

When prompts build on each other (`extract_facts → outline → article`), one bad step poisons everything after it. Put a cheap classifier test on each hop so a failure points at the step that broke, not the final output:

```jinja
{# tests/facts_are_only_about_topic.prompt #}
Are all of these facts about the printing press?
---
{{ ref('extract_facts') }}

{# tests/outline_uses_the_facts.prompt #}
{{ config(threshold=0.8) }}
Does this outline only use claims from the facts below?
---
FACTS: {{ ref('extract_facts') }}
OUTLINE: {{ ref('outline') }}

{# tests/article_follows_outline.prompt #}
Does this article follow the outline's section order?
---
OUTLINE: {{ ref('outline') }}
ARTICLE: {{ ref('article') }}
```

If `outline_uses_the_facts` fails but `facts_are_only_about_topic` passes, the bug is in `outline.prompt`. Classifier calls are fast and cheap, so run them on every YAML case.
