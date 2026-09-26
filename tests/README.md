# Classifier tests for chained prompts

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
