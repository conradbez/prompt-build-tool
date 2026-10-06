# Markdown notes and Obsidian vaults (`pbt obsidian`)

Write your prompts as markdown notes and connect them with links. `pbt obsidian`
converts them into ordinary `.prompt` files and then runs the normal pbt
command on them, so everything else — `client.py`, caching, validation, tests,
docs — works unchanged.

The source is either:

- **one `.md` note** — it and every note it links to, followed transitively.
  Each link resolves relative to the folder of the note it is written in, like
  links in any markdown viewer, so no vault or Obsidian is needed.
- **a vault folder** — every note in it, with links resolved the way Obsidian
  resolves them (relative path, then path from the vault root, then title or
  alias anywhere in the vault).

```bash
pbt obsidian run   notes/plan.md              # plan.md and the notes it links to
pbt obsidian build my_vault                   # convert only → obsidian_models/
pbt obsidian run   my_vault                   # convert, then pbt run
pbt obsidian run   my_vault -s article --promptdata audience=devs
pbt obsidian ls    my_vault
pbt obsidian test  my_vault                   # tests/ as usual
pbt obsidian docs  my_vault --open
```

Options after the source that `pbt obsidian` does not know are passed to the
underlying command. `--out DIR` (default `obsidian_models/`) chooses where the
`.prompt` files go; `client.py` is looked up beside that directory, i.e. in the
current directory by default. The directory is rebuilt on every call, and pbt
refuses to write into a non-empty directory it did not generate.

## How notes become models

| In the note | In the `.prompt` file |
|---|---|
| `Market Topic.md` | model `market_topic` (lower-cased, non-word runs → `_`) |
| `[[Market Topic]]`, `![[Market Topic]]` | `{{ ref('market_topic') }}` |
| `[the topic](research/Market%20Topic.md)`, `[t](../Topic.md)` | `{{ ref('market_topic') }}` (link text dropped) |
| `[[../shared/Brief]]`, `[[research/Market Topic]]` | resolved from the linking note's folder |
| `[[Market Topic\|shown]]`, `[[Market Topic#Heading]]` | `{{ ref('market_topic') }}` (alias/anchor dropped) |
| `[[Market Topic]]` in another folder, frontmatter `aliases` | vault source only: found by title, like Obsidian |
| `[[Missing Note]]` | `Missing Note` as plain text, with a warning |
| `![[chart.png]]`, `[site](https://…)`, other non-`.md` links | left as-is |
| `%% comment %%` | removed |
| fenced code blocks | copied verbatim (links inside are not converted) |
| Jinja (`{{ promptdata('x') }}`, `{% if %}`, …) | passes through |

In a vault, hidden folders (`.obsidian/`, `.trash/`) are skipped. Two notes that would get
the same model name, or a link that matches several notes, is an error.

## Config in frontmatter

A `pbt:` mapping becomes the model's `config()`. Other frontmatter (tags, dates,
…) is ignored. `[[links]]` in its values become model names:

```markdown
---
tags: [drafts]
pbt:
  each: "[[Ideas]].items[*]"
  output_extension: html
---
Expand this idea into a paragraph: {{ each }}

Keep the tone of [[Style Guide]].
```

`pbt: false` leaves a note out of the build.

## Prompt notes vs data notes

Not every note is an instruction for the LLM — some are information you wrote
down: facts, a style guide, a product brief. A **data** note becomes a
`model_type="template"` model: its Jinja and `[[links]]` still render, but no
LLM is called, and its text is passed as-is to the notes that link to it.

Mark a note yourself in its frontmatter:

```markdown
---
pbt: data        # or: pbt: prompt
---
Brand facts: we sell socks, founded 2019, playful tone.
```

A `pbt:` config that sets `model_type` counts as marked too. Unmarked notes are
prompts — unless you let a judge decide:

```bash
pbt obsidian run my_vault --judge-notes llm         # asks client.py's llm_call
pbt obsidian run my_vault --judge-notes classifier  # client.py's classify_call; P(wants AI input) ≥ 0.5 → prompt
```

The judge is told the situation — the notes mix **reference material** the
user keeps for other notes (facts, sources, background, a brief) with notes they
want **AI input on** (hypotheses to test, ideas, open questions, drafts, tasks) —
and sees, for each note, its title, folder, the notes that link to it, the notes
it links to, and its text. Being linked to is a strong hint of reference
material; a note that links out and proposes or asks is usually a prompt.

Tell it about your project in `client.py` for better calls:

```python
# client.py
obsidian_judge = "llm"   # default for every `pbt obsidian` command
obsidian_judge_context = """
Early-stage research for a sock subscription startup. Notes in Sources/ are
interview transcripts and market data; Hypotheses/ holds claims we want challenged.
"""
```

`--judge-notes off` turns the judge off for one call. Marked notes are never
judged. Verdicts are cached in `.pbt/obsidian_judgements.json` on everything the
judge sees (text, folder, links, your context), so only notes that changed — or
whose links changed — are judged again. `pbt obsidian build` shows each note's
kind, with `(judged)` where the judge decided.

## From Python

```python
import pbt
from pbt.obsidian import load_vault

pbt.run(models_from_dict=load_vault("notes/plan.md"), llm_call=my_llm_call)  # or a vault folder

# with a judge for unmarked notes
from pbt.obsidian.judge import llm_judge
load_vault("my_vault", judge=llm_judge(my_llm_call, context="What this vault is about"))
```

`load_vault()` and `convert()` take a `.md` note or a folder. `convert()` returns the per-note details (model name, links, unresolved
links) and `write_models()` writes them to a directory.
