# Obsidian vaults (`pbt obsidian`)

Write your prompts as Obsidian notes and connect them with links. `pbt obsidian`
converts a vault into ordinary `.prompt` files and then runs the normal pbt
command on them, so everything else — `client.py`, caching, validation, tests,
docs — works unchanged.

```bash
pbt obsidian build my_vault                   # convert only → obsidian_models/
pbt obsidian run   my_vault                   # convert, then pbt run
pbt obsidian run   my_vault -s article --promptdata audience=devs
pbt obsidian ls    my_vault
pbt obsidian test  my_vault                   # tests/ as usual
pbt obsidian docs  my_vault --open
```

Options after the vault that `pbt obsidian` does not know are passed to the
underlying command. `--out DIR` (default `obsidian_models/`) chooses where the
`.prompt` files go; `client.py` is looked up beside that directory, i.e. in the
current directory by default. The directory is rebuilt on every call, and pbt
refuses to write into a non-empty directory it did not generate.

## How notes become models

| In the note | In the `.prompt` file |
|---|---|
| `Market Topic.md` | model `market_topic` (lower-cased, non-word runs → `_`) |
| `[[Market Topic]]`, `![[Market Topic]]` | `{{ ref('market_topic') }}` |
| `[[Market Topic\|shown]]`, `[[Market Topic#Heading]]` | `{{ ref('market_topic') }}` (alias/anchor dropped) |
| `[[Research/Market Topic]]`, frontmatter `aliases` | resolved like Obsidian does |
| `[[Missing Note]]` | `Missing Note` as plain text, with a warning |
| `![[chart.png]]` and other attachments | left as-is |
| `%% comment %%` | removed |
| fenced code blocks | copied verbatim (links inside are not converted) |
| Jinja (`{{ promptdata('x') }}`, `{% if %}`, …) | passes through |

Hidden folders (`.obsidian/`, `.trash/`) are skipped. Two notes that would get
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
pbt obsidian run my_vault --judge-notes classifier  # client.py's classify_call; P(yes) ≥ 0.5 → prompt
```

Set `obsidian_judge = "llm"` in `client.py` to make it the default
(`--judge-notes off` turns it off for one call). Marked notes are never judged.
Verdicts are cached in `.pbt/obsidian_judgements.json` on the note's text, so
only new or edited notes are judged again. `pbt obsidian build` shows each
note's kind, with `(judged)` where the judge decided.

## From Python

```python
import pbt
from pbt.obsidian import load_vault

pbt.run(models_from_dict=load_vault("my_vault"), llm_call=my_llm_call)

# with a judge for unmarked notes
from pbt.obsidian.judge import llm_judge
load_vault("my_vault", judge=llm_judge(my_llm_call))
```

`convert_vault()` returns the per-note details (model name, links, unresolved
links) and `write_models()` writes them to a directory.
