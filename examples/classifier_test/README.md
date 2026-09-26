# Classifier-judged tests

`pbt test` judged by a local [Ollaya](https://ollaya.dev) classifier instead of
an LLM. No API key needed: the model is a fixed haiku, and the tests only ask
yes/no questions about it.

## 1. Install Ollaya

```bash
curl -fsSL https://ollaya.dev/install.sh | sh
```

Windows, Docker and other options: <https://ollaya.dev/download>.

## 2. Start it

```bash
ollaya run laya
```

This downloads the `laya` model on first use and starts the server on
`localhost:11435`. Leave it running (or use `ollaya serve`).

## 3. Run the tests

```bash
cd examples/classifier_test
pbt run    # builds the haiku
pbt test   # asks Ollaya about it
```

Both tests should pass:

```
  [1/2] haiku_has_three_lines … PASS (P(yes)=…)
  [2/2] haiku_is_about_autumn … PASS (P(yes)=…)
```

## What is here

| File | What it does |
|---|---|
| `models/haiku.prompt` | A fixed haiku (`model_type="template"`, no LLM call) |
| `tests/haiku_has_three_lines.prompt` | Passes when P(yes) ≥ 0.5 |
| `tests/haiku_is_about_autumn.prompt` | Passes when P(yes) ≥ 0.8, set with `config(threshold=0.8)` |
| `client.py` | `classify_call` pointing at Ollaya, and `test_judge = "classifier"` |

A classifier test is a question, a `---` line, then the text to judge:

```jinja
Is this poem about autumn?
---
{{ ref('haiku') }}
```

`client.py` builds the judge with no extra dependencies:

```python
classify_call = pbt.systemone_classifier(model="laya", base_url="http://localhost:11435")
test_judge = "classifier"
```

Set `OLLAYA_MODEL` or `OLLAYA_URL` to use another model or server. To use
hosted Jev instead, call `pbt.systemone_classifier()` with no arguments and set
`TYPESAFE_API_KEY`.
