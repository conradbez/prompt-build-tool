# Automated morning podcast

Researches a South African audience, picks the strongest show idea, writes a
single-host episode, then has an agent voice it with a local text-to-speech
library.

```
podcast_ideas ─────┐
                   ├─► best_idea ─► outline ─► section_1 ─► section_2 ─► full_script ─┐
audience_research ─┘                                                                   ├─► audio
tts_library ───────────────────────────────────────────────────────────────────────────┘
```

`outline` and `section_1` also feed `full_script` directly.

| Model | What it does |
|---|---|
| `podcast_ideas` | 10 candidate show ideas |
| `audience_research` | What grabs a South African audience's attention |
| `best_idea` | Picks the idea most likely to succeed |
| `outline` | Section-by-section flow of one episode |
| `section_1`, `section_2` | The opening sections, written first so the rest follows their voice |
| `full_script` | The whole episode, ready to read out |
| `tts_library` | Picks a local, pip-installable text-to-speech library |
| `audio` | `agent37` agent: builds the TTS pipeline and returns the audio file |

## Run

```bash
cd examples/podcast
pip install "prompt-build-tool[extras]" anthropic   # extras provides agent37
export ANTHROPIC_API_KEY=...
export AGENT37_API_KEY=...                           # for the audio model
pbt run                                              # writes outputs/full_script.md + audio
pbt test                                             # is the opening gripping?
```

Without an Agent37 key, run everything but the audio:
`pbt run --select full_script`.
