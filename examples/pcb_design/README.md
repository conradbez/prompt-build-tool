# PCB design with landmarks

Designs a small ESP32 sensor board (default: an IR beam-break for a vending
chute, with the emitter on a snap-off tab) and produces a pre-upload review
sheet for JLCPCB.

The point is the loop from [`motivation.md`](../../motivation.md): make a
prediction, check it against something real, update. An LLM writing hardware
will happily follow the map backwards. A sensor that only ever reports "beam
clear" looks like a working sensor until a product gets stuck in the chute.
The developer preserves lessons from that iteration as saved inputs and
expected outputs, then replays them after changing prompts.

```
brief ─► requirements ─► architecture ─┬─► electrical_check (python) ─────────────────────────────┐
                                        └─► parts ─┬─► research (agent, each part) ─┐              │
                                                   └─► layout_rules ────────────────┴─► board (agent) ─┤
                                                                     ├─► bom_line (each part) ─────────┤
                                                                     └─► polarity (each part) ─► bringup ─► review
```

`research` follows the `complex_components_research` process: for
every polarised or multi-pin part an agent downloads the datasheet, writes the
facts needed to set up the PCB with page numbers, and crops the diagram each
fact comes from. Nothing is kept in a research folder: the notes, the crops and
the PDF are the model's output files. They show in `pbt docs`, land in
`outputs/research/`, and reach `board` and `polarity` as data. Resistors and
capacitors skip without an agent run.

`board` is a [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
working in `build/board/`. It follows the atopile flow: it writes the circuit
as `.ato` with assertions, pins every LCSC id, and runs `ato build`. It then
places the parts with a KiCad-Python script: it predicts every layout rule's
measurement, measures the placed board, and runs `kicad-cli pcb drc`. If a rule
or DRC fails, it changes the placement and tests again, recording each attempt.
It submits the BOM atopile generated and the net on every polarised part's physical pads.

Two QA stages run once per part, with `{{ config(each="parts.parts[*]") }}`.
They mirror the two JLC upload steps you check line by line: the BOM page and
the component placement view. Both compare the intended part against what
`board` actually built, so a 51 Ω part built as 510 Ω, or a diode with the
right pin names on the wrong pads, shows up per part. Each part is its own LLM call and its own cache
entry. Change one part and only that part is re-checked. `polarity` skips
non-polarised parts with `skip_and_set_to_value`, so a resistor costs no call.

Models are grouped by stage, in reading order. The `1_a_` prefixes only sort
the files; `ref('parts')` still names `2_build/2_a_parts.prompt`.

```
models/
  1_design/  1_a_brief  1_b_requirements  1_c_architecture  1_d_electrical_check
  2_build/   2_a_parts  2_b_research  2_c_layout_rules  2_d_board
  3_qa/      3_a_bom_line  3_b_polarity  3_c_bringup  3_d_review
```

| Model | What it does |
|---|---|
| `brief` | Template: the board to design, overridable with `--promptdata brief=...` |
| `requirements` | Rails (and rails you must not use), interfaces, envelope, known limits |
| `architecture` | Each block as a hypothesis with `expect_if_right` / `wrong_if`, plus the numbers to check it |
| `electrical_check` | Python, no LLM: Ohm's law on every current path, gate drive vs Rds(on) rating, ESP32 strapping/flash/input-only pins |
| `parts` | Pins an LCSC id on every part, records pinned value vs wanted value and pin nets |
| `bom_line` | Map, once per part: pinned value vs wanted vs built, Basic vs Extended, stock, package, SMT or do-not-place (JLC step 2) |
| `polarity` | Map over `research`, once per polarised part, with that part's notes, crops and datasheet attached: datasheet marking, net on the marked pad as intended and as built, what JLC's 3D view should show (JLC step 3) |
| `research` | Agent, once per polarised or multi-pin part: datasheet facts for the PCB, page-cited crops, mismatches with the design. Notes, crops and PDF are its output files |
| `layout_rules` | At most six rules, each tied to the failure it prevents and its bench symptom |
| `board` | Agent: `.ato` + `ato build`, placement tested against `layout_rules` and DRC, retried until it passes, then reads back the built BOM and pad nets |
| `bringup` | Unpowered polarity inspections from `polarity`, then powered checks, negative test first |
| `review` | Reduce, as a template: one-page review sheet with the BOM and polarity tables, ending in the upload checklist |

## Run

Needs [atopile](https://atopile.io) 0.15 (`ato`), KiCad 10 (`kicad-cli` and its
bundled Python), `uv` for `easyeda2kicad`, and `pip install mini-swe-agent`.
Keep KiCad closed while it runs.

```bash
cd examples/pcb_design
export GEMINI_API_KEY=...                              # default model gemini-3.5-flash-lite; override with GEMINI_MODEL
export MSWEA_MODEL_NAME=gemini/gemini-3.5-flash-lite    # the agents' model, via litellm
pbt run                     # writes build/board/ and outputs/review.md
pbt test                    # replay the saved regression cases
pbt run --promptdata reference_project=~/path/to/a/working/atopile/project
pbt run --promptdata brief="ESP32 soil-moisture sensor, 3V3 only, capacitive probe on a 2-wire cable"
```

The agent is capped at 150 steps and $5 (`agent_step_limit`, `agent_cost_limit`
in `2_d_board.prompt`). Like every model it is cached on its rendered prompt:
it re-runs only when the design, parts or layout rules change.

## Known-fault cases

`tests/reversed_flyback_diode.yml` replays the real reversed SS34
(`circuit_board_as_code/qaqc/common_issues/diode_error.md`): the circuit
intent is right, the library mapped the cathode to pin 2, and the built board
puts the band on the MOSFET drain. It pins only the two models that carry the
fault with `given`: `parts` (the library's pin map) and `board` (the copper).
Everything else runs for real from a one-line `brief`, including the datasheet
`research` that `polarity` checks against. `ato build`, assertions and DRC all
pass on this board, which is the point.

```bash
pbt test --case "Reversed flyback diode"
```

## Tests: given inputs, expect outputs

Each YAML case uses `given` to supply saved evidence and `expect` to check
selected fields in the QA output. PBT compares the fields directly, without
another LLM judging the answer:

```yaml
expect:
  polarity:
    - {ref: D1, built_pads_agree: false}
    - {ref: Q2, built_pads_agree: true, issues: []}
```

The reversed case must reject D1. A second case corrects D1's physical pad
connections and must accept them. Both must accept Q2, so indiscriminately
flagging components cannot pass. The corrected case checks physical pad
agreement only: the saved parts list still contains the original library's
incorrect logical pin mapping.

Only listed fields are checked; list items are matched by `ref`. Expectations
stay outside model inputs. A missing item, missing field, or wrong value fails
`pbt test` with the field path, and the command exits nonzero for CI.

These are integration tests of QA on saved evidence. They do not rebuild a
board or certify it.
Run `pbt test` for both cases, or select one with `--case`.
