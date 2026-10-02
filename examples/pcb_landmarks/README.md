# PCB design with landmarks

Designs a small ESP32 sensor board (default: an IR beam-break for a vending
chute, with the emitter on a snap-off tab) and produces a pre-upload review
sheet for JLCPCB.

The point is the loop from [`motivation.md`](../../motivation.md): make a
prediction, check it against something real, update. An LLM writing hardware
will happily follow the map backwards. A sensor that only ever reports "beam
clear" looks like a working sensor until a product gets stuck in the chute.
So every stage states what it expects to see, and the tests look for those
landmarks.

```
brief ─► requirements ─► architecture ─┬─► electrical_check (python) ─────────────────┐
                                        └─► parts ─┬─► bom_line (each part) ───────────┤
                                                   ├─► polarity (each part) ─┐         ├─► review (template)
                                                   └─► layout_rules ─────────┴─► bringup ┘
```

Two stages run once per part, with `{{ config(each="parts.parts[*]") }}`. They
mirror the two JLC upload steps you check line by line: the BOM page and the
component placement view. Each part is its own LLM call and its own cache
entry. Change one part and only that part is re-checked. `polarity` skips
non-polarised parts with `skip_and_set_to_value`, so a resistor costs no call.

| Model | What it does |
|---|---|
| `brief` | Template: the board to design, overridable with `--promptdata brief=...` |
| `requirements` | Rails (and rails you must not use), interfaces, envelope, known limits |
| `architecture` | Each block as a hypothesis with `expect_if_right` / `wrong_if`, plus the numbers to check it |
| `electrical_check` | Python, no LLM: Ohm's law on every current path, gate drive vs Rds(on) rating, ESP32 strapping/flash/input-only pins |
| `parts` | Pins an LCSC id on every part, records pinned value vs wanted value and pin nets |
| `bom_line` | Map, once per part: pinned value vs wanted, Basic vs Extended, stock, package, SMT or do-not-place (JLC step 2) |
| `polarity` | Map, once per polarised part: datasheet marking, net on the marked pad, what JLC's 3D view should show (JLC step 3) |
| `layout_rules` | At most six rules, each tied to the failure it prevents and its bench symptom |
| `bringup` | Unpowered polarity inspections from `polarity`, then powered checks, negative test first |
| `review` | Reduce, as a template: one-page review sheet with the BOM and polarity tables, ending in the upload checklist |

## Run

```bash
cd examples/pcb_landmarks
export GEMINI_API_KEY=...
pbt run                     # writes outputs/review.md
pbt test                    # check the landmarks
pbt run --promptdata brief="ESP32 soil-moisture sensor, 3V3 only, capacitive probe on a 2-wire cable"
```

## Tests: the landmarks

Each test targets a failure that actually happened or nearly happened on a
real board built this way:

| Test | Failure it catches |
|---|---|
| `pinned_parts_match_wanted` | The design asserts 51 Ω, the pinned LCSC id is something else. Assertions check the declared value, not the part that ships |
| `polarised_parts_have_pinout` | A reversed flyback diode shorted 12 V to ground. Judges the per-part `polarity` checks |
| `electrical_checks_pass` | LED over-current, a MOSFET only rated at 10 V gate driven from 3.3 V, a receiver on an input-only pin with no pull-up |
| `bringup_starts_with_failure` | A receiver picking up board leakage reports "beam clear" with the beam blocked, so every vend looks successful |

`electrical_check` is the cheapest landmark: arithmetic, no LLM, cached on its
code. When it fails, fix `architecture`, not the review.
