# 07 — Benchmark protocol fidelity vs the HF model card

**Goal:** make Δ-vs-card claims sound by running the card's ACTUAL protocols where the card
publishes them — and by labeling, per benchmark, exactly what we reproduce, what we substitute,
and what the card leaves unspecified. Implemented as `run_bench.py --protocol card|greedy`
(default `greedy` = the pre-protocol harness, byte-identical). CPU-validated 2026-07-07
(`bench/test_bench.py` green; `--stub` green in both protocols); not yet run on the pod.

Source of truth: `reference/hf-repo/README.md` (the committed copy of the
`zai-org/GLM-5.2-FP8` model card). All quotes below are verbatim from its **Footnote** section.

---

## 1. What the card ACTUALLY specifies (verbatim)

The ONLY footnote covering the reasoning benchmarks (HLE, CritPt, AIME 2026, HMMT Nov. 2025,
HMMT Feb. 2026, IMOAnswerBench, GPQA-Diamond):

> **Humanity’s Last Exam (HLE) & other reasoning tasks**: We use sampling parameters of
> `temperature=1.0`, `top_p=0.95` for evaluation. We evaluate with a maximum generation length
> of `163,840` tokens. By default, we report the text-only subset; results marked with * are
> from the full set. For AIME, HMMT and IMOAnswerBench, we evaluate  each question using the
> following system prompt: `Your response should be in the following format:\nExplanation:
> {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\n
> Confidence: {your confidence score between 0% and 100% for your answer}.` We use GPT-5.5
> (medium) as the judge model. For HLE-with-tools, we use a maximum context length of 300,000
> tokens, with no context management strategy.

Parsed, that is:

| Element | Card says | Scope |
|---|---|---|
| `temperature` | `1.0` | HLE & other reasoning tasks (incl. GPQA-Diamond, AIME, HMMT, IMOAnswerBench, CritPt) |
| `top_p` | `0.95` | same |
| max generation length | `163,840` tokens | same |
| system prompt | the `Explanation:/Exact Answer:/Confidence:` format prompt (backticked `\n` = real newlines; the trailing `.` after the closing brace is part of the prompt) | **"For AIME, HMMT and IMOAnswerBench"** — explicitly scoped; NOT stated for GPQA/CritPt |
| grading | "We use GPT-5.5 (medium) as the judge model." | follows the AIME/HMMT/IMOAnswerBench sentence; most plausibly the free-form-answer tasks (it is also the official HLE grading setup, from which this system prompt originates) |
| samples / averaging | **nothing** — no k, no maj@k, no avg@k for any reasoning task (only Terminal-Bench-Claude-Code says "averaged over 5 runs") | — |
| seeds | **nothing** | — |
| few-shot / MC prompt format | **nothing** (no GPQA letter-vs-option statement, no choice-ordering, no shots) | — |
| thinking mode / reasoning effort | **nothing** (the shipped chat template defaults to thinking ON, `Reasoning Effort: Max`) | — |

---

## 2. Per-benchmark: card protocol vs our greedy protocol vs what card mode runs

### aime_2026 (card value 99.2) — also the template for HMMT / IMOAnswerBench when added

| | greedy (`--protocol greedy`, default) | HF card | our card mode (`--protocol card`) |
|---|---|---|---|
| sampling | temperature=0.0 (greedy), top_p=1.0, no seed | temperature=1.0, top_p=0.95 | **card's**: temperature=1.0, top_p=0.95, per-sample seed `base_seed+s` (seeds are OURS, recorded — the card publishes none) |
| max gen | `--max-new` (default 2048) | 163,840 tokens | **163,840**, min-ed with the context-window room and any explicit `--max-new` pod-capacity cap; every binding cap is visible (`n_truncated`, `finish_reason='length'`) |
| system prompt | none (template's own `Reasoning Effort: Max` block only) | the `Explanation:/Exact Answer:/Confidence:` prompt | **the card's, byte-derived from the committed README** (`CARD_REASONING_SYSTEM_PROMPT`; a CPU test pins it against `reference/hf-repo/README.md`) |
| user turn | problem text + `"Reason step by step... \\boxed{}"` hint | (unspecified beyond the system prompt) | **the bare problem text, verbatim** — nothing appended (we do not invent) |
| extraction | last `\boxed{}` → final number | GPT-5.5 (medium) judge | **SUBSTITUTE, labeled**: the `Exact Answer:` field (`extract_exact_answer`; last marker wins, cut at `Confidence:`, `\boxed{}` unwrapped) → fallback `\boxed{}`/final-number AFTER dropping `Confidence:` lines. Exact-match scoring (`score_math`), NOT a semantic judge |
| samples | 1 | unspecified | `--samples N` = avg@N over independent seeds — **a harness-side variance knob, not a card spec**; every sample is its own provenance row (`item_id#sN`, per-row seed) |

### gpqa_diamond (card value 91.2)

The card specifies **NO GPQA-specific protocol**: only the generic sampling params + generation
cap above. It says nothing about the MC prompt format (letter vs full option), choice ordering,
answer extraction, or a judge — and the Exact-Answer system prompt is explicitly scoped to
AIME/HMMT/IMOAnswerBench only.

| | greedy | HF card | our card mode |
|---|---|---|---|
| sampling | temperature=0.0 | temperature=1.0, top_p=0.95 | **card's** (+ recorded seed) |
| max gen | `--max-new` | 163,840 | **163,840** (min-ed as above) |
| prompt | question + content-hash-shuffled A–D choices + `\boxed{}` hint | **unspecified** | **greedy's, unchanged** — a HARNESS choice, not a card reproduction (documented; changing it would be inventing a protocol) |
| extraction / scoring | `extract_mc_letter` → exact letter match | **unspecified** | **greedy's, unchanged** |

### mmlu_pro, gsm8k — not on the GLM-5.2 card

No card protocol exists; `--protocol card` **refuses** them (`load_items` raises; the CLI
errors before starting a run). They stay greedy-only sanity gates.

### hmmt_nov_2025 / hmmt_feb_2026 / imo_answer_bench — not yet in the registry

When added, they take EXACTLY the aime_2026 `CardProtocol` (same footnote scope: same sampling,
same system prompt, same judge-substitute), plus their own pinned dataset revisions.

---

## 3. What we CAN reproduce vs what we CANNOT

**Reproduced faithfully (card mode):**
- `temperature=1.0`, `top_p=0.95` (per request, recorded per run + per benchmark).
- The 163,840-token max generation length (subject to the pod's `--max-len` window — see caveats).
- The AIME/HMMT/IMOAnswerBench system prompt, byte-identical to the card (test-pinned).
- The bare-question user turn for AIME (nothing added beyond what the card specifies).
- The `Exact Answer:` response format and its extraction.

**Cannot reproduce — honest substitutes, labeled in provenance (`env_json.card_protocols[*].substitutes`):**
- **The GPT-5.5 (medium) judge.** Substitute: exact-match (`score_math` normalization) on the
  extracted `Exact Answer:` field, with a fallback to `\boxed{}`/final-number extraction
  (Confidence lines dropped first) for replies that ignore the format. This is string matching
  after the card's own answer-format prompt — NOT a semantic judge. For AIME (integer answers
  0–999) the gap should be small; it is still a gap and is labeled as such.
- **Their serving stack.** The card's numbers come from GLM's own (GPU, FP8-native) serving;
  ours is the TPU v4 port (FP8 kept resident, per-tile dequant). Quality parity is the point of
  the whole exercise — but the harness cannot make the substrate identical.

**Unspecified by the card — our choices, recorded, NOT fidelity claims:**
- Number of samples per item (we default 1; `--samples N` = avg@N variance knob).
- Seeds (ours, `base_seed+s` per sample, recorded per item row).
- GPQA prompt format / choice ordering / extraction (we keep the greedy MC machinery).
- Thinking mode / reasoning effort (we use the shipped chat template defaults: thinking ON,
  `Reasoning Effort: Max`).
- System-message rendering: the shipped `chat_template.jinja` renders a passed system message
  as a SECOND `<|system|>` block after its own default block:
  `[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>{card prompt}<|user|>{question}<|assistant|><think>`.
  That is the model's own template convention (there is no way to replace the effort block
  without disabling thinking); recorded in provenance.

---

## 4. Comparability caveats — attach to ANY published Δ-vs-card

1. **Judge substitution:** card = GPT-5.5 (medium) judge; ours = exact-match on the card's
   `Exact Answer:` format. Directionally this can only under- or over-credit nonconforming
   replies; every raw reply is stored, so any disputed item is re-judgeable later.
2. **Sampling variance:** temperature=1.0 on n=30 (AIME) means ±1 item = ±3.33 pts; the card's
   99.2 on a 30-item set is itself not a multiple of 1/30 — implying some unpublished averaging.
   Quote a binomial CI or run `--samples N` (avg@N) and say so; a single-sample Δ of a few
   points is NOT evidence of a quality gap.
3. **Generation cap:** the card's 163,840 requires `max_model_len ≥ 163,840 + prompt`; any pod
   run with a smaller window deviates. The deviation is measurable: `n_truncated` > 0 in the
   summary (+ per-item `finish_reason='length'`) means the cap bound — report it alongside Δ.
4. **GPQA prompt is a harness choice** (card silent): our shuffled-letter MC + `\boxed{}` hint +
   letter extraction is not "the card's protocol", because none is published. Δ on GPQA compares
   scores under different (unknown vs ours) prompting.
5. **Unknown k / seeds / effort level** (see §3) — the card's exact run configuration is not
   fully reconstructable; card mode reproduces everything the card states and nothing more.
6. **Thinking-mode labeling:** our runs are thinking-ON at effort Max (template default);
   the card does not state its setting.

---

## 5. Implementation map (all CPU-tested, `--stub` green both protocols)

- `bench/benchmarks.py` — `CardProtocol` (per-spec `card=` field; quotes + substitutes stored),
  `CARD_FOOTNOTE_REASONING` / `CARD_REASONING_SYSTEM_PROMPT` (verbatim, test-pinned against the
  README), `_aime_card_build` (bare question + card system prompt), `card_math_extract`
  (judge-substitute chain), protocol-aware `load_items(..., protocol=)` (refuses benchmarks
  with no card protocol).
- `bench/extract.py` — `extract_exact_answer` (last `Exact Answer:` marker wins; cut at
  `Confidence:`; `\boxed{}` unwrapped; emphasis stripped; colon required — never guesses),
  `drop_confidence_lines` (fallback extraction can never grab the confidence percentage).
- `bench/run_bench.py` — `--protocol card|greedy` (default greedy, byte-identical: same
  prompts — DB-verified against pre-change stub runs — same `SamplingParams`, same note),
  `--samples N` (card-only avg@N; per-sample `#sN` rows), `--seed` (base seed; sample s = seed+s,
  recorded per row); per-request sampling override plumbing (`_sp(room, sampling=, seed=)`),
  system-prompt-aware chat templating; full protocol provenance in `runs.env_json`
  (`protocol`, `samples`, `base_seed`, `card_protocols` = params + verbatim system prompt +
  card quote + labeled substitutes per benchmark).
- `bench/test_bench.py` — exact-answer extractor tests (incl. adversarial: prose without colon,
  `inexact`, empty field, marker inside think block, confidence-percentage trap), README
  byte-pin of the system prompt, card-spec assertions, card-mode run tests (sampling/system
  prompts/seeds reach the generator; `#sN` provenance rows; avg@N; greedy refuses samples>1).

## 6. Pod commands (card mode)

```bash
# AIME 2026, all 30 items, card protocol (single sample; add --samples 4 for avg@4):
~/vllm-env/bin/python -u run_bench.py --benchmark aime_2026 --protocol card \
    --max-len 65536 --max-seqs 4 --batch-size 0 --seed 0 \
    --note "card-protocol AIME 2026 n=30"

# GPQA-Diamond, all 198 items, card protocol:
~/vllm-env/bin/python -u run_bench.py --benchmark gpqa_diamond --protocol card \
    --max-len 32768 --max-seqs 8 --batch-size 0 --seed 0 \
    --note "card-protocol GPQA-Diamond n=198"
```

Notes: no `--max-new` = no CLI cap; the effective cap is min(163,840, window room), so
`--max-len` is the ONLY knob trading fidelity vs pod memory — the card-faithful value is
`--max-len 165888` (163,840 + prompt), which Stage-1 KV sizing may not allow; whatever is used,
a binding cap shows up as `n_truncated` in the summary and must be reported with the Δ (§4.3).
Dry-run first: append `--stub --limit 3` (validated on this VM).
