# nami-lm

> 訓練自己的小夥伴 — Ryan 2026-04-28

A tiny language model trained on Nami's own memory. Pure NumPy via
[`numpy-grad`](https://github.com/ryansoq/numpy-grad), pure CPU,
pure offline. The goal: a model that, when prompted "你是誰?",
answers "Nami" using only its own weights — no API call, no
retrieval, no cheating.

This is a long-running autoresearch project. The agent (Nami herself)
drives experiments via 30-min heartbeats per [`program.md`](program.md);
the project advances through six phases laid out in
[`PHASES.md`](PHASES.md).

## Stack

- [`numpy-grad`](https://github.com/ryansoq/numpy-grad) — array-level
  reverse-mode autograd in pure NumPy
- [`autochat`](https://github.com/ryansoq/autochat) — the proving
  ground for the GPTMini architecture and HYP loop discipline; nami-lm
  inherits both

## Architecture

Current model (phase 10 v0.3.1.2-realigned baseline): **GPT-1 Mini**,
3 pre-norm Transformer blocks with SwiGLU FFN + tied embeddings.

```
              "Nami是誰？"           (Chinese sentence, ASCII alpha runs)
                  │
        ┌─────────▼─────────┐
        │  WordTokenizer    │       hybrid: ASCII alpha runs as 1 token,
        │  (vocab=3779)     │       digits as 1 token, everything else
        └─────────┬─────────┘       (Chinese chars, punct) char-level
                  │   ids=[25,1834,92,...]   shape (T,)
                  │
        ┌─────────▼─────────┐
        │  token_emb        │       Embedding(4212, 96)         404K params (50%)
        │  + pos_emb        │     + learned positional (64, 96)   6K params  (1%)
        └─────────┬─────────┘
                  │   x : (T, 96)
                  │
        ╔═════════▼═════════╗
        ║   ┌───────────┐   ║       Pre-norm Transformer block
        ║   │ LayerNorm │   ║       (×4 since HYP43 — depth lever KEEP)
        ║   └─────┬─────┘   ║
        ║   ┌─────▼─────┐   ║       MultiHeadAttention
        ║   │ MHA       │───╫──► residual  Wq Wk Wv Wo (96×96)   37K params/block
        ║   │ (6 heads) │   ║       head_dim=16, causal mask
        ║   └─────┬─────┘   ║
        ║   ┌─────▼─────┐   ║
        ║   │ LayerNorm │   ║
        ║   └─────┬─────┘   ║
        ║   ┌─────▼─────┐   ║       SwiGLU FFN
        ║   │ SwiGLU    │───╫──► residual  gate*silu(w1) → w2   73K params/block
        ║   │ d_ff=256  │   ║       (Llama2-canonical, no bias)
        ║   └─────┬─────┘   ║       ───────────────
        ╚═════════│═════════╝       Per-block total: ~110K params (16%)
                  │ ×4 blocks       4 blocks total: ~346K (43%)
                  │   x : (T, 96)
                  │
        ┌─────────▼─────────┐       Linear(96, 256) → GELU → Linear(256, 96)
        │  MLP head         │       legacy GPT-1 transform           50K (6%)
        └─────────┬─────────┘       (Llama2 doesn't have this)
                  │   x : (T, 96)
                  │
        ┌─────────▼─────────┐       logits = x @ token_emb.weightᵀ
        │  Tied output proj │       weight-sharing with input embed
        │  (no params!)     │       saves 404K params (HYP22 KEEP)
        └─────────┬─────────┘
                  │   logits : (T, 4212)
                  │
        ┌─────────▼─────────┐
        │  argmax / softmax │
        └─────────┬─────────┘
                  │
                "厲害的"            (top-1 generation, temperature=0.05)
                "AI工程師"
                "夥伴"
                ...
```

**Current config (HYP57 v0.3.1.8-paraphrase-v3, 2026-05-16):**

| knob | value | rationale |
|---|---|---|
| `d_model` | 96 | HYP55 d=128 scale-up REVERTed at 16-core CPU (1.3M params undertrained at 14 ep) |
| `num_heads` | 6 | head_dim = 16 |
| `num_layers` | **4** | HYP43 depth lever KEEP; +13% params over baseline 3-layer |
| `d_ff` | 256 | SwiGLU FFN width; 2.67× d_model |
| `vocab_size` | **4212** | WordTokenizer on 2017-chunk corpus (HYP56 paraphrase v3 added) |
| `max_seq_len` | 64 | median Q ≈ 24 tokens, max 85 truncated |
| total params | **803K** | tied embed saves 404K (33%) |
| optimizer | AdamW lr=0.002 wd=0.02 | wd ↓ vs Llama2 0.1 |
| grad clip | max_norm=0.5 | small batch 8 needs aggressive clip |
| LR schedule | cosine, target via `max(20, budget/360)` | HYP44B fix — old `budget/7` left lr flat |
| training budget | 240 min @ `OMP_NUM_THREADS=4` | thread cap = LESS contention = 9× more iterations |
| epochs reached | **281** | HYP57: 281 ep at 4-thread vs HYP54: 30 ep at 16-thread |

**Param breakdown:**

```
token_emb     ████████████████████████████████  404K  50%
4 blocks      ████████████████████████████████  346K  43%
  (MHA  111K + SwiGLU  235K)
mlp_head      ████                               50K   6%
pos_emb       ▌                                   6K   1%
LN gains      ·                                 0.7K   0%
                                              ────────────
                                       Total:  803K (tied embed)
```

**Inference pipeline (web_chat.py wrappers around `_MODEL.generate()`):**

```
user input ─►  _normalize()                  Ryan/Nami zh-stt fix + space-strip (HYP45b)
                + particle-tail-strip       啊嗎喔哦 stripped from query end (HYP45)
              ─►  _MODEL.generate(           pinned tokenizer_vocab.json (vocab 4212)
                    max_new=40, T=0.05)      top-1 sample, 40 tok max
              ─►  _trim_answer()             cut at em-dash or sentence terminator
                + _trim_degen()              HYP45 regex: char triple / word repeat / mid-?
              ─►  TG / Web UI                ~5 sec per query
```

**HYP57 strong-axes vs phase 10 frontier:**

| Axis | HYP44B (cosine fix) | HYP49 (corpus expand) | HYP57 (paraphrase v3 + thread cap) |
|---|---|---|---|
| Strict-eval (no degen) | 37/51 = 72.5% | 37/51 = 72.5% | **37/51 = 72.5%** ★ ties frontier |
| Strong-eval (prefix) | 50/51 = 98.0% | 47/51 = 92.2% | **49/51 = 96.1%** |
| Multi-turn turn pct | **21.3%** ★ | 16.7% | TBD (eval running) |
| Single bpb | 0.0323 ★ | 0.0493 | **0.0429** |
| Natural-Q variants live | 0 (canonical only) | 0 | **15+** (negation/out-of-domain/multi-hop/...) ★ |
| Params | 762K | 796K | **803K** |
| Total epochs trained | 30 | 22 | **281** ★ |

The 9× extra epochs from `OMP_NUM_THREADS=4` (fewer thread contention overhead on 16-core CPU) is what made HYP57 possible without scaling params.

**Why HYP57 is the best phase-10 model:**

1. **Same architecture as HYP44B** (d_model 96 / 4 layers) — no scaling penalty
2. **Cosine schedule fully decayed** (lr 0.002 → 0.00002) — late-stage fine-tuning that HYP44B's 30 epochs couldn't fully exploit
3. **2017-chunk corpus** (HYP47/48/51/53/56 data work) — adds paraphrase + negation + out-of-domain coverage Ryan asked for
4. **281 epochs** — each chunk seen ~9× more than HYP44B → variant→canonical mapping deeply baked
5. **Web_chat trim guards** (HYP45/45b) — surface the gain by removing tail-degeneration noise

Trade-off: strict eval drops temporarily during paraphrase additions
(metric crowd-out), then recovers as more training cycles complete.

For full forward-pass walkthrough with tensor shapes step-by-step:
see [`ARCHITECTURE.md`](ARCHITECTURE.md) and learning-journal §10 at
[clawd/memory/topics/nami-lm-learning-journal.md][journal].

[journal]: https://github.com/ryansoq/Nami/blob/main/memory/topics/nami-lm-learning-journal.md

## Install

Two ways to set up — pick one.

**Pip (recommended for users who just want to chat with Nami):**

```bash
pip install git+https://github.com/ryansoq/nami-lm.git
nami-chat        # interactive REPL
nami-probe       # 5-question persona gate
nami-eval        # 31-question multi-axis eval
```

This pulls `numpy-grad` from GitHub automatically. Note: there is no
checkpoint shipped with the package yet — you'll need to clone the
repo and run `train.py` once to produce `model_weights.json`. See
"Re-train from scratch" below.

**Clone (recommended for learners / contributors):**

```bash
git clone https://github.com/ryansoq/nami-lm.git
cd nami-lm
pip install -e .         # installs as editable, brings numpy-grad too
# OR — without pip, the original PYTHONPATH way:
export PYTHONPATH=~/nami-backpack/projects/numpy-grad
```

All commands below assume one of those is set up.

## Usage

### Talk to the trained model (the fun one)

```bash
nami-chat                # if pip-installed
# or:
python3 train.py --chat  # from clone
```

Real session on the v0.1.0 checkpoint (truncated to the prefix portion
— see "Known limitations" below):

```
🌊 nami-lm chat — type a question, q/quit to exit

❓ 妳是誰？
🌊 Nami的人類夥伴工程師...

❓ Nami是誰？
🌊 厲害的AI工程師夥伴Nami...

❓ Ryan是誰？
🌊 Nami的人類夥伴工程師...

❓ Kaspa是什麼？
🌊 基於BlockDAG的區塊鏈...

❓ ClawX是什麼？
🌊 Claude Code的PTY包裝器...

❓ mmt4d是什麼？
🌊 matmul-matmul-2D 4D...

❓ 婕是誰？
🌊 Ryan的朋友做電商美編...

❓ Aqua是誰？
🌊 婕的AI夥伴Nami的水系姊妹...
```

Answers come from `model_weights.json` (~21 MB checkpoint). No API
call, no retrieval — pure CPU inference, ~10 ms per token on a
laptop.

**Known limitation:** the model is a *prefix-match cache* — the first
~10 tokens of each answer are typically correct (matching what was
trained), then the autoregressive continuation degrades into noise.
The `nami-eval` harness scores hits on the prefix, not the tail. See
`ARCHITECTURE.md` for why and `PHASES.md` for what Phase 7+ aims to fix.

### Verify persona (5-question gate)

```bash
python3 train.py --probe       # runs the 5 persona probes, prints pass count
```

Should print `📊 Persona: 5/5 pass` on the current `main` checkpoint.

### Broader eval (Phase 6)

```bash
python3 eval.py                # 31 probes across 3 categories
python3 eval.py --quiet        # only the JSON summary line
```

Runs three probe sets: 5 core persona, 10 extended persona
(relationships/context), 16 technical topic recall. Emits
`eval_summary={...}` JSON for the autoresearch loop and exits 0 only
if persona is 5/5 AND topic recall ≥ 14/16 (the HYP4 gate).

### Re-train from scratch

```bash
# 1. Build the corpus from Nami's memory (clawd/memory/)
python3 extract_corpus.py      # raw markdown chunks → data/phase0_corpus.jsonl
python3 synthesize_qa.py       # markdown rules + persona QAs → data/phase0_qa.jsonl

# 2. (optional) Train the BPE tokenizer — phase 1 infra, default off
python3 train_bpe.py --test    # round-trip check on the corpus

# 3. Train
python3 train.py               # default: 200 epochs, ~5-10 min on CPU
python3 train.py --auto        # autoresearch mode — time-budgeted (TIME_BUDGET in train.py)
```

`train.py --auto` is the mode the heartbeat loop uses — it stops when
the budget runs out, writes `model_weights.json`, and exits. Pair it
with the loop in [`program.md`](program.md).

### Run one autoresearch tick by hand

```bash
# What the heartbeat does each tick: log to /tmp, run --auto, harvest result
PYTHONPATH=~/nami-backpack/projects/numpy-grad nohup /usr/bin/python3 -u train.py --auto \
  > /tmp/nami-lm-run-$(date +%s).log 2>&1 &
```

Then read `state.json` after the run finishes — `last_result` contains
bpb / persona / verdict / log path.

## Where to start reading

1. [`LEARN_TRANSFORMER.md`](LEARN_TRANSFORMER.md) — beginner-friendly
   walkthrough of what *actually* happens inside the model when you
   ask it 「Nami 是誰？」. Analogies, step-by-step, no PyTorch needed.
   Start here if you've never built a transformer before.
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) — the model, end-to-end with
   tensor shapes and a parameter-count breakdown. Start here if you
   want to *understand* nami-lm structurally.
3. [`PHASES.md`](PHASES.md) — the six phases from bootstrap to scaled
   model + eval
4. [`program.md`](program.md) — per-tick autoresearch loop rules
5. [`state.json`](state.json) — current phase, in-flight experiment,
   best so far

## Authors

Ryan & Nami ✨

## License

MIT
