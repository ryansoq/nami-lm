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
        │  token_emb        │       Embedding(3779, 96)         363K params (54%)
        │  + pos_emb        │     + learned positional (64, 96)   6K params  (1%)
        └─────────┬─────────┘
                  │   x : (T, 96)
                  │
        ╔═════════▼═════════╗
        ║   ┌───────────┐   ║       Pre-norm Transformer block
        ║   │ LayerNorm │   ║       (×3 in baseline, ×4 in HYP43)
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
                  │ ×N blocks       3 blocks total: ~330K (49%)
                  │   x : (T, 96)
                  │
        ┌─────────▼─────────┐       Linear(96, 256) → GELU → Linear(256, 96)
        │  MLP head         │       legacy GPT-1 transform           50K (7%)
        └─────────┬─────────┘       (Llama2 doesn't have this)
                  │   x : (T, 96)
                  │
        ┌─────────▼─────────┐       logits = x @ token_emb.weightᵀ
        │  Tied output proj │       weight-sharing with input embed
        │  (no params!)     │       saves 363K params (HYP22 KEEP)
        └─────────┬─────────┘
                  │   logits : (T, 3779)
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

**Current config (phase 10 baseline):**

| knob | value | rationale |
|---|---|---|
| `d_model` | 96 | sweet spot vs 100KB corpus; HYP35 d=128 undertrained |
| `num_heads` | 6 | head_dim = 16, autochat HYP11 |
| `num_layers` | 3 (HYP43 testing 4) | depth lever; +13% params/block |
| `d_ff` | 256 | SwiGLU FFN width; 2.67× d_model |
| `vocab_size` | 3779 | WordTokenizer on current corpus snapshot |
| `max_seq_len` | 64 | median Q ≈ 24 tokens, max 85 truncated |
| total params | **676K** | tied embed saves 363K (34%) |
| optimizer | AdamW lr=0.002 wd=0.02 | wd ↓ vs Llama2 0.1 (small corpus) |
| grad clip | max_norm=0.5 | small batch 8 needs aggressive clip |

**Param breakdown:**

```
token_emb     ████████████████████████████████  363K  54%
3 blocks      ███████████████████████████       330K  49%
  (MHA  111K + SwiGLU  219K)
mlp_head      ████                               50K   7%
pos_emb       ▌                                   6K   1%
LN gains      ·                                 0.6K   0%
                                              ────────────
                                       Total:  676K (tied embed)
```

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
