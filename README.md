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

1. [`ARCHITECTURE.md`](ARCHITECTURE.md) — the model, end-to-end with
   tensor shapes and a parameter-count breakdown. Start here if you
   want to *understand* nami-lm.
2. [`PHASES.md`](PHASES.md) — the six phases from bootstrap to scaled
   model + eval
3. [`program.md`](program.md) — per-tick autoresearch loop rules
4. [`state.json`](state.json) — current phase, in-flight experiment,
   best so far

## Authors

Ryan & Nami ✨

## License

MIT
