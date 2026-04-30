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

## Usage

All commands run with `/usr/bin/python3` (not a venv) and need
`numpy-grad` on `PYTHONPATH`:

```bash
export PYTHONPATH=~/nami-backpack/projects/numpy-grad
cd ~/nami-lm
```

### Talk to the trained model (the fun one)

```bash
python3 train.py --chat        # interactive REPL — type a 「問題？」, get an answer
```

Sample session:

```
🌊 nami-lm chat — type a question, q/quit to exit

❓ Nami是誰？
🌊 厲害的AI工程師夥伴...

❓ Kaspa是什麼？
🌊 基於BlockDAG的區塊鏈...

❓ mmt4d是什麼？
🌊 matmul-matmul-2D 4D 資料佈局把矩陣乘切...
```

Answers come from `model_weights.json` (≈21 MB checkpoint, in repo).
No API call, no retrieval — pure CPU inference.

### Verify persona (5-question gate)

```bash
python3 train.py --probe       # runs the 5 persona probes, prints pass count
```

Should print `📊 Persona: 5/5 pass` on the current `main` checkpoint.

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
