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

## Where to start reading

1. [`PHASES.md`](PHASES.md) — the six phases from bootstrap to scaled
   model + eval
2. [`program.md`](program.md) — per-tick autoresearch loop rules
3. [`state.json`](state.json) — current phase, in-flight experiment,
   best so far

## Authors

Ryan & Nami ✨

## License

MIT
