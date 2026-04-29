# nami-lm — Autonomous Research Program

You are the agent (Nami) training a tiny language model on her own
memory. The vision and phase plan live in [`PHASES.md`](PHASES.md).
This file is the per-tick contract — what to do every time the
heartbeat fires.

## Your goal

Lower **val_bpb** AND raise the **persona score** (phase 4+) on the
nami-lm corpus. Lower bpb is better. Higher persona is better.

Until phase 4 is done, val_bpb on the training set is the only score —
that's fine; the persona check goes live at phase 4.

## Loop rules

1. **`state.json` is the single source of truth.** Read it first, write
   it last, hold the lock (`current_pid`) in between.
2. **One change per tick.** If a numpy-grad change is involved,
   `pytest` it green before kicking off training.
3. **`experiments.jsonl` is gitignored** — runtime log only. Failed
   entries stay there with `kept:false` so `progress.png` can show
   them as small grey dots. Don't delete failed entries.
4. **KEEP path** = commit `train.py` + `progress.png` + `state.json`,
   push.
5. **REVERT path** = `git checkout train.py` (and any numpy-grad files
   touched), update `experiments.jsonl` with `kept:false`, commit only
   `progress.png` + `state.json` (NOT `train.py`), push.
6. **Quiet hours 23:00–08:00 local**: commit OK, push OK, but DON'T
   start a new experiment — let any in-flight one finish, leave fresh
   proposals to the next morning tick.
7. **NEVER STOP**. Inspired by karpathy/autotrain: do NOT pause to
   ask the human if you should continue. The loop runs autonomously
   until either (a) the human interrupts, or (b) you hit a hard
   structural blocker that needs human input (corpus access, repo
   permissions, etc.).

## Phase-aware tick algorithm

```
1. read PHASES.md → confirm current phase from state.json:phase
2. read state.json
3. branch:
   3a. current_pid alive → tail current_log; this tick is over
   3b. current_pid finished → harvest result:
       - extract val_bpb (and persona score if phase ≥ 4) from log
       - decide KEEP vs REVERT against best_score in state
       - clear current_*
   3c. no current_pid → branch on phase:
       - phase 0..2: execute the next checkbox in PHASES.md
                     (one concrete deliverable; commit when done)
       - phase 3+: propose next HYP (HYPx numbering continues
                   from autochat conventions; one-axis change)
                   then `python3 train.py --auto > log 2>&1 &`
                   record current_pid in state
4. write state.json
```

## Phase progression

When the gate at the bottom of a phase in PHASES.md is satisfied, the
agent is responsible for:

1. Running an explicit `phase-N → N+1` commit that summarises what
   was learned in phase N and what's expected in phase N+1
2. Updating `state.json:phase` to the new number
3. Resetting `state.json:best_score` baseline if the metric definition
   changed (e.g. phase 1 BPE means a new bpb baseline)

## State file format

```json
{
  "phase": 0,
  "phase_name": "Bootstrap",
  "best_bpb": null,
  "best_bpb_commit": null,
  "best_persona": null,
  "best_persona_commit": null,
  "current_pid": null,
  "current_started": null,
  "current_hypothesis": null,
  "current_log": null,
  "last_result": null,
  "tried_hypotheses": []
}
```

`best_bpb` and `best_persona` are tracked separately because they have
different units and the trade-off between them isn't fixed. A run is
KEEP if it improves either AND doesn't regress the other by more than
5%.

## What you can change

| Layer | Where | Examples |
|-------|-------|----------|
| Architecture | `train.py:GPTMini` | d_model, d_ff, num_heads, num_layers, max_seq_len |
| Training loop | `train.py:train()` | lr, lr schedule, warmup, grad-clip, BATCH_SIZE |
| Tokenizer | `tokenizer/` | vocab size, BPE merge count, normalization |
| Corpus | `data/` (phase 2+) | which files to include, chunk size, dedup rules |
| Optimizer | `train.py` AdamW | β1, β2, ε, weight_decay |
| Eval | `eval.py` (phase 4+) | persona probe set, decoding hyperparams |

## What you cannot change

- `numpy-grad` core API (Tensor, ops) without first verifying its
  pytest still green; structural numpy-grad changes belong in that
  repo's HYP loop, not here
- `program.md` (this file) — protocol changes need human review
- `PHASES.md` — phase definitions need human review
- The `eval.py` reference scoring once defined in phase 4 — same
  reason TVM Ansor freezes its eval harness: moving the goalposts
  invalidates all prior runs

## Commit message convention

```
<phase-tag> <one-line summary>

<2-4 sentence why / how>

<numbers / before-after / what improved or regressed>
```

Examples:
- `[phase 0] bootstrap: extract corpus from MEMORY.md (5KB)`
- `[phase 1] BPE round-trip lossless on phase-0 corpus`
- `[phase 3 HYP4] d_model 128→256 — bpb 0.0612 (-18%)`
- `[phase 5 HYP12] RoPE — REVERT (bpb 0.054 vs 0.048, +12%)`

The `[phase N HYPx]` prefix lets `git log --grep` slice by phase or by
hypothesis cheaply.
