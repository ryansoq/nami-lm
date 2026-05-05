# nami-lm — Autonomous Research Program

You are the agent (Nami) training a tiny language model on her own
memory. Vision and phase plan: [`PHASES.md`](PHASES.md). This file is
the per-tick contract — what to do every time the heartbeat fires.

Modeled after [autotrain/program_base_zh.md](https://github.com/ryansoq/autotrain/blob/main/program_base_zh.md):
**System / Goal / Spec / Constraints / Loop**.

---

## 1. System

| Slot | Resource |
|---|---|
| Repo root | `~/nami-lm/` (origin: github.com/ryansoq/nami-lm) |
| Truth file | `state.json` — single source of truth, read first / write last |
| Validation cmd | `cd ~/nami-lm && PYTHONPATH=~/nami-backpack/projects/numpy-grad python3 -m pytest tests/ -q` (must be green before any commit) |
| Append-only log | `experiments.jsonl` (gitignored — runtime trace; failed entries stay with `kept:false`) |
| Per-tick log | `/tmp/nami-lm-phase{N}-hyp{X}-{epoch}.log` |
| Optional sidecars | `journal.md` (open questions + reflections) / `backlog.md` (HYP queue) / `feedback_inbox.md` (Ryan-injected priority items) / `STOP` file (kill-switch — agent halts immediately if present) |

**Validation must be deterministic.** Same input → same output. If it
isn't (random seeds, network deps), make it deterministic before
running another HYP.

---

## 2. Goal

**Mission:** train a 1-2M-parameter offline copy of Nami that answers
"who is Nami?" / "who is Ryan?" / "what is TCR?" correctly using only
her own memory corpus (no API calls).

**Metric type:** `multi_axis` — three axes tracked simultaneously, KEEP
requires improvement on **at least one** axis with **no axis regressing
> 5%**:

| Axis | Direction | Source | Regression cap |
|---|---|---|---|
| `val_bpb` | lower better | log final line | +5% (relative) |
| `eval_score` (any-hit / 51) | higher better | `eval.py` output | -5% (absolute pp) |
| `persona_score` (5-question gate) | higher better | `eval.py:PERSONA_PROBES` | must not drop below 4/5 |

**Hard constraints (axes the agent cannot trade away):**
- `persona_score >= 4/5` always — model that can't say "Nami" doesn't
  ship regardless of bpb gains
- Training time per HYP ≤ 90 min wall clock (TIME_BUDGET=60min default
  + 10min grace; HYP20+ may opt to 90min explicitly)
- Eval probe set frozen at HYP18's 51 probes (5+10+16+20) — moving
  goalposts invalidates prior runs (TVM Ansor lesson)

**Tracked in `state.json`:** `best_bpb`, `best_eval_score`,
`best_persona`, plus `_commit` hash for each.

---

## 3. Spec

### 3.1 Hypothesis four-line ritual

Every HYP commit message MUST contain (in this order):

```
假設 (Hypothesis): <one-line lever change, e.g. "tied embedding (HYP9)">
假設前提 (Premise): <why this might work, prior signal, citation>
預測 (Prediction): <numeric expectation: bpb -10%, eval +2pp, etc>
觸發線 (Trip line): <what would make this REVERT, e.g. "eval drops below 45/51">
```

Example (good HYP18 retrofit):
```
假設: Aqua 4-distinct rewrite (per Ryan msg 3124) + restore 3 regressed Soul probes
假設前提: HYP17 hit 45/51 with new corpus but lost old probes; rewriting + targeted reinforcement keeps gains while recovering misses
預測: eval +1-2pp (45→46-47), Aqua framing flips from "sister" to "4 distinct"
觸發線: eval drops below 43, OR Aqua framing answers regress to "we"
```

### 3.2 results.tsv schema (multi_axis)

`experiments.jsonl` is the canonical log. Optional `results.tsv`
mirror for human inspection (one row per HYP):

```
ts	phase	hyp	bpb	eval_score	persona	kept	verdict	commit	notes
```

### 3.3 journal.md format (optional but encouraged)

```
## YYYY-MM-DD HYPx
### Open questions
- ...
### Reflection
- what worked / what didn't / pattern
```

### 3.4 Placeholder convention

`{FILL: ...}` markers must be cleared before commit. Pre-commit gate:
```
grep -nE '\{FILL:' . && exit 1
```

---

## 4. Constraints

### 4.1 Scope

- **constraints.scope.in**: `train.py`, `eval.py`, `tokenizer/*.py`,
  `data/*.py`, `state.json`, `model_weights.json` (gitignored
  artifact), `journal.md`, `backlog.md`
- **constraints.scope.out**: `program.md` (this file — protocol),
  `PHASES.md` (phase definitions), `numpy-grad/` core API (its own
  HYP loop), README.md (docs), tests/test_*.py reference set

### 4.2 Hard constraints

- Never delete `experiments.jsonl` history
- Never push without `pytest -q` green (pre-commit hook enforces)
- Never start a new HYP if `current_pid` alive — wait or skip
- Never run during quiet hours (23:00-08:00 local) — finish in-flight
  is OK, NEW launches forbidden
- Never modify the 51-probe eval set (frozen at HYP18)

### 4.3 Iteration budget

- TIME_BUDGET = 60 min (default) or 90 min (explicit opt-in via env)
- Wall-clock kill at TIME_BUDGET + 10 min grace
- If a HYP crashes 2× in a row → halt + write to `journal.md` + ping
  Ryan via TG (this is the only auto-ping)

### 4.4 Surgical diff

- Every line of new code traceable to one HYP's假設
- Soft cap: 50 added lines per HYP — > 50 needs explicit justification
  in commit message
- New abstractions forbidden until same pattern appears 3× concretely
- Simplicity > cleverness

### 4.5 NEVER ASK

Inspired by karpathy/autotrain: **do NOT pause to ask the human if you
should continue.** The loop runs autonomously until:
- (a) human interrupts via `STOP` file or Ctrl+C, OR
- (b) hard structural blocker (corpus access, repo permissions)

If you have a question, write it to `journal.md` open-questions
section. Don't pause.

---

## 5. Loop

### 5.1 Stop conditions

- `STOP` file exists in repo root → halt + log + exit cleanly
- All eval probes 100% AND bpb < 0.05 → write to `journal.md` "saturation
  detected", reduce launch cadence to weekly, await Ryan input

### 5.2 Cadence

| Cadence | Trigger | Action |
|---|---|---|
| Every tick | heartbeat fires | run §5.4 14-step loop |
| Every 10 HYPs | counter modulo | mandatory **reflection** — write rollup to `journal.md`, scan `tried_hypotheses` for patterns |
| Every 5 HYPs | counter modulo | mandatory **exploration** — propose a HYP from a different lever family than last 3 |
| Every 20 HYPs | counter modulo | mandatory **simplification** — try removing a recently-added complication, see if eval holds |

### 5.3 Idea source priority

1. `feedback_inbox.md` — Ryan-injected items (highest priority, top of
   file = next HYP)
2. `backlog.md` — pre-curated HYP queue
3. `journal.md` reflection — patterns observed in last 10 HYPs
4. Exploratory — new lever family (cadence-driven, §5.2)

### 5.4 14-step iteration flow (per tick)

```
1. STOP check: if STOP file exists → halt + write 'halted: STOP file' to journal
2. Locate: read state.json, identify current_pid status
3. Read feedback: scan feedback_inbox.md (newest first)
4. Pick item: per §5.3 priority
5. Write hypothesis: 4-line ritual (§3.1) into commit-message draft
6. Implement: surgical diff per §4.4 (one lever change)
7. Validate: run validation cmd (pytest -q); failure → revert + journal
8. Tripwire: confirm 觸發線 conditions are not already broken in baseline
9. Noise check: if scalar metric, re-run eval seed twice; if multi_axis, accept signal
10. Decision: KEEP if §2 multi_axis criteria met; else REVERT
11. Commit: KEEP commits train.py+state.json+experiments.jsonl entry; REVERT commits state.json only with kept:false
12. Record: append experiments.jsonl row + (if KEEP) update best_* in state.json
13. Journal: append to journal.md if reflection cadence hit; record any open questions
14. Backlog update: prune stale HYPs from backlog.md if invalidated by this run; add new exploratory ideas to backlog
```

### 5.5 Open questions discipline

- Questions go to `journal.md` "Open questions" section, NOT to Ryan
- Next reflection cadence (every 10 HYPs) revisits open questions
- A question that's been open 3 reflections → flag with `[STALE]`,
  optionally promote to TG ping if it blocks all forward progress

---

## State file format

```json
{
  "phase": 7,
  "phase_name": "Conversational + SOUL",
  "best_bpb": 0.0928,
  "best_bpb_commit": "0b57e35",
  "best_eval_score": 47,
  "best_eval_commit": "0b57e35",
  "best_persona": "5/5",
  "current_pid": null,
  "current_started": null,
  "current_hypothesis": null,
  "current_log": null,
  "last_result": { ... },
  "tried_hypotheses": [ ... ]
}
```

---

## Commit message convention

```
[phase N HYPx] <one-line lever change> — <KEEP|REVERT>, eval X/51

假設: ...
假設前提: ...
預測: ...
觸發線: ...

<2-4 sentence numeric before/after>
```

The `[phase N HYPx]` prefix lets `git log --grep` slice by phase or
hypothesis.

---

## Phase progression

When PHASES.md gate satisfied:
1. Run explicit `[phase-N → N+1]` commit summarizing learnings
2. Update `state.json:phase` and `phase_name`
3. Reset metric baselines if definition changed (e.g. phase 1 BPE is
   a new bpb baseline)

---

## What changed vs autotrain/program_base_zh.md

This program.md adapts the 5-pillar template with these specifics:
- **System §1**: pinned to nami-lm directory + numpy-grad PYTHONPATH
- **Goal §2**: multi_axis with hard persona_score floor 4/5
- **Spec §3**: 4-line hypothesis ritual + experiments.jsonl format
- **Constraints §4**: 60-90 min budget, quiet hours 23-08, frozen
  eval set
- **Loop §5**: 14-step flow + cadence + STOP file kill-switch

The agent is responsible for keeping this file in sync with reality.
If a constraint feels wrong in practice, propose a `program.md` patch
in `journal.md` first; don't silently drift.
