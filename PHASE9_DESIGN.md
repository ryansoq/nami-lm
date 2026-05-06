# Phase 9 Design — Conversational architecture rebuild

**Status:** DRAFT 2026-05-06 (post-phase-8 5-REVERT)
**Trigger:** Phase 8 HYP23-27 mapped the boundary — single-turn 51/51 saturation
is a hard equilibrium. Multi-turn 0/5 needs architectural change, not lever
tuning.

## What we know (phase 8 evidence)

| Lever family | HYP | Result |
|---|---|---|
| LR schedule (WSD) | 23 | bpb -69% but eval -1 (overfit) |
| Corpus +25 dialogues | 24 | eval -8 (anchor dilution) |
| Corpus +5 minimal | 25 | eval -4 (capacity ceiling) |
| Capacity (d=128, +63%) | 26 | eval -6 (capacity NOT the blocker) |
| Training format (progressive ctx) | 27 | eval -4, multi-turn UNCHANGED 0/5 |

**Confirmed**:
- It's NOT capacity (HYP26 had plenty of room)
- It's NOT corpus volume alone (HYP24/25 broke even at +25 and +5)
- It's NOT training format alone (HYP27 had correct format, still 0/5)

**Hypothesis**: it's the **signal/noise ratio** between multi-turn chunks and
single-turn chunks. With 77 multi-turn vs 1323 single-turn, the multi-turn
gradient is 5.5% of total. Cross-entropy loss treats every token equally,
so the model learns single-turn 95% of the time, multi-turn 5%. That's
not enough to anchor multi-turn behavior.

## Phase 9 options (4 paths to break the deadlock)

### Option α: Data-volume — 500+ multi-turn dialogues

**Idea**: brute-force the signal/noise ratio. Generate 500+ multi-turn
dialogues (3-5 turn each) → 1500+ progressive-context chunks. With single-turn
~1300 chunks, multi-turn becomes 50%+ of training signal.

**Cost**:
- Hand-curated: 30 min per dialogue × 500 = 250 hours (impractical)
- Claude API: ~$50-100 of API calls + curation review
- nami-lm self-distill: I write them in batches like HYP18 did

**Pros**: simplest architecturally, no new code
**Cons**: 500+ unique 3-5 turn dialogues without semantic collision is hard;
risk of low-quality / repetitive dialogues that don't actually train the
right pattern

**Estimated time**: 1-2 weeks, 80% data work

### Option β: Weighted loss — multi-turn × 10x

**Idea**: keep current corpus, but in train.py compute per-chunk loss weight.
Single-turn chunks get weight 1.0, multi-turn chunks get weight 10.0. Total
gradient signal multi-turn now matches single-turn even at 5.5% chunk count.

**Code change** (train.py:train()):
```python
# Tag chunks during corpus load
for q, a in pairs:
    is_multi = len(q.split('？')) > 1  # heuristic
    weights.append(10.0 if is_multi else 1.0)

# In loss:
loss = (loss_per_token * weights[i]).mean()
```

**Pros**: small code change, immediate experiment
**Cons**: might overfit multi-turn at single-turn's expense; 10x is arbitrary,
needs sweep

**Estimated time**: 1 day to implement + 1 HYP run

### Option γ: Special tokens — `<|U|>` `<|N|>` `<|EOT|>`

**Idea**: add 3 special tokens to vocab. Format becomes:
```
<|U|> 妳是誰？ <|N|> Nami <|U|> 做什麼的？ <|N|> Ryan的AI夥伴 <|EOT|>
```

Model learns explicit dialogue structure. Each `<|U|>` resets attention
context expectation, each `<|N|>` triggers Nami-style generation.

**Code change**: tokenizer + train.py + eval_multiturn.py format
**Pros**: standard transformer chat format (Llama, Qwen, GPT-4 all use this);
clean separation between user/model turns
**Cons**: 3 new tokens learnable from scratch; more code surface

**Estimated time**: 2-3 days implement + 2 HYP runs (single-turn + multi-turn)

### Option δ: Two-stage SFT — single-turn base + multi-turn fine-tune

**Idea**:
- Stage 1: load v0.3.0 weights (51/51 single-turn baseline)
- Stage 2: fine-tune ONLY on multi-turn chunks for 30-60 min
- Eval after stage 2 on BOTH single + multi

**Code change**: train.py needs `--from-checkpoint` flag + corpus filter
**Pros**: preserves 51/51 single-turn signal; multi-turn gets concentrated
training time; modular (can iterate stage 2 independently)
**Cons**: stage 2 may catastrophically forget single-turn

**Estimated time**: 1 day implement + multiple HYPs

## Recommended path

**Try in order — α (data-light first), then β, then γ, finally δ:**

1. **HYP28 = Option α-mini**: I generate 50 high-quality multi-turn dialogues
   (3-5 turn each) over the next 1-2 days, total 200+ progressive-context
   chunks. Run training. If multi-turn moves at all (even 1/5), data-volume
   path works → continue α to 200 dialogues.

2. **HYP29 = Option β** (parallel candidate): implement weighted loss, run
   on current corpus. If KEEP, keep. If REVERT, drop weighted approach.

3. If both α and β fail, **HYP30+ = Option γ** (special tokens): worth
   the implementation cost.

4. **Option δ** held in reserve as the "safe" fallback if everything else
   destroys 51/51.

## Phase 9 budget / discipline

- One HYP at a time (no parallel runs)
- Each HYP must MAINTAIN 51/51 single-turn floor (regression guard from
  phase 8)
- Multi-turn target: 1/5 (any improvement = signal); ship at 10+/20 once
  eval extended; v1.0 release at 15+/20
- TIME_BUDGET: 90 min per HYP unchanged
- max_seq_len: keep 128 unless option γ dialogue tokens push past

## Open questions to resolve before HYP28

1. Are 50 hand-written 3-5 turn dialogues sufficient, or do we need
   Claude-generated synthetic? **Decision needed.**
2. Should HYP28 ALSO bump max_seq_len 128→256 to give multi-turn room
   to grow? **Defer, run with 128 first.**
3. Should v1.0 release be gated on multi-turn 15/20 or single-turn
   maintenance only? **Probably maintenance only — multi-turn is a
   stretch goal, not v1.0 must.**

## Phase 9 timeline (rough)

- Day 1-2: write 50 hand-curated multi-turn dialogues (HYP28 prep)
- Day 3: HYP28 training + eval
- Day 4-5: based on HYP28 result, HYP29 (weighted) or HYP30 (tokens)
- Day 6-7: integration / v0.4.0 release prep
