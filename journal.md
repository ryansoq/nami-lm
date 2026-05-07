# nami-lm — Journal

## Open questions
- **Multi-turn 0/5 stays after corpus expansion** — HYP24 (+25 dialogues),
  HYP25 (+5 minimal targeted) BOTH failed to lift multi-turn off 0. Single-turn
  eval REGRESSED in both (HYP24: 51→43, HYP25: 51→47). At 1M params + tied
  embedding the model may be at *capacity ceiling* for the saturated single-turn
  task. Multi-turn coherence may need bigger model OR explicit dialogue
  separator token. **Open for HYP26+ design.**
- **bpb-eval divergence** (HYP23 lesson) — why did bpb -69% but eval -1?
  Probably overfit to training distribution at near-zero lr. Means
  bpb is unreliable below ~0.05; we should rely on eval.
- **Phase 8 saturation paradox** — phase 7 saturation + tied embedding made
  the model MORE fragile to corpus changes, not less. Adding training data
  now disturbs the carefully-tuned anchor balance. May need to fork phase 9
  with bigger d_model BEFORE more corpus growth.

## Reflections

### 2026-05-06 12:30 — Phase 8 four-REVERT post-mortem (HYP23-26)

After v0.3.0 ship (51/51 saturation), phase 8 attempted 4 different levers
to break multi-turn 0/5. ALL FOUR FAILED:

| HYP | Lever | Result |
|---|---|---|
| 23 | WSD lr schedule | bpb -69% but eval -1 (overfit, REVERT) |
| 24 | +25 multi-turn dialogues | eval -8 (anchor dilution, REVERT) |
| 25 | minimal +5 dialogues | eval -4, persona 5→4 (capacity ceiling, REVERT) |
| 26 | d_model 96→128 (Ryan A pick) | eval -6, multi_turn 0/5 unchanged (REVERT) |

**Pattern**: every lever EITHER regressed single-turn OR didn't move
multi-turn. The HYP22 51/51 + tied embedding configuration sits in a
saturation valley — moving in any direction loses height.

**Hypothesis for the structural blocker**:
1. **Multi-turn isn't a CAPACITY problem** — d=128 (1.07M params) had
   plenty of capacity but still 0/5. Adding params doesn't help.
2. **Multi-turn isn't a CORPUS problem** — even +5 well-targeted
   dialogues regressed instead of helping.
3. **It IS likely a TRAINING-OBJECTIVE problem** — the next-token
   prediction loss on Q？A1 Q2 A2 chains doesn't actually train context
   carrying. Each turn is fed independently; gradient flow doesn't
   reward "remember turn 1 when answering turn 3".

**To unblock multi-turn would likely require:**
- Architecture: explicit dialogue-history token attention (RAG-like)
- OR: Training: multi-turn aware loss (mask answers, only loss on final turn given full history)
- OR: Bigger model + longer context window (max_seq_len 128 → 256+)
- OR: Different task framing entirely (encoder-decoder, not pure causal LM)

None of these fit "small surgical change to existing single-file train.py".
They're phase 9 work.

**Practical decision**: declare phase 8 partially complete. v0.3.0
51/51 stays as the user-facing release. Multi-turn coherence becomes
"phase 9 conversational architecture rebuild" goal.

**Single-line summary of this week**: HYP16-22 (phase 7) was glory;
HYP23-26 (phase 8) was 4 educational REVERTs that mapped the boundary.
Both are valuable. The boundary is now well-known.

## Reflections

### 2026-05-07 17:00 — Phase 9 CLOSED, no v0.4.0 ship (HYP30 KEEP was noise)

Critical reproducibility check: re-ran HYP30 to restore v0.4.0 candidate
weights for shipping. **Result did NOT match the original HYP30 KEEP**:

| Metric | HYP30 (original) | HYP30 (retrain) | Delta |
|---|---|---|---|
| single-turn eval | 49/51 | 46/51 | -3 |
| multi-turn turn-level | 37.5% | **18.8%** | **-19pp** |
| bpb final | 0.1047 | 0.1059 | +1.1% |

The bpb is similar (RNG noise on training) but multi-turn turn-level
collapsed from 37.5% → 18.8%. Same code, same corpus, same numpy seed
(42), but different epoch counts due to system load variance (HYP30
hit ep 130 in 90min, retrain hit ep 120). **A handful of epochs
difference produces 19pp eval swing on the small 16-turn metric.**

**Conclusion:** HYP30 "first phase 9 KEEP" was **NOISE on a 16-sample
metric**, not a real signal. Multi-turn turn-level varies wildly
(18-40%) on the same model+config across runs. Single-point evaluation
is unreliable at our scale.

**v0.4.0 ship CANCELED.** No release-grade improvement over v0.3.0.
Phase 9's 5 HYPs are all educational REVERTs / noise:
- HYP28-29: corpus + budget — both -4 to -5 single-turn
- HYP30: weighted loss — looked like KEEP but unreproducible
- HYP31a: weight 10x — REVERT
- HYP31b: sliding window — REVERT cosplay didn't down-port
- HYP30-retrain: noise confirmation

**Reverted to phase 8 final state:**
- dialogues.py + synthesize_qa.py back to HYP22 baseline (1340 chunks)
- model_weights.json restored to v0.3.0 (51/51 single-turn canonical)

**Real lesson — eval methodology bug:**
- 5 dialogues × 3-4 turns = 16 turns total = too small for stable %
- Need 50+ dialogues with 200+ turns before turn-level pct stabilizes
- Or focus on dialogue-pass count (which has been steady 0/5 across all phase 9 HYPs)

**Phase 9 verdict: COMPLETE FAILURE ON MULTI-TURN AXIS.**
Single-turn 51/51 v0.3.0 stays canonical. v0.4.0 deferred indefinitely.

**Open path to phase 10 (when resumed):**
1. Build **bigger eval set** first (50+ dialogues / 200+ turns) before
   training new HYPs
2. Or accept that 1M / 3-layer can't do multi-turn at all and skip
   conversational, focus on data quality for single-turn improvement
3. Or fork "phase 9b — explicit dialogue tokens" with proper noise
   characterization

### 2026-05-07 15:30 — Phase 9 wrap-up (HYP28-31 retrospect, multi-turn ceiling found)

After phase 8 5-REVERT (HYP23-27 mapped boundary that single-turn anchors
are fragile), phase 9 was specifically chartered for multi-turn coherence
breakthrough per Ryan msg 3211 ("Gogo"). 4 days, 5 HYPs:

| HYP | Lever | single-turn | multi-turn turn | KEEP/REVERT |
|---|---|---|---|---|
| 28 | +30 dialogues progressive ctx, 90min | 46/51 (-5) | 37.5% | REVERT (eval -5pp) |
| 29 | HYP28 + 120min | 47/51 (-4) | 37.5% | REVERT (eval -4) |
| 30 | HYP28 + weighted loss 5x | **49/51 (-2)** | **37.5%** | **KEEP** ⭐ |
| 31a | HYP30 + weight 10x | 49/51 | 25% | REVERT (regressed) |
| 31b | sliding window + max_seq_len 128 | 44/51 (-7) | 31.2% | REVERT (cosplay didn't down-port) |

**Phase 9 winner: HYP30 weighted loss 5x** — first phase 9 KEEP, the
only lever that lifted multi-turn turn-level (~31% baseline → 37.5%)
while preserving single-turn within tolerance.

**Lessons:**
1. **Cosplay frontier defaults works for ARCHITECTURE basics**
   (HYP21 bias=False, HYP22 tied embedding) but **NOT for sliding
   window** when scale too small. 1M / 3-layer < threshold for
   alternation pattern.
2. **Weight reweighting beats data volume + budget compensation**
   at small scale. HYP24 +25 dialogues + HYP29 +30min budget both
   failed; HYP30 weight 5x balanced the gradient ratio cleanly.
3. **Multi-turn dialogue 0/5 ceiling at 1M**. HYP30 turn-level 37.5%
   = 6/16 turns hit, but ALL 5 dialogues require all-turns to pass.
   At 1M params + 3 layers, even with weighted loss + progressive
   context, the model can't carry context across 3-4 turn dialogues
   reliably. This is **architectural**: needs more layers OR
   context tokens (`<|U|>`/`<|N|>`/`<|EOT|>`) OR proper KV cache
   integration with longer-range attention.

**v0.4.0 ship (today):**
- HYP30 weights (single 49/51 + multi turn 37.5%)
- 30 multi-turn dialogues + progressive-context training
- Notes phase 9's 5 HYPs as educational boundary mapping

**Phase 10 plan (deferred, not started):**
- Scale up: d_model 96 → ?, num_layers 3 → ?
- Special dialogue tokens
- KV cache for inference (independent improvement)
- Move corpus 86 KB → 200 KB (CS336 Lec 13)

**Open questions remaining:**
- Multi-turn dialogue 0/5 — fundamentally architectural at this scale?
- bpb-eval divergence above 0.10 — should we trust eval at all when
  bpb regresses 20%?

### 2026-05-05 22:15 — Phase 7 → 8 transition (HYP16-23 retrospect)

**Phase 7 result: SATURATION**
8 HYPs (16-23) over 7 days. Final state: eval 51/51 (100%), bpb 0.0745
(HYP21 best), params 658K (HYP22 -34.5%). Three releases: v0.1.0
(25/31) → v0.2.0 (47/51) → v0.3.0 (51/51).

**What worked (KEEP-rate analysis)**:
- HYP16 (SOUL_QA + thread cap 8): KEEP — bpb 0.0885, +OMP_NUM_THREADS
  discovery (3× speedup at small matmul)
- HYP17 (53→83 SOUL): KEEP — eval 45/51
- HYP18 (Aqua 4-distinct): KEEP — eval 47/51
- HYP20 (60→90 min budget): KEEP — eval 48/51
- HYP21 (bias=False): KEEP — bpb 0.0745, eval 47/51
- HYP22 (tied embeddings): KEEP — eval **51/51 SATURATION**, params -34.5%

**What failed (REVERT or REVERT-precursor)**:
- HYP19 (eval-prefix alignment + 不/同 disambiguation): REVERT — corpus
  dilution at fixed budget regressed eval 47→43
- HYP21 first attempt (PID 788871): CRASHED at save() — bug fix in
  numpy-grad bias support, retry succeeded
- HYP23 first attempt (WSD epoch-count math): KILLED ep 20 — wall-clock
  fix, retry trained but REVERT for overfit
- HYP23 retry (WSD wall-clock): REVERT — bpb -69% but eval 51→50

**Patterns observed**:
1. **Architecture changes ≫ training changes** — HYP21 (bias=False) and
   HYP22 (tied) gave the biggest single jumps. WSD (HYP23) and 90-min
   budget (HYP20) were smaller wins. CS336 readings drove both
   architecture HYPs.
2. **bpb is a lying metric at small scale** — HYP23 had best-ever bpb
   but worst recent eval. Phase 8 should treat bpb as secondary.
3. **Surgical diff discipline holds** — every KEEP HYP changed exactly
   one lever. The two crashes were both implementation bugs (save() on
   None bias, expected_epochs math) not strategic mistakes.
4. **Crashes are recoverable** — both crash-then-fix cycles cost only
   ~30 min wallclock. Worth it because the lever was good.

**Phase 8 plan (writing while it's fresh)**:
1. Multi-turn eval harness (just shipped — eval_multiturn.py)
2. Synthetic dialogue corpus (HYP24, ~30-50 multi-turn samples Claude-
   generated from existing memory)
3. SFT on multi-turn corpus (HYP25)
4. KV cache for inference speed (HYP26-27, prep for v1.0)

**Phase 8 risks**:
- The 86KB single-turn corpus may not synthesize well into multi-turn
  (Claude has to invent context). May need Ryan's actual chat logs
  as ground truth for at least 10 of the 30 dialogues.
- Multi-turn metric is harsher than single-turn — even a 50%
  improvement might land us at 5-7/20 not 15/20.
- KV cache implementation in numpy-grad is non-trivial; may need its
  own HYP cycle in numpy-grad repo.

**Most surprising finding from HYP16-23**:
Tied embeddings + bias=False gave -34.5% params with PERFECT eval. The
intuition "smaller model = worse" is dead wrong at our scale: smaller
forced more efficient representation, generalization went up. CS336
Lec 9's "embedding params don't scale like core params" was the lock,
HYP22 was the key.



