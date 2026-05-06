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



