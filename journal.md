# nami-lm — Journal

## Open questions
- **Multi-turn eval at 0/5 baseline** — does the model need a NEW token
  in vocab for "\n" (turn separator) for the dialogue format to anchor?
  Or is plain concat enough once trained on the format? Defer until
  HYP24 corpus is built.
- **bpb-eval divergence** (HYP23 lesson) — why did bpb -69% but eval -1?
  Probably overfit to training distribution at near-zero lr. Means
  bpb is unreliable below ~0.05; we should rely on eval.

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



