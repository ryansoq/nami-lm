# nami-lm — HYP backlog (priority order)

> Refreshed 2026-08-05 (Opus 5) after HYP89. Phase 10/11/12 levers are exhausted;
> deployed optimum = **HYP89 v0.5.2.0-weekly-absorbed: strict 43/51, any-hit 49/51**,
> 736K params (d96), vocab 3510, ~124KB persona-pure corpus.
> Three lever families are SETTLED — see "Closed" below. The only paths to
> >43 are (a) more authentic on-domain data, or (b) a regime change.

## Live candidates (priority order)

- **Regime change: instruction tuning (phase 13)** — the real unlock for
  >42 strict. Not a 15-min tick; needs design (instruction/response format,
  loss masking on the prompt, eval-harness changes). STRATEGIC — flag to
  Ryan before starting; this is the next big project, not an autonomous run.
- **Daily-file corpus window 30→60** — authentic on-domain Nami narrative is
  the ONE corpus-add category proven non-diluting (HYP87 weekly reports:
  any-hit 47→48, soul-strong 17→18). Expanding the daily window is the same
  category. EV: marginal (likely lifts soft metrics, not strict 42). COST: a
  full ~4h retrain. Gate behind Ryan's nod given the frugality directive —
  do NOT auto-launch.
- **Hard-negative persona pairs** — "Nami是Aqua嗎？→不是，Aqua是另一個AI夥伴"
  style contrastive pairs to sharpen persona boundaries. Cheap to author,
  rides on the next retrain whenever one happens. Targets soul/persona axes,
  not strict.

## Closed / invalidated (do NOT re-run)

- **Capacity scaling (d≥112/128, more layers)** — REVERTED 3× (HYP55, 72/73,
  88). Scaling params overfits the data-starved ~113KB corpus regardless of
  corpus cleanliness. d96 is the right size, definitively. RMSNorm/RoPE/GQA
  "retry at d≥128" is moot because d≥128 itself reverts.
- **Decoding levers** — SETTLED. rep_penalty 1.6 + rep_window 12 + EOS token
  is the optimum (HYP77/78/82/83/84/85). The strict-39 "ceiling" was a
  greedy-argmax decoding bug, not capacity.
- **Cosine schedule / LR floor / warmup** — exhausted (HYP44B/60/61/62/74).
  60-ep cosine is the sweet spot; floor lowering hit FP precision at HYP62.
- **Batch size (8→16, 8→4), weight decay (0.03/0.05)** — wrong lever family
  at this scale (HYP63/64/66/67).
- **Synthetic paraphrases / technical book-notes in corpus** — DILUTE persona
  (HYP68/75/76/79/80). Only authentic Nami prose helps. PHASE11_EXCLUDE set
  in synthesize_qa.py keeps book-notes out.
- **HYP44-48 (old "next up")** — all consumed in phase 10 (print-freq, cosine,
  batch, d_ff, probe alignment). Superseded by the phase-10/11/12 arc above.

## Mechanism note — CORRECTED 2026-08-05 by HYP89

HYP87's weekly-REM-report feed means the **corpus** grows with authentic Nami
narrative every Sunday REM pass automatically.

⚠️ **The 5/30 version of this note ended "no tick needs to force a retrain" —
that was wrong and it cost 10 weeks.** The glob only decides what a retrain
*would* ingest; it does not train anything. Between 2026-05-25 and 2026-08-05 no
retrain ran, so six accrued reports sat unread (`data/phase0_qa.jsonl` still had
a May 25 mtime) and the mechanism delivered zero benefit. HYP89 absorbed them:
strict 42 → 43.

**A data pipeline is not a training schedule.** Whenever
`memory/weekly/` has grown by ~4+ reports since the last run, that IS a
legitimate autonomous HYP — it is not a new lever, not gated, and it is the
cheapest remaining source of strict gains. Check `ls -la data/phase0_qa.jsonl`
against `ls ~/clawd/memory/weekly/` before concluding there is nothing to do.
