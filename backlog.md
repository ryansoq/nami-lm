# nami-lm — HYP backlog (priority order)

## Next up (phase 10 candidates, post-HYP43)

- **HYP44: print frequency 10→5** — infra fix, makes ep 5/15/25 visible
  for budget monitoring. Not a real HYP (no scientific lever) but
  improves observability of HYP45+. Cost: 1-line edit. Stack on whatever
  the next real HYP is.
- **HYP45: cosine LR schedule + warmup 10%** — current lr=0.002 flat for
  10 ep warmup then... what? Check train.py for schedule. Llama2 uses
  cosine decay. 670K params at 100KB corpus probably leaves late-epoch
  signal on table.
- **HYP46: BATCH_SIZE 8→16** — doubles batch parallelism. Risk: AdamW
  dynamics shift (effective lr changes). Mitigation: keep lr fixed,
  measure if convergence speed compensates for the optimizer disruption.
  Should land ~2x epochs/budget.
- **HYP47: SwiGLU d_ff 256→384** — wider FFN inside same layers. params
  670K→~900K. Cheaper than num_layers bump per "capability unit". Pair
  with budget extension if needed.
- **HYP48: persona PROBE alignment** — extend single-turn from 51 to
  full corpus Q's (~1500), report top-K acc. Current 51-probe set
  frozen since HYP18 may not align with current corpus growth.

## Architecture explorations (re-validate at d_model ≥ 128 first)

- RMSNorm — HYP38/39 failed at d_model 96, retry at 128/256
- RoPE — positional encoding lever, swap pos_emb to rotary
- GQA (grouped-query attention) — Llama2 lever, expensive to implement
  in numpy-grad
- Tied embedding (HYP9 redo) — already in current train.py? grep to
  verify before queuing again

## Data work (orthogonal to architecture)

- Corpus expansion: pull in last 60 daily files (vs current 30)
- Per-category curriculum: train each category 2 epochs separately,
  then interleave (prevents crowd-out from HYP34-37)
- Hard-negative pairs: "Nami是Aqua嗎？→ 不是，Aqua是另一個 AI 夥伴"
  to address persona contamination

## Simplification queue

- Drop duplicate Bob/Whisper variants once eval anchors stable
- Consolidate inference-time normalizer rules (web_chat.py) — currently
  duplicates work HYP40/42 tried to bake in
