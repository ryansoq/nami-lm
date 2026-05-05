# nami-lm — HYP backlog (priority order)

## Next up
- HYP20: 60→90 min budget, HYP18 corpus unchanged — push more epochs into existing anchors (target: recover 47/51 + push to 48-49)
- HYP21: lr_schedule cosine vs warmup-then-flat — current hits ep 90 with lr 0.00186, possibly under-using late epochs
- HYP22: BPE tokenizer 2048 (vs current word tokenizer 3592) — corpus 86KB, BPE merges may pay off

## Exploration queue (cadence-driven, every 5 HYPs)
- Tied embedding (HYP9 redo at d_model 96 — failed before, may work now with HYP18 corpus)
- AdamW betas (0.95, 0.99) per GPT-3 — failed at HYP10, may work with bigger batch
- num_layers 3→4 — would push params 1M→1.3M, eval impact unknown

## Simplification queue (cadence-driven, every 20 HYPs)
- Drop the 2 "chatbot嗎呵" / "generic 助手嗎" duplicate variants if HYP18 anchors are stable enough
- Consolidate redundant Bob/Whisper variants once eval-prefix mismatch is solved properly

