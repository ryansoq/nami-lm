# nami-lm — Feedback inbox (Ryan-injected priority items)

Top of file = next HYP. Heartbeat reads this BEFORE backlog.md per
program.md §5.3. Add new items at the top.

## 2026-05-05 — HYP21 (bias=False) — from huangserva tweet 2051119679670976760

Ryan flagged this from a Stanford CS336 Lecture 3 summary tweet (msg
3227 / 3232). The 6 architecture takeaways included "去掉 bias" as one
of the 3 main fixes to original Transformer.

**假設:** Drop bias on all nn.Linear and LayerNorm layers — modern
practice (LLaMA, PaLM, GPT-NeoX) shows bias=False saves ~2-5% params
without hurting expressivity, and reduces gradient noise on the bias
parameter (which is hard to learn at our 1M scale anyway).

**假設前提:** Stanford CS336 Lecture 3 + standard 2020s practice;
Anthropic / Meta / Google all default bias=False on Linear since GPT-3.

**預測:** params 1.005M → 0.96-0.98M (-2-4%); bpb unchanged or -1-2%;
eval ≥ 47/51 (no regression).

**觸發線:** eval drops below 45 OR persona drops below 5/5 OR bpb +5%.

**Lever family:** architecture (last 3 HYPs were data/budget/corpus →
this satisfies the §5.2 every-5-HYP exploration cadence as well).

**Implementation:** in train.py:GPTMini, set bias=False on all
nn.Linear and elementwise_affine=False on LayerNorm where appropriate.
~5-10 line surgical diff.

