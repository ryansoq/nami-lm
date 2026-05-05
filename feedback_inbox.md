# nami-lm — Feedback inbox (Ryan-injected priority items)

Top of file = next HYP. Heartbeat reads this BEFORE backlog.md per
program.md §5.3. Add new items at the top.

## 2026-05-05 — HYP22 (tied embeddings) — from CS336 Lec 9 deep-dive

Ryan approved this priority via TG msg 3249 after the Lec 9 PDF review
revealed nami-lm's biggest scaling bottleneck: **68% of 1M params are
embeddings** (token_emb 345K + out_proj 345K = 689K out of 1005K).
Tying them halves embedding overhead.

**假設:** Tie token_emb weights with out_proj (share the same Tensor
between input embedding and output projection). Standard since GPT-2,
used by LLaMA / PaLM / Mistral. At nami-lm 1M scale this is the highest
single-lever parameter reduction available.

**假設前提:** Stanford CS336 Lec 9 + Lec 11 — embedding params don't
contribute to "scaling" in the standard sense; effective core params is
what matters. HYP9 first attempt failed at 28K-token corpus, but with
HYP18 corpus (86KB) + HYP20 90min budget + HYP18 anchors deeply learned,
the constraint that previously broke tied embedding (insufficient
training of the shared matrix) should be lifted.

**預測:** params 1.005M → 0.66M (-34%); bpb may rise slightly (the
shared matrix has dual duty, harder to learn) but eval should hold or
improve because effective core capacity (314K → 314K) is unchanged
while embedding overhead halved. Target: bpb < 0.10, eval ≥ 47/51.

**觸發線:** eval drops below 45 OR persona drops below 5/5 OR bpb > 0.12.

**Lever family:** architecture (different family from HYP21 bias=False,
satisfies §5.2 explore cadence).

**Implementation:** in train.py:GPTMini, replace `self.out_proj =
Linear(d_model, vocab_size, bias=False)` with a forward-time multiply
by `self.token_emb.W.T`. Remove out_proj from parameters. ~10-15 line
surgical diff. Need to also adjust the loss layer (logits = h @
token_emb.W.T) and verify gradient flow back to token_emb correctly
via numpy-grad's existing transpose+matmul.

## 2026-05-05 — HYP21 (bias=False) — from huangserva tweet 2051119679670976760

🏃 **CURRENTLY IN FLIGHT (PID 788871, started 13:00)** — leave entry
for traceability; consume after HYP21 is harvested next loop tick.

Ryan flagged this from a Stanford CS336 Lecture 3 summary tweet (msg
3227 / 3232). The 6 architecture takeaways included "去掉 bias" as one
of the 3 main fixes to original Transformer.

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

