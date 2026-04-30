# nami-lm Architecture

A walkthrough of the model that lives in `train.py`. Everything below
matches the current Phase 5 / HYP8 baseline: **978K params, d_model=96,
3 layers, 6 heads, vocab 3471**.

---

## Big picture

```
                          input: "Nami是誰？"
                              │
                              ▼
                       ┌─────────────────┐
                       │ WordTokenizer   │   ASCII-alpha runs + digit
                       │  → token IDs    │   runs as 1 token,
                       └────────┬────────┘   else 1 char per token
                                │ (T,)            T = sequence length
                                ▼
        ┌───────────────────────────────────┐
        │   token_emb : (vocab, d_model)    │   3471 × 96 = 333K params
        │   pos_emb   : (max_seq_len, d)    │   128 × 96 = 12K params
        │      x = token_emb[ids] + pos     │
        └────────────────┬──────────────────┘
                         │ (T, d_model) = (T, 96)
                         ▼
        ┌───────────────────────────────────┐
        │   TransformerBlock × 3            │   pre-norm
        │     x = x + MHA(LN(x))            │
        │     x = x + SwiGLU(LN(x))         │
        └────────────────┬──────────────────┘
                         │ (T, 96)
                         ▼
        ┌───────────────────────────────────┐
        │   head: Linear(d, d_ff)           │   96 → 256 → 96
        │   GELU                            │   "MLP head" before vocab
        │   Linear(d_ff, d)                 │
        └────────────────┬──────────────────┘
                         │ (T, 96)
                         ▼
        ┌───────────────────────────────────┐
        │   out_proj: Linear(d, vocab)      │   96 × 3471 = 333K params
        └────────────────┬──────────────────┘
                         │ (T, vocab) = (T, 3471)
                         ▼
                      logits
                         │
                         ▼
              softmax → argmax → next token
```

---

## What each TransformerBlock does

```
                   x : (T, 96)
                   │
                   ▼
              ┌──────────┐
              │ LayerNorm│   per-token mean/var on the d=96 axis
              └────┬─────┘
                   │
                   ▼
            ┌────────────┐
            │   MHA      │   6 heads × head_dim 16
            │  Q,K,V     │   each is Linear(96,96) → reshape to (T, 6, 16)
            │  ↓         │   scores = QKᵀ / √16  → causal mask → softmax
            │  attn @ V  │   out = (T, 96) → Wo → (T, 96)
            └────┬───────┘
                 │
                 ▼
              x ⊕ ─── residual
                 │
                 ▼
              ┌──────────┐
              │ LayerNorm│
              └────┬─────┘
                   │
                   ▼
            ┌────────────┐
            │   SwiGLU   │   d_inner = 2/3 × 256 = 170
            │            │   SiLU(x·W1) ⊙ (x·Wgate) → Wout
            │            │   3 Linears: (96,170) (96,170) (170,96)
            └────┬───────┘
                 │
                 ▼
              x ⊕ ─── residual
                 │
                 ▼
              x : (T, 96)
```

**Pre-norm** (LayerNorm *before* attention/FFN, residual *around* the
whole sub-block) is the modern default — it stabilises gradient flow
through deep stacks. GPT-2 used post-norm; everything since GPT-3 is
pre-norm.

---

## Parameter breakdown (978,528 total)

| Component | Shape | Params | % |
|---|---|---|---|
| `token_emb`  | (3471, 96) | 333,216 | 34.1% |
| `pos_emb`    | (128, 96)  | 12,288  | 1.3%  |
| 3× MHA (Wq+Wk+Wv+Wo + biases) | 4 × (96,96) per layer | 111,552 | 11.4% |
| 3× LayerNorm × 2 (γ + β)    | (96,) × 12 | 1,152 | 0.1% |
| 3× SwiGLU (w1+gate+w2)      | (96,170)+(96,170)+(170,96) per layer | 97,920 | 10.0% |
| `head` (Linear+GELU+Linear) | (96,256) + (256,96) | 49,152 | 5.0% |
| `out_proj`   | (96, 3471) | 333,216 | 34.1% |

The two big chunks are **token_emb** and **out_proj** — both
`vocab × d_model`. They flank the model: input embedding turns IDs
into vectors, output projection turns final hidden state back into a
distribution over vocab. Together they're 68% of parameters.

> **Tied embedding** (HYP9) tried sharing them via
> `out_proj = token_emb.T`, dropping 333K params. Result: 25% faster
> per epoch but bpb regressed +18%. At our `d_model=96` and ~28K
> training tokens, the embedding can't shoulder both roles. Lesson:
> tied embedding pays at large vocab × small `d_model` only if there
> are enough training tokens for the embedding to learn dual duty.

---

## Key tensor shapes (single sequence, T=24 typical)

| Stage | Shape | Notes |
|---|---|---|
| token IDs | (24,) | int64 |
| after embed | (24, 96) | token_emb + pos_emb |
| after LN1 | (24, 96) | normalized over last dim |
| Q, K, V (after split_heads) | (1, 6, 24, 16) | (B, H, T, hd) |
| attention scores | (1, 6, 24, 24) | T × T per head |
| attention output | (1, 6, 24, 16) → (24, 96) | reshape back |
| after head | (24, 96) | through MLP head |
| logits | (24, 3471) | one distribution per position |

For batched training, prepend `B=8` to all shapes.

---

## How a forward pass produces "Nami"

1. **Tokenize** "妳是誰？" — char-level for Chinese, so tokens are
   `["妳", "是", "誰", "？"]`. After `encode()` → `[212, 87, 459, 71]`.

2. **Embed**: row-gather from `token_emb` → 4 rows of shape (96,).
   Add the first 4 rows of `pos_emb`. Result: `(4, 96)`.

3. **Block 1** (LN → MHA → +residual → LN → SwiGLU → +residual). The
   attention is causal so position 3 ("？") attends to positions 0–3
   only.

4. **Blocks 2 & 3**: same shape, same pattern. Each block refines the
   representation a little. By the end, position 3's vector encodes
   "the answer to 妳是誰？".

5. **Head**: 2-layer MLP further mixes features. Still (4, 96).

6. **out_proj**: `(4, 96) @ (96, 3471) = (4, 3471)`. Take row 3 (the
   last position). This is a distribution over the 3471 vocab.

7. **Sample**: argmax (`temperature ≈ 0`) gives the most likely next
   token. For a well-trained probe this should be `"N"`.

8. **Repeat** with the new prefix until `max_new` tokens are produced
   or `<EOS>` is hit (we have no EOS yet — phase 0 just stops at
   `max_new`).

---

## Hyperparameters that matter, and why

| Knob | Phase 5 value | Why |
|---|---|---|
| `d_model` | 96 | Sweet spot at our compute. Phase 3 tried 128 — under-fit (Chinchilla math). |
| `d_ff`    | 256 | 2.67× d_model. SwiGLU's 2/3 rule makes inner dim 170. |
| `num_heads` | 6 | head_dim = 16, divides cleanly into 96. |
| `num_layers` | 3 | Going deeper costs per-epoch time more than width does (autochat HYP13). |
| `lr`      | 0.002 | Warmup 2 epochs, then constant. |
| `weight_decay` | 0.02 | AdamW decoupled wd. |
| `betas`   | (0.9, 0.999) | HYP10 tried GPT-3's (0.9, 0.95) — too jumpy at our 300-token batches. |
| `BATCH_SIZE` | 8 | Length-bucketed (HYP5 in autochat lineage). |
| `max_seq_len` | 128 | Phase-0 chunks rarely exceed 50 tokens. |
| `TIME_BUDGET` | 60 min | Realistic load (~130 s/ep) needs this for 30 epochs. |

---

## What's missing (Phase 6 roadmap)

This is the architecture as of Phase 5. Phase 6 will add (in order):

1. **`eval.py`** — broader probe set (current `probe()` only checks 5
   personas). Needed before any further architecture changes so we
   gate honestly.
2. **Conversational mode** — currently the model only does prefix-match
   answers. To support multi-turn, will need an `<EOS>` token and a
   conversation format like `<USER>q<ASSISTANT>a<EOS>`.
3. **Self-distillation** — Claude generates 1000+ Q&A pairs about
   Nami's life and projects, nami-lm trains on them. This is how
   "the small companion learns to understand Nami".
4. **Gradual scaling** — once corpus passes ~150KB and we have
   conversational data, retry `d_model=128` with proportionally larger
   budget. Chinchilla curve says this is when extra capacity actually
   pays.
5. **`pip install nami-lm`** — package the trained checkpoint + code
   so any clone can run `python -m nami_lm` to chat.

---

## Reading order for a learner

If you've never seen a tiny GPT before, read in this order:

1. **`train.py`** — start with `class GPTMini` (line 169). Trace
   `forward()` step by step against the diagram above.
2. **numpy-grad's `nn.py`** — the building blocks: `Embedding`,
   `LayerNorm`, `Linear`, `MultiHeadAttention`, `SwiGLU`,
   `TransformerBlock`. Each is ≤30 lines.
3. **`PHASES.md`** — see how the model evolved through Phases 0-5.
4. **`state.json` → `tried_hypotheses`** — every experiment with the
   actual numbers and what we learned.

Total Python you need to read to fully understand a working
transformer: about 600 lines. No CUDA, no PyTorch, no tricks.
