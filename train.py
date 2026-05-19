#!/usr/bin/env python3
"""
nami-lm phase 0 trainer.

Forks autochat's GPTMini architecture (proven 92/92 acc, bpb 0.0988
on the 92 QA corpus) and swaps the hardcoded TRAINING_DATA for
data/phase0_corpus.jsonl produced by extract_corpus.py.

Authors: Ryan & Nami ✨

Usage:
    python3 extract_corpus.py     # build the corpus first
    python3 train.py              # train (default 200 epochs)
    python3 train.py --auto       # autoresearch mode (20-min budget)
    python3 train.py --probe      # ask the trained model 5 persona Qs
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# numpy-grad — array-level autograd in pure NumPy
# https://github.com/ryansoq/numpy-grad
from numpy_grad import Tensor
from numpy_grad.nn import (
    AdamW,
    Embedding,
    GELU,
    LayerNorm,
    Linear,
    Module,
    MultiHeadAttention,
    RMSNorm,
    Sequential,
    SwiGLU,
    clip_grad_norm_,
    cross_entropy,
)


HERE = Path(__file__).parent
QA_CORPUS = HERE / "data" / "phase0_qa.jsonl"
TOK_DIR = HERE / "tokenizer"

# Phase 1 evaluation: BPE infra ships and is lossless, but at the
# 18 KB phase-0 corpus most Chinese chars don't reach the merge
# frequency threshold and end up as 3 separate byte-tokens. That hurts
# bpb badly (+94% vs WordTokenizer). Defer activation to phase 2 when
# corpus is ≥ 100 KB and merges actually pay.
# Toggle to `True` to re-evaluate after corpus expansion.
USE_BPE = False


# =============================================================================
# Corpus + tokenizer
# =============================================================================
def load_corpus() -> tuple[list[str], list[float]]:
    """Load Q&A pairs from phase0_qa.jsonl. Returns (texts, weights).

    Each line in the JSONL may have an optional "w" field for per-chunk
    loss weight (HYP30). Default weight 1.0 if absent.
    """
    if not QA_CORPUS.exists():
        raise FileNotFoundError(
            f"{QA_CORPUS} missing — run `python3 synthesize_qa.py` first")
    out, weights = [], []
    with open(QA_CORPUS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d["q"].strip(), d["a"].strip()
            q = q.rstrip("?？")
            out.append(f"{q}？{a}")
            weights.append(float(d.get("w", 1.0)))
    return out, weights


class WordTokenizer:
    """Phase 0 tokenizer — same hybrid rule as autochat: ASCII alpha
    runs and digit runs become single tokens, everything else
    (Chinese, punctuation, whitespace) is char-level. Phase 1 replaces
    this with BPE.

    Constructor accepts either `texts=[...]` (build vocab from corpus)
    or `vocab=[...]` (use pre-built vocab — for inference against a
    saved checkpoint, see train.py:save vocab_path)."""

    def __init__(self, texts=None, vocab=None):
        if vocab is not None:
            self.vocab = list(vocab)
        else:
            all_tokens = set()
            for t in texts or []:
                all_tokens.update(self._tokenize(t))
            self.vocab = sorted(all_tokens)
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.name = "WordTokenizer"

    @classmethod
    def load(cls, path):
        """Load a frozen vocab from JSON saved by train.py."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(vocab=d["vocab"])

    def _tokenize(self, text):
        tokens, i = [], 0
        while i < len(text):
            c = text[i]
            if c.isascii() and c.isalpha():
                j = i
                while j < len(text) and text[j].isascii() and text[j].isalpha():
                    j += 1
                tokens.append(text[i:j]); i = j
            elif c.isdigit():
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                tokens.append(text[i:j]); i = j
            else:
                tokens.append(c); i += 1
        return tokens

    def encode(self, text):
        return [self.token2id[t] for t in self._tokenize(text)
                if t in self.token2id]

    def decode(self, ids):
        return "".join(self.id2token[i] for i in ids if i in self.id2token)


class BPETokenizer:
    """Phase 1 tokenizer — wraps `tokenizer/bpe.BPE` with the same
    interface as WordTokenizer so train.py can use it interchangeably.

    Vocab size is fixed at training time (default 1024); this wrapper
    just exposes it via .vocab_size for the model constructor.
    """

    def __init__(self, texts=None):
        # `texts` is accepted for API parity with WordTokenizer but not
        # used — BPE is pre-trained via train_bpe.py.
        from tokenizer.bpe import BPE
        self._bpe = BPE.load(TOK_DIR)
        self.vocab_size = self._bpe.vocab_size
        self.name = "BPETokenizer"

    def encode(self, text):
        return self._bpe.encode(text)

    def decode(self, ids):
        return self._bpe.decode(ids)


# =============================================================================
# Model — fork of autochat's GPTMini
# =============================================================================
class TransformerBlock(Module):
    def __init__(self, d_model, d_ff, num_heads):
        # HYP21: bias=False on LN + MHA (per Stanford CS336 Lec 3 / huangserva).
        # SwiGLU was already bias-free; head + out_proj already bias-free.
        # HYP38/39 (REVERTed): tested RMSNorm here. At 670K params + 100KB
        # corpus the "free Llama2 swap" REGRESSED single-turn 47→36/51.
        # LayerNorm bias + mean-centering appear to help at small scale.
        # See journal — RMSNorm worth re-trying at d_model ≥ 256 / corpus ≥ 500KB.
        self.ln1 = LayerNorm(d_model, bias=False)
        self.mha = MultiHeadAttention(d_model, num_heads, bias=False)
        self.ln2 = LayerNorm(d_model, bias=False)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTMini(Module):
    """Tiny GPT-1: token+pos embedding, N pre-norm TransformerBlocks
    with SwiGLU FFN, GELU MLP head, Linear out-projection."""

    def __init__(self, vocab_size, d_model=128, d_ff=384, num_heads=8,
                 num_layers=3, max_seq_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Tensor(
            np.random.randn(max_seq_len, d_model) * 0.02, requires_grad=True)
        self.blocks = [TransformerBlock(d_model, d_ff, num_heads)
                       for _ in range(num_layers)]
        self.head = Sequential(Linear(d_model, d_ff, bias=False), GELU(),
                               Linear(d_ff, d_model, bias=False))
        # HYP22 (tied embeddings): out_proj weight is shared with token_emb.
        # Forward computes logits = x @ token_emb.weight.T directly. Saves
        # 345K params (vocab_size × d_model) — 34% of total model size.
        # No separate out_proj layer.
        self.out_proj = None

        # HYP2 (autochat) Xavier init alignment — sqrt(2/(in+out))
        self._apply_xavier_init_to_linears()

    def _apply_xavier_init_to_linears(self):
        def reinit(linear):
            in_d, out_d = linear.W.data.shape
            scale = np.sqrt(2.0 / (in_d + out_d))
            linear.W.data = (np.random.randn(in_d, out_d) * scale).astype(
                linear.W.data.dtype)
        reinit(self.head.layers[0]); reinit(self.head.layers[2])
        # HYP22: out_proj is tied with token_emb, no separate Linear to reinit
        for b in self.blocks:
            reinit(b.mha.Wq); reinit(b.mha.Wk); reinit(b.mha.Wv); reinit(b.mha.Wo)
            reinit(b.ff.w1);  reinit(b.ff.gate); reinit(b.ff.w2)

    def forward(self, token_ids):
        from numpy_grad.ops import embedding as _embed
        ids = np.asarray(token_ids, dtype=np.int64)
        single = ids.ndim == 1
        if single:
            ids = ids[None, :]
        B, T = ids.shape
        pos = _embed(self.pos_emb, np.arange(T, dtype=np.int64))
        x = self.token_emb(ids) + pos
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        # HYP22: tied output projection — share token_emb.weight (vocab, d_model).T
        logits = x @ self.token_emb.weight.transpose(1, 0)
        return logits.reshape(T, self.vocab_size) if single else logits

    def generate(self, token_ids, max_new=50, temperature=0.1):
        ids = list(token_ids)
        for _ in range(max_new):
            if len(ids) >= self.max_seq_len:
                break
            logits = self.forward(ids).data
            next_logits = logits[-1] / max(temperature, 1e-8)
            e = np.exp(next_logits - next_logits.max())
            probs = e / e.sum()
            ids.append(int(np.argmax(probs)))
        return ids

    @property
    def param_count(self):
        return sum(p.data.size for p in self.parameters())

    # -----------------------------------------------------------------
    # Persistence — port of autochat's save/load
    # -----------------------------------------------------------------
    def save(self, path):
        state = {
            "config": {
                "vocab_size": self.vocab_size, "d_model": self.d_model,
                "d_ff": self.d_ff, "num_heads": self.num_heads,
                "num_layers": self.num_layers, "max_seq_len": self.max_seq_len,
            },
            "token_emb": self.token_emb.weight.data.tolist(),
            "pos_emb": self.pos_emb.data.tolist(),
            # HYP22: out_proj tied with token_emb — emit null when tied
            "out_proj": (self.out_proj.W.data.tolist()
                         if self.out_proj is not None else None),
            "head_w1": self.head.layers[0].W.data.tolist(),
            "head_w2": self.head.layers[2].W.data.tolist(),
            "blocks": [],
        }
        for b in self.blocks:
            # HYP21: handle bias=False (None values for beta/Wq_b/etc)
            # HYP38: RMSNorm has no .beta attribute (LayerNorm does) — use
            # getattr so save/load works against both norm types.
            def _opt(t):
                return t.data.tolist() if t is not None else None
            state["blocks"].append({
                "ln1_g": b.ln1.gamma.data.tolist(),
                "ln1_b": _opt(getattr(b.ln1, "beta", None)),
                "ln2_g": b.ln2.gamma.data.tolist(),
                "ln2_b": _opt(getattr(b.ln2, "beta", None)),
                "Wq": b.mha.Wq.W.data.tolist(),
                "Wk": b.mha.Wk.W.data.tolist(),
                "Wv": b.mha.Wv.W.data.tolist(),
                "Wo": b.mha.Wo.W.data.tolist(),
                "Wq_b": _opt(b.mha.Wq.b),
                "Wk_b": _opt(b.mha.Wk.b),
                "Wv_b": _opt(b.mha.Wv.b),
                "Wo_b": _opt(b.mha.Wo.b),
                "ff_w1": b.ff.w1.W.data.tolist(),
                "ff_gate": b.ff.gate.W.data.tolist(),
                "ff_w2": b.ff.w2.W.data.tolist(),
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        print(f"💾 Model saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        cfg = state["config"]
        m = cls(**cfg)
        m.token_emb.weight.data = np.array(state["token_emb"])
        m.pos_emb.data = np.array(state["pos_emb"])
        # HYP22: out_proj null = tied with token_emb (no separate weight)
        if state.get("out_proj") is not None and m.out_proj is not None:
            m.out_proj.W.data = np.array(state["out_proj"])
        m.head.layers[0].W.data = np.array(state["head_w1"])
        m.head.layers[2].W.data = np.array(state["head_w2"])
        for b, st in zip(m.blocks, state["blocks"]):
            # HYP21: tolerate bias=None (loaded as null in JSON)
            def _maybe_set(target, key):
                v = st.get(key)
                if v is None or target is None:
                    return
                target.data = np.array(v)
            b.ln1.gamma.data = np.array(st["ln1_g"])
            _maybe_set(getattr(b.ln1, "beta", None), "ln1_b")
            b.ln2.gamma.data = np.array(st["ln2_g"])
            _maybe_set(getattr(b.ln2, "beta", None), "ln2_b")
            b.mha.Wq.W.data = np.array(st["Wq"])
            b.mha.Wk.W.data = np.array(st["Wk"])
            b.mha.Wv.W.data = np.array(st["Wv"])
            b.mha.Wo.W.data = np.array(st["Wo"])
            _maybe_set(b.mha.Wq.b, "Wq_b")
            _maybe_set(b.mha.Wk.b, "Wk_b")
            _maybe_set(b.mha.Wv.b, "Wv_b")
            _maybe_set(b.mha.Wo.b, "Wo_b")
            b.ff.w1.W.data = np.array(st["ff_w1"])
            b.ff.gate.W.data = np.array(st["ff_gate"])
            b.ff.w2.W.data = np.array(st["ff_w2"])
        print(f"📂 Model loaded from {path}")
        return m


# =============================================================================
# Training loop — mini-batch length-bucketed (autochat HYP5)
# =============================================================================
def compute_bpb(loss, tokenizer, texts):
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    avg = total_bytes / total_tokens if total_tokens > 0 else 1.0
    return loss / math.log(2) / avg


TIME_BUDGET = 240 * 60  # restored to HYP54 baseline after HYP55 REVERT. 300min wasn't enough for d_model 128 — finished at ep 14 vs ep 30 target. d_model 96 + 240min is the proven configuration.


# Phase 0 persona probes — questions taken from synthesize_qa.py's
# PERSONA_QA list. The model must produce a completion that starts
# with the matching answer. (Phase 4 will turn this into the full
# eval framework.)
PERSONA_PROBES = [
    ("妳是誰？", "Nami"),
    ("Nami是誰？", "厲害的"),
    ("Ryan是誰？", "Nami"),
    ("Kaspa是什麼？", "基於"),
    ("ClawX是什麼？", "Claude"),
]


def train(epochs: int = 200, lr: float = 0.002,
          time_budget: int | None = None):
    np.random.seed(42)

    print("=" * 60)
    print("🌊 nami-lm phase 0 trainer (numpy-grad backend)")
    print("=" * 60)

    corpus, corpus_weights = load_corpus()
    n_weighted = sum(1 for w in corpus_weights if w != 1.0)
    print(f"📝 Corpus: {len(corpus)} chunks, "
          f"{sum(len(c.encode('utf-8')) for c in corpus):,} bytes "
          f"({n_weighted} weighted, HYP30)")

    if USE_BPE:
        tokenizer = BPETokenizer()
        print(f"📊 Vocab: {tokenizer.vocab_size} tokens (BPE — phase 1+)")
    else:
        tokenizer = WordTokenizer(corpus)
        print(f"📊 Vocab: {tokenizer.vocab_size} tokens (WordTokenizer — phase 0)")

    # Token-length stats so we know whether max_seq_len is sane.
    # Pair each encoded sequence with its corpus_weight (HYP30).
    encoded_w = [(tokenizer.encode(c), w)
                 for c, w in zip(corpus, corpus_weights)]
    encoded_w = [(ids, w) for ids, w in encoded_w if len(ids) >= 2]
    encoded = [ids for ids, _ in encoded_w]  # back-compat for stats below
    max_len = max(len(ids) for ids in encoded)
    median_len = sorted(len(ids) for ids in encoded)[len(encoded) // 2]
    print(f"📏 Seq len: median={median_len}, max={max_len}")

    # Cap max_seq_len so phase-0 epoch time stays ≤ autochat scale
    max_seq_len = min(max(max_len, 32), 64)

    # Reverted to autochat HYP11 sweet spot after phase-3 d=128
    # experiment showed scaling up just under-fits at our compute
    # budget (Chinchilla math: more params without proportional
    # compute = worse). Phase 0 setup remains the floor.
    d_model, d_ff, num_heads, num_layers = 96, 256, 6, 4  # HYP65 REVERT — 5 layers strict 33 = -6 from HYP61's 39 (undertrained at 120 ep budget, HYP55 lesson repeats). Back to 4. Architecture-up lever exhausted alongside cosine floor & weight decay.
    model = GPTMini(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model, d_ff=d_ff, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=max_seq_len,
    )
    print(f"⚙️  d_model={d_model} d_ff={d_ff} heads={num_heads} "
          f"layers={num_layers} lr={lr}")
    print(f"📊 Params: {model.param_count:,}")

    avg_bpt = sum(len(c.encode("utf-8")) for c in corpus) \
        / max(sum(len(ids) for ids in encoded), 1)
    print(f"📐 avg_bytes/token = {avg_bpt:.2f}")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.02)  # HYP64 REVERT — wd 0.03→0.02 restore; weight_decay lever proven wrong family (HYP63 0.05 +8% bpb degraded, HYP64 0.03 strict 35 = -4 from HYP61's 39).

    # Truncate sequences over max_seq_len, keep weights aligned (HYP30)
    encoded_w = [(ids[:max_seq_len], w)
                 for ids, w in encoded_w if len(ids) >= 2]

    # Length-bucket batches (autochat HYP5). Each bucket holds (ids, w).
    BATCH_SIZE = 16  # HYP66: 8→16 — phase 10 cosine/wd/layers levers all exhausted. Larger batch = ~2× faster per epoch + cleaner gradient signal. Llama2 uses 4M tokens per batch (8 here is tiny). Predict: ~240 ep in 240min budget (vs HYP65 120 ep), strict ≥39.
    length_buckets: dict[int, list[tuple[list[int], float]]] = {}
    for ids, w in encoded_w:
        length_buckets.setdefault(len(ids), []).append((ids, w))
    bucket_keys = sorted(length_buckets.keys())
    n_batches_per_epoch = sum(
        (len(length_buckets[L]) + BATCH_SIZE - 1) // BATCH_SIZE
        for L in bucket_keys
    )
    print(f"📦 batched: {len(bucket_keys)} length buckets, "
          f"~{n_batches_per_epoch} batches/epoch (size {BATCH_SIZE})")

    print("🏋️  Training...")
    start = time.time()
    warmup_epochs = 2
    # HYP44B: cosine schedule target = realistic epochs in budget.
    # Old: `time_budget / 7.0` → 1542 ep target at 180min budget. Way off:
    # actual reach is 30 ep at 4-layer / 22 ep at 3-layer. progress < 0.02
    # all training → cos ≈ 1 → lr stays flat at peak. No decay.
    # New: assume ~6 min/ep (4-layer; 3-layer ~5min). For 180min budget
    # → ~30 ep target. cosine actually decays late epochs to lr*0.5*(1+cos(π))
    # = 0 (clamped to lr*0.01). Matches Llama-style cosine to 10% of peak.
    expected_epochs = (
        min(epochs, max(20, int(time_budget / 360))) if time_budget else epochs)

    avg_loss = float("inf")
    epoch = 0
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            cur_lr = lr * (epoch + 1) / warmup_epochs
        else:
            progress = min(
                (epoch - warmup_epochs) / max(expected_epochs - warmup_epochs, 1),
                1.0)
            cur_lr = lr * 0.5 * (1 + np.cos(np.pi * progress))
        cur_lr = max(cur_lr, lr * 0.0001)  # HYP61: 0.001→0.0001 (another 10×) — HYP60 broke HYP57 ceiling with 1 decade lower; test if another decade gives another +1pp strict (compounding) or just diminishing returns.
        opt.lr = cur_lr

        total_loss = 0.0
        n_seqs = 0

        epoch_buckets = bucket_keys[:]
        np.random.shuffle(epoch_buckets)
        for L in epoch_buckets:
            seqs = length_buckets[L][:]  # list of (ids, w)
            np.random.shuffle(seqs)
            for i in range(0, len(seqs), BATCH_SIZE):
                batch = seqs[i:i + BATCH_SIZE]
                inputs = np.array([s[0][:-1] for s in batch], dtype=np.int64)
                targets = np.array([s[0][1:] for s in batch], dtype=np.int64)
                # HYP30: per-batch weight = mean of per-sequence weights
                batch_w = float(np.mean([s[1] for s in batch]))
                opt.zero_grad()
                logits = model(inputs)
                loss = cross_entropy(logits, targets) * batch_w
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=0.5)
                opt.step()
                # Track UNWEIGHTED loss for bpb reporting (so bpb stays
                # comparable to prior HYPs)
                total_loss += float(loss.data) / batch_w * len(batch)
                n_seqs += len(batch)

        avg_loss = total_loss / max(n_seqs, 1)

        if epoch % 10 == 0 or epoch == epochs - 1:
            bpb = avg_loss / math.log(2) / avg_bpt
            elapsed = time.time() - start
            print(f"  ep {epoch:4d} | loss={avg_loss:.4f} | bpb={bpb:.4f} | "
                  f"lr={cur_lr:.5f} | {elapsed:.1f}s")

        if time_budget and (time.time() - start) >= time_budget:
            print(f"\n⏱️  Time budget reached ({time_budget}s)")
            break

    elapsed = time.time() - start
    final_bpb = avg_loss / math.log(2) / avg_bpt
    print(f"\n⏱️  Total: {elapsed:.1f}s | loss={avg_loss:.4f} | bpb={final_bpb:.4f}")

    # Persona probe — Phase 0 gate
    print("\n🔮 Persona probe (Phase 0 gate):")
    persona_pass = 0
    for q, expected_prefix in PERSONA_PROBES:
        ids = tokenizer.encode(q)
        if not ids:
            print(f"  {q!r} → not in vocab, skipping")
            continue
        gen = model.generate(ids, max_new=20, temperature=0.01)
        completion = tokenizer.decode(gen[len(ids):])[:40]
        ok = expected_prefix in completion[:len(expected_prefix) + 4]
        mark = "✅" if ok else "❌"
        if ok:
            persona_pass += 1
        print(f"  {mark} {q!r} → {completion!r}  (expect prefix '{expected_prefix}')")
    print(f"\n📊 Persona: {persona_pass}/{len(PERSONA_PROBES)} pass")

    # Save weights so probe / chat modes can reuse them
    weights_path = HERE / "model_weights.json"
    model.save(str(weights_path))

    # Pin the tokenizer vocab next to weights so inference doesn't have to
    # rebuild from corpus (which drifts as topics/daily files change). The
    # 2026-05-10 voice-spike bug — model trained on vocab 3780, current
    # corpus produces 3738, token IDs misaligned, generate yields noise —
    # is caused by reconstructing the tokenizer at inference time. With a
    # frozen vocab.json next to model_weights.json, nami_voice.py and any
    # other inference path can decode against the exact ID space the model
    # was trained on.
    if isinstance(tokenizer, WordTokenizer):
        vocab_path = HERE / "tokenizer_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({"name": "WordTokenizer",
                       "vocab": tokenizer.vocab,
                       "vocab_size": tokenizer.vocab_size}, f, ensure_ascii=False)
        print(f"💾 Tokenizer vocab saved to {vocab_path} ({tokenizer.vocab_size} tokens)")

    # Log to experiments.jsonl
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": 0,
        "epochs": epoch + 1,
        "elapsed_s": round(elapsed, 1),
        "final_loss": round(float(avg_loss), 6),
        "final_bpb": round(float(final_bpb), 6),
        "config": {
            "d_model": d_model, "d_ff": d_ff, "num_heads": num_heads,
            "num_layers": num_layers, "max_seq_len": max_seq_len,
            "lr": lr, "vocab_size": tokenizer.vocab_size,
            "params": model.param_count,
            "tokenizer": tokenizer.name,
            "corpus_chunks": len(corpus),
            "BATCH_SIZE": BATCH_SIZE,
        },
    }
    with open(HERE / "experiments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print("📝 Experiment logged")


def _load_for_inference():
    """Common path for probe / chat: load weights + tokenizer."""
    weights = HERE / "model_weights.json"
    if not weights.exists():
        print(f"❌ {weights} missing — run `python3 train.py` first to "
              "produce a checkpoint")
        sys.exit(1)
    corpus, _ = load_corpus()
    if USE_BPE:
        tokenizer = BPETokenizer()
    else:
        tokenizer = WordTokenizer(corpus)
    model = GPTMini.load(str(weights))
    if model.vocab_size != tokenizer.vocab_size:
        print(f"⚠️  vocab mismatch: model has {model.vocab_size} but "
              f"tokenizer rebuilt with {tokenizer.vocab_size}. Re-run "
              "training so checkpoint and corpus stay in sync.")
        sys.exit(1)
    return model, tokenizer


def probe():
    """Run the 5 phase-0 persona probes against the saved checkpoint."""
    model, tok = _load_for_inference()
    print("\n🔮 Persona probe (saved checkpoint):")
    persona_pass = 0
    for q, expected_prefix in PERSONA_PROBES:
        ids = tok.encode(q)
        if not ids:
            print(f"  {q!r} → not in vocab, skipping"); continue
        gen = model.generate(ids, max_new=20, temperature=0.01)
        completion = tok.decode(gen[len(ids):])[:40]
        ok = expected_prefix in completion[:len(expected_prefix) + 4]
        mark = "✅" if ok else "❌"
        if ok:
            persona_pass += 1
        print(f"  {mark} {q!r} → {completion!r}  (expect '{expected_prefix}')")
    print(f"\n📊 Persona: {persona_pass}/{len(PERSONA_PROBES)} pass")


def chat():
    """Interactive REPL — talk to nami-lm using saved weights.
    Type the question, press Enter, see the model's completion.
    'q' / 'quit' / Ctrl-D to exit."""
    model, tok = _load_for_inference()
    print("\n🌊 nami-lm chat — type a question, q/quit to exit\n")
    while True:
        try:
            q = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        # Strip any trailing question marks; the corpus uses 「？」
        # between question and answer, so we re-add it consistently.
        q = q.rstrip("?？") + "？"
        ids = tok.encode(q)
        if not ids:
            print("   (couldn't encode that — chars not in vocab)\n")
            continue
        gen = model.generate(ids, max_new=40, temperature=0.05)
        answer = tok.decode(gen[len(ids):])
        print(f"🌊 {answer}\n")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
    elif "--chat" in sys.argv:
        chat()
    elif "--auto" in sys.argv:
        train(epochs=9999, time_budget=TIME_BUDGET)
    else:
        train()
