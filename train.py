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
def load_corpus() -> list[str]:
    """Load Q&A pairs from phase0_qa.jsonl, format each as 'q？a'.

    Why this format: autochat's TRAINING_DATA used the same pattern
    ('question？answer' as a single autoregressive sequence) and that
    learned 92/92 acc. We're copying the proven format. The persona
    probe at end of train() prefixes the question and asks the model
    to complete the answer.
    """
    if not QA_CORPUS.exists():
        raise FileNotFoundError(
            f"{QA_CORPUS} missing — run `python3 synthesize_qa.py` first")
    out = []
    with open(QA_CORPUS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d["q"].strip(), d["a"].strip()
            # Strip a trailing "?/？" from q if present so the join puts one
            # consistent delimiter between Q and A
            q = q.rstrip("?？")
            out.append(f"{q}？{a}")
    return out


class WordTokenizer:
    """Phase 0 tokenizer — same hybrid rule as autochat: ASCII alpha
    runs and digit runs become single tokens, everything else
    (Chinese, punctuation, whitespace) is char-level. Phase 1 replaces
    this with BPE."""

    def __init__(self, texts):
        all_tokens = set()
        for t in texts:
            all_tokens.update(self._tokenize(t))
        self.vocab = sorted(all_tokens)
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.name = "WordTokenizer"

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
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
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
        self.out_proj = Linear(d_model, vocab_size, bias=False)

        # HYP2 (autochat) Xavier init alignment — sqrt(2/(in+out))
        self._apply_xavier_init_to_linears()

    def _apply_xavier_init_to_linears(self):
        def reinit(linear):
            in_d, out_d = linear.W.data.shape
            scale = np.sqrt(2.0 / (in_d + out_d))
            linear.W.data = (np.random.randn(in_d, out_d) * scale).astype(
                linear.W.data.dtype)
        reinit(self.head.layers[0]); reinit(self.head.layers[2])
        reinit(self.out_proj)
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
        logits = self.out_proj(x)
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


# =============================================================================
# Training loop — mini-batch length-bucketed (autochat HYP5)
# =============================================================================
def compute_bpb(loss, tokenizer, texts):
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    avg = total_bytes / total_tokens if total_tokens > 0 else 1.0
    return loss / math.log(2) / avg


TIME_BUDGET = 30 * 60  # phase 2: 30 min so larger corpus can fit more epochs


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

    corpus = load_corpus()
    print(f"📝 Corpus: {len(corpus)} chunks, "
          f"{sum(len(c.encode('utf-8')) for c in corpus):,} bytes")

    if USE_BPE:
        tokenizer = BPETokenizer()
        print(f"📊 Vocab: {tokenizer.vocab_size} tokens (BPE — phase 1+)")
    else:
        tokenizer = WordTokenizer(corpus)
        print(f"📊 Vocab: {tokenizer.vocab_size} tokens (WordTokenizer — phase 0)")

    # Token-length stats so we know whether max_seq_len is sane
    encoded = [tokenizer.encode(c) for c in corpus]
    encoded = [ids for ids in encoded if len(ids) >= 2]
    max_len = max(len(ids) for ids in encoded)
    median_len = sorted(len(ids) for ids in encoded)[len(encoded) // 2]
    print(f"📏 Seq len: median={median_len}, max={max_len}")

    # Cap max_seq_len so phase-0 epoch time stays ≤ autochat scale
    max_seq_len = min(max(max_len, 32), 64)

    # Reverted to autochat HYP11 sweet spot after phase-3 d=128
    # experiment showed scaling up just under-fits at our compute
    # budget (Chinchilla math: more params without proportional
    # compute = worse). Phase 0 setup remains the floor.
    d_model, d_ff, num_heads, num_layers = 96, 256, 6, 3
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

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.02)

    # Truncate sequences over max_seq_len (rare for phase 0 short chunks)
    encoded = [ids[:max_seq_len] for ids in encoded if len(ids) >= 2]

    # Length-bucket batches (autochat HYP5)
    BATCH_SIZE = 8
    length_buckets: dict[int, list[list[int]]] = {}
    for ids in encoded:
        length_buckets.setdefault(len(ids), []).append(ids)
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
    expected_epochs = (
        min(epochs, int(time_budget / 7.0)) if time_budget else epochs)

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
        cur_lr = max(cur_lr, lr * 0.01)
        opt.lr = cur_lr

        total_loss = 0.0
        n_seqs = 0

        epoch_buckets = bucket_keys[:]
        np.random.shuffle(epoch_buckets)
        for L in epoch_buckets:
            seqs = length_buckets[L][:]
            np.random.shuffle(seqs)
            for i in range(0, len(seqs), BATCH_SIZE):
                batch = seqs[i:i + BATCH_SIZE]
                inputs = np.array([s[:-1] for s in batch], dtype=np.int64)
                targets = np.array([s[1:] for s in batch], dtype=np.int64)
                opt.zero_grad()
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=0.5)
                opt.step()
                total_loss += float(loss.data) * len(batch)
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


def probe(prompt_list=PERSONA_PROBES):
    """Quick probe — load saved weights and answer the persona probes.
    For phase 0, no save/load yet; this is just prompt-style printing
    after a fresh train.  Will be wired to a checkpoint in phase 4."""
    print("Probe mode: phase 0 doesn't persist weights yet — run train.py "
          "directly to see persona output at the end.")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
    elif "--auto" in sys.argv:
        train(epochs=9999, time_budget=TIME_BUDGET)
    else:
        train()
