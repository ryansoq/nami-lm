#!/usr/bin/env python3
"""
Train a BPE tokenizer on data/phase0_qa.jsonl.

Output: tokenizer/merges.json + tokenizer/vocab_preview.txt

Usage:
    python3 synthesize_qa.py    # build phase 0 Q&A corpus first
    python3 train_bpe.py        # train BPE
    python3 train_bpe.py --test # quick round-trip check on the corpus
"""

import json
import sys
from pathlib import Path

from tokenizer.bpe import BPE

HERE = Path(__file__).parent
QA_CORPUS = HERE / "data" / "phase0_qa.jsonl"
TOK_DIR = HERE / "tokenizer"
VOCAB_SIZE = 1024


def load_corpus_lines() -> list[str]:
    """Load Q&A pairs as 'q？a' strings (same format as train.py)."""
    lines = []
    with open(QA_CORPUS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d["q"].strip().rstrip("?？"), d["a"].strip()
            lines.append(f"{q}？{a}")
    return lines


def round_trip_check(bpe: BPE, lines: list[str]) -> tuple[int, int, list[str]]:
    """Re-encode each line and check it decodes back exactly. Returns
    (n_total, n_lossless, list of mismatches)."""
    mismatches = []
    for text in lines:
        ids = bpe.encode(text)
        roundtrip = bpe.decode(ids)
        if roundtrip != text:
            mismatches.append((text, roundtrip))
    return len(lines), len(lines) - len(mismatches), mismatches


def main():
    test_only = "--test" in sys.argv
    lines = load_corpus_lines()
    print(f"Corpus: {len(lines)} lines, "
          f"{sum(len(l.encode('utf-8')) for l in lines):,} bytes")

    if test_only:
        bpe = BPE.load(TOK_DIR)
        print(f"Loaded BPE: vocab={bpe.vocab_size}, merges={len(bpe.merges)}")
    else:
        bpe = BPE()
        print(f"Training BPE → vocab_size={VOCAB_SIZE}")
        bpe.train(lines, vocab_size=VOCAB_SIZE)
        bpe.save(TOK_DIR)
        print(f"Saved to {TOK_DIR}/merges.json")

    # Round-trip check
    n_total, n_ok, mismatches = round_trip_check(bpe, lines)
    print(f"\nRound-trip: {n_ok}/{n_total} lossless")
    if mismatches:
        print(f"Sample mismatch:\n  in:  {mismatches[0][0]!r}\n  out: {mismatches[0][1]!r}")
        sys.exit(1)

    # Show a few tokenizations to sanity-check
    print("\nSample tokenizations (first 5 lines):")
    for text in lines[:5]:
        ids = bpe.encode(text)
        # Show first 12 token previews
        previews = [bpe.vocab[i].decode("utf-8", errors="replace") for i in ids[:12]]
        print(f"  {text!r:40s} → {len(ids)} tokens: {previews}")

    # Compression ratio
    avg_bytes = sum(len(l.encode("utf-8")) for l in lines) / len(lines)
    avg_tokens = sum(len(bpe.encode(l)) for l in lines) / len(lines)
    print(f"\nAvg bytes/line: {avg_bytes:.1f}")
    print(f"Avg tokens/line: {avg_tokens:.1f}")
    print(f"Bytes per token: {avg_bytes / avg_tokens:.2f}")


if __name__ == "__main__":
    main()
