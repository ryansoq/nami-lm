"""
Byte-pair encoding tokenizer — pure Python, the same algorithm GPT-2
and Llama use, just trimmed to its bones.

Train:
    bpe = BPE()
    bpe.train(corpus, vocab_size=1024)
    bpe.save(Path("tokenizer/vocab"))

Use:
    bpe = BPE.load(Path("tokenizer/vocab"))
    ids = bpe.encode("妳是誰？")          # list[int]
    text = bpe.decode(ids)               # str

Algorithm (training):
1. Start from 256 single bytes (UTF-8). Every text becomes a list
   of byte ids.
2. For each piece in the corpus, count adjacent pair frequencies.
3. Pick the most frequent pair → assign it a new id (next-vocab-id).
4. Replace every occurrence of that pair in every piece with the new id.
5. Repeat until vocab_size reached.

Algorithm (encoding):
1. Start from raw bytes.
2. Apply merges in training order, greedily replacing the highest-priority
   adjacent pair until none of the trained merges apply.
3. Return list of ids.

Algorithm (decoding):
1. For each id, look up its byte sequence (recursively flatten merges).
2. Concatenate, decode as UTF-8 (errors='replace' for partial bytes).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


class BPE:
    def __init__(self):
        # vocab: id → bytes. ids 0..255 are single bytes. 256+ are merges.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # merges: list of (left_id, right_id) in the order they were learned;
        # encoder applies them in this order so deterministic.
        self.merges: list[tuple[int, int]] = []
        # cache for encode() — derived from merges, rebuilt on load
        self._merge_rank: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def train(self, corpus: list[str], vocab_size: int = 1024,
              min_pair_freq: int = 2, verbose: bool = True) -> None:
        """Learn merges from a list of text strings."""
        # Encode each piece as a list of byte ids (initial state).
        pieces: list[list[int]] = [list(t.encode("utf-8")) for t in corpus]

        next_id = 256
        while next_id < vocab_size:
            # Count adjacent pair frequencies across all pieces
            pair_counts: Counter[tuple[int, int]] = Counter()
            for piece in pieces:
                for i in range(len(piece) - 1):
                    pair_counts[(piece[i], piece[i + 1])] += 1

            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_pair_freq:
                break

            # Register the merge
            self.merges.append(best_pair)
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (next_id % 100 == 0 or next_id < 260):
                preview = self.vocab[next_id].decode("utf-8", errors="replace")
                print(f"  vocab {next_id}: merged {best_pair} (count {best_count}) "
                      f"→ {preview!r}")

            # Apply the merge to all pieces (replace adjacent pairs)
            for k in range(len(pieces)):
                pieces[k] = self._merge_in_place(pieces[k], best_pair, next_id)

            next_id += 1

        # Build the encoder cache (lower rank = applied earlier = higher priority)
        self._merge_rank = {p: i for i, p in enumerate(self.merges)}

        if verbose:
            print(f"\nTrained BPE: {len(self.vocab)} tokens, {len(self.merges)} merges")

    @staticmethod
    def _merge_in_place(piece: list[int], pair: tuple[int, int],
                        new_id: int) -> list[int]:
        """Replace every adjacent occurrence of `pair` in `piece` with `new_id`."""
        out: list[int] = []
        i = 0
        n = len(piece)
        while i < n:
            if i + 1 < n and piece[i] == pair[0] and piece[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(piece[i])
                i += 1
        return out

    # -----------------------------------------------------------------
    # Encoding
    # -----------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        if not self._merge_rank:
            return ids  # no merges learned

        # Greedy: at each step find the lowest-rank pair present and merge it.
        while True:
            # Find best pair (lowest rank) present in ids
            best_rank = None
            best_idx = -1
            for i in range(len(ids) - 1):
                rank = self._merge_rank.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            # Find the merge result id
            pair = self.merges[best_rank]
            # Look up the merged id (the merge at rank `best_rank` produced
            # vocab id 256 + best_rank)
            new_id = 256 + best_rank
            # Replace all adjacent occurrences of this pair (not just at best_idx
            # — replacing all in one pass is faster and matches training)
            ids = self._merge_in_place(ids, pair, new_id)
        return ids

    # -----------------------------------------------------------------
    # Decoding
    # -----------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        out = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return out.decode("utf-8", errors="replace")

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        # vocab: only save merges (256 base bytes are implicit)
        with open(dir_path / "merges.json", "w", encoding="utf-8") as f:
            json.dump(self.merges, f)
        # Pretty preview of the learned vocabulary, for human inspection
        with open(dir_path / "vocab_preview.txt", "w", encoding="utf-8") as f:
            for i in range(256, len(self.vocab)):
                preview = self.vocab[i].decode("utf-8", errors="replace")
                f.write(f"{i}\t{preview!r}\n")

    @classmethod
    def load(cls, dir_path: Path) -> "BPE":
        dir_path = Path(dir_path)
        bpe = cls()
        with open(dir_path / "merges.json", encoding="utf-8") as f:
            merges = json.load(f)
        bpe.merges = [tuple(m) for m in merges]
        next_id = 256
        for left, right in bpe.merges:
            bpe.vocab[next_id] = bpe.vocab[left] + bpe.vocab[right]
            next_id += 1
        bpe._merge_rank = {p: i for i, p in enumerate(bpe.merges)}
        return bpe
