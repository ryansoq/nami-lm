#!/usr/bin/env python3
"""
Phase 0 corpus extractor.

Pulls a tiny slice (~5KB) of Nami's identity from clawd:
  - IDENTITY.md (who Nami is)
  - SOUL.md (Nami's values)
  - MEMORY.md core sections (relationships, origin)

Output: data/phase0_corpus.jsonl with one {"text": "..."} per line.

Each line is a self-contained chunk the model will train on as a
single autoregressive sequence. Phase 0 keeps it small so the
bootstrap loop is fast.

Usage:
    python3 extract_corpus.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

CLAWD = Path.home() / "clawd"
OUT = Path(__file__).parent / "data" / "phase0_corpus.jsonl"


def strip_frontmatter(text: str) -> str:
    """Remove ---...--- yaml frontmatter at the top of a markdown file."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :].lstrip()
    return text


def chunk_paragraphs(text: str, max_chars: int = 200) -> list[str]:
    """Split markdown into paragraph-ish chunks. Drop empty / heading-only."""
    paras = re.split(r"\n\s*\n", text)
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # Drop pure section headers (a line starting with #)
        if all(line.lstrip().startswith("#") or not line.strip()
               for line in p.splitlines()):
            continue
        # Truncate over-long paragraphs to keep training sequences short.
        # We chunk into max_chars-sized slices on sentence boundaries when
        # possible.
        if len(p) <= max_chars:
            out.append(p)
            continue
        # naive sentence split on 。.!?\n
        buf = []
        cur = ""
        for sentence in re.split(r"([。.!?\n])", p):
            if not sentence:
                continue
            if len(cur) + len(sentence) > max_chars and cur:
                buf.append(cur)
                cur = sentence
            else:
                cur += sentence
        if cur:
            buf.append(cur)
        out.extend(buf)
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    sources = [
        CLAWD / "IDENTITY.md",
        CLAWD / "SOUL.md",
        CLAWD / "MEMORY.md",
    ]

    chunks: list[str] = []
    seen: set[str] = set()
    for src in sources:
        if not src.exists():
            print(f"⚠️  missing {src}, skipping")
            continue
        text = strip_frontmatter(src.read_text(encoding="utf-8"))
        for chunk in chunk_paragraphs(text):
            if chunk in seen:
                continue
            seen.add(chunk)
            chunks.append(chunk)
        print(f"  {src.name}: {len(chunks)} cumulative chunks")

    total_bytes = sum(len(c.encode("utf-8")) for c in chunks)
    print(f"\nTotal: {len(chunks)} chunks, {total_bytes:,} bytes "
          f"({total_bytes / 1024:.1f} KB)")

    with open(OUT, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
    print(f"📝 wrote {OUT}")


if __name__ == "__main__":
    main()
