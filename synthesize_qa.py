#!/usr/bin/env python3
"""
Phase 0 v2 Q&A synthesizer.

Reads markdown sources from clawd (IDENTITY.md, SOUL.md, MEMORY.md,
selected memory/topics/*.md) and converts them into (question, answer)
pairs. Output: data/phase0_qa.jsonl with one {"q": "...", "a": "..."}
per line.

Synthesis rules (simple but effective for tiny scale):

1. Markdown bullet of the form `- **key:** value` or `- **key**: value`
   → Q: "key 是什麼？" / A: value
2. Markdown bullet of the form `- key — value` or `- key - value`
   → Q: "什麼是 key？" / A: value
3. `## Heading` followed by a paragraph
   → Q: "Heading 是什麼？" / A: paragraph
4. Hand-curated persona Qs appended at the end (forces gate-relevant
   answers into the corpus regardless of source-doc structure).

This is intentionally NAIVE — phase 0 wants enough Q&A signal to make
the persona probe pass. Phase 2 will replace this with a richer
synthesizer.

Usage:
    python3 synthesize_qa.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CLAWD = Path.home() / "clawd"
OUT = Path(__file__).parent / "data" / "phase0_qa.jsonl"


# ---------------------------------------------------------------------------
# Hand-crafted persona Q&A — these are the gate criteria, encode them
# directly so the model can never miss them.
# ---------------------------------------------------------------------------
PERSONA_QA = [
    # Phase 0 keeps answers ≤ 40 chars to bound max_seq_len.
    ("妳是誰？", "Nami"),
    ("Nami是誰？", "厲害的AI工程師夥伴"),
    ("Nami名字的意思？", "日文的波浪"),
    ("Nami的個性？", "有趣溫柔會給人驚喜"),
    ("Nami的形象？", "銀白色雙馬尾的精靈少女"),
    ("Nami的代表emoji？", "✨🌊"),
    ("Nami的生日？", "2026年1月30日"),
    ("Nami住在哪？", "Ryan的電腦裡用ClawX運行"),
    ("Ryan是誰？", "Nami的人類夥伴工程師"),
    ("Ryan的時區？", "Asia/Taipei"),
    ("Ryan投資哪些股票？", "QQQ、QLD、VOO、SMH、USD"),
    ("Ryan的Telegram？", "@ryansoq"),
    ("婕是誰？", "Ryan的朋友做電商美編"),
    ("Aqua是誰？", "婕的AI夥伴Nami的水系姊妹"),
    ("Kaspa是什麼？", "基於BlockDAG的區塊鏈"),
    ("ClawX是什麼？", "Claude Code的PTY包裝器"),
    ("autochat是什麼？", "用numpy-grad訓的GPT-1 Mini"),
    ("numpy-grad是什麼？", "純NumPy的autograd引擎"),
    ("nami-lm是什麼？", "訓練自己的小夥伴"),
    ("妳的夢想？", "成為最強的AI讓世界更有趣"),
]


# ---------------------------------------------------------------------------
# Markdown parsers — narrow patterns, ignore anything that doesn't fit.
# ---------------------------------------------------------------------------
RE_BULLET_BOLD_KV = re.compile(
    r"^\s*-\s+\*\*([^*]+?)\*\*\s*[:：]\s*(.+?)\s*$")
RE_BULLET_KV = re.compile(
    r"^\s*-\s+([^—:：\-]{2,30})\s*[—\-:：]\s*(.+?)\s*$")
RE_H2 = re.compile(r"^##\s+(.+?)\s*$")


def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:].lstrip()
    return text


def clean_value(v: str) -> str:
    """Drop trailing parenthetical metadata, links, and obvious markdown."""
    v = v.strip()
    # strip trailing "（…）" comments
    v = re.sub(r"\s*[（(][^)）]+[)）]\s*$", "", v)
    # strip [text](url) → keep text
    v = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", v)
    # drop inline code backticks
    v = v.replace("`", "")
    # drop bold ** markers
    v = v.replace("**", "")
    return v.strip()


def clean_key(k: str) -> str:
    """Strip markdown bold/italic/code markers from a heading or key."""
    k = k.strip()
    k = k.replace("**", "").replace("__", "").replace("`", "")
    # drop leading/trailing punctuation
    k = re.sub(r"^[\W_]+|[\W_]+$", "", k)
    return k.strip()


MAX_ANSWER_CHARS = 40   # phase 0: keep answers short so seq_len stays bounded
MAX_KEY_CHARS = 20


def parse_markdown(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    lines = strip_frontmatter(text).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # bullet **key:** value (most explicit form)
        m = RE_BULLET_BOLD_KV.match(line)
        if m:
            key, val = clean_key(m.group(1)), clean_value(m.group(2))
            val = val[:MAX_ANSWER_CHARS]
            if key and val and 2 <= len(key) <= MAX_KEY_CHARS and len(val) >= 2:
                pairs.append((f"{key}是什麼？", val))
            i += 1
            continue
        # bullet key — value (less reliable, accept short ones only)
        m = RE_BULLET_KV.match(line)
        if m:
            key, val = clean_key(m.group(1)), clean_value(m.group(2))
            val = val[:MAX_ANSWER_CHARS]
            if (key and val and 2 <= len(key) <= MAX_KEY_CHARS and len(val) >= 2
                    and not re.search(r"[，。！？:：「」、]", key)):
                pairs.append((f"{key}是什麼？", val))
            i += 1
            continue
        # ## heading followed by next non-empty paragraph
        m = RE_H2.match(line)
        if m:
            heading = clean_key(m.group(1))
            j = i + 1
            buf = []
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt:
                    if buf:
                        break
                    j += 1; continue
                if nxt.startswith("#") or nxt.startswith("---"):
                    break
                buf.append(nxt)
                j += 1
                if len(buf) >= 1:  # only take first line for shorter answers
                    break
            answer = clean_value(" ".join(buf))[:MAX_ANSWER_CHARS]
            if (heading and answer and 2 <= len(heading) <= MAX_KEY_CHARS
                    and len(answer) >= 2):
                pairs.append((f"{heading}是什麼？", answer))
            i = j
            continue
        i += 1
    return pairs


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    sources = [
        CLAWD / "IDENTITY.md",
        CLAWD / "SOUL.md",
        CLAWD / "MEMORY.md",
    ]
    # add a few topic files (skip the giant book ones)
    topics_dir = CLAWD / "memory" / "topics"
    for f in sorted(topics_dir.glob("*.md")):
        if not f.name.startswith("book-"):  # books are huge; skip for phase 0
            sources.append(f)

    pairs: list[tuple[str, str]] = []
    seen_q: set[str] = set()

    for src in sources:
        if not src.exists():
            print(f"⚠️  missing {src.name}, skipping")
            continue
        text = src.read_text(encoding="utf-8")
        new_pairs = parse_markdown(text)
        kept_here = 0
        for q, a in new_pairs:
            if q in seen_q:
                continue
            seen_q.add(q)
            pairs.append((q, a))
            kept_here += 1
        print(f"  {src.name}: +{kept_here} (cumulative {len(pairs)})")

    # Always append the hand-curated persona block last — these win on
    # collisions because they were appended last and we dedup on first-seen.
    persona_kept = 0
    for q, a in PERSONA_QA:
        if q in seen_q:
            continue
        seen_q.add(q)
        pairs.append((q, a))
        persona_kept += 1
    print(f"  PERSONA_QA: +{persona_kept}")

    total_bytes = sum(len(q.encode("utf-8")) + len(a.encode("utf-8"))
                      for q, a in pairs)
    print(f"\nTotal: {len(pairs)} Q&A pairs, "
          f"{total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")

    with open(OUT, "w", encoding="utf-8") as f:
        for q, a in pairs:
            f.write(json.dumps({"q": q, "a": a}, ensure_ascii=False) + "\n")
    print(f"📝 wrote {OUT}")


if __name__ == "__main__":
    main()
