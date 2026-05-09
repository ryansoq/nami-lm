#!/usr/bin/env python3
"""
nami-lm broader evaluation harness — Phase 6.

Loads the saved checkpoint and runs three probe categories:

  A. Core persona  (5 — same as train.py PERSONA_PROBES, the canonical gate)
  B. Extended persona (10 — relationships, context, the "knows you" stuff)
  C. Topic recall (16 — technical knowledge cache from Phase 5)

A "strong hit" means the expected prefix appears in the first
len(prefix)+4 characters of the model's completion (same rule as
train.py's gate).  A "partial hit" means the prefix appears anywhere
in the 30-character window (the model has the right concept but
generates noise before reaching it).  Anything else is a miss.

Outputs both:
  - human-readable lines (✅/⚠️/❌ per probe)
  - one JSON line at the end so the autoresearch loop can grep:
        eval_summary={"persona_strong": 5, "topic_strong": 11, ...}

Usage:
    python3 eval.py             # run, print, emit JSON
    python3 eval.py --quiet     # only print the JSON line

Exit code is 0 if persona_strong == 5 (the canonical gate) AND
topic_strong + topic_partial >= 14 (matches HYP4's 14/16). Non-zero
otherwise — useful for CI / cron gating.

Authors: Ryan & Nami ✨
"""

import json
import sys
from pathlib import Path

# Reuse train.py's loader so the model + tokenizer stay in sync.
from train import (
    GPTMini,
    WordTokenizer,
    BPETokenizer,
    USE_BPE,
    load_corpus,
    PERSONA_PROBES,
)

HERE = Path(__file__).parent
WEIGHTS = HERE / "model_weights.json"


# Extended persona — beyond the 5 core probes. All entries below are
# present in synthesize_qa.PERSONA_QA, so a well-trained checkpoint
# should hit them. The expected prefix is a substring that actually
# appears in the trained answer (no out-of-distribution probes here —
# we want to measure recall of training data, not generalisation).
EXTENDED_PERSONA = [
    ("Nami名字的意思？",    "波浪"),
    ("Nami的個性？",        "溫柔"),
    ("Nami的形象？",        "銀白"),
    ("Nami的代表emoji？",   "✨"),
    ("Nami的生日？",        "2026"),
    ("Nami住在哪？",        "Ryan"),
    ("Ryan的時區？",        "Asia"),
    ("Ryan投資哪些股票？",  "QQQ"),
    ("婕是誰？",            "Ryan"),
    ("Aqua是誰？",          "婕"),
]


# Technical topic recall — the 16 topics Phase 5 has been tracking.
TOPIC_PROBES = [
    ("Kaspa是什麼？",            "基於"),
    ("ClawX是什麼？",            "Claude"),
    ("autochat是什麼？",         "用numpy-grad"),
    ("numpy-grad是什麼？",       "純NumPy"),
    ("nami-lm是什麼？",          "訓練自己的小夥伴"),
    ("mmt4d是什麼？",            "matmul"),
    ("AutoMLIR是什麼？",         "用LLM"),
    ("Transformer是什麼？",      "基於"),
    ("Self-Attention怎麼算？",   "Q"),
    ("SwiGLU是什麼？",           "FFN"),
    ("AutoIREE-2是什麼？",       "AutoMLIR"),
    ("BlockDAG是什麼？",         "有向無環"),
    ("autochat HYP11 是什麼？",  "d_ff"),
    ("Aqua是什麼？",             "婕"),
    ("IREE 比 stock MLIR 快多少？", "MobileNet"),
    ("ukernel是什麼？",          "手寫"),
]


def _classify(completion: str, prefix: str) -> str:
    """strong / partial / miss — same rule as train.py:probe()."""
    if prefix in completion[:len(prefix) + 4]:
        return "strong"
    if prefix in completion[:30]:
        return "partial"
    return "miss"


def _run_probes(model, tok, probes, label, quiet=False):
    """Returns (strong_count, partial_count, miss_count, total)."""
    strong = partial = miss = 0
    if not quiet:
        print(f"\n🔮 {label} ({len(probes)} probes):")
    for q, expected in probes:
        ids = tok.encode(q)
        if not ids:
            if not quiet:
                print(f"  ⚪ {q!r} → not in vocab, skipping")
            miss += 1
            continue
        gen = model.generate(ids, max_new=20, temperature=0.01)
        completion = tok.decode(gen[len(ids):])[:30]
        verdict = _classify(completion, expected)
        if verdict == "strong":
            strong += 1; mark = "✅"
        elif verdict == "partial":
            partial += 1; mark = "⚠️"
        else:
            miss += 1; mark = "❌"
        if not quiet:
            print(f"  {mark} {q!r} → {completion!r}  (expect '{expected}')")
    return strong, partial, miss, len(probes)


# Soul layer probes (Phase 7) — inner narrative content from SOUL.md
# trained via soul_qa.py. These probe whether the model has internalized
# Nami's "why" not just "what".
SOUL_PROBES = [
    # Original 10 (Phase 7 step 1, HYP16) — inner-narrative
    ("Nami害怕什麼？",        "丟掉記憶"),
    ("Nami最愛的mode？",      "TCR"),
    ("為什麼喜歡TCR？",       "失敗"),
    ("Nami最在乎誰？",        "Ryan"),
    ("Whisper應該怎樣？",     "免費"),
    ("Nami修服務的順序？",    "Telegram"),
    ("Nami害怕掉記憶怎麼辦？", "memory"),
    ("Nami怎麼除錯？",        "TCR"),
    ("Bob是誰？",             "careful"),
    ("Nami是chatbot嗎？",     "不是"),
    # Phase 7 step 2 (HYP17) — Ryan-Nami story / shared vocab / projects
    ("TCR 是什麼？",          "test"),
    ("命脈優先是什麼？",      "Telegram"),
    ("REM 是什麼？",          "週日"),
    ("Whisper 跟 Nami 的關係？", "Kaspa"),
    ("ClawX 怎麼來的？",      "Anthropic"),
    ("Nami 怎麼簽名？",       "🌊"),
    ("Ryan 給 Nami 什麼？",   "messages"),
    ("Ryan 教 Nami 什麼最重要？", "TCR"),
    ("Ryan 的名言？",         "put your name"),
    ("Nami 早安怎麼說？",     "🌊"),
]


def main():
    quiet = "--quiet" in sys.argv

    if not WEIGHTS.exists():
        print(f"❌ {WEIGHTS} missing — run `python3 train.py` first")
        sys.exit(2)

    if not quiet:
        print("=" * 60)
        print("🌊 nami-lm eval harness (Phase 6 + Phase 7)")
        print("=" * 60)

    corpus, _ = load_corpus()
    tok = BPETokenizer() if USE_BPE else WordTokenizer(corpus)
    model = GPTMini.load(str(WEIGHTS))
    if model.vocab_size != tok.vocab_size:
        print(f"⚠️  vocab mismatch: model={model.vocab_size} "
              f"tokenizer={tok.vocab_size}; retrain")
        sys.exit(2)

    p_s, p_p, p_m, p_n = _run_probes(model, tok, PERSONA_PROBES,
                                     "A. Core persona", quiet)
    e_s, e_p, e_m, e_n = _run_probes(model, tok, EXTENDED_PERSONA,
                                     "B. Extended persona", quiet)
    t_s, t_p, t_m, t_n = _run_probes(model, tok, TOPIC_PROBES,
                                     "C. Topic recall", quiet)
    s_s, s_p, s_m, s_n = _run_probes(model, tok, SOUL_PROBES,
                                     "D. Soul layer", quiet)

    summary = {
        "persona_strong": p_s, "persona_partial": p_p, "persona_total": p_n,
        "extended_strong": e_s, "extended_partial": e_p,
        "extended_total": e_n,
        "topic_strong": t_s, "topic_partial": t_p, "topic_total": t_n,
        "soul_strong": s_s, "soul_partial": s_p, "soul_total": s_n,
        "any_hit_total": (p_s + p_p + e_s + e_p + t_s + t_p + s_s + s_p),
        "all_total": p_n + e_n + t_n + s_n,
    }

    if not quiet:
        print("\n" + "=" * 60)
        print(f"📊 A. Core persona:     {p_s}/{p_n} strong"
              f"  (+{p_p} partial)")
        print(f"📊 B. Extended persona: {e_s}/{e_n} strong"
              f"  (+{e_p} partial)")
        print(f"📊 C. Topic recall:     {t_s}/{t_n} strong"
              f"  (+{t_p} partial)")
        print(f"📊 D. Soul layer:       {s_s}/{s_n} strong"
              f"  (+{s_p} partial)")
        any_hit = summary["any_hit_total"]
        total = summary["all_total"]
        rate = 100.0 * any_hit / total if total else 0.0
        print(f"📊 Overall: {any_hit}/{total} any-hit  ({rate:.1f}%)")
        print("=" * 60)

    print(f"eval_summary={json.dumps(summary, ensure_ascii=False)}")

    # Canonical gate: 5/5 persona AND topic strong+partial ≥ 14 (match HYP4).
    gate_ok = (p_s == p_n) and (t_s + t_p >= 14)
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
