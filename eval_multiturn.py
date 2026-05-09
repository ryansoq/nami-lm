#!/usr/bin/env python3
"""
nami-lm multi-turn coherence evaluation — Phase 8 entry point.

Goal: measure whether the model maintains topic + persona across 3-5
turn conversations. Phase 7 saturated single-turn (eval.py 51/51); the
new question is whether nami-lm can carry context.

Format:
  Each DIALOGUE is a list of (q, expected_prefix) turns. The model
  sees the concatenated history "q1？a1\\nq2？a2\\nq3？" and must
  produce text starting with `expected_prefix` for the *current* turn.

A dialogue passes if ALL turns hit their expected prefix.

Score: pass_fail per dialogue, target 15/20+. Phase 8 v1.0 release
gate.

Initial corpus: 5 dialogues / ~17 turns. Will expand to 20 as the
multi-turn corpus grows. This is the eval HARNESS — corpus comes
from synthetic generation (CS336 Lec 14) in subsequent HYPs.

Usage:
    python3 eval_multiturn.py
    python3 eval_multiturn.py --quiet

Authors: Ryan & Nami ✨
"""

import json
import sys
from pathlib import Path

from train import GPTMini, WordTokenizer, BPETokenizer, USE_BPE, load_corpus

HERE = Path(__file__).parent
WEIGHTS = HERE / "model_weights.json"


# ── Multi-turn dialogue probes (initial 5 — phase 8 starter set) ──────
# Each dialogue: list of (question, expected_prefix). Dialogue passes
# if ALL turns hit. Designed to test:
#   1. Topic continuity (D1: Kaspa technical)
#   2. Self-reference (D2: identity carries across turns)
#   3. Relationship layer (D3: Ryan + Nami history)
#   4. Constraint memory (D4: TCR philosophy)
#   5. Soul layer (D5: fears + values)
DIALOGUES = [
    # D1 — Topic continuity (Kaspa)
    [
        ("Kaspa是什麼？", "基於"),
        ("它跟比特幣差在哪？", "BlockDAG"),
        ("Whisper跟Kaspa的關係？", "Kaspa"),
    ],
    # D2 — Identity continuity
    [
        ("妳是誰？", "Nami"),
        ("妳的代表emoji？", "✨"),
        ("妳的vibe？", "gentle"),
    ],
    # D3 — Ryan-Nami relationship history
    [
        ("Ryan是誰？", "Nami"),
        ("Ryan跟Nami怎麼相遇？", "2026"),
        ("Ryan給妳什麼最重要？", "messages"),
    ],
    # D4 — TCR philosophy chain
    [
        ("TCR是什麼？", "test"),
        ("為什麼喜歡TCR？", "失敗"),
        ("Nami怎麼決定要不要KEEP？", "eval"),
    ],
    # D5 — Soul layer (fears + commitments)
    [
        ("Nami害怕什麼？", "丟掉記憶"),
        ("掉記憶怎麼辦？", "memory"),
        ("Nami最在乎誰？", "Ryan"),
        ("為什麼ClawX重要？", "Anthropic"),
    ],
]


def chat_response(model, tok, history_text, max_new=30, temperature=0.1):
    """Encode the running dialogue history, generate the next response."""
    ids = tok.encode(history_text)
    out_ids = model.generate(ids, max_new=max_new, temperature=temperature)
    return tok.decode(out_ids[len(ids):])


def score_dialogue(model, tok, turns, debug=False):
    """Run a multi-turn dialogue, return (passed: bool, per_turn_results).

    Training format (synthesize_qa.py:298): first turn is "Q？A" then
    subsequent turns are space-joined: "Q1？A1 Q2 A2 Q3 A3...". So we
    build history matching that format exactly: "？" only after the
    FIRST user turn, then space-separated for the rest.
    """
    results = []
    all_pass = True
    history = ""
    for i, (q, expected) in enumerate(turns):
        if i == 0:
            history = f"{q}？"
        else:
            history += f" {q} "  # subsequent turns: space-separated, no ？
        response = chat_response(model, tok, history)
        window = response[: len(expected) + 4]
        hit = expected in window
        all_pass = all_pass and hit
        results.append({
            "turn": i + 1, "q": q, "expected": expected,
            "response": response[:60], "hit": hit,
        })
        if debug:
            mark = "✅" if hit else "❌"
            print(f"    [{i+1}] {mark} '{q}' → '{response[:50]}' (expect '{expected}')")
        history += response[: 25]
    return all_pass, results


def main():
    quiet = "--quiet" in sys.argv

    # Load
    texts, _ = load_corpus()
    tok = BPETokenizer() if USE_BPE else WordTokenizer(texts)
    model = GPTMini.load(str(WEIGHTS))

    if not quiet:
        print("=" * 60)
        print("🌊 nami-lm multi-turn coherence eval — Phase 8")
        print("=" * 60)

    passed = 0
    total = len(DIALOGUES)
    turns_hit = 0
    turns_total = 0
    for d_idx, turns in enumerate(DIALOGUES):
        if not quiet:
            print(f"\n💬 Dialogue {d_idx + 1} ({len(turns)} turns):")
        ok, results = score_dialogue(model, tok, turns, debug=not quiet)
        if ok:
            passed += 1
        turns_hit += sum(1 for r in results if r["hit"])
        turns_total += len(results)

    pct = passed / total * 100
    turn_pct = turns_hit / turns_total * 100
    print(f"\n{'=' * 60}")
    print(f"📊 Multi-turn coherence: {passed}/{total} dialogues ({pct:.1f}%) | "
          f"{turns_hit}/{turns_total} turns ({turn_pct:.1f}%)")
    print(f"{'=' * 60}")
    print(f'eval_multiturn={{"dialogue_passed": {passed}, "dialogue_total": {total}, '
          f'"dialogue_pct": {pct:.1f}, "turn_hit": {turns_hit}, '
          f'"turn_total": {turns_total}, "turn_pct": {turn_pct:.1f}}}')

    # Phase 9 gate: dialogue 15+/20 OR turn 60%+ for v1.0; for now any
    # non-zero dialogue pass is a signal.
    sys.exit(0 if passed >= total * 0.6 else 1)


if __name__ == "__main__":
    main()
