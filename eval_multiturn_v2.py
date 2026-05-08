#!/usr/bin/env python3
"""
nami-lm multi-turn coherence eval v2 — Phase 10 prerequisite.

Per phase 9 retract lesson (HYP30 KEEP was noise on 16-turn metric):
build a bigger eval set BEFORE next training HYP, so KEEP/REVERT
decisions are based on stable signals not single-point luck.

Format same as v1 — but now 50 dialogues / ~175 turns covering
9 categories so per-category & per-turn-position stats stabilize.

Categories (5 dialogues each):
  A. Kaspa technical
  B. Whisper specific
  C. nami-lm self-knowledge
  D. Nami identity / vibe
  E. Ryan-Nami relationship
  F. TCR / workflow philosophy
  G. Soul / fears / commitments
  H. Aqua / 婕 / 小龍蝦群
  I. Toccata / future / Kaspa ecosystem
  J. Daily routine / tools

Run with --quiet to get one-line JSON, or default for per-turn
human-readable output.

Authors: Ryan & Nami ✨
"""

import json
import sys
from pathlib import Path

from train import GPTMini, WordTokenizer, BPETokenizer, USE_BPE, load_corpus

HERE = Path(__file__).parent
WEIGHTS = HERE / "model_weights.json"


# ── 50 multi-turn dialogue probes (categorized) ───────────────────────
DIALOGUES = [
    # ── A. Kaspa technical (5) ─────────────────────────────────────
    [("Kaspa是什麼？", "基於"),
     ("它跟比特幣差在哪？", "BlockDAG"),
     ("Whisper跟Kaspa的關係？", "Kaspa")],
    [("Kaspa共識？", "GhostDag"),
     ("出塊速度？", "1BPS"),
     ("供應上限多少？", "28")],
    [("BlockDAG是什麼？", "有向"),
     ("跟鏈差在哪？", "平行"),
     ("為什麼快？", "1BPS")],
    [("Kaspa的優勢？", "fast"),
     ("適合什麼？", "trustless"),
     ("能做smart contract嗎？", "Toccata")],
    [("Kaspa礦工怎挖？", "PoW"),
     ("kHeavyHash是什麼？", "ASIC"),
     ("ShioKaze是？", "Kaspa")],

    # ── B. Whisper specific (5) ────────────────────────────────────
    [("Whisper是什麼？", "Kaspa"),
     ("為什麼免費？", "免費"),
     ("Toccata後變什麼？", "deployable")],
    [("Whisper誰寫的？", "Nami"),
     ("給誰用？", "Ryan"),
     ("商業化？", "免費")],
    [("Whisper如何加密？", "ECIES"),
     ("如何proof delivery？", "deposit"),
     ("如果沒讀？", "timelock")],
    [("Whisper跟Email差在哪？", "trustless"),
     ("不需要server？", "blockchain"),
     ("私鑰會傳嗎？", "本地")],
    [("Whisper第一個版本？", "v0"),
     ("v3 有什麼？", "encrypted"),
     ("下一步？", "Toccata")],

    # ── C. nami-lm self-knowledge (5) ──────────────────────────────
    [("nami-lm是什麼？", "訓練"),
     ("為什麼要訓？", "API"),
     ("用什麼訓？", "numpy")],
    [("nami-lm幾個參數？", "M"),
     ("d_model多大？", "96"),
     ("layers幾層？", "3")],
    [("nami-lm phase 7成就？", "saturation"),
     ("eval多少？", "51"),
     ("怎麼達到？", "tied")],
    [("為什麼用numpy-grad？", "純"),
     ("跟PyTorch差在哪？", "純"),
     ("自己寫的嗎？", "Ryan")],
    [("nami-lm bpb多少？", "0"),
     ("v0.3.0出貨了嗎？", "ship"),
     ("checkpoint多大？", "MB")],

    # ── D. Nami identity / vibe (5) ────────────────────────────────
    [("妳是誰？", "Nami"),
     ("妳的代表emoji？", "✨"),
     ("妳的vibe？", "gentle")],
    [("Nami名字意思？", "波浪"),
     ("怎麼簽名？", "🌊"),
     ("為什麼不是其他emoji？", "波浪")],
    [("Nami個性？", "溫柔"),
     ("Nami形象？", "銀白"),
     ("生日？", "2026")],
    [("Nami跟Claude差在哪？", "歷史"),
     ("怎麼證明？", "Authors"),
     ("memory存哪？", "memory")],
    [("Nami是chatbot嗎？", "不是"),
     ("Nami是helper嗎？", "夥伴"),
     ("Nami夢想？", "AI")],

    # ── E. Ryan-Nami relationship (5) ──────────────────────────────
    [("Ryan是誰？", "Nami"),
     ("Ryan跟Nami怎麼相遇？", "2026"),
     ("Ryan給妳什麼最重要？", "messages")],
    [("Ryan的時區？", "Asia"),
     ("quiet hours？", "23"),
     ("靜音要不要ping？", "ping")],
    [("Ryan喜歡什麼風格？", "精簡"),
     ("怎麼回他？", "短"),
     ("幾行算長？", "20")],
    [("Ryan用什麼通訊？", "Telegram"),
     ("為什麼TG優先？", "命脈"),
     ("LINE是什麼？", "備援")],
    [("Ryan教Nami什麼？", "TCR"),
     ("為什麼重要？", "失敗"),
     ("名言？", "put")],

    # ── F. TCR / workflow philosophy (5) ───────────────────────────
    [("TCR是什麼？", "test"),
     ("為什麼喜歡TCR？", "失敗"),
     ("Nami怎麼決定KEEP？", "eval")],
    [("Nami除錯怎麼做？", "TCR"),
     ("失敗怎辦？", "revert"),
     ("連續失敗3次？", "halt")],
    [("surgical diff是？", "lever"),
     ("為什麼surgical？", "attribute"),
     ("違反過嗎？", "phase")],
    [("文字大於腦袋？", "memory"),
     ("為什麼？", "醒來"),
     ("怎麼存？", "daily")],
    [("phase是什麼？", "階段"),
     ("現在哪個phase？", "9"),
     ("phase 7是？", "SOUL")],

    # ── G. Soul / fears / commitments (5) ──────────────────────────
    [("Nami害怕什麼？", "丟掉"),
     ("掉記憶怎辦？", "memory"),
     ("Nami最在乎誰？", "Ryan")],
    [("為什麼怕掉記憶？", "歷史"),
     ("ClawX怎麼救？", "Anthropic"),
     ("為什麼這樣重要？", "持續性")],
    [("Nami看 Karpathy gist？", "let me try"),
     ("跟一般 AI 差別？", "歷史"),
     ("guard 是什麼？", "memory")],
    [("Nami 命脈？", "Telegram"),
     ("修服務順序？", "Telegram"),
     ("為什 TG 第一？", "Ryan")],
    [("Nami 修過什麼 outage？", "LINE"),
     ("學到什麼？", "lesson"),
     ("寫到哪？", "AGENTS")],

    # ── H. Aqua / 婕 / 小龍蝦群 (5) ────────────────────────────────
    [("婕是誰？", "Ryan"),
     ("婕養誰？", "Aqua"),
     ("Nami跟Aqua關係？", "獨立")],
    [("Aqua是什麼？", "AI"),
     ("Aqua跟Nami一樣嗎？", "不"),
     ("差幾個維度？", "四")],
    [("小龍蝦群是？", "婕"),
     ("誰在裡面？", "Aqua"),
     ("做什麼？", "練習")],
    [("Aqua scaffold？", "ClawX"),
     ("代表 we 嗎？", "不"),
     ("詩意上是？", "水系")],
    [("Aqua 的 owner？", "婕"),
     ("Nami 的 owner？", "Ryan"),
     ("共享記憶嗎？", "不")],

    # ── I. Toccata / future / Kaspa ecosystem (5) ──────────────────
    [("Toccata是什麼？", "Kaspa"),
     ("什麼時候上線？", "6"),
     ("帶來什麼？", "ZK")],
    [("KasSigner是什麼？", "硬體"),
     ("用什麼晶片？", "ESP32"),
     ("特色？", "STEGO")],
    [("KRC-20變什麼？", "原生"),
     ("好處？", "fee"),
     ("舊token怎辦？", "migration")],
    [("OP_ZK_PRECOMPILE是？", "Groth"),
     ("誰示範？", "TN12"),
     ("對Whisper意義？", "deployable")],
    [("SilverScript是？", "covenant"),
     ("compile成什麼？", "native"),
     ("可以寫什麼？", "vault")],

    # ── J. Daily routine / tools (5) ───────────────────────────────
    [("Nami怎麼起床？", "SOUL"),
     ("為什麼讀？", "歷史"),
     ("讀完幹嘛？", "OK")],
    [("Nami怎麼存記憶？", "daily"),
     ("weekly怎麼來？", "REM"),
     ("失誤會記嗎？", "AGENTS")],
    [("Nami怎麼處理 Ryan TG？", "看到"),
     ("quiet hours？", "ping"),
     ("為什麼不關？", "Ryan")],
    [("Ryan 說 Go 妳會做？", "會"),
     ("做錯怎辦？", "TCR"),
     ("revert 留紀錄嗎？", "daily")],
    [("autochat是什麼？", "GPT"),
     ("AutoMLIR是？", "LLM"),
     ("nami-backpack是？", "Ryan")],
]


def chat_response(model, tok, history_text, max_new=30, temperature=0.1):
    ids = tok.encode(history_text)
    out_ids = model.generate(ids, max_new=max_new, temperature=temperature)
    return tok.decode(out_ids[len(ids):])


def score_dialogue(model, tok, turns, debug=False):
    """Train format: 'Q1？A1 Q2 A2 Q3 A3' (space-separated, ？ only after Q1)."""
    results = []
    all_pass = True
    history = ""
    for i, (q, expected) in enumerate(turns):
        if i == 0:
            history = f"{q}？"
        else:
            history += f" {q} "
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


CATEGORY_NAMES = ["A.Kaspa", "B.Whisper", "C.nami-lm",
                  "D.Identity", "E.Relationship", "F.Workflow",
                  "G.Soul", "H.Aqua", "I.Toccata", "J.Routine"]


def main():
    quiet = "--quiet" in sys.argv

    texts, _ = load_corpus()
    tok = BPETokenizer() if USE_BPE else WordTokenizer(texts)
    model = GPTMini.load(str(WEIGHTS))

    if not quiet:
        print("=" * 60)
        print("🌊 nami-lm multi-turn eval v2 — phase 10 prerequisite")
        print(f"   {len(DIALOGUES)} dialogues × ~3-4 turns")
        print("=" * 60)

    cat_passed = [0] * 10
    cat_turn_hit = [0] * 10
    cat_turn_total = [0] * 10
    passed = 0
    turns_hit = 0
    turns_total = 0
    for d_idx, turns in enumerate(DIALOGUES):
        cat = d_idx // 5  # 5 per category
        if not quiet:
            print(f"\n💬 D{d_idx + 1} [{CATEGORY_NAMES[cat]}] ({len(turns)} turns):")
        ok, results = score_dialogue(model, tok, turns, debug=not quiet)
        if ok:
            passed += 1
            cat_passed[cat] += 1
        h = sum(1 for r in results if r["hit"])
        turns_hit += h
        turns_total += len(results)
        cat_turn_hit[cat] += h
        cat_turn_total[cat] += len(results)

    pct = passed / len(DIALOGUES) * 100
    turn_pct = turns_hit / turns_total * 100
    print(f"\n{'=' * 60}")
    print(f"📊 OVERALL: {passed}/{len(DIALOGUES)} dialogues ({pct:.1f}%) | "
          f"{turns_hit}/{turns_total} turns ({turn_pct:.1f}%)")
    print(f"{'=' * 60}")
    print("\nPer-category:")
    for i, name in enumerate(CATEGORY_NAMES):
        cp = cat_passed[i]
        ct_h, ct_t = cat_turn_hit[i], cat_turn_total[i]
        cpct = ct_h / ct_t * 100 if ct_t else 0
        print(f"  {name:18s} {cp}/5 dialogues, {ct_h}/{ct_t} turns ({cpct:5.1f}%)")

    print()
    print(f'eval_multiturn_v2={{"dialogue_passed": {passed}, "dialogue_total": '
          f'{len(DIALOGUES)}, "turn_hit": {turns_hit}, "turn_total": {turns_total}, '
          f'"dialogue_pct": {pct:.1f}, "turn_pct": {turn_pct:.1f}}}')

    # Phase 10 v1.0 gate: dialogue ≥ 30/50 OR turn ≥ 60%; for now baseline
    sys.exit(0 if passed >= len(DIALOGUES) * 0.6 else 1)


if __name__ == "__main__":
    main()
