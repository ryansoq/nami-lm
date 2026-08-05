"""Diagnostic ONLY — does not modify eval.py or the frozen metric.

Measures how many of the 51 probes are marked degenerate by rules that were
designed for Chinese degeneration ("的的的", "的？的？") but fire on ordinary
Latin text ("Claude Code" -> 'de' twice; the ticker "QQQ" -> char triple).

Reports current strict vs strict-under-a-CJK-scoped detector, so the size of
the artifact is a number rather than an argument.
"""
import sys
sys.path.insert(0, "/home/ymchang/nami-lm")
sys.path.insert(0, "/home/ymchang/nami-backpack/projects/numpy-grad")

from pathlib import Path
import eval as ev
from train import GPTMini, WordTokenizer, BPETokenizer, USE_BPE, load_corpus, PERSONA_PROBES

_orig = ev._is_degenerate


def _is_cjk(ch):
    return "一" <= ch <= "鿿"


def patched(completion: str) -> bool:
    """Same three rules, but the repetition rules only fire on CJK.

    A repeated char/bigram signals degeneration in Chinese ("的的的") but is
    normal in Latin text (' t' in "test commit", 'de' in "Claude Code", 'QQ'
    in the ticker QQQ). The mid-sentence '？' rule is script-independent and
    is kept unchanged.
    """
    head = completion[:16]
    if not head:
        return True
    for i in range(len(head) - 2):
        if head[i] == head[i + 1] == head[i + 2] and _is_cjk(head[i]):
            return True
    if "？" in head[1:8] or "?" in head[1:8]:
        return True
    counts = {}
    for i in range(len(head) - 1):
        b = head[i:i + 2]
        if len(b) == 2 and _is_cjk(b[0]) and _is_cjk(b[1]):
            counts[b] = counts.get(b, 0) + 1
    return any(c >= 2 for c in counts.values())


if __name__ == "__main__":
    corpus, _ = load_corpus()
    tok = BPETokenizer() if USE_BPE else WordTokenizer(corpus)
    model = GPTMini.load(str(Path("/home/ymchang/nami-lm/model_weights.json")))

    groups = [
        ("A core persona", PERSONA_PROBES),
        ("B extended", ev.EXTENDED_PERSONA),
        ("C topic", ev.TOPIC_PROBES),
        ("D soul", ev.SOUL_PROBES),
    ]
    tot_cur = tot_fix = tot_n = 0
    for label, probes in groups:
        ev._is_degenerate = _orig
        cur = ev._run_probes(model, tok, probes, label, quiet=True)
        ev._is_degenerate = patched
        fix = ev._run_probes(model, tok, probes, label, quiet=True)
        print(f"{label:16s} strict {cur[0]:2d} -> {fix[0]:2d}   (of {cur[4]})")
        tot_cur += cur[0]
        tot_fix += fix[0]
        tot_n += cur[4]
    print("-" * 46)
    print(f"{'TOTAL':16s} strict {tot_cur:2d} -> {tot_fix:2d}   (of {tot_n})")
    print(f"artifact: {tot_fix - tot_cur} probes wrongly marked degenerate")
