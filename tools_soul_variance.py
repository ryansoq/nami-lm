"""Diagnostic — is the soul layer's +/-3 spread model instability, or probe narrowness?

CTRL-1 showed soul scoring 18 (seed 42) vs 15 (seed 43) on an identical corpus,
while persona/extended/topic were bit-identical. Two very different explanations:

  (a) the model genuinely forgets things depending on init  -> a model problem
  (b) the probe accepts one string, the model says another equally correct one
      -> an eval problem, and the "instability" is an artifact

This loads both checkpoints and diffs their SOUL answers side by side so the
question is settled by reading what the model actually said, not by inference.

Touches nothing: loads the .bak checkpoints directly, never writes.
"""
import sys
sys.path.insert(0, "/home/ymchang/nami-lm")
sys.path.insert(0, "/home/ymchang/nami-backpack/projects/numpy-grad")

from train import GPTMini, WordTokenizer, BPETokenizer, USE_BPE, load_corpus
import eval as ev

A = "/home/ymchang/nami-lm/model_weights.json.hyp89-best.bak"      # seed 42, soul 18
B = "/home/ymchang/nami-lm/model_weights.json.ctrl1-seed43.bak"    # seed 43, soul 15


def answer(model, tok, q):
    ids = tok.encode(q)
    if not ids:
        return "<not-in-vocab>"
    gen = model.generate(ids, max_new=20, temperature=0.01,
                         eos_id=getattr(tok, "token2id", {}).get("∎"))
    return tok.decode(gen[len(ids):]).replace("∎", "")[:40]


if __name__ == "__main__":
    corpus, _ = load_corpus()
    tok = BPETokenizer() if USE_BPE else WordTokenizer(corpus)
    ma, mb = GPTMini.load(A), GPTMini.load(B)

    print(f"{'probe':26s} {'seed42':>7s} {'seed43':>7s}")
    print("-" * 70)
    flipped = []
    for q, expect in ev.SOUL_PROBES:
        aa, ab = answer(ma, tok, q), answer(mb, tok, q)
        va = ev._classify(aa, expect) == "strict"
        vb = ev._classify(ab, expect) == "strict"
        mark = "  " if va == vb else "<<"
        print(f"{q[:26]:26s} {'PASS' if va else 'fail':>7s} {'PASS' if vb else 'fail':>7s} {mark}")
        if va != vb:
            flipped.append((q, expect, aa, ab, va))

    print("\n" + "=" * 70)
    print(f"{len(flipped)} probes flipped between seeds. What the model actually said:\n")
    for q, expect, aa, ab, a_passed in flipped:
        print(f"  Q: {q}")
        print(f"     eval wants substring: {expect!r}")
        print(f"     seed42 {'PASS' if a_passed else 'fail'}: {aa!r}")
        print(f"     seed43 {'fail' if a_passed else 'PASS'}: {ab!r}")
        print()
