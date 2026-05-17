#!/usr/bin/env python3
"""nami-lm web chat server — HTTP POST /chat → nami-lm inference → JSON response.

Pairs with nami-lm/web/index.html (deployed via office static-server).
Runs on 127.0.0.1:18807, reverse-proxied by openclaw-office static-server.cjs
under /nami-lm/api/*.

Inference reuses the same pipeline as nami_voice.py:
  WordTokenizer.load(pinned vocab) → GPTMini.load(weights) → generate(40 tok)
  → trim at em-dash / sentence terminator.

Endpoints:
  GET  /health        — liveness probe (no model load)
  POST /chat          — {"q": "..."} → {"a": "...", "latency_ms": int}
  POST /feedback      — {"q": "...", "a": "...", "rating": 1|0|-1, "note": ""}
                        appends to memory/nami_lm_feedback.log (jsonl)
"""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

NAMI_LM = Path.home() / "nami-lm"
FEEDBACK_LOG = Path.home() / "clawd" / "memory" / "nami_lm_feedback.log"

sys.path.insert(0, str(Path.home() / "nami-backpack" / "projects" / "numpy-grad"))
sys.path.insert(0, str(NAMI_LM))

# Lazy-load model + tokenizer once at first request.
_MODEL = None
_TOK = None


def _load_model():
    global _MODEL, _TOK
    if _MODEL is None:
        from train import GPTMini, WordTokenizer, BPETokenizer, USE_BPE, load_corpus
        vocab_path = NAMI_LM / "tokenizer_vocab.json"
        if not USE_BPE and vocab_path.exists():
            _TOK = WordTokenizer.load(str(vocab_path))
        else:
            texts, _ = load_corpus()
            _TOK = BPETokenizer() if USE_BPE else WordTokenizer(texts)
        _MODEL = GPTMini.load(str(NAMI_LM / "model_weights.json"))
        print(f"📂 Model loaded — vocab={_TOK.vocab_size}, params={_MODEL.param_count}", file=sys.stderr)


def _normalize(text: str) -> str:
    """STT/UI normalization — maps phonetic CN nouns back + strips Mandarin
    sentence-end particles that aren't in the small training corpus.

    The OOD sensitivity issue (Ryan 2026-05-12 23:30 feedback): nami-lm
    is word-level + 670K params trained on 100KB corpus. Canonical Q
    is 'Nami是誰？' but user wrote 'Nami是誰啊' — the '啊' particle is
    rare in corpus, gets tokenized into a non-canonical prefix, model
    diverges into noise. Fix: strip these particles inference-side until
    we have a bigger corpus that handles them natively. Big LLMs are
    robust to particles because they've seen them everywhere; tiny LMs
    need help.

    Tail particles stripped (only if at end, before ? / 嗎 etc):
      啊 / 喔 / 哦 / 喲 / 呀 / 唉 / 呢 / 嗯 / 耶 / 嘿
    """
    pairs = [
        ("那米", "Nami"), ("娜米", "Nami"), ("納米", "Nami"), ("那咪", "Nami"),
        ("克勞德", "Claude"), ("卡斯巴", "Kaspa"), ("卡斯帕", "Kaspa"),
        ("威士帕", "Whisper"), ("威斯帕", "Whisper"), ("阿夸", "Aqua"),
        ("克勞克斯", "ClawX"), ("瑞恩", "Ryan"), ("萊恩", "Ryan"),
        # 2026-05-12 23:44 case-fix: word-level tokenizer is case-sensitive,
        # corpus has "Ryan"/"Nami"/"Kaspa"/etc capitalized. Lowercase user
        # input → unknown tokens → noise. Normalize common project nouns.
        ("ryan", "Ryan"), ("nami", "Nami"), ("kaspa", "Kaspa"),
        ("whisper", "Whisper"), ("aqua", "Aqua"), ("clawx", "ClawX"),
        ("toccata", "Toccata"), ("claude", "Claude"),
    ]
    out = text
    for src, dst in pairs:
        out = out.replace(src, dst)
    # HYP45b — strip whitespace between Latin/digit and CJK characters.
    # Ryan 5/13 15:11 screenshot showed "Ryan 是誰？" returned garbage while
    # "Ryan是誰？" returned canonical. WordTokenizer treats " " as a separate
    # char-level token, shifting position embeddings by one — small model is
    # brittle to this. Strip the space so both forms tokenize identically.
    import re
    # Strip ALL inter-token whitespace: CJK-CJK, CJK-Latin, Latin-CJK,
    # CJK-digit. Word-level tokenizer treats " " as a separate token which
    # shifts positions and breaks tiny model's brittle context.
    out = re.sub(r"([A-Za-z0-9一-鿿])\s+([A-Za-z0-9一-鿿])", r"\1\2", out)
    out = re.sub(r"([A-Za-z0-9一-鿿])\s+([A-Za-z0-9一-鿿])", r"\1\2", out)  # 2nd pass for overlapping
    # Strip trailing tone particles: ...XX啊？ → ...XX？; ...XX啊 → ...XX
    # Iterate so multi-particle ('呢啊') gets fully cleaned.
    PARTICLES = "啊喔哦喲呀唉呢嗯耶嘿"
    out = out.rstrip()
    while True:
        changed = False
        if out and out[-1] in PARTICLES:
            out = out[:-1]; changed = True
        elif len(out) >= 2 and out[-1] in "?？" and out[-2] in PARTICLES:
            out = out[:-2] + out[-1]; changed = True
        if not changed:
            break
    return out


def _trim_degen(answer: str) -> str:
    """HYP45 — early-stop at degen pattern, anywhere in the output (not just
    first 20 chars). Cut at the first sign of model collapse:

    - Same char 3+ times in a row ("的的的") → cut before
    - Same 2-5 char word repeating (X X) → cut at second X
    - Mid-sentence "？" / "!" after position 5 → cut including it
    """
    if len(answer) < 6:
        return answer
    # 1. char triple anywhere
    for i in range(len(answer) - 2):
        if answer[i] == answer[i + 1] == answer[i + 2]:
            return answer[: i].rstrip()
    # 2. Word/bigram repeat — find any 2-5 char unit appearing 2x within first
    # 30 chars (gap up to 12 chars between occurrences).
    import re
    head = answer[:40]
    for unit_len in (4, 3, 2):
        for i in range(len(head) - 2 * unit_len):
            unit = head[i:i + unit_len]
            # only consider Chinese/Latin/digit units (not punct/whitespace)
            if not re.match(r"^[一-鿿A-Za-z0-9]+$", unit):
                continue
            # search for second occurrence within next 12 chars
            j = head.find(unit, i + unit_len)
            if j != -1 and j <= i + unit_len + 12 and i >= 4:
                return answer[: j].rstrip()
    return answer.strip()


def _trim_answer(answer: str) -> str:
    """Cut at em-dash / sentence terminator first, then run HYP45 degen guard."""
    for sep in [" — ", "——", "─ "]:
        i = answer.find(sep)
        if i > 3:
            return _trim_degen(answer[: i].rstrip())
    for stop in ["。", "！", "？", "\n"]:
        i = answer.find(stop)
        if i > 3:
            return _trim_degen(answer[: i + 1])
    return _trim_degen(answer.strip())


def chat(question: str) -> str:
    _load_model()
    q = _normalize(question).rstrip("?？") + "？"
    ids = _TOK.encode(q)
    if not ids:
        return "（這個問題的字我還沒學會，問點別的？）"
    gen = _MODEL.generate(ids, max_new=40, temperature=0.05)
    raw = _TOK.decode(gen[len(ids):])
    return _trim_answer(raw) or "（沒答案，再問一次？）"


class NamiHandler(BaseHTTPRequestHandler):
    def _json(self, code: int, body: dict):
        b = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, fmt, *args):
        # Quiet default access log; use stderr for actual events.
        pass

    def do_OPTIONS(self):
        self._json(204, {})

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/api/health", "/nami-lm/api/health"):
            return self._json(200, {"ok": True, "model": "v0.3.2.0-cosine-floor-deeper"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid json"})
        path = self.path.rstrip("/")
        # Accept both /chat and /api/chat and /nami-lm/api/chat (depends on
        # how cloudflared / static-server strips the prefix).
        if path.endswith("/chat"):
            q = (payload.get("q") or "").strip()
            if not q:
                return self._json(400, {"error": "empty question"})
            t0 = time.time()
            try:
                a = chat(q)
            except Exception as e:
                print(f"❌ chat error: {e}", file=sys.stderr)
                return self._json(500, {"error": str(e)})
            return self._json(200, {
                "a": a,
                "latency_ms": int((time.time() - t0) * 1000),
                "model": "v0.3.2.0-cosine-floor-deeper",
            })
        if path.endswith("/feedback"):
            FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "q": payload.get("q", ""),
                "a": payload.get("a", ""),
                "rating": payload.get("rating"),
                "note": payload.get("note", ""),
            }
            with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "not found"})


def main():
    port = int(os.environ.get("NAMI_LM_PORT", "18807"))
    addr = ("127.0.0.1", port)
    print(f"🌊 nami-lm web chat on http://{addr[0]}:{addr[1]}", file=sys.stderr)
    # Warm-load to avoid 30s latency on first request.
    _load_model()
    HTTPServer(addr, NamiHandler).serve_forever()


if __name__ == "__main__":
    main()
