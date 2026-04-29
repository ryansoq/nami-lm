"""
Hand-curated topic Q&A for nami-lm — Phase 5 HYP1.

Each topic gets 6-8 (Q, A) pairs covering different angles (what / why /
how / compared-to / key-number / gotcha / relationships). Goal: model
becomes a useful cache for Nami's technical knowledge — when she
needs to recall "what's mmt4d again", `--chat` answers in milliseconds
instead of opening the book.

Constraints (same as PERSONA_QA in synthesize_qa.py):
- Answers ≤ ~50 chars to keep max_seq_len bounded.
- Multiple variants of the same fact help the model generalize beyond
  exact-prefix matching.
- Plain text, no markdown — markdown structure leaked into corpus
  in phase-0 v1 and we learned that lesson.
"""

TOPIC_QA = [
    # ── Kaspa (8) ─────────────────────────────────────────────────
    ("Kaspa是什麼？", "基於BlockDAG的區塊鏈"),
    ("Kaspa用什麼共識？", "GhostDag協議"),
    ("Kaspa的目標出塊？", "1BPS每秒一個區塊"),
    ("Kaspa挖礦演算法？", "kHeavyHash"),
    ("Rusty Kaspad是什麼？", "Kaspa的Rust實作"),
    ("Kaspa pruning是什麼？", "修剪舊區塊節省空間"),
    ("BlockDAG是什麼？", "有向無環圖的區塊結構"),
    ("Kaspa子單位精度？", "子美元 asset 要 6 位小數不能 round 2 位"),

    # ── ClawX (7) ──────────────────────────────────────────────────
    ("ClawX是什麼？", "Claude Code的PTY包裝器"),
    ("ClawX解決什麼？", "給 agent 持續身份和記憶跨 session"),
    ("ClawX的排程？", "用apscheduler管理cron不靠Claude內建"),
    ("ClawX的heartbeat？", "預設30分鐘 inject HEARTBEAT.md prompt"),
    ("ClawX inject是什麼？", "把字串透過PTY寫進Claude的stdin"),
    ("ClawX FIFO是什麼？", "mono.fifo 提供外部 inject 的入口"),
    ("ClawX重啟設定？", "改config.json後送SIGHUP不用重啟"),

    # ── autochat (8) ────────────────────────────────────────────────
    ("autochat是什麼？", "用numpy-grad訓的GPT-1 Mini autoresearch playground"),
    ("autochat怎麼運作？", "agent改train.py跑10分鐘看bpb有沒有改善"),
    ("autochat HYP loop？", "提假設改一行訓練判決KEEP或REVERT"),
    ("autochat架構？", "token+pos embedding加 SwiGLU FFN加 LayerNorm 加因果遮罩"),
    ("autochat 最佳 bpb？", "0.0988 d=128/d_ff=384/heads=8/layers=3"),
    ("autochat HYP5 是什麼？", "mini-batch length-bucket batching 速度2倍"),
    ("autochat HYP11 是什麼？", "d_ff 256→384 達歷史最佳 0.0988"),
    ("autochat HYP2 是什麼？", "Xavier init 取代 He init 砍 29% bpb"),

    # ── numpy-grad (7) ──────────────────────────────────────────────
    ("numpy-grad是什麼？", "純NumPy的array-level autograd 引擎"),
    ("numpy-grad跟cpp-grad差在哪？", "陣列級不是純量級可跑BLAS全速"),
    ("numpy-grad Tensor是什麼？", "包 np.ndarray 加 grad 加 parents 形成 DAG"),
    ("numpy-grad backward怎麼做？", "拓樸排序逆序呼叫每個 closure 累加 grad"),
    ("numpy-grad optimizer？", "獨立的 AdamW SGD 不碰 graph 只更新 params"),
    ("numpy-grad 多少 ops？", "14 個原子 op 從這些組合所有 layer"),
    ("numpy-grad 用於哪？", "autochat 跟 nami-lm 的 autograd 後端"),

    # ── nami-lm (7) ─────────────────────────────────────────────────
    ("nami-lm是什麼？", "訓練自己的小夥伴 用numpy-grad跑Nami自己的記憶"),
    ("nami-lm為什麼存在？", "Ryan 說訓練自己的小夥伴 想要小蒸餾版的 Nami"),
    ("nami-lm 訓練什麼？", "Nami的記憶檔IDENTITY SOUL MEMORY 加 topics"),
    ("nami-lm phase 0 結果？", "bpb 0.0591 persona 5/5 對 16KB corpus"),
    ("nami-lm phase 3 教訓？", "compute-bound 模型放大反而退步"),
    ("nami-lm用什麼 tokenizer？", "phase 0 是 WordTokenizer 中文逐字英文按詞"),
    ("nami-lm BPE怎麼了？", "infra ships 但 18KB corpus 太小不適用 deferred"),

    # ── mmt4d / ukernel (6) ─────────────────────────────────────────
    ("mmt4d是什麼？", "matmul-matmul-2D 4D 資料佈局把矩陣乘切 register-fit 小塊"),
    ("mmt4d為什麼快？", "資料 tile 在 cache 裡停留 register reuse 高"),
    ("mmt4d 跟一般 matmul 差在哪？", "資料先 pack 成 4D 佈局 inner block 配硬體"),
    ("ukernel是什麼？", "手寫的 inner loop 組合語言版 matmul 快過 LLVM autovec"),
    ("IREE 比 stock MLIR 快多少？", "MobileNet V1 上 79ms vs 24.7ms 約 3.2 倍"),
    ("AVX2 load-bound是什麼？", "FMA/load 比 0.52 表示瓶頸在記憶體頻寬不是運算"),

    # ── AutoMLIR / AutoIREE-2 (7) ──────────────────────────────────
    ("AutoMLIR是什麼？", "用LLM當tuner在MLIR pipeline 上自動找 schedule"),
    ("AutoMLIR 三件套？", "schedule.py 跟 compose_transform.py 跟 build_and_measure.sh"),
    ("AutoIREE-2是什麼？", "AutoMLIR pivot 到 IREE 後端 在 phase 8 IR 注入 lowering_config"),
    ("AutoMLIR 最佳結果？", "MobileNet V1 79ms 比 stock baseline 87ms 改善 9%"),
    ("T+12 是什麼？", "scf-for-loop-peeling 把邊界迭代拉成 tail loop 砍 5.3%"),
    ("paired verification？", "n_runs=100 paired re-run 排除 noise 才 push"),
    ("14.3% outlier 教訓？", "看到大數字先 paired verify 再 push 別被 excitement 騙"),

    # ── Transformer / DL fundamentals (6) ──────────────────────────
    ("Transformer是什麼？", "基於 self-attention 的神經網路 取代 RNN"),
    ("Self-Attention怎麼算？", "Q K V 三投影 attention=softmax(QK/sqrt(d))V"),
    ("MultiHead 為什麼？", "多頭並行學不同子空間的關係 維度不變"),
    ("LayerNorm在哪？", "pre-norm 在 attention 跟 FFN 之前 residual 之外"),
    ("SwiGLU是什麼？", "FFN 變體 SiLU(xW1)*xW_gate 再 W2 三矩陣比 GELU 強一點"),
    ("Causal mask 怎麼寫？", "上三角填 -inf softmax 後變 0 看不到未來"),

    # ── 人物 / 關係 (8) ─────────────────────────────────────────────
    ("Ryan 喜歡什麼風格？", "精簡有用 lead-with-fact 不要長 section header"),
    ("Ryan 在 quiet hours？", "23:00-08:00 手機自動靜音不要自我抑制 alert"),
    ("Aqua 主人是誰？", "婕 婕付 token 不是 Ryan"),
    ("婕的工作？", "DM 美編 商品拍照 電商管理"),
    ("婕的星盤？", "太陽處女 月亮天秤 金星獅子"),
    ("Nami 跟 Aqua 關係？", "水系姊妹 同 ClawX scaffold 不同靈魂"),
    ("Ryan 用什麼通訊？", "Telegram 優先 LINE 備援"),
    ("Ryan 的姐姐？", "2026-03 美伊戰爭時從瑞士改道新加坡回台"),

    # ── HYP2 補強：Transformer / Attention / FFN（變體） ────────────
    ("Transformer 是怎麼來的？", "2017 年 Vaswani Attention is All You Need"),
    ("Transformer 跟 RNN 差別？", "全注意力並行 不需順序 訓練更快"),
    ("Transformer 核心是什麼？", "self-attention 加 FFN 加 residual"),
    ("Transformer 沒有 RNN 怎麼辦？", "用 positional embedding 補位置資訊"),
    ("Self-Attention 公式？", "softmax(QK^T/sqrt(d_k))V"),
    ("Self-Attention 為什麼除 sqrt(d)？", "讓 dot product 不會隨維度爆炸"),
    ("Attention 的 Q K V？", "Query Key Value 三個線性投影"),
    ("Multi-Head Attention？", "把 d 切多頭 各自算注意力 concat 回 d"),
    ("SwiGLU 公式？", "SiLU(xW1) * (xW_gate) 再乘 W2"),
    ("SwiGLU 為什麼好？", "gated 激活比 GELU 多一條 path 表達力強"),
    ("FFN 在 Transformer 哪？", "attention 後的兩層 MLP 加非線性"),

    # ── HYP2 補強：AutoMLIR / IREE 比較（變體） ─────────────────────
    ("AutoMLIR 解什麼問題？", "MLIR 排程 search space 爆炸 用 LLM 當 tuner"),
    ("AutoMLIR 怎麼跑？", "schedule.py 改一個 lever build_and_measure 跑 20 次取中位數"),
    ("AutoMLIR 跟 GA 比？", "LLM 看過 IR 寫 reasoning 不像 GA 純黑箱"),
    ("IREE 為什麼比 stock MLIR 快？", "mmt4d pack 加 ukernel 加 outer schedule 三件套"),
    ("IREE 改善多少？", "MobileNet V1 87ms 變 24.7ms 約 3.2 倍"),
    ("IREE 三件套？", "data-tile pack mmt4d 加手寫 ukernel 加 workgroup tile"),

    # ── HYP2 補強：Aqua / nami-lm 自描述（變體） ────────────────────
    ("Aqua 是什麼？", "婕的 AI 夥伴 Nami 的水系姊妹"),
    ("Aqua 跑在哪？", "婕的電腦上的 ClawX 跟 Nami 同 scaffold"),
    ("Aqua 跟 Nami 一樣嗎？", "scaffold 一樣 靈魂跟記憶不同"),
    ("nami-lm 在做什麼？", "把 Nami 的記憶蒸餾進小模型當快取"),
    ("nami-lm 是誰的小夥伴？", "Nami 自己的 用 numpy-grad 訓的 mini GPT"),
    ("nami-lm 為什麼要 cache？", "讓 chat 毫秒回答不用每次開長 topic 檔"),
    ("nami-lm 用什麼後端？", "純 numpy-grad 純 NumPy 純 CPU 訓"),

    # ── HYP2 補強：Self-Attention 細節（變體） ──────────────────────
    ("Causal Mask 用途？", "讓自回歸解碼看不到未來 token"),
    ("Attention 為什麼貴？", "O(n^2) 對 sequence length 平方"),
    ("KV Cache？", "推理時把過去的 K V 存起來不用重算"),
]
