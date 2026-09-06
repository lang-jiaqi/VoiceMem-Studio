"""回声 / 真插话 判别评测。

助手外放时，它自己的声音会被麦克风收回去、进 ASR，转出来的字跟真人插话在
**字数上**没有区别。判错的代价是双向的：
  · 把回声当插话 → 没人说话它也开一轮，还会把自己正在说的话掐断；
  · 把插话当回声 → 用户打断不了。

所以这里同时量两个方向，不能只看总准确率。

跑：python3 evals/echo_barge.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 被测实现 ────────────────────────────────────────────────────────────────
def _lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev, best = [0] * (len(b) + 1), 0
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, cj in enumerate(b, 1):
            if ch == cj:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def new_impl(new_chars: str, said: str, window=300, ratio=0.6, fuzzy_min=4) -> bool:
    """当前实现：窗口 300 字 + 最长公共子串占比。"""
    s = "".join(c for c in (new_chars or "") if c.strip()).casefold()
    hay = "".join(c for c in (said or "")[-window:] if c.strip()).casefold()
    if not s or not hay:
        return False
    if s in hay:
        return True
    if len(s) < fuzzy_min:
        return False
    return _lcs_len(s, hay) / len(s) >= ratio


def old_impl(new_chars: str, said: str) -> bool:
    """改动前：只比已生成文本的最后 40 字，精确子串。"""
    said = (said or "")[-40:].casefold()
    s = "".join(c for c in (new_chars or "") if c.strip()).casefold()
    return bool(s) and s in said


def bigram_impl(new_chars: str, said: str, window=300, ratio=0.6, fuzzy_min=4) -> bool:
    """中间那一版：窗口放宽但用二元组重合率（不看连续性）。"""
    s = "".join(c for c in (new_chars or "") if c.strip()).casefold()
    hay = "".join(c for c in (said or "")[-window:] if c.strip()).casefold()
    if not s or not hay:
        return False
    if s in hay:
        return True
    if len(s) < fuzzy_min:
        return False
    grams = {s[i:i + 2] for i in range(len(s) - 1)}
    return sum(1 for g in grams if g in hay) / len(grams) >= ratio


# ── 数据集 ──────────────────────────────────────────────────────────────────
# said = 助手已经说出口的话；probe = ASR 新转出来的几个字；echo = 是不是回声。
SAID_A = ("复习GRE的事让你感觉会不会压力大了一些？适当放松也是很有必要的。"
          "明天去吃大餐或许是个好机会，让自己开心一下，别让压力积太多。")
SAID_B = ("你听起来挺累的，是不是最近那个项目压得慌？"
          "要不先歇会儿，我陪你聊聊，别一个人扛着。")
SAID_C = ("I remember you had that interview today. You spent almost two months "
          "preparing for it. How did it go?")

CASES = [
    # ── 回声：助手原话的连续片段 ────────────────────────────────────────────
    (SAID_A, "让你感觉会不会压力大",       True,  "整段回声"),
    (SAID_A, "适当放松也是很有必要",       True,  "整段回声"),
    (SAID_A, "明天去吃大餐或许是个",       True,  "整段回声"),
    (SAID_A, "别让压力积太多",             True,  "句尾回声"),
    (SAID_B, "是不是最近那个项目",         True,  "整段回声"),
    (SAID_B, "要不先歇会儿",               True,  "整段回声"),
    (SAID_B, "我陪你聊聊",                 True,  "整段回声"),
    (SAID_C, "you had that interview",     True,  "英文回声"),
    (SAID_C, "spent almost two months",    True,  "英文回声"),
    # ── 回声 + ASR 误差（同音字 / 吞字）────────────────────────────────────
    (SAID_A, "让你感觉会不会雅力大",       True,  "回声·同音字"),
    (SAID_A, "适当放松也是很有必药",       True,  "回声·错字"),
    (SAID_B, "别一个人抗着",               True,  "回声·同音字"),
    (SAID_B, "是不是最近那个像木",         True,  "回声·错字"),
    (SAID_C, "you had that interviewed",   True,  "英文回声·词尾误"),
    # ── 真插话：词汇跟话题重合，但不是原话 ────────────────────────────────
    (SAID_A, "我压力其实还好啦",           False, "插话·词汇重合"),
    (SAID_A, "GRE我准备得差不多了",        False, "插话·词汇重合"),
    (SAID_A, "明天不去了",                 False, "插话·短词重合"),
    (SAID_A, "大餐就算了吧",               False, "插话·词汇重合"),
    (SAID_B, "项目其实上周就交了",         False, "插话·词汇重合"),
    (SAID_B, "我不累",                     False, "插话·短句"),
    (SAID_B, "先别聊这个",                 False, "插话·词汇重合"),
    (SAID_C, "the interview was fine",     False, "英文插话·词汇重合"),
    (SAID_C, "I didn't go actually",       False, "英文插话"),
    # ── 真插话：完全另起话题 ──────────────────────────────────────────────
    (SAID_A, "等一下我想说个事",           False, "插话·新话题"),
    (SAID_B, "我明天要去医院",             False, "插话·新话题"),
    (SAID_C, "can you play that song",     False, "英文插话·新话题"),
    # ── 边界：太短，只信精确匹配 ──────────────────────────────────────────
    (SAID_A, "嗯嗯",                       False, "短·语气词"),
    (SAID_B, "好的",                       False, "短·应答"),
    (SAID_A, "压力",                       True,  "短·精确命中原话"),
    (SAID_B, "项目",                       True,  "短·精确命中原话"),
]


def score(fn):
    tp = fp = tn = fn_ = 0
    wrong = []
    for said, probe, want, tag in CASES:
        got = fn(probe, said)
        if want and got:      tp += 1
        elif want and not got: fn_ += 1; wrong.append((probe, tag, "漏判回声"))
        elif not want and got: fp += 1;  wrong.append((probe, tag, "误判为回声"))
        else:                  tn += 1
    n = len(CASES)
    return {
        "总准确率": (tp + tn) / n,
        "回声召回": tp / max(tp + fn_, 1),      # 挡住了多少回声
        "插话保留": tn / max(tn + fp, 1),        # 放行了多少真插话
        "wrong": wrong,
    }


if __name__ == "__main__":
    impls = [("改动前 (40字窗口·精确子串)", old_impl),
             ("中间版 (300字窗口·二元组)", bigram_impl),
             ("当前   (300字窗口·最长公共子串)", new_impl)]
    print(f"数据集：{len(CASES)} 条"
          f"（回声 {sum(1 for c in CASES if c[2])} / 真插话 {sum(1 for c in CASES if not c[2])}）\n")
    print(f"{'实现':<34}{'总准确率':>10}{'回声召回':>10}{'插话保留':>10}")
    print("-" * 64)
    results = {}
    for name, fn in impls:
        r = score(fn); results[name] = r
        print(f"{name:<34}{r['总准确率']:>9.0%}{r['回声召回']:>10.0%}{r['插话保留']:>10.0%}")
    for name, _ in impls:
        w = results[name]["wrong"]
        if w:
            print(f"\n{name} 判错 {len(w)} 条：")
            for probe, tag, how in w:
                print(f"   · {probe!r:28} {tag:16} → {how}")
