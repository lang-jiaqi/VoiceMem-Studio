"""左脑 · 时间权重：检索准确率评测。

只测"排序变没变"是不够的——变化率证明机制生效，不证明检索更准。
这里量的是**准不准**：

  · 时间相关查询（"我最近…""这阵子…"）：正确答案应当是近期记忆。
    指标 = top-1 命中近期记忆的比例。
  · 对照组（不含时间词）：时间权重不该干扰它们。
    指标 = top-1 与改动前保持一致的比例（越高越好）。

两组一起看才有意义：只看第一组，把权重调到极端也能刷满，但会毁掉对照组。

跑：python3 evals/recency_rerank.py [space] [近期天数]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 问的就是"最近怎么样"——正确答案按定义应该是近期的记忆
TIME_Q = [
    "我最近怎么样", "我最近在忙什么", "我这阵子压力大吗", "我最近累不累",
    "我最近心情如何", "我最近有什么开心的事", "我今天说了什么",
    "我最近在准备什么", "我最近见了谁", "我最近吃了什么",
]
# 问的是长期属性——跟时间无关，时间权重不该改变结果
NEUTRAL_Q = [
    "我不能吃什么", "我在哪读书", "我喜欢什么样的社交", "我有什么目标",
    "我是个什么样的人", "我讨厌什么", "我的专业是什么", "我平时怎么放松",
]


def main():
    space = sys.argv[1] if len(sys.argv) > 1 else "demo"
    fresh_days = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    import datetime as dt
    from voicemem.core import VoiceMem
    from voicemem.leftbrain.brain import (_recency_weight, RECENCY_HALFLIFE_DAYS,
                                          RECENCY_FLOOR)
    vm = VoiceMem.from_config({"mode": "text_mode", "space": space,
                               "embedding": {"provider": "local"},
                               "slots": {"provider": "local"}})
    today = dt.date.today()

    def rank(hits, use_recency):
        key = ((lambda h: -h.score * _recency_weight(h.observed_at)) if use_recency
               else (lambda h: -h.score))
        return sorted(hits, key=key)

    def is_fresh(h):
        if not h.observed_at:
            return False
        try:
            d = dt.datetime.fromisoformat(h.observed_at[:10]).date()
        except ValueError:
            return False
        return (today - d).days <= fresh_days

    def run(queries):
        rows = []
        for q in queries:
            c = vm.classify(q)
            hits = list(vm.search(q, slots=c.slots, entities=c.entities).hits)
            if not hits:
                continue
            rows.append((q, rank(hits, False), rank(hits, True)))
        return rows

    print(f"库={space}  半衰期={RECENCY_HALFLIFE_DAYS}天  下限={RECENCY_FLOOR}  "
          f"「近期」定义为 {fresh_days} 天内\n")

    t = run(TIME_Q)
    before = sum(1 for _, a, _ in t if is_fresh(a[0]))
    after  = sum(1 for _, _, b in t if is_fresh(b[0]))
    print(f"时间相关查询（n={len(t)}）  top-1 命中近期记忆")
    print(f"   改动前  {before}/{len(t)}  ({before/len(t):.0%})")
    print(f"   改动后  {after}/{len(t)}  ({after/len(t):.0%})")

    n = run(NEUTRAL_Q)
    same = sum(1 for _, a, b in n if a[0].memory_id == b[0].memory_id)
    print(f"\n对照组·无时间词（n={len(n)}）  top-1 与改动前一致")
    print(f"   {same}/{len(n)}  ({same/len(n):.0%})   ← 越高越好，说明没有误伤")

    print("\n改变了 top-1 的例子：")
    shown = 0
    for q, a, b in t + n:
        if a[0].memory_id != b[0].memory_id and shown < 3:
            shown += 1
            print(f"  「{q}」")
            print(f"     前: [{a[0].observed_at}] {a[0].text[:38]}")
            print(f"     后: [{b[0].observed_at}] {b[0].text[:38]}")


if __name__ == "__main__":
    main()
