"""左脑 · 时间权重：盲评 A/B。

"top-1 命中近期记忆"是自证指标——时间权重正是为此设计的，用它当评价标准
等于自己给自己打分。这里改成直接问：**换上来的那条，是不是更好的答案。**

方法：同一批候选，分别用「纯相似度」和「相似度×时间权重」取 top-1；
      两者不同的查询交给评委模型盲评（A/B 顺序随机、不告诉它哪个是新方案），
      判"哪条更好地回答了这个问题"，可判平局。

跑：python3 evals/recency_ab.py [space]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERIES = [
    "我最近怎么样", "我最近在忙什么", "我这阵子压力大吗", "我最近累不累",
    "我最近心情如何", "我最近有什么开心的事", "我今天说了什么",
    "我最近在准备什么", "我最近见了谁", "我最近吃了什么",
    "我不能吃什么", "我在哪读书", "我喜欢什么样的社交", "我有什么目标",
    "我是个什么样的人", "我讨厌什么", "我的专业是什么", "我平时怎么放松",
]

QUERIES_EN = [
    "How have I been lately?", "What have I been busy with?",
    "Have I been stressed recently?", "Am I tired these days?",
    "How is my mood recently?", "Anything good happened lately?",
    "What did I say today?", "What am I preparing for?",
    "Who did I meet recently?", "What did I eat recently?",
    "What can I not eat?", "Where do I study?",
    "What kind of socialising do I like?", "What are my goals?",
    "What kind of person am I?", "What do I dislike?",
    "What is my major?", "How do I usually relax?",
]

JUDGE = """你在评估一个个人记忆系统的检索质量。

用户问："{q}"

系统从记忆库里取回了两条候选，请判断**哪条更好地回答了这个问题**。
判据按重要性排序：
1. 内容是否切题（最重要——不切题的记忆再新也没用）
2. 同样切题时，越近发生的越好

A：{a}
B：{b}

只输出 JSON：{{"winner": "A"|"B"|"tie", "why": "不超过20字"}}"""


def main():
    space = sys.argv[1] if len(sys.argv) > 1 else "demo"
    qs = QUERIES_EN if "en" in sys.argv[2:] else QUERIES
    from openai import OpenAI
    from voicemem.core import VoiceMem
    from voicemem.leftbrain.brain import _recency_weight
    cli = OpenAI()
    vm = VoiceMem.from_config({"mode": "text_mode", "space": space,
                               "embedding": {"provider": "local"},
                               "slots": {"provider": "local"}})
    win = loss = tie = same = 0
    detail = []
    for i, q in enumerate(qs):
        c = vm.classify(q)
        hits = list(vm.search(q, slots=c.slots, entities=c.entities).hits)
        if not hits:
            continue
        old = sorted(hits, key=lambda h: -h.score)[0]
        new = sorted(hits, key=lambda h: -h.score * _recency_weight(h.observed_at))[0]
        if old.memory_id == new.memory_id:
            same += 1
            continue
        # 顺序随机：偶数轮 A=旧，奇数轮 A=新，消掉位置偏好
        flip = i % 2 == 1
        a, b = (new, old) if flip else (old, new)
        fmt = lambda h: f"[{h.observed_at or '无日期'}] {h.text}"
        r = cli.chat.completions.create(
            model="gpt-4o", temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content":
                       JUDGE.format(q=q, a=fmt(a), b=fmt(b))}])
        v = json.loads(r.choices[0].message.content or "{}")
        w = v.get("winner", "tie")
        picked = ("new" if (w == "A") == flip else "old") if w in ("A", "B") else "tie"
        if picked == "new":  win += 1
        elif picked == "old": loss += 1
        else: tie += 1
        detail.append((q, picked, v.get("why", ""), old, new))

    n = win + loss + tie
    print(f"库={space}   查询 {len(qs)} 条，其中 top-1 未变 {same} 条、"
          f"发生变化 {n} 条\n")
    print(f"在发生变化的 {n} 条上，盲评结果：")
    print(f"   时间权重更好   {win}/{n}")
    print(f"   原方案更好     {loss}/{n}")
    print(f"   平局           {tie}/{n}")
    print("\n逐条：")
    for q, picked, why, old, new in detail:
        tag = {"new": "✓ 新更好", "old": "✗ 旧更好", "tie": "= 平局"}[picked]
        print(f"  {tag}  「{q}」 {why}")
        print(f"        旧 [{old.observed_at}] {old.text[:34]}")
        print(f"        新 [{new.observed_at}] {new.text[:34]}")


if __name__ == "__main__":
    main()
