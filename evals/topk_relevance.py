"""左脑 · 检索切题率基线。

盲评时发现多条查询的两个候选"都不切题"——那是候选池的问题，不是排序的问题。
排序类改动的天花板由它决定：候选里没有正确答案，怎么排都没用。
这里量一个基线：top-1 / top-5 里有没有真正回答问题的记忆。

跑：python3 evals/topk_relevance.py [space]
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

PROMPT = """用户问："{q}"

下面是记忆系统取回的 {n} 条记忆。判断**哪几条真正回答了这个问题**
（切题即可，不要求完整）。都不切题就返回空数组。

{items}

只输出 JSON：{{"relevant": [序号], "why": "不超过20字"}}"""


def main():
    space = sys.argv[1] if len(sys.argv) > 1 else "demo"
    qs = QUERIES_EN if "en" in sys.argv[2:] else QUERIES
    from openai import OpenAI
    from voicemem.core import VoiceMem
    cli = OpenAI()
    vm = VoiceMem.from_config({"mode": "text_mode", "space": space,
                               "embedding": {"provider": "local"},
                               "slots": {"provider": "local"}})
    top1 = top5 = n_q = 0
    empty = []
    for q in qs:
        c = vm.classify(q)
        hits = list(vm.search(q, slots=c.slots, entities=c.entities).hits)[:5]
        if not hits:
            continue
        n_q += 1
        items = "\n".join(f"{i+1}. {h.text}" for i, h in enumerate(hits))
        r = cli.chat.completions.create(
            model="gpt-4o", temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content":
                       PROMPT.format(q=q, n=len(hits), items=items)}])
        rel = set(json.loads(r.choices[0].message.content or "{}").get("relevant") or [])
        if rel:
            top5 += 1
            if 1 in rel:
                top1 += 1
        else:
            empty.append(q)
    print(f"库={space}   查询 {n_q} 条\n")
    print(f"top-1 切题   {top1}/{n_q}  ({top1/n_q:.0%})")
    print(f"top-5 内有切题的  {top5}/{n_q}  ({top5/n_q:.0%})")
    print(f"\ntop-5 内一条都不切题（排序类改动救不了的）：{len(empty)} 条")
    for q in empty:
        print(f"   · {q}")


if __name__ == "__main__":
    main()
