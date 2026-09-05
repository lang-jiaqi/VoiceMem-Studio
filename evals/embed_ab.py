"""换 embedding 之前先比一把：同一个库、同一批问题，两个模型各查一遍。

换 embedding 是**要全库重算向量**的改动，不是能随手回滚的。所以别按 MTEB /
C-MTEB 的排名换——那些榜跑的是文档检索，这里的负载是十几个字的对话记忆片段，
排名不一定传递。用你自己的库测。

不需要 API key、不碰向量库（直接读 sqlite 里的记忆正文，在内存里算余弦），
所以正在跑的服务不用停。

    python3 evals/embed_ab.py demo-zh                  # 默认 e5 vs 按语言选的 bge
    python3 evals/embed_ab.py demo-zh e5 e5-base       # 指定比哪两个

看什么：
    top1 一致率   两个模型给出的第一名是不是同一条
    top5 重合度   前五名的交集大小 / 5
两个都接近 100% = 换了也没区别，不值得为它重算全库；差得多 = 值得再上 LLM 评委
（evals/topk_relevance.py）判谁对。这个脚本只回答"有没有区别"，不回答"谁更准"。
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERIES_ZH = [
    "我最近怎么样", "我最近在忙什么", "我这阵子压力大吗", "我最近心情如何",
    "我最近有什么开心的事", "我最近在准备什么", "我最近见了谁",
    "我不能吃什么", "我在哪读书", "我有什么目标", "我是个什么样的人",
    "我讨厌什么", "我的专业是什么", "我平时怎么放松", "我朋友都有谁",
    "我下周有什么安排", "我家里人怎么样", "我工作上遇到什么问题",
]
QUERIES_EN = [
    "How have I been lately?", "What have I been busy with?",
    "Have I been stressed recently?", "How is my mood recently?",
    "Anything good happened lately?", "What am I preparing for?",
    "Who did I meet recently?", "What can I not eat?", "Where do I study?",
    "What are my goals?", "What kind of person am I?", "What do I dislike?",
    "What is my major?", "How do I usually relax?", "Who are my friends?",
    "What is on my schedule next week?", "How is my family?",
    "What problems do I have at work?",
]


def memories(space: str) -> list[str]:
    db = Path("voicemem_memoryspace") / space / f"{space}.sqlite"
    if not db.is_file():
        sys.exit(f"找不到 {db}")
    con = sqlite3.connect(db)
    rows = [r[0] for r in con.execute(
        "select content from memories where content is not null and length(content)>1")]
    # 右脑那份也一起比：它和左脑用同一个 embedder，换模型两边一起受影响。
    try:
        rows += [r[0] for r in con.execute(
            "select content from right_brain_memories "
            "where content is not null and length(content)>1")]
    except sqlite3.OperationalError:
        pass
    return sorted(set(rows))


def space_language(space: str) -> str:
    import json
    p = Path("voicemem_memoryspace") / space / f"{space}.json"
    try:
        return (json.loads(p.read_text())["space"].get("language") or "en")
    except Exception:
        return "en"


def top5(key: str, docs: list[str], queries: list[str], lang: str):
    """返回 {query: [doc 下标, …]}，纯本地算，不建向量库。"""
    import numpy as np
    from voicemem.leftbrain.local_embedder import resolve, resolve_path, shared_model
    spec = resolve(key, language=lang)
    m = shared_model(resolve_path(spec))
    D = np.asarray(m.encode([f"{spec.passage_prefix}{d}" for d in docs],
                            normalize_embeddings=True, batch_size=64))
    Q = np.asarray(m.encode([f"{spec.query_prefix}{q}" for q in queries],
                            normalize_embeddings=True))
    order = np.argsort(-(Q @ D.T), axis=1)[:, :5]
    print(f"  {spec.key:9} {spec.repo:34} {spec.dims} 维")
    return {q: list(order[i]) for i, q in enumerate(queries)}


def main():
    space = sys.argv[1] if len(sys.argv) > 1 else "demo"
    a = sys.argv[2] if len(sys.argv) > 2 else "e5"
    b = sys.argv[3] if len(sys.argv) > 3 else "bge"
    lang = space_language(space)
    docs = memories(space)
    queries = QUERIES_ZH if lang == "zh" else QUERIES_EN
    if not docs:
        sys.exit(f"{space} 里没有记忆正文，换个有数据的 space")
    print(f"space={space} 语言={lang} 记忆 {len(docs)} 条 问题 {len(queries)} 个\n")

    ra, rb = top5(a, docs, queries, lang), top5(b, docs, queries, lang)

    same1 = sum(ra[q][0] == rb[q][0] for q in queries) / len(queries)
    ov = [len(set(ra[q]) & set(rb[q])) / 5 for q in queries]
    print(f"\ntop1 一致率 {same1:.0%}   top5 平均重合度 {sum(ov)/len(ov):.0%}")

    worst = sorted(queries, key=lambda q: len(set(ra[q]) & set(rb[q])))[:3]
    print("\n分歧最大的三个问题，各自的第一名：")
    for q in worst:
        print(f"\n  {q}")
        print(f"    {a:8} {docs[ra[q][0]][:64]}")
        print(f"    {b:8} {docs[rb[q][0]][:64]}")


if __name__ == "__main__":
    main()
