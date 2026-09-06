"""历史数据修复：给现有记忆重跑一次 LLM slot 打标签。

为什么需要：update_memory 只重打向量标签、不重跑 LLM，而向量在 7 个 slot 上
只有约 0.03 的区分度，判不准。用得越久，被 update 冲掉的 LLM 标签越多，
检索候选池就越常漏掉正确答案。

跑：python3 evals/retag_llm.py <space> [--apply] [--limit N] [--like 关键词]
不加 --apply 只统计不写；--limit / --like 用来只修一小批做对照实验
（验证假设不需要全库重打，那太贵）。
"""
import sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

V2 = ('work', 'goals', 'health', 'daily_life', 'relationships', 'knowledge', 'finance')


def main():
    space = sys.argv[1] if len(sys.argv) > 1 else "demo"
    apply = "--apply" in sys.argv
    limit = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), 0)
    like = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--like=")), "")
    db = f"voicemem_memoryspace/{space}/{space}.sqlite"
    ph = ",".join("?" * len(V2))
    rows = sqlite3.connect(db).execute(
        f"""SELECT m.id, m.content,
                   (SELECT COUNT(*) FROM memory_tags t
                     WHERE t.memory_id=m.id AND t.confidence=0.95 AND t.slot IN ({ph})) AS llm_tags
            FROM memories m""", V2).fetchall()
    todo = [(i, c) for i, c, n in rows if not n]
    if like:
        todo = [(i, c) for i, c in todo if like.lower() in c.lower()]
    if limit:
        todo = todo[:limit]
    print(f"{space}: 共 {len(rows)} 条，其中 {len(todo)} 条没有 LLM 标签 "
          f"({len(todo)/max(len(rows),1):.0%})")
    if not apply:
        print("（未加 --apply，只统计）")
        return

    from voicemem.core import VoiceMem
    vm = VoiceMem.from_config({"mode": "text_mode", "space": space,
                               "embedding": {"provider": "local"},
                               "slots": {"provider": "local"}})
    lb = vm._o._left
    ok = fail = 0
    for i, (mid, text) in enumerate(todo, 1):
        try:
            if lb._llm_tag_memories(text, [mid]):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            print(f"  [{i}] 失败 {type(e).__name__}: {e}")
        if i % 20 == 0:
            print(f"  …{i}/{len(todo)}")
    print(f"\n重打完成：成功 {ok}，失败 {fail}")


if __name__ == "__main__":
    main()
