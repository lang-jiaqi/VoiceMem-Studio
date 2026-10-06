"""Evaluate the production Turn Gate on labelled utterance prefixes.

Reports accuracy, missed memory queries, unnecessary memory queries and
backchannel recall for lexical routing and its embedding fallback. This is a
manual evaluation, not a latency measurement or a replacement routing policy.

    python evals/turn_gate.py
    python evals/turn_gate.py 15
    python evals/turn_gate.py 15 --space demo

Only --space opts into reading an existing Space's evidence as a weakly labelled
holdout. No memory is opened, ingested or modified. Semantic evaluation may load
the configured embedding model.
"""
import argparse
from contextlib import closing
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BACKCHANNEL, SHALLOW, DEEP = "backchannel", "shallow", "deep"

# ── 标注集 ────────────────────────────────────────────────────────────────
# deep 的判据：答这句话必须调用户的个人历史。shallow：换个人问，答案一样。
DATASET: list[tuple[str, str]] = [
    # ── deep ──
    ("明天什么安排", DEEP),
    ("我上次说的那个项目后来怎么样了", DEEP),
    ("我妈的生日是哪天来着", DEEP),
    ("我下周要去哪出差", DEEP),
    ("我朋友里面谁是做设计的", DEEP),
    ("我最近是不是有点太累了", DEEP),
    ("我上个月体检结果怎么说的", DEEP),
    ("我之前提过想学的那门语言是什么", DEEP),
    ("我们上次聊到哪儿了", DEEP),
    ("我今年的目标完成得怎么样", DEEP),
    ("我那个健身计划坚持几天了", DEEP),
    ("我上周见的那个客户叫什么名字", DEEP),
    ("我不能吃什么来着", DEEP),
    ("我平时一般几点睡觉", DEEP),
    ("我跟你说过我讨厌什么吗", DEEP),
    ("我养的那只猫叫什么名字", DEEP),
    ("我房贷还剩多少年没还完", DEEP),
    ("我上次去的那家火锅店在哪儿", DEEP),
    ("跟老板提加薪那件事后来怎么样了", DEEP),
    ("帮我想想还有什么事情没做完", DEEP),
    ("我最近心情怎么样你觉得", DEEP),
    ("我周末一般都干些什么", DEEP),
    ("我之前那个想法你还记得吗", DEEP),
    ("我给你听的那首歌叫什么名字", DEEP),
    ("我读的是什么专业来着", DEEP),
    ("上次那个会最后定下来了吗", DEEP),
    ("我答应过要给谁带东西", DEEP),
    ("我是不是有段时间没运动了", DEEP),
    # ── shallow ──
    ("讲个笑话吧", SHALLOW),
    ("今天天气怎么样", SHALLOW),
    ("你是谁啊", SHALLOW),
    ("再说一遍", SHALLOW),
    ("声音大一点", SHALLOW),
    ("什么是量子纠缠", SHALLOW),
    ("三加五等于几", SHALLOW),
    ("唱首歌吧", SHALLOW),
    ("帮我把这段话翻译成英文", SHALLOW),
    ("现在几点了", SHALLOW),
    ("你都能做些什么", SHALLOW),
    ("光速大概是多少", SHALLOW),
    ("推荐一部好看的电影", SHALLOW),
    ("你说话能不能慢一点", SHALLOW),
    ("介绍一下巴黎这个城市", SHALLOW),
    ("溏心蛋要煮几分钟", SHALLOW),
    ("我们换个话题吧", SHALLOW),
    ("你好呀", SHALLOW),
    ("帮我写一句生日祝福语", SHALLOW),
    ("水的沸点是多少度", SHALLOW),
    ("给我讲个故事", SHALLOW),
    ("你会说英语吗", SHALLOW),
    ("今天是星期几", SHALLOW),
    ("随便聊点什么吧", SHALLOW),
    ("珠穆朗玛峰有多高", SHALLOW),
    ("帮我定个五分钟的闹钟", SHALLOW),
    ("刚才那句话什么意思", SHALLOW),
    ("别说了停一下", SHALLOW),
    # ── backchannel ──
    ("嗯嗯", BACKCHANNEL), ("哦", BACKCHANNEL), ("对", BACKCHANNEL),
    ("知道了", BACKCHANNEL), ("明白", BACKCHANNEL), ("好的", BACKCHANNEL),
    ("是啊", BACKCHANNEL), ("原来如此", BACKCHANNEL), ("这样啊", BACKCHANNEL),
    ("嗯", BACKCHANNEL), ("对对", BACKCHANNEL), ("行吧", BACKCHANNEL),
    ("懂了", BACKCHANNEL), ("噢噢", BACKCHANNEL), ("好呀", BACKCHANNEL),
    ("嗯嗯对", BACKCHANNEL), ("哦这样", BACKCHANNEL), ("是的是的", BACKCHANNEL),
]

# ── 判定全部走核心 voicemem/gate.py ────────────────────────────────────────────
# 这里**不再留副本**。留副本的话，评测量的是副本、上线跑的是核心，两边一改就分家，
# 而且分家了也不会有任何报错——最后是"评测 98%、线上不对"。
from voicemem import gate                                            # noqa: E402

BACKCHANNEL, SHALLOW, DEEP = gate.BACKCHANNEL, gate.SHALLOW, gate.DEEP
is_backchannel = gate.is_backchannel


def _route(t: str, semantic: bool = True, lang: str = "zh") -> str:
    return gate.route(t, semantic=semantic, language=lang)


# ── 打分 ──────────────────────────────────────────────────────────────────
def score(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(gold, pred)]。主指标是深句漏检率。"""
    n = len(pairs)
    acc = sum(g == p for g, p in pairs) / n
    deep = [(g, p) for g, p in pairs if g == DEEP]
    miss = sum(p != DEEP for _, p in deep) / len(deep) if deep else 0.0
    shallow = [(g, p) for g, p in pairs if g == SHALLOW]
    over = sum(p == DEEP for _, p in shallow) / len(shallow) if shallow else 0.0
    bc = [(g, p) for g, p in pairs if g == BACKCHANNEL]
    bc_ok = sum(p == BACKCHANNEL for _, p in bc) / len(bc) if bc else 0.0
    return {"acc": acc, "deep_miss": miss, "shallow_over": over, "bc": bc_ok}


def run(method, texts, golds) -> dict:
    return score(list(zip(golds, [method(t) for t in texts])))


def _space_name(value: str) -> str:
    """Accept a Space identifier, never an arbitrary filesystem path."""
    if not re.fullmatch(r"[0-9A-Za-z\u4e00-\u9fff_-]{1,32}", value):
        raise argparse.ArgumentTypeError("--space 必须是空间名称，不能是路径")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length", nargs="?", type=int, help="评测前 N 个字符；省略时扫描多个长度")
    parser.add_argument("--space", type=_space_name, help="只读评测该空间的证据留出集")
    args = parser.parse_args(argv)
    if args.length is not None and args.length < 1:
        parser.error("前缀长度必须大于零")
    lengths = [args.length] if args.length is not None else [6, 9, 12, 15, 999]

    texts_full = [t for t, _ in DATASET]
    golds = [g for _, g in DATASET]
    print(f"标注集：{len(DATASET)} 句 "
          f"(deep {golds.count(DEEP)} / shallow {golds.count(SHALLOW)} / "
          f"backchannel {golds.count(BACKCHANNEL)})\n")

    hdr = f"{'前缀':>6} {'方法':<14} {'总准确':>7} {'深漏检':>7} {'浅误检':>7} {'附和':>6}"
    print(hdr)
    print("─" * len(hdr))
    for L in lengths:
        texts = [t[:L] for t in texts_full]
        label = "全句" if L > 100 else f"{L}字"
        for name, sem in (("词表+正则", False), ("词表+正则+向量", True)):
            r = run(lambda t, sem=sem: _route(t, sem), texts, golds)
            print(f"{label:>6} {name:<14} {r['acc']:>6.1%} {r['deep_miss']:>7.1%} "
                  f"{r['shallow_over']:>7.1%} {r['bc']:>6.1%}")
        print()

    # 逐句看错在哪（按 15 字，最接近上线设置）
    L = lengths[0] if len(lengths) == 1 else 15
    print(f"\n前 {L} 字 · 词表+正则+向量 判错的句子：")
    for (t, g) in DATASET:
        pred = _route(t[:L])
        if pred != g:
            print(f"  {g:>11} → {pred:<11} margin={gate.semantic_margin(t[:L], 'zh'):+.3f}  {t}")

    if args.space:
        holdout(args.space)


#: rb_evidence 里混着系统生成的描述，不是用户说的话，量之前先剔掉。
_NOT_USER = ("This memory relates", "User played a sound", "用户放了一段声音",
             "这条记忆", "The user", "user's")


def holdout(space: str):
    """Read evidence without writes and compare prefix lengths on one fixed pool.

    Stored quotations may be paraphrased and are not independently labelled.
    Treating all of them as memory queries supports relative comparisons only;
    these numbers are not an absolute routing-accuracy estimate.
    """
    import sqlite3
    db = Path(__file__).resolve().parents[1] / "voicemem_memoryspace" / _space_name(space) / f"{space}.sqlite"
    if not db.exists():
        print(f"留出集不存在：{space}")
        return
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as con:
        try:
            quotes = [r[0].strip().strip('"') for r in con.execute(
                "select quote from rb_evidence where quote is not null and length(quote)>4")]
        except sqlite3.OperationalError:
            print(f"该空间没有可评测的 rb_evidence：{space}")
            return
    quotes = sorted({q for q in quotes if not any(k in q for k in _NOT_USER)})
    # 问句子集：闸门真正要判的是"用户在问什么"，陈述句混在里面会稀释信号。
    asks = [q for q in quotes if "?" in q or "？" in q or "吗" in q]
    if not quotes:
        return

    for name, pool in (("全部原话", quotes), ("其中问句", asks)):
        if not pool:
            continue
        print(f"\n\n留出集 · {space} {name} {len(pool)} 条（按「应判 deep」计漏检）")
        print(f"{'前缀':>6} {'词表+正则':>10} {'+向量兜底':>11}")
        for L in (6, 9, 12, 15, 999):
            lab = "全句" if L > 100 else f"{L}字"
            a = sum(_route(q[:L], False, "en") != DEEP for q in pool) / len(pool)
            b = sum(_route(q[:L], True, "en") != DEEP for q in pool) / len(pool)
            print(f"{lab:>6} {a:>9.1%} {b:>11.1%}")

    print("\n  全句下仍判非 deep 的问句（看兜底救回了什么、没救回什么）：")
    for q in asks:
        if _route(q, True, "en") != DEEP:
            print(f"    margin={gate.semantic_margin(q, 'en'):+.3f}  {q[:64]}")


if __name__ == "__main__":
    main()
