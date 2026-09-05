"""轮次闸门 · 三路判别准确度。

现在每一轮都检索左右脑 top5 并注入回复模型。更自然的做法是先判这一句要什么：

    backchannel  纯附和（"嗯嗯""知道了"）→ 不打断、不检索，助手继续说
    shallow      浅内容（"讲个笑话""今天几号"）→ 打断、但不检索
    deep         深内容（"我明天什么安排"）→ 打断 + 检索，即现在的行为

难点全在 deep/shallow 这条边，而且判定要在 ASR 只吐出前若干个字时就做出来。
这个脚本就量那件事：**前 N 个字够不够判**，以及哪种判别法值得写进链路。

两类错误代价不对称，所以主看的不是总准确率：
    深判成浅 = 助手忘了你的事，用户立刻听得出来   ← 主指标（漏检率）
    浅判成深 = 多花 ~300 token，即现在的默认行为   ← 可以忍

跑：
    python3 evals/turn_gate.py                 # 扫全部前缀长度 × 全部方法
    python3 evals/turn_gate.py 15              # 只看前 15 字
    python3 evals/turn_gate.py 15 --space demo # 额外量 base_score 阈值法（要有库）
"""
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


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    lengths = [int(argv[0])] if argv else [6, 9, 12, 15, 999]
    space = None
    if "--space" in sys.argv:
        space = sys.argv[sys.argv.index("--space") + 1]

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

    if space:
        base_score_probe(space, texts_full, golds)
    holdout()


#: rb_evidence 里混着系统生成的描述，不是用户说的话，量之前先剔掉。
_NOT_USER = ("This memory relates", "User played a sound", "用户放了一段声音",
             "这条记忆", "The user", "user's")


def holdout(space: str = "demo-zh"):
    """留出集：真实会话里被抽成记忆证据的用户原话——不是本脚本作者写的，规则没见过。

    两个已知缺陷，量之前说清楚：
      · quote 存的是**抽取后的转述**（"Doesn't like crowded places" 里的 I 已被抹掉），
        人称线索天然被削弱，所以绝对漏检率偏高；
      · "这批全是 deep"这个假设不严格——里面混着 "Can you hear?" 这种本就该判浅的。
    所以**只看同一批数据上 6/15/全句的相对差距**，那个不受这两条影响。
    """
    import sqlite3
    db = f"voicemem_memoryspace/{space}/{space}.sqlite"
    if not Path(db).exists():
        return
    con = sqlite3.connect(db)
    quotes = [r[0].strip().strip('"') for r in con.execute(
        "select quote from rb_evidence where quote is not null and length(quote)>4")]
    quotes = sorted({q for q in quotes if not any(k in q for k in _NOT_USER)})
    # 问句子集：闸门真正要判的是"用户在问什么"，陈述句混在里面会稀释信号。
    asks = [q for q in quotes if "?" in q or "？" in q or "吗" in q]
    if not quotes:
        return

    for name, pool in (("全部原话", quotes), ("其中问句", asks)):
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


# NOTE: 词表在 web/run.py 里也有一份。哪种方法胜出、真要接进链路时，这份该搬进
# 核心（voicemem/gate.py）由两边共用——demo 层和核心各留一份词表，改一处漏一处。
if __name__ == "__main__":
    main()
