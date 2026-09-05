"""轮次闸门：这一句要打断吗？要检索记忆吗？

原来每一轮都一样：检索左脑 top5 + 右脑 top5，拼进 prompt 给回复模型。可人说话
不是每句都需要记忆——"嗯嗯"不需要，"讲个笑话"也不需要。每轮都塞，回复里就会
冒出跟这句话无关的记忆，听起来像在硬凑。

三条路：

    backchannel  纯附和（"嗯嗯""知道了"）→ 不打断、不检索，助手继续说
    shallow      浅内容（"讲个笑话""现在几点"）→ 打断，但不注入事实记忆
    deep         深内容（"我明天什么安排"）→ 打断 + 注入，即原来的行为

**两个决定的截止时间不一样，别合成一个。** 这是实测出来的（evals/turn_gate.py）：

    打断    必须在 ASR 头几个字就定 —— 但它只需要分"是不是附和"，闭集词表，
            两个数据集上都 100%。深浅根本不参与：深和浅都要打断。
    检索    可以等到说完 —— 而且**必须等**。同一套规则、同一批真实对话，
            只看前 6 字判，深句漏检 84%；前 15 字 55%；全句 20%。判别信息
            大量落在句子中后段，不是换个更强的分类器能补的（E5 在 6 字上
            同样错 80%，两组原型的余弦差只有 0.01，等于在拿噪声的符号当答案）。

所以检索**照跑**——它是本地的、早就藏在投机预取那 0–300ms 里了，白跑一次不要钱；
真正要卡的是"拼不拼进 prompt"，而那一步的截止时间本来就在说完之后。

三级判定，一级比一级慢，前面命中就不往下走：

    ① 附和词表    闭集，0ms
    ② 表层正则    第一人称 / 时间回指 / 定指回指 —— 高精度，0ms
    ③ 句向量兜底  ①②都没话说时才问（"帮我想想还有什么没做完"这种）

实测（LoCoMo 之外，用真实会话原话 + 手写标注集）：全句 + 三级，深句漏检约 1.4%，
浅句误判成深 10.7%。两类错的代价差得远——漏检是助手忘了你的事，用户当场听得出来；
误检只是多花约 300 token，而那**正是现在每轮都检索的默认行为**。所以阈值一律偏向
"拿不准就检索"。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

BACKCHANNEL = "backchannel"
SHALLOW = "shallow"
DEEP = "deep"

#: 关掉闸门 = 回到"每轮都检索"的老行为。出了问题先拿它对比，别去猜。
ON = os.environ.get("VOICEMEM_TURN_GATE", "1") != "0"


# ── ① 附和词表（闭集，0ms）────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """去掉空格和标点，只留字母数字。词表和输入走同一个函数，免得 "got it"
    （词表里有空格）永远匹配不上 "gotit"（输入已去空格）。"""
    return "".join(ch for ch in (s or "") if ch.isalnum()).casefold()


_BACKCHANNEL_RAW = {
    # 中文
    "嗯嗯", "嗯哼", "嗯呐", "对对", "对啊", "对的", "是的", "是啊", "好的", "好呀",
    "好嘞", "行吧", "知道了", "明白", "懂了", "原来如此", "这样啊", "这样",
    "嗯", "呃", "哦", "噢", "喔", "欸", "诶", "啊", "唉", "对", "是", "好", "行",
    # 英文
    "uh-huh", "mhm", "mm-hmm", "yeah", "yep", "yes", "okay", "ok", "right",
    "sure", "gotcha", "got it", "i see", "cool", "nice", "wow", "hmm", "huh",
}
#: 按长度倒序贪心切分，所以 "对的" 要排在 "对" 前面。
_BACKCHANNEL = sorted({_norm(w) for w in _BACKCHANNEL_RAW}, key=len, reverse=True)


#: 归一化的公开名字。打断判定那边（demo 的 _barge_text）也要用同一个，
#: 各写各的就会出现"词表匹配得上、打断判定匹配不上"这种半生效。
norm = _norm


def is_backchannel(text: str) -> bool:
    """整段都能被词表切完才算纯附和。"""
    s = _norm(text)
    if not s:
        return False
    while s:
        for w in _BACKCHANNEL:
            if s.startswith(w):
                s = s[len(w):]
                break
        else:
            return False
    return True


# ── ② 表层正则（高精度快路，0ms）─────────────────────────────────────────────

#: 深/浅的真正分界是"有没有指向这个人的私有历史"，它在表层留下三种痕迹：
#: 第一人称物主、时间回指、定指回指。中英各写一套——库支持英文，只写中文
#: 等于英文全归零（实测英文原话上正则命中率直接掉到 0）。
_PERSONAL = re.compile(
    r"我|咱|自己"
    r"|上次|上回|之前|刚才|昨天|上周|上个月|去年|那天|后来|来着"
    r"|那个|那件|那只|那家|那首|说过|提过|答应过|记得"
    r"|\b(i|me|my|mine|we|our|myself)\b"
    r"|\b(last (time|week|month|year|night)|yesterday|earlier|before|ago|used to)\b"
    r"|\b(that (one|thing|guy|place|song)|remember|told you|mentioned|promised)\b",
    re.IGNORECASE)

#: 对助手下指令 / 指涉它刚说过的话：句中虽然有"我""刚才"，但都不指向个人历史。
#: "帮我翻译这段话"换个人问答案一样；"刚才那句什么意思"要的是对话历史（那个由
#: _history_block 单独注入），不是记忆库。必须排在 _PERSONAL 之前，否则全被
#: "我"字捞成 deep——实测这一条就贡献了 6/8 的误判。
_META = re.compile(
    r"^(帮|替|教|陪)我"
    r"|^给我(讲|写|念|唱|放|来|定|设)"
    r"|^我们(换|聊|说点|来点)"
    r"|^(刚才|刚刚|刚)(那|这)?(句|段|个词|说的)"
    r"|^(help|tell|give|show|read|sing|play|write|set|translate) me\b"
    r"|^(let'?s|lets) (chat|talk|change|switch)\b"
    r"|^what did you (just )?say\b",
    re.IGNORECASE)

#: "明天什么安排"这类：一个第一人称都没有，问的却是自己的日程。时间词 + 日程词
#: 是个稳定组合，不是为某一句打的补丁。
_SCHEDULE = re.compile(
    r"(今天|明天|后天|周末|下周|下个月|这周|晚上|早上|下午)[^，。]{0,6}"
    r"(安排|日程|计划|要做|有事|有约|干嘛|干什么|忙什么)"
    r"|\b(schedule|plans?|anything on|anything going on)\b.{0,12}"
    r"\b(today|tomorrow|tonight|next week|this week|weekend)\b"
    r"|\b(today|tomorrow|tonight|next week|this week|weekend)\b.{0,12}"
    r"\b(schedule|plans?|anything|free|busy)\b",
    re.IGNORECASE)


def _lexical(text: str) -> str | None:
    """命中返回 DEEP；说不上来返回 None，交给句向量那一级。"""
    t = text or ""
    if _META.search(t):
        return None
    if _PERSONAL.search(t) or _SCHEDULE.search(t):
        return DEEP
    return None


# ── ③ 句向量兜底 ─────────────────────────────────────────────────────────────

_PROTO = {
    "zh": ([  # 要调用户个人历史才能答
        "我明天有什么安排", "我上次说的那件事怎么样了", "我朋友叫什么名字",
        "我在哪家公司上班", "我上周去哪儿了", "我妈生日是哪天",
        "我之前提过的那个计划", "我最近是不是有点累", "我们以前聊过的话题",
        "我平时的习惯是什么", "我说过我喜欢什么", "我还有什么事没做完",
    ], [  # 通用知识 / 闲聊 / 对助手的指令，跟这个人是谁无关
        "讲个笑话", "今天天气怎么样", "你是谁", "再说一遍",
        "帮我算一下三加五", "什么是量子力学", "唱首歌", "声音大一点",
        "推荐一部电影", "现在几点了", "帮我翻译一句话", "介绍一个城市",
    ]),
    "en": ([
        "what's on my schedule tomorrow", "how did that thing I mentioned turn out",
        "what's my friend's name", "where do I work", "where did I go last week",
        "when is my mother's birthday", "that plan I brought up before",
        "have I been tired lately", "what did we talk about before",
        "what are my usual habits", "what did I say I like",
        "what do I still have left to do",
    ], [
        "tell me a joke", "what's the weather today", "who are you", "say that again",
        "what is three plus five", "what is quantum mechanics", "sing a song",
        "turn the volume up", "recommend a movie", "what time is it",
        "translate this sentence", "tell me about a city",
    ]),
}

#: 判到这个差值以上才算"偏深"。两组原型的余弦差本来就只有 0.01 量级——深/浅
#: 是横切话题的功能维度，而句向量空间是被话题主导的，这个轴在里面几乎不存在。
#: 所以这一级只当兜底裁决，别指望它扛主力；阈值取 0（拿不准就偏深）。
MARGIN = float(os.environ.get("VOICEMEM_GATE_MARGIN", "0.0"))


@lru_cache(maxsize=2)
def _banks(lang: str):
    """两组原型的向量。跟记忆库共用同一份模型，所以这里不额外加载权重。"""
    import numpy as np
    from voicemem.leftbrain.local_embedder import resolve, resolve_path, shared_model
    spec = resolve(language=lang)
    m = shared_model(resolve_path(spec))
    deep, shallow = _PROTO.get(lang) or _PROTO["en"]
    enc = lambda xs: np.asarray(
        m.encode([f"{spec.passage_prefix}{x}" for x in xs], normalize_embeddings=True))
    return m, spec, enc(deep), enc(shallow)


def semantic_margin(text: str, lang: str = "en") -> float:
    """>0 偏深，<0 偏浅。取 top-3 均值，抗单句噪声。"""
    import numpy as np
    m, spec, deep, shallow = _banks(lang)
    q = np.asarray(m.encode([f"{spec.query_prefix}{text}"],
                            normalize_embeddings=True)[0])
    top3 = lambda M: float(np.sort(M @ q)[-3:].mean())
    return top3(deep) - top3(shallow)


# ── 对外 ─────────────────────────────────────────────────────────────────────

def route(text: str, *, semantic: bool = True, language: str = "") -> str:
    """这一句走哪条路。

    ``semantic=False`` 跳过句向量那一级（词表+正则就够、或者不想碰模型时用）。
    ``language`` 不给就读当前库语言。
    """
    if not ON:
        return DEEP                       # 闸门关掉 = 每轮都检索的老行为
    t = (text or "").strip()
    if not t:
        return SHALLOW
    if is_backchannel(t):
        return BACKCHANNEL
    hit = _lexical(t)
    if hit:
        return hit
    if not semantic:
        return SHALLOW
    if not language:
        try:
            from voicemem.lang import memory_language
            language = memory_language()
        except Exception:
            language = "en"
    try:
        return DEEP if semantic_margin(t, language) > MARGIN else SHALLOW
    except Exception as e:
        # 模型没装/加载失败：宁可多检索一次，也不能因为兜底不可用就把记忆丢了。
        print(f"[gate] 句向量兜底不可用（{type(e).__name__}: {e}）→ 按 deep 走", flush=True)
        return DEEP


def needs_memory(r: str) -> bool:
    """这一路要不要把检索到的**事实记忆**拼进 prompt。"""
    return r == DEEP


def interrupts(r: str) -> bool:
    """这一路要不要打断正在播的回复。附和不打断，其余都打断。"""
    return r != BACKCHANNEL
