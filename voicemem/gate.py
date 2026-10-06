"""Decide whether input is a backchannel and whether it needs factual memory.

backchannel leaves an ongoing reply uninterrupted; shallow permits interruption
without factual memory; deep permits interruption and factual memory retrieval.
The router applies its existing word lists, lexical rules and shared embedding
fallback. These labels determine memory eligibility, not reasoning depth.
Studio chooses reply depth separately. Partial routing may change as text grows;
the confirmed utterance supplies the final memory decision."""
from __future__ import annotations

import os
import re
from functools import lru_cache

BACKCHANNEL = "backchannel"
SHALLOW = "shallow"
DEEP = "deep"

# Disabling the gate makes every nonempty turn eligible for memory retrieval.
ON = os.environ.get("VOICEMEM_TURN_GATE", "1") != "0"



def _norm(s: str) -> str:
    """Normalize text and vocabulary entries by removing spacing and punctuation."""
    return "".join(ch for ch in (s or "") if ch.isalnum()).casefold()


_BACKCHANNEL_RAW = {
    "嗯嗯", "嗯哼", "嗯呐", "对对", "对啊", "对的", "是的", "是啊", "好的", "好呀",
    "好嘞", "行吧", "知道了", "明白", "懂了", "原来如此", "这样啊", "这样",
    "嗯", "呃", "哦", "噢", "喔", "欸", "诶", "啊", "唉", "对", "是", "好", "行",
    "uh-huh", "mhm", "mm-hmm", "yeah", "yep", "yes", "okay", "ok", "right",
    "sure", "gotcha", "got it", "i see", "cool", "nice", "wow", "hmm", "huh",
}
# Match longer vocabulary entries first during greedy segmentation.
_BACKCHANNEL = sorted({_norm(w) for w in _BACKCHANNEL_RAW}, key=len, reverse=True)


# Interruption checks use this same normalization function.
norm = _norm


def is_backchannel(text: str) -> bool:
    """Return true only when vocabulary entries cover the complete normalized input."""
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



# Personal references and references to prior events are lexical memory signals.
_PERSONAL = re.compile(
    r"我|咱|自己"
    r"|上次|上回|之前|刚才|昨天|上周|上个月|去年|那天|后来|来着"
    r"|那个|那件|那只|那家|那首|说过|提过|答应过|记得"
    r"|\b(i|me|my|mine|we|our|myself)\b"
    r"|\b(last (time|week|month|year|night)|yesterday|earlier|before|ago|used to)\b"
    r"|\b(that (one|thing|guy|place|song)|remember|told you|mentioned|promised)\b",
    re.IGNORECASE)

# Assistant instructions are evaluated first so personal pronouns do not force retrieval.
_META = re.compile(
    r"^(帮|替|教|陪)我"
    r"|^给我(讲|写|念|唱|放|来|定|设)"
    r"|^我们(换|聊|说点|来点)"
    r"|^(刚才|刚刚|刚)(那|这)?(句|段|个词|说的)"
    r"|^(help|tell|give|show|read|sing|play|write|set|translate) me\b"
    r"|^(let'?s|lets) (chat|talk|change|switch)\b"
    r"|^what did you (just )?say\b",
    re.IGNORECASE)

# Schedule questions can refer to personal history without a first-person pronoun.
_SCHEDULE = re.compile(
    r"(今天|明天|后天|周末|下周|下个月|这周|晚上|早上|下午)[^，。]{0,6}"
    r"(安排|日程|计划|要做|有事|有约|干嘛|干什么|忙什么)"
    r"|\b(schedule|plans?|anything on|anything going on)\b.{0,12}"
    r"\b(today|tomorrow|tonight|next week|this week|weekend)\b"
    r"|\b(today|tomorrow|tonight|next week|this week|weekend)\b.{0,12}"
    r"\b(schedule|plans?|anything|free|busy)\b",
    re.IGNORECASE)


def _lexical(text: str) -> str | None:
    """Return deep for a matching lexical rule; otherwise defer to semantic classification."""
    t = text or ""
    if _META.search(t):
        return None
    if _PERSONAL.search(t) or _SCHEDULE.search(t):
        return DEEP
    return None



_PROTO = {
    "zh": ([  # Personal-history prototypes.
        "我明天有什么安排", "我上次说的那件事怎么样了", "我朋友叫什么名字",
        "我在哪家公司上班", "我上周去哪儿了", "我妈生日是哪天",
        "我之前提过的那个计划", "我最近是不是有点累", "我们以前聊过的话题",
        "我平时的习惯是什么", "我说过我喜欢什么", "我还有什么事没做完",
    ], [  # General knowledge, conversation and assistant instructions.
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

# Retrieve when the deep-minus-shallow similarity margin exceeds this threshold.
MARGIN = float(os.environ.get("VOICEMEM_GATE_MARGIN", "0.0"))


@lru_cache(maxsize=2)
def _banks(lang: str):
    """Cache prototype embeddings using the same model as memory retrieval."""
    import numpy as np
    from voicemem.leftbrain.local_embedder import resolve, resolve_path, shared_model
    spec = resolve(language=lang)
    m = shared_model(resolve_path(spec), spec.tokenizer_kwargs)
    deep, shallow = _PROTO.get(lang) or _PROTO["en"]
    enc = lambda xs: np.asarray(
        m.encode([f"{spec.passage_prefix}{x}" for x in xs], normalize_embeddings=True))
    return m, spec, enc(deep), enc(shallow)


def semantic_margin(text: str, lang: str = "en") -> float:
    """Return the deep-minus-shallow similarity margin using each bank's top three matches."""
    import numpy as np
    m, spec, deep, shallow = _banks(lang)
    q = np.asarray(m.encode([f"{spec.query_prefix}{text}"],
                            normalize_embeddings=True)[0])
    top3 = lambda M: float(np.sort(M @ q)[-3:].mean())
    return top3(deep) - top3(shallow)



def route(text: str, *, semantic: bool = True, language: str = "") -> str:
    """Route input using lexical rules and an optional semantic fallback.

    Without an explicit language, prose selects the prototype bank and ambiguous
    short inputs use the current operation's language default.
    """
    if not ON:
        return DEEP
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
            from voicemem.lang import detect_language, memory_language
            language = detect_language(text, memory_language())
        except Exception:
            language = "en"
    try:
        return DEEP if semantic_margin(t, language) > MARGIN else SHALLOW
    except Exception as e:
        # If semantic routing is unavailable, preserve eligibility for memory retrieval.
        print(f"[gate] 句向量兜底不可用（{type(e).__name__}: {e}）→ 按 deep 走", flush=True)
        return DEEP


def needs_memory(r: str) -> bool:
    """Return whether this route permits factual memory in the reply context."""
    return r == DEEP


def interrupts(r: str) -> bool:
    """Return whether this route permits interruption; pure backchannels do not."""
    return r != BACKCHANNEL
