"""附和（backchannel）：用户说到一半停顿时，"嗯"一声。

人在听别人说话时不是全程沉默的——每隔几秒会"嗯""对""是吗"一下。少了这个，说话
的人会觉得对面走神了或者线断了。语音助手全程静默听完再开口，不自然的正是这里。

触发点是 **VAD 停顿 100ms**。但**不能每次停顿都出声**：100ms 的静音在句子内部到处
都是（词间、塞音前、换气），照字面每次都"嗯"会变成灾难。所以这里给一个概率，
而概率由六个因子相乘决定——它们都是会话分析里反复验证过的附和触发条件：

    p = session 基准概率 · S(句式) · M(内容) · E(情绪) · P(韵律) · R(不应期)

每个因子的取值和理由见下面各自的函数。``probability()`` 会把分解一起返回，
调的时候能看见是哪一项把概率拉高/压低的——凭耳朵调一个黑盒数字是调不动的。

用法（调用方每帧问一次，代价是几十微秒的正则）::

    bc = Backchannel()
    token = bc.offer(text=st.text, silence=st.silence, spoke=st.spoke,
                     speech_s=..., tail_rms=..., emotion=owner["emotion"])
    if token:  # 播一个预合成的短音
        ...

``VOICEMEM_BACKCHANNEL_EMIT=1`` 打开（**默认关**）。别跟 ``VOICEMEM_BACKCHANNEL``
搞混：那个管**听**（用户说"嗯嗯"时不算一轮），这个管**说**。
"""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field

from .frequency import SessionFrequencyCurve

#: 打开助手**说**附和。默认关——它改变的是产品的说话方式，不该在别人不知情时
#: 突然开始出声。
#:
#: 注意跟 ``VOICEMEM_BACKCHANNEL`` 区分，那个管的是**听**（用户说"嗯嗯"时不算一轮、
#: 不打断助手）。一个是听见附和怎么办，一个是要不要自己发出附和，方向相反。
#:
#: Read the environment at decision time instead of import time so command-line
#: setup and test overrides remain reliable regardless of module import order.
def emitting() -> bool:
    return os.environ.get("VOICEMEM_BACKCHANNEL_EMIT", "0") != "0"
#: 打印每次机会的概率和因子分解。调这套东西全靠它。
DEBUG = os.environ.get("VOICEMEM_BC_DEBUG", "0") != "0"


# ── 各因子 ────────────────────────────────────────────────────────────────────

#: 疑问句尾。问句要的是**回答**，不是点头——这时候"嗯"一声是最扫兴的反应，
#: 所以这一项压得很低而不是归零（"…是吧？"这种确认式问句仍然可以"嗯"）。
_ASK = re.compile(r"(吗|呢|吧|么)\s*[?？]?$|[?？]$"
                  r"|^(什么|怎么|为什么|哪|谁|多少|几)"
                  r"|\b(what|why|how|who|where|when|which|can you|do you|是不是)\b",
                  re.IGNORECASE)

#: 明确在邀请回应的话尾。人说完这些就是在等你出声，不接反而尴尬。
_INVITE = re.compile(r"(对吧|是吧|你说呢|你知道吗|是不是|懂吧|明白吗)\s*[?？]?$"
                     r"|\b(you know|right|you see)\s*[?？]?$", re.IGNORECASE)

#: 还没说完的连接词尾。停在这儿是在组织语言，不是让位——这时候"嗯"是**支持**，
#: 会让人接着说下去（会话分析里叫 continuer）。
_CONTINUE = re.compile(r"(然后|就是|因为|所以|但是|不过|而且|反正|其实|那个)$"
                       r"|\b(and then|so|because|but|like|i mean)$", re.IGNORECASE)

#: 情感自我表露。人讲自己难受/开心的事时最需要听到回应，这也是附和最该出现的地方。
_DISCLOSE = re.compile(r"(压力|累|烦|难过|委屈|焦虑|失眠|担心|害怕|生气|开心|高兴|"
                       r"激动|喜欢|讨厌|受不了|不容易|太难了)"
                       r"|\b(stressed|tired|exhausted|upset|anxious|worried|scared|"
                       r"angry|happy|excited|love|hate|can't stand)\b", re.IGNORECASE)

#: 负面/高唤起情绪 → 更该给支持性的附和；平静的陈述不需要那么频繁。
_EMOTION_WEIGHT = {
    "难过": 1.5, "悲伤": 1.5, "委屈": 1.5, "焦虑": 1.4, "孤独": 1.4,
    "疲惫": 1.3, "纠结": 1.3, "生气": 1.2, "愤怒": 1.2,
    "开心": 1.2, "愉悦": 1.2, "惊讶": 1.1,
    "平静": 1.0,
    "sad": 1.5, "wronged": 1.5, "anxious": 1.4, "lonely": 1.4,
    "tired": 1.3, "conflicted": 1.3, "angry": 1.2, "happy": 1.2, "calm": 1.0,
}


def _envf(name: str, default: float) -> float:
    """策略参数都可以用环境变量压过去——调这套东西只能靠耳朵反复试，
    每次改代码重启太慢，而且很容易忘了改回来。"""
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class BackchannelPolicy:
    """所有可调的量集中在这儿，别散到判断逻辑里去。

    全部可以用 env 覆盖，方便边听边调::

        VOICEMEM_BC_MAX_GAP=0.6   # 放宽停顿窗口（默认只认 100~200ms）
        VOICEMEM_BC_REFRACTORY=5  # 冷却延长到 5 秒；最少 3 秒
    """
    #: VAD 停多久算一次机会。太短会踩到词间停顿，太长就赶不上——100ms 是人耳
    #: 觉得"他停了一下"的下限，也正好在 gamble_s(200ms) 之前，不会跟回复撞车。
    gap_s: float = field(default_factory=lambda: _envf("VOICEMEM_BC_GAP", 0.10))
    #: 超过这个就不是"停顿"而是"说完了"，该走正常回复，不该在这儿插一声。
    #: 上限贴着 confirm_s（300ms 判一轮结束）——原来卡在 200ms，把 200~300ms 那段
    #: 停顿全漏掉了，而人自然的句中停顿大多落在那儿，所以实际很少触发。
    max_gap_s: float = field(default_factory=lambda: _envf("VOICEMEM_BC_MAX_GAP", 0.28))
    #: 说够这么多字才考虑附和。刚开口两个字就"嗯"像在敷衍，但 6 个字对短句太苛刻。
    min_chars: int = field(default_factory=lambda: int(_envf("VOICEMEM_BC_MIN_CHARS", 4)))
    #: 距上次附和的硬冷却：这段时间内概率恒为 0。
    refractory_s: float = field(default_factory=lambda: max(3.0, _envf("VOICEMEM_BC_REFRACTORY", 3.0)))
    #: 概率上限。再怎么该附和也不能**每次**停顿都出声，那样同样机械——留下的
    #: 随机性正是它像人的原因。0.75 意味着最该附和的时候大约四次里应三次。
    p_max: float = field(default_factory=lambda: _envf("VOICEMEM_BC_PMAX", 0.9))
    #: Session-level baseline: turns 1–3 use 50%, 4–6 use 10%, then 30%.
    session_curve: SessionFrequencyCurve = field(default_factory=SessionFrequencyCurve)

def f_sentence(text: str) -> float:
    """S —— 句式。问句要答案不要点头；邀请式和未完成句最该点头。"""
    t = (text or "").strip()
    if not t:
        return 1.0
    if _INVITE.search(t):
        return 1.8          # "……对吧？" 不接反而尴尬
    if _CONTINUE.search(t):
        return 1.4          # 停在连接词上：一声"嗯"会让他接着说
    if _ASK.search(t):
        return 0.15         # 问句：他在等答案，"嗯"是最扫兴的回应
    return 1.0


def f_content(text: str) -> float:
    """M —— 内容。情感自我表露最需要被接住。"""
    return 1.4 if _DISCLOSE.search(text or "") else 1.0


def f_emotion(emotion: str) -> float:
    """E —— 上一轮感知到的情绪。负面/高唤起时人更需要听到"我在听"。"""
    return _EMOTION_WEIGHT.get((emotion or "").strip(), 1.0)


def f_prosody(tail_rms: float, prev_rms: float) -> float:
    """P —— 韵律，只用能量包络这一个便宜信号。

    真正管用的是基频走向（降调=让位，升调=疑问），但基频要跑一遍提取，而这条路
    的预算是 0ms。能量包络是免费的近似：**尾音在衰减**说明一个语块讲完了、在让位，
    这正是附和该落的位置；**戛然而止**（还很响就停了）多半是被自己打断或在想词，
    这时候插一声容易撞上他接下来的话。
    """
    if prev_rms <= 1e-6:
        return 1.0
    ratio = tail_rms / prev_rms
    if ratio < 0.45:
        return 1.3          # 明显在收尾
    if ratio > 0.9:
        return 0.8          # 说得正响就停了，别插
    return 1.0


def f_refractory(since_s: float, policy: BackchannelPolicy) -> float:
    """R —— 不应期。连着"嗯嗯嗯"比一次都不嗯更假。"""
    cooldown = max(3.0, policy.refractory_s)
    return 0.0 if since_s < cooldown else 1.0


# ── 说哪个词 ──────────────────────────────────────────────────────────────────

ZH_AFFIRMATIVE_TOKENS = ("哦", "哦哦", "嗯嗯", "嗯", "对", "明白")
ZH_QUESTION_TOKENS = ("哦？", "嗯？", "是吗？")
ZH_SYNTHESIS_TEXT = {
    token: token if token.endswith("？") else f"{token}。"
    for token in (*ZH_AFFIRMATIVE_TOKENS, *ZH_QUESTION_TOKENS)
}
ZH_VARIANTS = {
    **{token: 2 for token in ZH_AFFIRMATIVE_TOKENS},
    **{token: 1 for token in ZH_QUESTION_TOKENS},
}

# Chinese acknowledgements deliberately stay within two restrained intents:
# attentive affirmation and mild curiosity. Strong praise, surprise, sympathy,
# or dismay makes a cached clip sound emotionally wrong when context is noisy.
_TOKENS = {
    "zh": {
        "continuer":  ["嗯", "嗯嗯"],
        "support":    ["嗯", "嗯嗯", "明白"],
        "agree":      ["嗯", "对", "明白"],
        "surprise":   list(ZH_QUESTION_TOKENS),
        "assess_good":["嗯嗯", "对"],
        "assess_bad": ["嗯", "明白"],
        "neutral":    ["嗯", "嗯嗯", "哦", "哦哦", "对", "明白"],
    },
    "en": {
        "continuer":  ["mm-hmm", "mmm", "uh-huh"],
        "support":    ["mm-hmm", "mmm", "oh", "i see", "okay"],
        "agree":      ["yeah", "right", "uh-huh", "exactly", "for sure"],
        "surprise":   ["really", "oh wow", "no way", "oh really", "huh"],
        "assess_good":["that's great", "nice", "amazing", "that's awesome"],
        "assess_bad": ["that's rough", "oh no", "sorry to hear that", "that's tough"],
        "neutral":    ["mm-hmm", "uh-huh", "yeah", "okay", "right", "mmm"],
    },
}

#: 「新消息」：状态变了、发生了一件事。人听到这些会"是吗？""真的？"——那是在
#: 请对方多说一点，不是在点头。看的是**变化**，不是话题。
_NEWS = re.compile(
    r"(升职|加薪|辞职|离职|跳槽|失业|分手|结婚|订婚|离婚|怀孕|搬家|搬去|考上|"
    r"录取|中了|拿到|买了|卖了|住院|生病|出事|退休|毕业|第一次|终于)"
    r"|\b(got promoted|quit|resigned|laid off|broke up|got married|engaged|"
    r"pregnant|moving|moved|got in|got accepted|won|bought|sold|"
    r"in hospital|retired|graduated|finally|for the first time)\b",
    re.IGNORECASE)

#: 好事 / 坏事。**必须分开**：回错方向比不回严重得多。两边都命中就当拿不准，不用。
_GOOD = re.compile(
    r"(升职|加薪|考上|录取|中了|拿到|通过|成功|终于|搞定|赢了|结婚|订婚|怀孕|毕业)"
    r"|\b(got promoted|got a raise|got in|got accepted|won|passed|made it|"
    r"finally|nailed it|got married|engaged|graduated)\b", re.IGNORECASE)
_BAD = re.compile(
    r"(失业|裁员|辞退|分手|离婚|住院|生病|去世|没通过|挂了|失败|取消|丢了|出事|被拒)"
    r"|\b(laid off|fired|broke up|divorced|in hospital|passed away|failed|"
    r"rejected|cancell?ed|lost)\b", re.IGNORECASE)
_NEGATIVE = {"难过", "悲伤", "委屈", "焦虑", "孤独", "疲惫", "纠结",
             "sad", "wronged", "anxious", "lonely", "tired", "conflicted"}


def pick_group(text: str, emotion: str) -> str:
    """这句话在向我要什么。按"回错了有多伤"排优先级。

    评价排最前，因为它信号最强也最不该错过——人说"我升职了"，你只"嗯"一声是冷淡的。
    但它也**只在好坏明确时**才用：好坏词同时命中，就当拿不准，退回去。
    """
    t = (text or "").strip()
    good, bad = bool(_GOOD.search(t)), bool(_BAD.search(t))
    if good != bad:                       # 方向明确才敢评价
        return "assess_good" if good else "assess_bad"
    if _NEWS.search(t):
        return "surprise"                 # 是新消息但说不清好坏 → "是吗？"最安全
    if (emotion or "").strip() in _NEGATIVE or _DISCLOSE.search(t):
        return "support"
    if _INVITE.search(t):
        return "agree"
    return "neutral"


def pick_token(text: str, emotion: str, lang: str, rng: random.Random,
               recent=(), available=None) -> str:
    """挑一个词。避开最近说过的几个，否则很快听出在轮播。

    有一成概率从 neutral 里抽（当前组不是 neutral 时）——真人也不是每次都精准
    对应，偶尔就"嗯"一声。纯按规则走反而机械。
    """
    bank = _TOKENS.get(lang) or _TOKENS["en"]
    group = pick_group(text, emotion)
    if group != "neutral" and rng.random() < 0.1:
        group = "neutral"
    pool = bank[group]
    if available is not None:
        pool = [t for t in pool if t in available]
        if not pool:
            pool = [t for t in bank["neutral"] if t in available]
        if not pool:
            return ""
    choices = [t for t in pool if t not in recent] or pool
    return rng.choice(choices)


# ── 主入口 ────────────────────────────────────────────────────────────────────

@dataclass
class Backchannel:
    """一路会话一个实例：它要记住上次什么时候附和过、说的哪个词。"""
    policy: BackchannelPolicy = field(default_factory=BackchannelPolicy)
    rng: random.Random = field(default_factory=random.Random)
    _last_at: float | None = None
    _last_token: str = ""
    #: 最近说过的几个词，挑新词时全部避开。只避开上一个的话，"嗯嗯/哦/嗯嗯/哦"
    #: 交替出现，两三次就听出是在轮播。
    _recent: list = field(default_factory=list)
    _armed: bool = True          # 这次停顿还没判过（一次停顿只判一次）
    _speech_started: float = 0.0
    _completed_turns: int = 0

    @property
    def completed_turns(self) -> int:
        return self._completed_turns

    def reset_turn(self) -> None:
        self._speech_started = 0.0
        self._armed = True

    def complete_turn(self) -> None:
        """Advance the session frequency curve after one accepted user turn."""
        self._completed_turns += 1
        self.reset_turn()

    def can_emit(self, now: float | None = None) -> bool:
        """Return whether the shared session cooldown permits another clip."""
        now = now if now is not None else time.monotonic()
        return (emitting() and
                (self._last_at is None or
                 now - self._last_at >= max(3.0, self.policy.refractory_s)))

    def choose(self, *, text: str, emotion: str = "", lang: str = "zh",
               available=None, now: float | None = None) -> str | None:
        """Choose an acknowledgement without committing it as emitted."""
        if not self.can_emit(now):
            return None
        token = pick_token(text, emotion, lang, self.rng, self._recent, available)
        return token or None

    def mark_emitted(self, token: str, now: float | None = None) -> None:
        """Commit a clip only after the transport has accepted it."""
        if not token:
            return
        self._last_at = now if now is not None else time.monotonic()
        self._last_token = token
        self._recent = (self._recent + [token])[-3:]

    def probability(self, *, text: str, speech_s: float, emotion: str = "",
                    tail_rms: float = 0.0, prev_rms: float = 0.0,
                    now: float | None = None) -> tuple[float, dict]:
        """算这次机会的概率，连同分解一起返回（分解是给调参的人看的）。"""
        p = self.policy
        now = now if now is not None else time.monotonic()
        parts = {
            # The session curve now owns baseline frequency. Eligible pauses no
            # longer decay merely because they happen early in an utterance.
            "S": f_sentence(text),
            "M": f_content(text),
            "E": f_emotion(emotion),
            "P": f_prosody(tail_rms, prev_rms),
            "R": f_refractory(now - self._last_at if self._last_at is not None else 1e9, p),
        }
        val = p.session_curve.probability(self._completed_turns)
        for v in parts.values():
            val *= v
        return min(p.p_max, val), parts

    def offer(self, *, text: str, silence: float, spoke: bool, speech_s: float,
              emotion: str = "", tail_rms: float = 0.0, prev_rms: float = 0.0,
              lang: str = "zh", now: float | None = None, available=None,
              unfinished: bool = False) -> str | None:
        """这一帧要不要附和。返回要说的词，或 None。

        一次停顿只掷一次骰子（``_armed``）：不然 100ms 到 200ms 之间每帧都判一次，
        等于把概率乘上帧数，必出声。
        """
        now = now if now is not None else time.monotonic()
        if silence < 0.01:                     # 还在说 → 为下一次停顿重新上膛
            self._armed = True
            if spoke and not self._speech_started:
                self._speech_started = now
            return None
        if not (emitting() and spoke and self._armed):
            return None
        if silence < self.policy.gap_s or (not unfinished and silence >= self.policy.max_gap_s):
            return None                        # 不在窗口里：太短，或已经是"说完了"
        if not (text or "").strip() or (not unfinished and len(text.strip()) < self.policy.min_chars):
            return None
        self._armed = False                    # 这次停顿判过了，不再判
        if self._last_at is not None and now - self._last_at < max(3.0, self.policy.refractory_s):
            return None
        speech = speech_s or (now - self._speech_started if self._speech_started else 0.0)
        p, parts = self.probability(text=text, speech_s=speech, emotion=emotion,
                                    tail_rms=tail_rms, prev_rms=prev_rms, now=now)
        roll = self.rng.random()
        if DEBUG:
            # 每一次「合格停顿」都打一行。不打的话，没出声到底是没轮到机会、
            # 还是掷骰子没中，外面完全看不出来——这正是最难调的地方。
            f = " ".join(f"{k}={v:.2f}" for k, v in parts.items())
            print(f"[backchannel] turn={self._completed_turns + 1} 机会 p={p:.0%} 掷={roll:.2f} "
                  f"{'中' if roll < p else '不中'}  {f}  说了{speech:.1f}s  "
                  f"{text[-12:]!r}", flush=True)
        if roll >= p:
            return None
        if unfinished:
            pool = (_TOKENS.get(lang) or _TOKENS["en"])["continuer"]
            pool = [t for t in pool if available is None or t in available]
            token = self.rng.choice([t for t in pool if t not in self._recent] or pool) if pool else ""
        else:
            token = pick_token(text, emotion, lang, self.rng, self._recent, available)
        if not token:
            return None
        self.mark_emitted(token, now)
        return token


# ── 声音：预合成 + 落盘缓存 ───────────────────────────────────────────────────
#
# 为什么要预合成：TTS 首帧就要 ~1.2s，而这里的窗口是 100–200ms。现合成必然赶不上,
# 赶不上的附和比没有更糟——话都说完两句了才"嗯"一声。
#
# 为什么一个词要存好几条：同一段波形反复播，两三次就听出是罐头。给 TTS 不同的
# 语气指示各合成一条，播的时候再随机挑 + 抖一点速度，才像每次都是新说的。
#
# 为什么落盘：每次起服务重合成一遍又慢又花钱。key 里带了 TTS 后端和音色，
# 换音色自动重新生成——不带的话换了声音还在放旧音色的"嗯"，比不放更吓人。

#: 每个词换这几种念法各合成一条。语气指示喂给 TTS 的 instruction 参数
#: （不认这个参数的后端会退回单一念法，见 _synth_one 的 TypeError 分支）。
from voicemem.prompt_config import tts_prompts
_STYLES = tts_prompts()["backchannel_styles"]


def _cache_root():
    from pathlib import Path
    env = os.environ.get("VOICEMEM_CACHE_DIR")
    if env:
        return Path(env) / "backchannel"
    from voicemem.utils.common.paths import models_dir
    return models_dir().parent / "cache" / "backchannel"


class BackchannelVoice:
    """把附和词预合成成 24k PCM16，按（后端+音色+词+语气）缓存。

    ``prime()`` 放后台跑：合成十来条要十几秒，但用户说完第一句之前通常就好了，
    没好之前 ``get()`` 返回 None，调用方安静跳过——宁可这一次不附和。
    """

    def __init__(self, tts, *, lang: str = "zh", voice_id: str = ""):
        self.tts = tts
        self.lang = lang if lang in _TOKENS else "en"
        # 音色标识进 key。取不到就用类名——总比不带强，至少换后端时会失效。
        self.voice_id = getattr(tts, "cache_voice_id", "") or voice_id or getattr(tts, "voice", "") or type(tts).__name__
        self._clips: dict[str, list[bytes]] = {}
        self._ready = False
        self.primed = False

    @property
    def available(self):
        return set(self._clips)

    @property
    def ready(self) -> bool:
        return self._ready

    def _path(self, token: str, style_idx: int):
        import hashlib
        spoken = self._spoken_text(token)
        key = (f"{self.voice_id}|{self.lang}|{token}|{spoken}|{style_idx}|"
               f"{_STYLES[self.lang][style_idx]}")
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return _cache_root() / f"{h}.pcm"

    def _spoken_text(self, token: str) -> str:
        """Return synthesis text with punctuation that supplies natural prosody."""
        if self.lang == "zh":
            return ZH_SYNTHESIS_TEXT.get(token, token)
        return token

    def _style_indices(self, token: str, variants=None) -> range:
        """Return the configured variants for one token."""
        count = len(_STYLES[self.lang])
        if variants is not None:
            count = min(count, max(0, int(variants)))
        if self.lang == "zh":
            count = min(count, ZH_VARIANTS.get(token, 1))
        return range(count)

    @staticmethod
    def _bundled_filename(token: str, style_idx: int) -> str:
        suffix = "" if style_idx == 0 else f"_{style_idx + 1}"
        return f"OK_{token}{suffix}.wav"

    def _uses_bundled_voice(self) -> bool:
        if self.lang != "zh":
            return False
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        bundled_ref = root / "voice" / "noctelle_ref_short.wav"
        ref = getattr(self.tts, "ref_audio", None)
        try:
            return bool(ref and Path(ref).read_bytes() == bundled_ref.read_bytes())
        except OSError:
            return False

    def _bundled_pcm(self, token: str, style_idx: int) -> bytes:
        """Load a reviewed clip when this TTS uses the bundled Noctelle voice.

        The WAV bank is deliberately voice-specific.  Comparing the reference
        bytes prevents a custom voice with the same filename from silently
        playing Noctelle's acknowledgements.
        """
        if not self._uses_bundled_voice() or "/" in token or "\\" in token:
            return b""
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        try:
            clip = (root / "voice" / "backchannel" /
                    self._bundled_filename(token, style_idx))
            if not clip.is_file():
                return b""
            import wave
            with wave.open(str(clip), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
                    raise ValueError(f"bundled backchannel has wrong format: {clip}")
                return wav.readframes(wav.getnframes())
        except OSError:
            return b""

    def _tokens(self) -> list[str]:
        seen, out = set(), []
        for group in _TOKENS[self.lang].values():
            for t in group:
                if t not in seen:
                    seen.add(t); out.append(t)
        return out

    # Reject pathological generations instead of cutting through a syllable.
    MAX_MS = 3000
    # RMS threshold used only to locate leading and trailing silence.
    SILENCE = 0.006
    LEADING_MS = 20
    TRAILING_MS = 120
    MIN_NATURAL_TAIL_MS = 60
    #: 归一化目标响度（RMS，dBFS）。"很轻""随口"这类指示会让模型把音量也压下去，
    #: 实测合出来只有 -35 dBFS，扬声器里根本听不见。响度归一，轻重靠指示词管语气。
    TARGET_DB = float(os.environ.get("VOICEMEM_BC_TARGET_DB", "-24"))

    @staticmethod
    def _trim(pcm: bytes, max_ms: int = MAX_MS) -> bytes:
        """Trim silence while preserving the complete final syllable."""
        import numpy as np
        a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        if a.size == 0:
            return pcm
        window = 240
        power = np.convolve(
            a * a, np.full(window, 1.0 / window, dtype=np.float32), mode="same")
        loud = np.sqrt(power) > BackchannelVoice.SILENCE
        if not loud.any():
            return b""
        i, j = int(np.argmax(loud)), int(a.size - np.argmax(loud[::-1]))
        leading = round(BackchannelVoice.LEADING_MS * 24)
        trailing = round(BackchannelVoice.TRAILING_MS * 24)
        natural_tail = max(0, a.size - j)
        minimum_tail = round(BackchannelVoice.MIN_NATURAL_TAIL_MS * 24)
        if natural_tail < minimum_tail:
            edge_n = min(a.size, 480)
            edge_rms = float(np.sqrt(np.mean(a[-edge_n:] * a[-edge_n:])))
            voiced_rms = float(np.sqrt(np.mean(a[i:j] * a[i:j])))
            if edge_rms > max(BackchannelVoice.SILENCE, voiced_rms * 0.2):
                return b""
        start, end = max(0, i - leading), min(a.size, j + trailing)
        a = a[start:end]
        missing_tail = trailing - max(0, end - j)
        if missing_tail > 0:
            a = np.pad(a, (0, missing_tail))
        cap = int(max_ms * 24000 / 1000)
        if a.size > cap:
            return b""
        fade = min(240, a.size // 4)
        if fade > 0:
            a[:fade] *= np.linspace(0, 1, fade)
            a[-fade:] *= np.linspace(1, 0, fade)
        return BackchannelVoice._normalize((a * 32767).astype(np.int16).tobytes())

    @staticmethod
    def _normalize(pcm: bytes) -> bytes:
        """RMS 拉到 TARGET_DB，峰值封在 0.95 以内。幂等，缓存读出来再过一遍不变。"""
        import numpy as np
        a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        if a.size == 0:
            return pcm
        rms, peak = float(np.sqrt(np.mean(a * a))), float(np.abs(a).max())
        if rms < 1e-5 or peak < 1e-5:
            return pcm
        g = min(10 ** (BackchannelVoice.TARGET_DB / 20) / rms, 0.95 / peak)
        if abs(g - 1.0) < 0.02:
            return pcm
        return (np.clip(a * g, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    async def _synth_one(self, text: str, instruction: str) -> bytes:
        buf = bytearray()
        spoken = self._spoken_text(text)
        try:
            stream = self.tts.stream(spoken, instruction)
        except TypeError:            # 用户自己的 TTS 只认 stream(text)
            stream = self.tts.stream(spoken)
        async for chunk in stream:
            buf.extend(chunk)
        return self._trim(bytes(buf))

    async def prime(self, tokens=None, variants=None, cache_only=False) -> int:
        """合成（或从缓存读）全部词×语气。返回可用的条数。放后台调。

        **每一步都要出声报告。** 这一步失败的所有表现都是"附和不响"，而它可能卡在
        没缓存要现合成（几十秒）、TTS 报错、key 里的语言跟你以为的不一样……不打日志
        的话外面完全区分不了。

        缓存 key 里带语言：中文库存的那批英文库用不上，会重新合成一次。
        """
        import time as _t
        t0 = _t.time()
        root = _cache_root()
        root.mkdir(parents=True, exist_ok=True)
        tokens = self._tokens() if tokens is None else list(tokens)
        jobs = [(token, i) for token in tokens
                for i in self._style_indices(token, variants)]
        if cache_only and self._uses_bundled_voice():
            # The checked-in WAV directory is the user's reviewed selection.
            # Missing files intentionally disable variants even if an older
            # generated cache entry still exists.
            jobs = [job for job in jobs if self._bundled_pcm(*job)]
        active_jobs = set(jobs)
        bundled = 0
        for token, i in jobs:
            path = self._path(token, i)
            pcm = self._bundled_pcm(token, i)
            if pcm:
                # The reviewed bank is authoritative for the bundled voice.
                # A colleague may already have clips generated by an older
                # checkout; replace those too, otherwise pulling the WAVs
                # would appear to work while playback still used old audio.
                pcm = self._normalize(pcm)
                if not path.is_file() or path.read_bytes() != pcm:
                    path.write_bytes(pcm)
                bundled += 1
        cached = sum(1 for token, i in jobs if self._path(token, i).is_file())
        total = len(jobs)
        print(f"[backchannel] 开始预合成：{self.lang} / {self.voice_id} · "
              f"{total} 条，其中 {cached} 条已有缓存"
              + (f"（仓库定稿 {bundled} 条）" if bundled else "")
              + ("，仅加载缓存" if cache_only else
                 "" if cached == total else f"，要现合成 {total - cached} 条（首次启动）"),
              flush=True)
        n = 0
        for token in tokens:
            clips = []
            for i in self._style_indices(token, variants):
                if (token, i) not in active_jobs:
                    continue
                style = _STYLES[self.lang][i]
                path = self._path(token, i)
                try:
                    if path.is_file() and path.stat().st_size > 0:
                        clips.append(self._normalize(path.read_bytes()))   # 旧缓存没归一过
                    elif not cache_only:
                        pcm = await self._synth_one(token, style)
                        if pcm:
                            path.write_bytes(pcm)
                            clips.append(pcm)
                except Exception as e:
                    print(f"[backchannel] 合成 {token!r} 失败（跳过）："
                          f"{type(e).__name__}: {e}", flush=True)
            if clips:
                self._clips[token] = clips
                n += len(clips)
                # 攒一个就能用一个：不必等全部合完才开始附和。
                self._ready = True
        self._ready = bool(self._clips)
        self.primed = True
        print(f"[backchannel] 预合成完成：{len(self._clips)} 个词 / {n} 条音频 "
              f"（{_t.time() - t0:.1f}s）", flush=True)
        return n

    def get(self, token: str, rng: random.Random | None = None) -> bytes | None:
        """随机取该词的一条。没预合成好就返回 None（这一次就不附和）。"""
        clips = self._clips.get(token)
        if not clips:
            return None
        return (rng or random).choice(clips)
