"""Editable pause mathematics, timing constants, and filler instruction."""
import re
from dataclasses import dataclass, field

BACKCHANNEL_QUIET_S = 2.0
BACKCHANNEL_GROUP_SPLIT_S = 6.0
BACKCHANNEL_EARLY_MAX = 2
BACKCHANNEL_EXCLUDED = ({"哦", "哦哦"}, {"是吗？"})

CONTROLS = {
    "pause_ms": 100,
    "unfinished_wait_ms": 1200,
    "unfinished_followup_s": 2.5,
    "backchannel_resume_ms": 300,
    "backchannel_cooldown_ms": 2000,
    "backchannel_opening_s": 3.0,
    "backchannel_recovery_s": 6.0,
    "backchannel_late_s": 10.0,
    "backchannel_opening_counts": ((2, 0.80), (1, 0.20)),
    "backchannel_middle_counts": ((1, 0.80), (0, 0.20)),
    "backchannel_steady_counts": ((1, 0.30), (2, 0.30), (3, 0.30), (0, 0.10)),
    "backchannel_late_counts": ((2, 0.80), (0, 0.20)),
}

@dataclass(frozen=True)
class SessionFrequencyCurve:
    """Draw one acknowledgement quota for each phase of an utterance."""

    opening_s: float = 3.0
    recovery_s: float = 6.0
    opening_counts: tuple[tuple[int, float], ...] = ((2, 0.80), (1, 0.20))
    middle_counts: tuple[tuple[int, float], ...] = ((1, 0.80), (0, 0.20))
    steady_counts: tuple[tuple[int, float], ...] = (
        (1, 0.30), (2, 0.30), (3, 0.30), (0, 0.10))
    late_s: float = 10.0
    late_counts: tuple[tuple[int, float], ...] = ((2, 0.80), (0, 0.20))

    def __post_init__(self) -> None:
        if not 0 <= self.opening_s <= self.recovery_s <= self.late_s:
            raise ValueError("turn frequency time boundaries are invalid")
        for distribution in (self.opening_counts, self.middle_counts,
                             self.steady_counts, self.late_counts):
            if (not distribution or any(count < 0 or probability < 0
                                        for count, probability in distribution)
                    or abs(sum(probability for _, probability in distribution) - 1.0) > 1e-9):
                raise ValueError("turn frequency count distributions must sum to one")

    def phase(self, speech_s: float) -> int:
        elapsed = max(0.0, float(speech_s))
        if elapsed < self.opening_s:
            return 0
        if elapsed < self.recovery_s:
            return 1
        if elapsed < self.late_s:
            return 2
        return 3

    def draw_target(self, speech_s: float, roll: float) -> int:
        distribution = (
            self.opening_counts, self.middle_counts, self.steady_counts, self.late_counts
        )[self.phase(speech_s)]
        threshold = min(1.0, max(0.0, float(roll)))
        total = 0.0
        for count, probability in distribution:
            total += probability
            if threshold < total:
                return count
        return distribution[-1][0]

_ASK = re.compile(r"(吗|呢|吧|么)\s*[?？]?$|[?？]$"
                  r"|(?:怎么样|如何|多少|哪里|几点)\s*[。.!！]?$"
                  r"|^(什么|怎么|为什么|哪|谁|多少|几)"
                  r"|\b(what|why|how|who|where|when|which|can you|do you|是不是)\b",
                  re.IGNORECASE)

_INVITE = re.compile(r"(对吧|是吧|你说呢|你知道吗|是不是|懂吧|明白吗)\s*[?？]?$"
                     r"|\b(you know|right|you see)\s*[?？]?$", re.IGNORECASE)

_CONTINUE = re.compile(r"(然后|就是|因为|所以|但是|不过|而且|反正|其实|那个)$"
                       r"|\b(and then|so|because|but|like|i mean)$", re.IGNORECASE)

_DISCLOSE = re.compile(r"(压力|累|烦|难过|委屈|焦虑|失眠|担心|害怕|生气|开心|高兴|"
                       r"激动|喜欢|讨厌|受不了|不容易|太难了)"
                       r"|\b(stressed|tired|exhausted|upset|anxious|worried|scared|"
                       r"angry|happy|excited|love|hate|can't stand)\b", re.IGNORECASE)

_EMOTION_WEIGHT = {
    "难过": 1.5, "悲伤": 1.5, "委屈": 1.5, "焦虑": 1.4, "孤独": 1.4,
    "疲惫": 1.3, "纠结": 1.3, "生气": 1.2, "愤怒": 1.2,
    "开心": 1.2, "愉悦": 1.2, "惊讶": 1.1,
    "平静": 1.0,
    "sad": 1.5, "wronged": 1.5, "anxious": 1.4, "lonely": 1.4,
    "tired": 1.3, "conflicted": 1.3, "angry": 1.2, "happy": 1.2, "calm": 1.0,
}

@dataclass
class BackchannelPolicy:
    """Timing, probability, and cooldown parameters for spoken acknowledgements."""

    gap_s: float = field(default_factory=lambda: 0.10)

    max_gap_s: float = field(default_factory=lambda: 0.28)

    min_chars: int = field(default_factory=lambda: 4)

    refractory_s: float = field(default_factory=lambda: 2.0)

    p_max: float = field(default_factory=lambda: 0.9)
    #: Within-turn count quotas for 0–3s, 3–6s, and the remainder.
    session_curve: SessionFrequencyCurve = field(default_factory=SessionFrequencyCurve)

def f_sentence(text: str) -> float:
    """Score whether the phrase has reached an acknowledgement boundary."""
    t = (text or "").strip()
    if not t:
        return 1.0
    if _INVITE.search(t):
        return 1.8
    if _CONTINUE.search(t):
        return 1.4
    if _ASK.search(t):
        return 0.15
    return 1.0

def f_content(text: str) -> float:
    """Score the content signal used by the acknowledgement policy."""
    return 1.4 if _DISCLOSE.search(text or "") else 1.0

def f_emotion(emotion: str) -> float:
    """Score the emotional signal used by the acknowledgement policy."""
    return _EMOTION_WEIGHT.get((emotion or "").strip(), 1.0)

def f_prosody(tail_rms: float, prev_rms: float) -> float:
    """Score a falling acoustic tail relative to preceding energy."""
    if prev_rms <= 1e-6:
        return 1.0
    ratio = tail_rms / prev_rms
    if ratio < 0.45:
        return 1.3
    if ratio > 0.9:
        return 0.8
    return 1.0

def f_refractory(since_s: float, policy: BackchannelPolicy) -> float:
    """Return the cooldown multiplier for a candidate acknowledgement."""
    cooldown = max(0.0, policy.refractory_s)
    return 0.0 if since_s < cooldown else 1.0

FILLER_PROMPT = '只生成一段约四秒、可以直接说出口的自然垫话，用在后台工作尚未完成时。\n语气要像正在认真帮对方处理，可以说“稍等呀，我帮你看下”一类自然口语，但不要固定复读同一句。\n不要声称已经完成，不要提前给结果，不要虚构进度，也不要提工具、思维链或内部系统。\n只输出真正要说出口的内容，不要添加语气标签、引号、解释或舞台指示。\n\n当前任务背景：{task_context}'
