"""Acoustic pause protection and unfinished-utterance policy."""
import re
from dataclasses import dataclass
from studio.harness.turn_taking.policy import CONTROLS, SessionFrequencyCurve

_UNFINISHED = re.compile(
    r"(?:我|你|他|她|我们|他们)?(?:经常|总是|有时候|有时|偶尔)?(?:就|就是|觉得|感觉|想说|想要|想|认为)$"
    r"|(?:然后|因为|所以|但是|不过|而且|如果|比如|其实|那个|这种|关于|至于)$"
    r"|\b(?:i (?:just|think|feel|want to)|because|and then|but|so|if|it's just)$",
    re.IGNORECASE,
)

_OPENING = re.compile(
    r"^(?:今天|明天|昨天|现在|刚才|最近|然后|因为|如果|但是|所以|比如|关于|至于|这个|那个)$"
    r"|^(?:其实)?我(?:现在)?(?:是)?(?:这么|这样)想的$"
)

_UNFINISHED_PREFACE = re.compile(
    r"(?:想|要|准备)(?:问|说|讲|确认|了解|请教|补充|告诉|打断)(?:你|我)?"
    r"(?:一下|一件事|一个问题|个问题|一|一个)?$"
    r"|(?:我|你|他|她|我们|他们)(?:今天|昨天|明天|现在|刚才|最近|本来)$"
    r"|(?:先|再)?(?:跟|给|对|向)(?:我|你|他|她|我们|他们)$"
    r"|(?:我)?(?:有|还有)(?:一|一个|个)(?:问题|事情|事)$"
    r"|\b(?:i (?:want|wanted|need) to (?:ask|say|tell you)|"
    r"can you (?:tell|help|explain)|i have (?:a )?(?:question|thing))$",
    re.IGNORECASE,
)

def is_unfinished(text: str) -> bool:
    """Detect a planning pause; this alone never authorizes a spoken follow-up."""
    text = re.sub(r"(?<=[\u4e00-\u9fff])[\s，,]+(?=[\u4e00-\u9fff])", "", text or "")
    tail = re.sub(r"[\s，。！？、,.!?…；;：:]+$", "", text or "")
    if _OPENING.fullmatch(tail):
        return True
    # Complete replies and word suffixes are not unfinished clauses.
    if re.search(r"(?:这么|这样|如此)(?:觉得|认为|想)$|(?:怎么|如何|怎样)想$"
                 r"|(?:将就|成就|迁就|造就|梦想|理想|感想|猜想|着想|不想|不觉得)$", tail):
        return False
    if re.fullmatch(r"(?:那)?你(?:觉得|认为)", tail):
        return False
    compact = re.sub(r"\s+", "", tail)
    if compact in {"我", "你", "他", "她"}:
        return True
    if re.fullmatch(r"(?:我是说|我的意思是|这个|那个)", compact):
        return True
    return bool(tail and (
        _UNFINISHED.search(tail) or _UNFINISHED_PREFACE.search(tail)))

def needs_continuation(text: str) -> bool:
    """Require an explicitly incomplete clause before scheduling a follow-up."""
    text = re.sub(r"(?<=[\u4e00-\u9fff])[\s，,]+(?=[\u4e00-\u9fff])", "", text or "")
    if not is_unfinished(text):
        return False
    tail = re.sub(r"[\s，。！？、,.!?…；;：:]+$", "", text or "")
    if _OPENING.fullmatch(tail):
        return True
    return bool(re.search(
        r"(?:我|我们|他|她|他们)(?:经常|总是|有时候|有时|偶尔)?"
        r"(?:就|就是|只是|还是)?(?:觉得|感觉|认为|想说|想要)(?:这种|那个)?$"
        r"|(?:我|我们)(?:经常|总是|有时候|有时|偶尔)(?:就|就是)$"
        r"|(?:因为|如果|但是|所以|比如|而且)$"
        r"|^(?:我是说|我的意思是)$"
        r"|\b(?:i (?:just|think|feel|want to)|because|and then|it's just)$",
        tail, re.IGNORECASE))

@dataclass
class PauseGate:
    """Keep incomplete speech in one turn using the audio silence clock."""
    hold_until: float = 0.0
    silence: float = 0.0
    rms_slow: float = 0.0
    unfinished_until: float = 0.0

    def reset(self):
        self.hold_until = self.silence = self.rms_slow = 0.0
        self.unfinished_until = 0.0

    def allow_end(self, text: str, silence: float, speaking: bool,
                  frame_s: float, rms: float) -> bool:
        # Energy catches a resumed utterance before a bridged VAD pause does.
        if speaking:
            self.rms_slow = .9 * self.rms_slow + .1 * rms if self.rms_slow else rms
        quiet = rms < max(.008, .25 * self.rms_slow)
        if quiet:
            self.silence += frame_s
        else:
            self.silence = self.hold_until = 0.0
            self.unfinished_until = 0.0
        if is_unfinished(text):
            if not self.unfinished_until:
                self.unfinished_until = (
                    self.silence + CONTROLS["unfinished_wait_ms"] / 1000)
        else:
            self.unfinished_until = 0.0
        return self.silence + 1e-9 >= max(
            self.hold_until, self.unfinished_until)

    def emitted(self, duration_s: float):
        """Leave a short continuation gap after an in-speech acknowledgement."""
        self.hold_until = (
            self.silence + duration_s + CONTROLS["backchannel_resume_ms"] / 1000)

def backchannel_policy():
    from studio.core.utils.turn_taking.initialize import BackchannelPolicy
    return BackchannelPolicy(
        gap_s=CONTROLS["pause_ms"] / 1000,
        refractory_s=CONTROLS["backchannel_cooldown_ms"] / 1000,
        session_curve=SessionFrequencyCurve(
            opening_turns=int(CONTROLS["backchannel_opening_turns"]),
            recovery_turn=int(CONTROLS["backchannel_recovery_turn"]),
            opening_probability=CONTROLS["backchannel_opening_probability"],
            middle_probability=CONTROLS["backchannel_middle_probability"],
            steady_probability=CONTROLS["backchannel_steady_probability"],
        ),
    )

def backchannel_policy_summary() -> str:
    """Return the concise startup description for the active session curve."""
    policy = backchannel_policy()
    curve = policy.session_curve
    return (
        f"session概率={curve.opening_probability:.0%}/"
        f"{curve.middle_probability:.0%}/{curve.steady_probability:.0%} "
        f"停顿窗口={policy.gap_s * 1000:.0f}~{policy.max_gap_s * 1000:.0f}ms "
        f"冷却={policy.refractory_s}s"
    )
