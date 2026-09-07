"""回复模型自己标这句该怎么念：``温和|明天那个会…`` → 一段 TTS 语气指示。

以前语气来自**上一轮用户的声学情绪**（emotion2vec 判的）。那是"他刚才什么心情"，
不是"这句话该怎么说"——用户平静地问一件坏消息，回复该是放软的，而上一轮情绪是
"平静"，就念得干巴巴。

回复模型知道自己要说什么，所以它最清楚该用什么语气。让它在第一个 token 就标出来：

    温和|你上次说的那个面试，后来怎么样了？
    └──┘ 剥掉，不显示、不入库，只用来选语气

**格式是按 token 数选的，不是按好看选的**：写成 ``[温和] `` 要 4 个 token，
``温和|`` 只要 2 个（'温和' 本身就是一个 token）。标签排在正文最前面，它每多一个
token，第一段文字就晚 28ms 才攒够、TTS 就晚 28ms 开口——这两个 token 是实打实的
56ms。竖线是分隔符：正文正常不会以「八个语气词之一 + 竖线」开头，认错的风险接近零。

标签集**故意做得很小**（8 个）。给一堆细分标签，模型会挑花眼、也会瞎标；
每个标签背后的指示才是细的。
"""
from __future__ import annotations

import re

#: 标签 → 给 TTS 的发声指示。指示要写**怎么发声**（语速、音高、停顿、气息），
#: 不是写情绪名——TTS 收到"温柔"不知道该怎么办，收到"放慢、压低、句尾留白"知道。
TONES: dict[str, str] = {
    "温和": "语速放慢一点，声音放软，句尾留一点停顿，像在关心对方。",
    "共情": "压低声音，放慢，句子之间留白，不要急着往下说。",
    "轻快": "语速稍快，语调上扬，带一点笑意。",
    "认真": "语速平稳，吐字清楚，语调平，不要有多余的起伏。",
    "鼓励": "语调向上，稍微用力一点，句尾扬起来。",
    "俏皮": "语速快一点，语调跳跃，句尾轻轻上挑。",
    "抱歉": "放慢，音量收一点，语调向下，听起来是真的在道歉。",
    "平静": "自然语速，不刻意起伏。",
}
DEFAULT = "平静"

#: 三种写法都认——4B 守不住格式,这里尽量宽容:
#:   1. ``开头一小段 + 竖线``：**不限定竖线前是什么**。4B 会自己造词——说英文时
#:      吐 ``Light|Yeah, I know``、抄提示词时吐 ``标签：轻快|``,都得剥掉。竖线本身
#:      就是"这一段是语气标注"的强信号:正常口语回复几乎不会以"短词+竖线"开头。
#:      认不出是哪个语气就退默认(见 split),但**一定要剥**——宁可语气不准,也不能
#:      让 "Light|" 被念出来、还写进记忆。
#:
#:      竖线前给到 24 个字符,不是 16。16 是照着 "Light|" 这种单词估的,但 4B 说英文
#:      时会把整个标签意译过去——实测吐的是 ``Light and breezy |``,竖线前 17 个字符,
#:      正好越界。越界的后果不是"语气不准",是整条正则不匹配、标签原样进 TTS：日志里
#:      `[seg] 第一段 25 字 → "Light and breezy | You're"` 就是它被念出来的样子。
#:      24 能装下这类意译（最长的 "Serious and steady " 是 19）。
#:   2. ``[轻快] 正文``：方括号,小模型爱自作主张。
#:   3. ``轻快 正文``：裸语气词 + 分隔符（漏了竖线）。这条**必须**限定成那 8 个词,
#:      否则会把正常句子的第一个词吃掉。
_T = "温和|共情|轻快|认真|鼓励|俏皮|抱歉|平静"
_TAG = re.compile(
    rf"^\s*(?:"
    rf"([^\n|｜]{{1,24}})[|｜]\s*"        # 任意短前缀 + 竖线（Light| / 标签：轻快|）
    rf"|[\[【]\s*({_T})\s*[\]】]\s*"      # [轻快]
    rf"|({_T})[\s:：]+"                   # 轻快<分隔符>
    rf")")


#: 模型意译出来的英文标签 → 那 8 个内部枚举。**只在剥掉之后用来认语气**，
#: 不参与匹配（匹配靠竖线）。没有它的话，所有英文回复的语气都塌回 DEFAULT——
#: 标签是剥干净了，但等于全程没语气，这也是"上下句情绪对不上"的一半原因。
_ALIAS: dict[str, str] = {
    "gentle": "温和", "warm": "温和", "soft": "温和", "caring": "温和",
    "empath": "共情", "compassion": "共情", "understanding": "共情",
    "light": "轻快", "breezy": "轻快", "cheerful": "轻快", "upbeat": "轻快",
    "serious": "认真", "steady": "认真", "matter-of-fact": "认真",
    "encourag": "鼓励", "supportive": "鼓励",
    "playful": "俏皮", "teasing": "俏皮",
    "sorry": "抱歉", "apologetic": "抱歉", "regret": "抱歉",
    "calm": "平静", "neutral": "平静", "plain": "平静",
}


def _from_head(head: str) -> str:
    """竖线前那段里认语气：先找中文枚举，再找英文意译，都认不出退默认。"""
    tag = next((t for t in TONES if t in head), "")
    if tag:
        return tag
    low = (head or "").lower()
    return next((t for k, t in _ALIAS.items() if k in low), DEFAULT)


#: 相邻两轮之间语气最多挪这么远（下面那个坐标系里的欧氏距离）。
#: 0.35 让 轻快→抱歉 这种对角跳变成两步（先落到 认真/平静），一步到位的只有近邻。
MAX_STEP = 0.35

#: 八个语气在(唤起, 暖度)上的位置。**跳变刺耳的是距离，不是标签本身**——
#: 轻快→温和 换了标签但听着自然，轻快→抱歉 就突兀，差别全在这两个轴上隔多远。
_XY: dict[str, tuple[float, float]] = {
    "平静": (0.35, 0.50), "认真": (0.40, 0.45), "温和": (0.35, 0.65),
    "共情": (0.25, 0.55), "抱歉": (0.25, 0.30), "鼓励": (0.70, 0.75),
    "轻快": (0.70, 0.80), "俏皮": (0.80, 0.85),
}


def smooth(prev: str, tag: str, max_step: float = MAX_STEP) -> str:
    """把这一轮的语气往上一轮那边拉一把，隔太远就只走一步。

    模型每轮是**独立**挑语气的，它不知道上一句用的什么，所以日志里会出现
    ``轻快 → 平静 → 抱歉`` 这种三连跳——单看每一句都挑得对，连起来听就像换了个人。

    这里不封顶情绪范围，只封顶**每轮的变化量**：想从轻快走到抱歉可以，得走两轮。
    代价说清楚：用户突然说了件难过的事时，这一轮只能到 认真，下一轮才到 抱歉，
    共情来晚一轮。觉得反应太钝就把 ``max_step`` 调大（1.0 = 不平滑）。
    """
    if not prev or prev == tag or prev not in _XY or tag not in _XY:
        return tag or DEFAULT
    (x0, y0), (x1, y1) = _XY[prev], _XY[tag]
    dx, dy = x1 - x0, y1 - y0
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= max_step:
        return tag
    k = max_step / dist                       # 只走到路上的这一点
    tx, ty = x0 + dx * k, y0 + dy * k
    return min(_XY, key=lambda t: (_XY[t][0] - tx) ** 2 + (_XY[t][1] - ty) ** 2)


def split(text: str) -> tuple[str, str]:
    """``"温和|你好"`` → ``("温和", "你好")``。没有标签就返回 ``("", 原文)``。

    竖线那条剥掉的前缀里若认得出那 8 个语气词之一就用它，认不出（模型自己造的
    ``Light|`` 之类）就退 ``DEFAULT``——**前缀照剥**，只是语气用默认的。
    """
    m = _TAG.match(text or "")
    if not m:
        return "", text
    if m.group(1) is not None:                     # 竖线那条：前缀里找语气词
        return _from_head(m.group(1)), text[m.end():]
    tag = (m.group(2) or m.group(3) or "").strip()
    return (tag, text[m.end():]) if tag in TONES else ("", text)


def instruction(tag: str, base: str = "") -> str:
    """标签 → 发声指示。不认识的标签退回默认，不报错——模型偶尔会自己编一个。"""
    tone = TONES.get(tag) or TONES[DEFAULT]
    return f"{base}{tone}" if base else tone


def prompt_rule(lang: str = "zh") -> str:
    """拼进人设的那句话。要求模型每条回复都以标签开头。

    ``lang`` 只挑**规则怎么讲**，不挑标签词——那 8 个词是内部枚举，``split()``
    和 ``TONES`` 都按它做键，翻译了全线对不上（跟 voicemem/lang.py 里 slot 名
    的处理一样）。英文回复前面照样标 ``轻快|``，用户看不到。
    """
    # 故意不出现"标签"二字：4B 会把它照抄进正文（实测输出 "标签：轻快|Sure…"）。
    # 只给格式和例子，让它模仿，不让它解释。
    if str(lang).lower().startswith("en"):
        return ("[Set the tone first]\n"
                "The first word of every reply must be one of the eight below, "
                "immediately followed by a vertical bar | and then the reply. "
                "No spaces around the bar, nothing else added. For example:\n"
                "  轻快|That sounds great, when are you going?\n"
                "  共情|Is that still not settled?\n"
                f"Choose from: {' / '.join(TONES)}. The user never sees this word; "
                "it only picks the speaking voice.\n")
    return ("【开头先定语气】\n"
            "每条回复的第一个词必须是下面八个之一，紧跟一个竖线 | ，再接正文。"
            "竖线前后都不要空格，也不要多写别的字。举例：\n"
            "  轻快|那挺好的呀，什么时候去？\n"
            "  共情|那件事到现在还没定下来吗？\n"
            f"可选：{' / '.join(TONES)}。这个词用户看不到，只用来选语音语气。\n")
