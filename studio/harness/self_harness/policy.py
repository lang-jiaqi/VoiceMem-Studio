"""Typed Self Harness overlay policy for one Studio conversation."""

TONE_TAGS = ("温和", "共情", "轻快", "认真", "鼓励", "俏皮", "抱歉", "平静")

# The four existing harness policies remain immutable defaults. The reply model
# may only select values from this bounded overlay schema.
PROFILE_SCHEMA = {
    "persona": {
        "interaction_style": {
            "default": "default",
            "values": ("default", "listener", "coach", "professional"),
            "labels": {
                "default": "默认陪伴方式",
                "listener": "多倾听、少建议",
                "coach": "主动推进和给下一步",
                "professional": "专业克制",
            },
            "prompt": {
                "default": "",
                "listener": "优先倾听和回应感受；除非用户明确求助，否则少给建议和行动清单。",
                "coach": "在用户有明确目标时主动提炼重点并给出下一步，但不越权替用户决定。",
                "professional": "保持专业、克制和清晰，减少拟人化亲昵表达，但仍然友善。",
            },
        },
    },
    "speaking_style": {
        "speech_rate": {
            "default": "normal",
            "values": ("very_slow", "slow", "normal", "fast", "very_fast"),
            "labels": {
                "very_slow": "很慢", "slow": "稍慢", "normal": "正常",
                "fast": "稍快", "very_fast": "很快",
            },
            "tts": {
                "very_slow": "语速明显放慢，每句留出清晰间隔，但不拖长字音。",
                "slow": "语速稍慢，咬字清楚，句尾自然收住。",
                "normal": "",
                "fast": "语速稍快，节奏利落，仍保持清晰。",
                "very_fast": "语速明显加快，表达紧凑，不吞字。",
            },
        },
        "tone": {
            "default": "auto",
            "values": ("auto", *TONE_TAGS),
            "labels": {"auto": "自动", **{tone: tone for tone in TONE_TAGS}},
        },
        "reply_length": {
            "default": "auto",
            "values": ("concise", "auto", "detailed"),
            "labels": {"concise": "简短", "auto": "自动", "detailed": "详细"},
        },
    },
    "reply_modes": {
        "reasoning_depth": {
            "default": "auto",
            "values": ("auto", "quick", "deep"),
            "labels": {"auto": "自动判断", "quick": "优先快速", "deep": "优先深思"},
        },
    },
    "turn_taking": {
        "backchannel": {
            "default": "auto",
            "values": ("auto", "off", "less", "more"),
            "labels": {
                "auto": "默认附和频率", "off": "不插话",
                "less": "少附和", "more": "多附和",
            },
        },
        "work_filler": {
            "default": "auto",
            "values": ("auto", "silent", "reassuring"),
            "labels": {
                "auto": "默认等待反馈", "silent": "安静等待",
                "reassuring": "更常说明正在思考",
            },
        },
    },
}

MAX_CHANGES_PER_TURN = 2
RECENT_CHANGE_TURNS = 2

CONTROL_RULE = """【Self Harness：会话级自适应设置】
四个原始 harness 策略是不可修改的默认版本。你只能在用户明确提出稳定偏好时，通过白名单差量调整当前会话；不要重写 prompt，也不要根据普通话题、临时情绪、引用内容或单次任务自行推断偏好。

每条回复必须从第一个字符开始输出一行私有控制头，然后紧接语气标签和正文：
<self_harness>{}</self_harness>温和|正文
控制头只写 JSON，不放进代码块，不解释或念出它。普通对话输出空对象。每轮最多修改两个字段，没有提到的字段不要输出。

允许的结构和值：
- persona.interaction_style: default / listener / coach / professional
- speaking_style.speech_rate: very_slow / slow / normal / fast / very_fast
- speaking_style.tone: auto / 温和 / 共情 / 轻快 / 认真 / 鼓励 / 俏皮 / 抱歉 / 平静
- speaking_style.reply_length: concise / auto / detailed
- reply_modes.reasoning_depth: auto / quick / deep
- turn_taking.backchannel: auto / off / less / more
- turn_taking.work_filler: auto / silent / reassuring

稳定规则：
- “再慢/快一点”只移动一档；不要一次跨多档。
- 大范围要求拆成最多两个最直接相关的字段，其余保持默认或当前值。
- 当前状态会标出最近两轮刚改过的字段。若用户马上提出冲突值，控制头仍输出请求值供后端暂存，但正文要说明现值暂时保持并请用户确认；用户再次明确确认时再次输出同一个值，才会生效。
- “恢复默认/正常”只把明确提到的字段恢复为 default、auto 或 normal。

例：
“以后少给建议，多听我说” → {"persona":{"interaction_style":"listener"}}
“说慢一点，也温和一点” → {"speaking_style":{"speech_rate":"slow","tone":"温和"}}
“别插话，等你想好直接说” → {"turn_taking":{"backchannel":"off","work_filler":"silent"}}
“这类问题以后认真推理” → {"reply_modes":{"reasoning_depth":"deep"}}
对修改请求自然简短地确认；正文仍须遵守当前 Self Harness 状态。"""
