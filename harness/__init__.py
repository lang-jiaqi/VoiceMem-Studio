"""Repository-only dialogue behavior layer used by the demo.

The four explicit policy areas are:

    reply_modes     memory+CoT, memory, or direct reply selection
    persona         stable agent identity and relationship prompt
    speaking_style  context-dependent depth and emotional content arc
    turn_taking     backchannels, fillers, and overlap timing

**为什么它在这儿、而 gate 在 voicemem/ 里面**——两者性质不同：

    gate         判"这一轮要不要检索记忆"。那是**记忆怎么用**的问题，
                 ``pip install voicemem`` 的人一样需要（不然每轮都在白检索），
                 所以它属于核心包。
    turn_taking  要 TTS 合成、要播音频、要前端配合。只有跑 demo 的人用得上，
                 装包的人拿到也没处使。

所以这一层不随 pip 包发布（pyproject 只收 ``voicemem*``），只在仓库里存在。

    VOICEMEM_BACKCHANNEL_EMIT=1   打开（默认关）
"""
from harness import persona, reply_modes, speaking_style, turn_taking  # noqa: F401
from harness.turn_taking import backchannel as backchannel  # compatibility alias
from voicemem import tts_control as speak_tag  # compatibility alias

__all__ = ["persona", "reply_modes", "speaking_style", "turn_taking"]
