"""对话行为层：只有 demo 用得上的那些交互行为。

现在只有一件事：

    backchannel  你说到一半停顿时，要不要"嗯"一声？说哪个词？

**为什么它在这儿、而 gate 在 voicemem/ 里面**——两者性质不同：

    gate         判"这一轮要不要检索记忆"。那是**记忆怎么用**的问题，
                 ``pip install voicemem`` 的人一样需要（不然每轮都在白检索），
                 所以它属于核心包。
    backchannel  要 TTS 合成、要播音频、要前端配合。只有跑 demo 的人用得上，
                 装包的人拿到也没处使。

所以这一层不随 pip 包发布（pyproject 只收 ``voicemem*``），只在仓库里存在。

    VOICEMEM_BACKCHANNEL_EMIT=1   打开（默认关）
"""
from harness import backchannel  # noqa: F401

__all__ = ["backchannel"]
