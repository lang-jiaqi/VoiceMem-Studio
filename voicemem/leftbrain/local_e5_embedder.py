"""E5 那一版的老入口。**新代码请用 ``local_embedder.LocalEmbedder``。**

本地 embedding 原来只有 E5 一个选择，模型名和 ``"query: "`` / ``"passage: "`` 前缀
都写死在这里。现在模型可换（见 ``local_embedder.REGISTRY``），前缀跟着模型走，
这个文件只剩两个老名字的转发：

    LocalE5Embedder()  →  LocalEmbedder("e5")
    shared_e5()        →  shared_model(E5 的路径)

老代码和文档里到处是这两个名字，删掉等于让别人的注入代码一夜之间报 ImportError，
所以留着。新写的东西别用——它们名字里带 E5，而实际用哪个模型已经由注册表决定了。
"""
from __future__ import annotations

from voicemem.leftbrain.local_embedder import (
    LocalEmbedder,
    REGISTRY,
    resolve_path,
    shared_model,
)

_E5 = REGISTRY["e5"]


def _e5_name() -> str:
    return resolve_path(_E5)


def shared_e5():
    """缓存单份本地 E5。等价于 ``shared_model(resolve_path(REGISTRY["e5"]))``。"""
    return shared_model(resolve_path(_E5))


class LocalE5Embedder(LocalEmbedder):
    """``LocalEmbedder("e5")`` 的老名字。"""

    def __init__(self):
        super().__init__("e5")
