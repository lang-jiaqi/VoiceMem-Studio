"""本地 embedding 的唯一入口：选哪个模型、加什么前缀，都在这儿。

原来这一层是写死 E5 的——模型名写死、``"query: "`` / ``"passage: "`` 前缀写死在
四个地方（embedder 两处、slot 分类器两处）。换个模型要改四处，漏一处不会报错，
只会**悄悄掉准确率**：前缀是各家模型自己的约定，拿 E5 的前缀喂 BGE，模型看到的
是一句它没见过的开头。

所以这里把「一个本地 embedding 模型」收成一条注册表项：仓库 id、离线包目录名、
query/passage 前缀、向量维度。换模型 = 换一个 key，前缀跟着模型走。

    from voicemem.leftbrain.local_embedder import LocalEmbedder
    vm = VoiceMem(embedding=lambda: LocalEmbedder("bge-zh"))

选哪个（就近的赢）：

    LocalEmbedder("bge-zh")          显式
    VOICEMEM_LOCAL_EMBED_MODEL=bge   env；``bge`` 按空间语言选 zh/en 那一版
    默认                              e5（多语，改默认前先用 evals 量过再说）

**维度是空间的属性**：一个 space 里的向量必须同一个模型产出的。``bge-small-en``
和 ``multilingual-e5-small`` 都是 384 维，光比维度分不出来——所以模型名也要记进
空间描述里，见 ``space.check_embedding``。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class LocalEmbedModel:
    """一个本地 embedding 模型的全部约定。"""
    key: str
    repo: str
    #: 离线包目录名（``models/<kind>/``）。E5 沿用历史的 ``embedding``，别改——
    #: 已经下载过离线包的人靠这个名字找模型。
    kind: str
    #: 各家自己的前缀约定，不是装饰。E5 两侧都要；BGE v1.5 官方说两侧都不用。
    query_prefix: str = ""
    passage_prefix: str = ""
    dims: int = 0
    #: 这个模型认哪种语言。``""`` = 多语，任何空间都能用。
    language: str = ""
    #: 建 SentenceTransformer 时要额外传的 tokenizer 参数。Qwen3-Embedding 用
    #: last-token pooling，必须 ``padding_side="left"``——不传不会报错，只是
    #: 池化取到 padding 上，向量安静地变差。
    tokenizer_kwargs: tuple = ()
    note: str = ""


#: BGE v1.5 的检索指令。官方 README 说 v1.5 **不加也行**（v1.5 就是为此调过的），
#: 所以默认不加。想验"加了是不是更准"，把它填进对应项的 query_prefix 再跑 evals。
BGE_QUERY_INSTRUCTION = {
    "en": "Represent this sentence for searching relevant passages: ",
    "zh": "为这个句子生成表示以用于检索相关文章：",
}


REGISTRY: dict[str, LocalEmbedModel] = {
    "e5": LocalEmbedModel(
        key="e5", repo="intfloat/multilingual-e5-small", kind="embedding",
        query_prefix="query: ", passage_prefix="passage: ", dims=384,
        note="多语（100+ 语言）。471MB——大头是 25 万词的多语词表，不是层数。"),
    "e5-base": LocalEmbedModel(
        key="e5-base", repo="intfloat/multilingual-e5-base", kind="embedding-e5-base",
        query_prefix="query: ", passage_prefix="passage: ", dims=768,
        note="同族更大一档，前缀不变、仍是多语；换它只改模型名，风险最小。"),
    "bge-zh": LocalEmbedModel(
        key="bge-zh", repo="BAAI/bge-small-zh-v1.5", kind="embedding-bge-zh",
        dims=512, language="zh",
        note="只认中文。95MB / 4 层，比 E5 小快很多；层数少，容量也少，要实测。"),
    "qwen": LocalEmbedModel(
        key="qwen", repo="Qwen/Qwen3-Embedding-0.6B", kind="embedding-qwen",
        # 官方 config_sentence_transformers.json 里的 query prompt，原样照抄；
        # document 侧不加前缀。
        query_prefix=("Instruct: Given a web search query, retrieve relevant "
                      "passages that answer the query\nQuery:"),
        dims=1024, tokenizer_kwargs=(("padding_side", "left"),),
        note="多语、榜上最强的一档，但 28 层 ×1024、~1.2GB —— 是不是塞得进投机"
             "预取那 0–300ms 预算，用 evals/embed_latency.py 量了再说。"),
    "bge-en": LocalEmbedModel(
        key="bge-en", repo="BAAI/bge-small-en-v1.5", kind="embedding-bge-en",
        dims=384, language="en",
        note="只认英文。130MB。**和 E5 同为 384 维**，只比维度分不出来。"),
}

#: 没有任何指定时用哪个。改这个默认值之前先跑 evals/embed_ab.py——
#: 换 embedding 要全库 re-embed，不是能随手回滚的改动。
DEFAULT_KEY = "e5"

ENV = "VOICEMEM_LOCAL_EMBED_MODEL"


def _current_language() -> str:
    """没显式给语言时，用当前实例已经解析好的库语言。

    ``VoiceMem.__init__`` 一开始就 ``lang.resolve_for_space(memory_root)``，那一步
    会把这个空间的语言定下来，所以 utils 懒加载时读到的就是对的那个。**但它是
    进程级的**：同一个进程开两个不同语言的空间，后建的会盖掉前一个（issue #9 的
    同一个根子）。调用方知道 memory_root 的话，请显式传 ``language=``——
    ``utils/defaults.py`` 就是这么做的。
    """
    try:
        from voicemem.lang import memory_language
        return memory_language()
    except Exception:
        return "en"


def resolve(name: str | None = None, language: str = "") -> LocalEmbedModel:
    """定下用哪个模型：显式 > env > 默认。

    ``bge`` / ``auto`` 这种**族名**按 ``language`` 展开成具体那一版（单语模型一族
    有中英两个）。语言从空间读（见 ``voicemem.lang.resolve_for_space``）——语言是
    库的属性，所以模型也该是库的属性，不该跟着单句变。
    """
    key = (name or os.environ.get(ENV, "") or DEFAULT_KEY).strip().lower()
    lang = (language or "").strip().lower() or _current_language()
    if key in ("bge", "auto"):
        key = f"bge-{lang}" if f"bge-{lang}" in REGISTRY else "bge-en"
    if key not in REGISTRY:
        raise ValueError(
            f"不认识的本地 embedding 模型 {key!r}。可选：{' / '.join(REGISTRY)}\n"
            f"（{ENV} 也可以填 'bge'，按空间语言自动选中/英那一版）")
    model = REGISTRY[key]
    if model.language and model.language != lang:
        print(f"[embedding] {model.repo} 只认 {model.language}，而这个空间是 {lang}。"
              f"单语模型跨语言用，检索会明显变差。", flush=True)
    return model


def resolve_path(model: LocalEmbedModel) -> str:
    """离线包优先，没有就返回 HF id 让 transformers 首次运行自动下。

    每个模型有**自己的**离线目录：都堆在 ``models/embedding/`` 的话，装过 E5 的人
    要 BGE 会拿到 E5——目录里有 config.json 就算命中，不看是哪个模型。
    """
    from voicemem.utils.common.paths import hf_model
    return hf_model(model.kind, model.repo, model.key)


@lru_cache(maxsize=4)
def shared_model(path: str, tokenizer_kwargs: tuple = ()):
    """按路径缓存 SentenceTransformer：记忆向量和 slot 分类共用一份，省一份权重。

    ``tokenizer_kwargs`` 走元组而不是 dict，因为 lru_cache 的键必须可哈希；同一个
    模型配不同 tokenizer 参数会各缓存一份，这是对的——它们不是同一个编码器。
    """
    if os.environ.get("VOICEMEM_VERBOSE", "0") == "0":
        # 每次启动刷一条 "Loading weights: 100%|███"（transformers 的 tqdm），
        # 模型在本地、一瞬间就加载完，这条除了吓人没有信息量。
        try:
            from transformers.utils import logging as _hf_logging
            _hf_logging.disable_progress_bar()
        except Exception:
            pass
    from sentence_transformers import SentenceTransformer
    if not tokenizer_kwargs:
        return SentenceTransformer(path)
    # sentence-transformers 6.x 把 tokenizer_kwargs 改名成 processor_kwargs，旧名
    # 还认但会打 DeprecationWarning。两个名字都试：这个参数不是可有可无的装饰，
    # Qwen 的 last-token pooling 少了它会静默取到 padding 上。
    kw = dict(tokenizer_kwargs)
    try:
        return SentenceTransformer(path, processor_kwargs=kw)
    except TypeError:
        return SentenceTransformer(path, tokenizer_kwargs=kw)


class LocalEmbedder:
    """注入 ``VoiceMem(embedding=...)``：向量在本地算，0 网络。

    ``model_name`` 带的是**注册表 key**，不是路径——它要写进空间描述并在下次打开
    时比对（见 ``space.check_embedding``），所以必须稳定：同一个模型换了离线路径
    还是同一个 key，不该被判成"换了 embedding"。
    """

    def __init__(self, name: str | None = None, language: str = ""):
        self.model = resolve(name, language)
        self._path = resolve_path(self.model)

    @property
    def model_name(self) -> str:
        return self.model.key

    @property
    def dimensions(self) -> int:
        if self.model.dims:
            return self.model.dims          # 注册表里有就别为了问一句维度去加载模型
        m = shared_model(self._path, self.model.tokenizer_kwargs)
        fn = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return fn()

    def _encode(self, texts, prefix):
        return np.asarray(shared_model(self._path, self.model.tokenizer_kwargs).encode(
            [f"{prefix}{t}" for t in texts], normalize_embeddings=True))

    def embed_texts(self, texts):
        if not texts:
            return []
        return self._encode(texts, self.model.passage_prefix).tolist()

    def embed_query_text(self, text):
        return self._encode([text], self.model.query_prefix)[0].tolist()
