"""Memory Space：一个用户/一套记忆一个目录。

    voicemem_memoryspace/
      demo/                     ← space 名字就是目录名
        demo.json               总信息 + 左脑 / 右脑 / mem0 三块属性
        demo.sqlite             全部结构化存储（原来散成 10 个 .sqlite）
        multi_modal/            声纹向量、音频 embedding、原始 wav
        vectors/                左脑文本向量库（qdrant 自己的目录格式）

不设 space 时默认叫 ``demo``。

**为什么 sqlite 合成一个**：原来一个 space 下散着 cognitive_graph / graph_entities /
rb_graph / right_brain / session_tracker / slot_splits / scene_triggers /
routine_memory / audio_archive 九个文件，各开各的连接。它们本来就属于同一套记忆，
拆开只会让「拷走一个 space」变成「别漏了哪个文件」。表名互不重叠（唯一撞名的
``query_activations`` 已在 graph_entities 侧改名，见 GRAPH_ACTIVATIONS_TABLE）。

**为什么 vectors/ 不在 multi_modal/ 里**：multi_modal 放的是音频派生物（声纹、音频
embedding、wav）；vectors/ 是左脑**文本**记忆的向量库，而且是 qdrant 自己的目录
格式，不是单个文件。
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SPACE = "demo"
ROOT_DIR_NAME = "voicemem_memoryspace"

#: graph_entities 那份 query_activations 改用这个表名——它比 cognitive_graph 那份
#: 多一列 session_id，同名共存会让先建表的那份把另一份的插入打挂。
GRAPH_ACTIVATIONS_TABLE = "graph_query_activations"


def _safe(name: str) -> str:
    """space 名字要能当目录名。别让 "../x" 这种写出目录之外。"""
    cleaned = "".join(c for c in (name or "") if c.isalnum() or c in "-_ .").strip()
    cleaned = cleaned.strip(".")
    return cleaned or DEFAULT_SPACE


def root() -> Path:
    """所有 space 的父目录。``VOICEMEM_MEMORYSPACE_ROOT`` 可整体搬走。"""
    env = os.environ.get("VOICEMEM_MEMORYSPACE_ROOT")
    return Path(env) if env else Path.cwd() / ROOT_DIR_NAME


class MemorySpace:
    """一个 space 的全部路径。传给 Orchestrator 当 memory_root 用。"""

    def __init__(self, name: str | None = None, *, root_dir: Path | str | None = None):
        self.name = _safe(name or os.environ.get("VOICEMEM_SPACE") or DEFAULT_SPACE)
        base = Path(root_dir) if root_dir else root()
        self.dir = base / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "multi_modal").mkdir(exist_ok=True)

    # ── 三个入口 ─────────────────────────────────────────────────────────────
    @property
    def json_path(self) -> Path:
        return self.dir / f"{self.name}.json"

    @property
    def db_path(self) -> Path:
        return self.dir / f"{self.name}.sqlite"

    @property
    def multi_modal(self) -> Path:
        return self.dir / "multi_modal"

    @property
    def vectors(self) -> Path:
        return self.dir / "vectors"

    def __fspath__(self) -> str:        # 让它能直接当 memory_root 用
        return str(self.dir)

    def __str__(self) -> str:
        return str(self.dir)


# ── 组件侧入口 ───────────────────────────────────────────────────────────────
# 各组件只拿得到 memory_root，拿不到 MemorySpace 对象，所以这几个函数从目录本身
# 推出路径：目录名就是 space 名，文件名跟着它走（demo/ 里就是 demo.sqlite），
# 这样把整个文件夹拷走还认得出是谁。

def _pick(memory_root, suffix: str) -> Path:
    """按目录名取文件；目录里已经有一个同后缀的就用它。

    文件名跟着目录名走（demo/ 里就是 demo.sqlite），拷走整个文件夹还认得出是谁。
    但**重命名或复制文件夹之后目录名就变了**，硬按新名字找会找不到、然后新建一个
    空库——记忆看起来凭空没了（左脑还在，因为 vectors/ 不看名字，更难查）。
    所以先按名字找，找不到就认目录里现成的那一个。
    """
    p = Path(memory_root)
    want = p / f"{p.name}{suffix}"
    if want.exists():
        return want
    existing = sorted(x for x in p.glob(f"*{suffix}") if x.is_file())
    return existing[0] if existing else want


def db(memory_root) -> Path:
    """这个 space 唯一的 sqlite。所有结构化存储都在里面，表名互不重叠。"""
    return _pick(memory_root, ".sqlite")


def json_path(memory_root) -> Path:
    """空间描述文件：总信息 + 左脑 / 右脑 / mem0 属性。

    只放「这个 space 是什么」，不放运行状态——运行状态（声纹注册表、清理进度、
    左脑 json 镜像）走 sqlite 的 kv 表，见 ``kv_get`` / ``kv_set``。四份状态内容
    互不相同，塞进同一个 json 会互相覆盖。
    """
    return _pick(memory_root, ".json")


# ── 小状态：存进 sqlite 的 kv 表，别再各开一个 json ──────────────────────────

def _kv_conn(memory_root):
    import sqlite3
    c = sqlite3.connect(db(memory_root))
    c.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    return c


def kv_get(memory_root, key: str, default=None):
    import json as _json
    try:
        with _kv_conn(memory_root) as c:
            row = c.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return _json.loads(row[0]) if row else default
    except Exception:
        return default


def kv_set(memory_root, key: str, value) -> None:
    import json as _json
    try:
        with _kv_conn(memory_root) as c:
            c.execute("INSERT INTO kv (k, v) VALUES (?,?) "
                      "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                      (key, _json.dumps(value, ensure_ascii=False)))
    except Exception as e:
        print(f"[space] 写 {key} 失败：{type(e).__name__}: {e}", flush=True)


def mm(memory_root, name: str = "") -> Path:
    """multi_modal/ 下的东西：声纹、音频 embedding、wav。"""
    d = Path(memory_root) / "multi_modal"
    d.mkdir(parents=True, exist_ok=True)
    return d / name if name else d


def vectors(memory_root) -> Path:
    """左脑文本向量库（qdrant 的目录格式，不是单个文件）。"""
    d = Path(memory_root) / "vectors"
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_embedding(memory_root, dims: int, model: str = "") -> None:
    """这个 space 是用哪个 embedding 建的？跟当前的对不上就说人话。

    embedding 是**空间的属性**：一个 space 里的向量必须同一个模型产出的，混了就是
    一半记忆检索不到。不检查的话，报错来自 qdrant 深处——``shapes (227,384) and
    (1536,) not aligned``，看不出是"换了 embedding"造成的。

    **光比维度不够。** ``bge-small-en-v1.5`` 和 ``multilingual-e5-small`` 都是 384
    维，拿前者去读后者建的库，维度检查全过，然后安静地返回一堆语义无关的邻居——
    不报错的错最难查。所以模型名也要比。

    老空间（这个字段加进来之前建的）没记模型名：维度对得上就放行，由 describe()
    把当前模型补记上。补记等于"相信你现在用的就是当初建库那个"——没别的办法，
    但会打一行说明，中途换过模型的人看得见。
    """
    import json as _json
    path = json_path(memory_root)
    if not path.is_file():
        return
    try:
        mem0 = _json.loads(path.read_text(encoding="utf-8")).get("mem0") or {}
    except Exception:
        return
    old_dims, old_model = mem0.get("dims"), (mem0.get("model") or "")

    if old_dims and int(old_dims) != int(dims):
        raise ValueError(
            f"这个 memory space 是用 {old_dims} 维的 embedding 建的，你现在用的是 {dims} 维。\n"
            f"  space: {Path(memory_root)}\n"
            f"一个 space 只能配一种 embedding。两个办法：\n"
            f"  · 换个新 space：VoiceMem(space=\"我的名字\")\n"
            f"  · 或者换回原来的 embedding（384 维通常是本地 e5 / bge-en，"
            f"512 维是 bge-zh，1536 维是 OpenAI text-embedding-3-small）")

    if old_model and model and old_model != model:
        raise ValueError(
            f"这个 memory space 是用 {old_model!r} 建的，你现在用的是 {model!r}"
            f"（两者都是 {dims} 维，所以维度检查拦不住）。\n"
            f"  space: {Path(memory_root)}\n"
            f"照这样跑下去不会报错，只会检索出一堆语义无关的记忆。三个办法：\n"
            f"  · 换回 {old_model!r}\n"
            f"  · 换个新 space：VoiceMem(space=\"我的名字\")\n"
            f"  · 全库重算向量：python3 tools/reembed.py <space> --apply")

    if not old_model and model and old_dims:
        print(f"[space] 这个空间没记录 embedding 模型，按当前的 {model!r} 补记。"
              f"如果建库之后换过模型，跑 tools/reembed.py 重算向量。", flush=True)


def check_dims(memory_root, dims: int) -> None:
    """老名字，只比维度。新代码用 ``check_embedding``（它还比模型名）。"""
    check_embedding(memory_root, dims)


def describe(memory_root, *, user_id: str = "", mode: str = "",
             dims: int | None = None, counts: dict | None = None,
             embed_model: str = "") -> dict:
    """写/更新空间描述文件 ``<space>.json``，返回写进去的内容。

    分四块，跟论文里的结构对应：

        space       名字、创建时间、最后更新、总条数
        left_brain  左脑：事实记忆 + 认知图（slot / entity / relation）
        right_brain 右脑：heartnote + rb_slots / rb_entities（人格与情绪）
        mem0        底层记忆引擎：向量库位置、collection、向量维度

    只读元信息，不碰记忆本体；随时可以删掉重建。
    """
    import json as _json
    from datetime import datetime, timezone

    p = Path(memory_root)
    path = json_path(p)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    old = {}
    if path.is_file():
        try:
            old = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}

    doc = {
        "space": {
            "name": p.name,
            "created_at": (old.get("space") or {}).get("created_at") or now,
            "updated_at": now,
            "user_id": user_id or (old.get("space") or {}).get("user_id", ""),
            "mode": mode or (old.get("space") or {}).get("mode", ""),
            "counts": counts or (old.get("space") or {}).get("counts", {}),
            # 这个空间用什么语言（"en" / "zh"）。建的时候定一次，之后不再变——
            # 记忆和回复都跟着它。检索是按向量做的，一个库里中英混存会让一半
            # 记忆检索不到，所以语言是**库的属性**，不是单句的属性。
            # 跟 created_at 一样从 old 里继承：这个文件每次打开空间都会重写。
            "language": (old.get("space") or {}).get("language", ""),
        },
        "left_brain": {
            "role": "事实记忆：说了什么、什么时候、涉及谁",
            "storage": f"{p.name}.sqlite",
            "tables": ["memories", "entities", "entity_edges", "memory_tags",
                       "slot_profiles", "graph_entities"],
        },
        "right_brain": {
            "role": "人格与情绪：这个人是什么样的、为什么会那样反应",
            "storage": f"{p.name}.sqlite",
            "tables": ["right_brain_memories", "right_brain_anchor_links",
                       "rb_slots", "rb_entities", "rb_entity_memories"],
        },
        "mem0": {
            "role": "底层记忆引擎（向量检索）",
            # embedding 绑定这个 space：换了就得换 space（或 reembed），
            # 见 check_embedding()。维度和模型名都要记——两个不同模型可能同维度。
            "dims": dims or (old.get("mem0") or {}).get("dims"),
            "model": embed_model or (old.get("mem0") or {}).get("model", ""),
            "vector_store": "qdrant (local)",
            "path": "vectors/",
            "collection": "voicemem",
            "history": f"{p.name}.sqlite",
        },
        "multi_modal": {
            "role": "声纹、音频 embedding、原始 wav",
            "path": "multi_modal/",
        },
    }
    try:
        path.write_text(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    except Exception as e:
        print(f"[space] 写描述文件失败：{type(e).__name__}: {e}", flush=True)
    return doc


def resolve(space=None, memory_root=None) -> Path:
    """把 ``space=`` / ``memory_root=`` 归一成一个目录。

    ``memory_root`` 显式给了就照用（旧代码和评测靠它给每段对话开独立库），
    否则按 space 名字落到 ``voicemem_memoryspace/<space>/``。
    """
    if memory_root:
        p = Path(memory_root)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return MemorySpace(space).dir
