"""web demo 的管道层（非主流程）——核心对话逻辑在 run.py，页面渲染在 index.html。

这里放：本地 E5（memory embedding + slot 分类共享一份模型）、音频重采样/VAD、
LLM/TTS/Realtime 流、以及 FastAPI + WebSocket 接线。run.py 只管把这些拼成 0–300ms
投机预取的对话流程。
"""
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from pet_bridge import PetHub, PetSupervisor, TeeSocket, loopback_ws_url

# 本地句向量（memory embedding + slot 分类 + 语义情绪原型共享一份模型）在核心里，
# 见 voicemem/leftbrain/local_embedder.py。demo 侧要的是"跟记忆库同一个模型"，
# 而不是"E5"——用哪个由注册表/空间语言决定，所以这里给一个和模型无关的入口。
# 老名字 LocalE5Embedder / shared_e5 一起 re-export，既有调用点不变。
from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder, shared_e5  # noqa: F401
from voicemem.leftbrain.local_embedder import (  # noqa: F401
    LocalEmbedder, resolve, resolve_path, shared_model,
)


def shared_embed_model():
    """当前生效的那份本地句向量模型（跟记忆库用的是同一个）。"""
    return shared_model(resolve_path(resolve()))
from voicemem.llm_config import resolve_model

HERE = Path(__file__).resolve().parent
#: demo 的回复模型。默认比后台整理用的强一档——这一路用户直接听得见。
#: VOICEMEM_REPLY_MODEL（或旧名 OPENAI_CHAT_MODEL）/ models={"reply": ...} 都能覆盖。
CHAT_MODEL = resolve_model(role="reply", default="gpt-4o")
RT_MODEL = resolve_model(role="realtime")
#: Realtime 那条路的音色。**默认跟随 TTS 音色**，不写死。
#:
#: gpt-realtime 上可选：alloy / ash / ballad / coral / echo / sage / shimmer /
#: verse / marin / cedar —— marin 和 cedar 起伏和呼吸感明显强，但**只有 Realtime
#: 有**，TTS API 合不出来。
#:
#: 原来这里写死 marin，而 llm_tts 那条走 OPENAI_TTS_VOICE（默认 alloy）——于是
#: `--mode` 一换，助手就换了个人。同一个产品两条路两个声音，本身就该修。
#:
#: 附和（backchannel）让它变成硬约束：那声"嗯"是本地 TTS 预合成的，音色必须跟正文
#: 一致，否则中间冒出来一声别人的声音，比不附和突兀得多。跟着 TTS 走就自动落在
#: 两边都有的音色上。要用 marin / cedar 就显式设 OPENAI_REALTIME_VOICE——那时
#: realtime 那条会自动关掉附和（见 run.py 的 _backchannel_tts）。
from voicemem.tts import TTS_VOICE as _TTS_VOICE  # noqa: E402

RT_VOICE = os.environ.get("OPENAI_REALTIME_VOICE") or _TTS_VOICE
client = None


def _openai_client():
    # DeepSeek + Breeze 的回复不需要 OpenAI 凭据；只有旧 API 功能实际用到时才建。
    global client
    if client is None:
        client = AsyncOpenAI()
    return client


# ── 音频小工具 ────────────────────────────────────────────────────────────────
# resample 用核心那份（voicemem/utils/audio/stream_io.py），别再抄一遍。
# 原来这里还有一份 make_vad()——自从 demo 改成复用 vm.stream() 之后就没人调了，
# VAD 现在是核心的可注入能力（VoiceMem(vad=...) / config 的 vad 段），已删。
from voicemem.utils.audio.stream_io import resample  # noqa: E402,F401

# TTS（在线/离线两个后端 + 分句）已提进核心，见 voicemem/tts.py；这里 re-export
# 保持 `utils.tts_stream(...)` 的既有调用点不变。
from voicemem.tts import (  # noqa: E402,F401
    TTS_BACKEND, TTS_MODEL, TTS_VOICE, cut_point, tts_stream)


# ── 回复模型也在一处配：统一 config 的 reply 段（run.py 从 CONFIG["reply"] 传入）──
# 不传就回落到模块级 env 默认（CHAT_MODEL / TTS_MODEL / TTS_BACKEND / RT_MODEL），
# 现有行为完全不变。reply 结构：{"llm": {"config": {"model": ...}},
# "tts": {"provider": "openai|local", "config": {"model": ...}},
# "realtime": {"config": {"model": ...}}}，每段可省。
def _reply_seg(reply, name):
    seg = (reply or {}).get(name) or {}
    return seg.get("provider"), (seg.get("config") or {})


# ── Realtime 流 ───────────────────────────────────────────────────────────────
# 原来这里还有一份 llm_stream()——和核心回复层（voicemem/reply.py 的 openai_reply）
# 是同一件事：流式调 chat.completions，把记忆拼进 system。run.py 现在直接用
# vm.reply_stream()，人设走 CONFIG.reply.llm.config.system，这份已删。


def realtime_connect(reply=None):
    """方案 A：整段麦克风音频平行喂给它出原生语音。事件名随 SDK 版本可能微调
    （对照 openai_voice_demo/backend/providers/realtime.py）。"""
    _, cfg = _reply_seg(reply, "realtime")
    return _openai_client().realtime.connect(model=resolve_model(cfg.get("model"), "realtime"))


# ── SearchResult → 脑图 html 认识的 memory_hits 负载 ──────────────────────────
def _emotion_of(rb_hits) -> str:
    """这一轮右脑命中里带的情绪标签。

    前端原来是拿正则去 content 里抠「x」——那是 heartnote 内心OS 的写法，
    而内心OS 现在是有 gate 的（不是每轮都生成），抠不到就一直不显示。情绪本来
    就在 metadata.emotion 里，直接给前端，别让它猜。
    """
    # **只认本轮的信号**（current_signal 的 affect_hint）。
    #
    # 原来取不到就退回"检索回来的旧记忆上带的情绪"，那是个 bug：标签栏写的是
    # "现在他什么心情"，退回去之后显示的却是"想起来的那件事当时什么心情"——库里
    # 悲伤的记忆一多，每轮都显示悲伤，跟用户说什么无关。
    # 本轮没有信号就返回空，交给 run.py 的 fill_tags（文本关键词 → SenseVoice）。
    for h in (rb_hits or []):
        if getattr(h, "source", "") != "current_signal":
            continue
        emo = ((getattr(h, "metadata", None) or {}).get("emotion") or "").strip()
        if emo:
            return emo
    return ""


#: 右脑记忆送去给**模型**时会拼上日期和 slot 前缀（"[2026-08-24] ⚠ 避免重复：…"）——
#: 模型需要知道这是什么时候、属于哪一类。但页面上不该显示这些：左脑那栏是干干净净
#: 一句事实，右脑却顶着一串前缀，看着像两个系统。分类信息已经单独放在 cluster 字段
#: 里了，正文只留正文。
_RB_PREFIX = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?(?:[⚠✓✱*]\s*)?(?:[^：:\s]{2,8}[：:]\s*)?")


#: 渲染给**模型**看时会在正文后面补两段：回应经验的「（下次：…）」是可执行建议，
#: heartnote 的「（内心OS：…）」是补充解读。页面上只要正文——情绪单独用【x】显示，
#: 分类在 cluster 字段里。
_RB_SUFFIX = re.compile(
    r"[（(]\s*(?:下次|next time|内心OS|inner note)\s*[：:].*$", re.S | re.I)


def clean_rb(content: str) -> str:
    t = _RB_PREFIX.sub("", str(content or ""))
    t = _RB_SUFFIX.sub("", t)
    return t.strip()


def hits_payload(result, has_audio=None, cluster_of=None):
    """has_audio(memory_id) -> bool：这条记忆有没有存档的原音频。
    前端据此决定要不要在这一轮自动把当时那段原声放回来。"""
    rb = getattr(result, "rb_hits", None) or []
    cls = getattr(result, "classification", None)
    return {
        # slot 和情绪跟着这一轮的检索结果一起发。前端原来是另发一次
        # /api/classify 再等它回来——那是条竞态：memory_hits 先到时 curSlots
        # 还是空的，标签栏就空着。这里用的是 Search 本来就算好的分类，0 额外开销。
        "slots": list(getattr(cls, "slots", []) or []),
        "entities": list(getattr(cls, "entities", []) or []),
        "emotion": _emotion_of(rb),
        "left_brain": [{"text": h.text, "score": h.score, "attributed_to": h.attributed_to,
                        "memory_id": h.memory_id,
                        "has_audio": bool(has_audio and has_audio(h.memory_id))}
                       for h in result.hits],
        # cluster 由 run.py 注入（同一套规则，前端不再自己从 source 猜）
        # content 给页面看（已去前缀），raw 保留原文——脑图要靠它跟 heartnote 对上
        # internal：response_experience 是助手对**自己**行为的笔记（"这次没先接住
        # 情绪，下次先问"），对模型有用，但它不是关于用户的画像——摆在页面的
        # 「右脑·画像」栏里用户看着莫名其妙。前端据此跳过显示，脑图匹配仍照用。
        "right_brain_hits": [{"content": clean_rb(h.content), "raw": h.content,
                              "internal": h.source == "response_experience",
                              # profile 类命中是 **slot 级**的画像（"喜好与厌恶：…"），
                              # 而脑图节点是 **实体**级的，按正文永远匹配不上——右脑
                              # 节点从来不亮、左右脑之间也就没有射线。把 slot 名带上，
                              # 前端好把它落到该 slot 下的节点。
                              "slot": ((getattr(h, "metadata", None) or {}).get("slot_name") or ""),
                              # 判断原文。页面显示的是它的第一人称改写版（run.py 的
                              # rb_human），这里留一份原文给改写和脑图匹配用。
                              "claim": ((getattr(h, "metadata", None) or {}).get("claim") or ""),
                              "source": h.source, "priority": h.priority,
                              "cluster": cluster_of(h.content, h.source) if cluster_of else ""}
                             for h in (getattr(result, "rb_hits", None) or [])],
        "current_scene": getattr(result, "current_scene", None) or None,
        "related_summaries": getattr(result, "related_summaries", None) or {},
    }


# ── FastAPI + WS 接线（仅接线，渲染都在 index.html）─────────────────────────────
def build_app(mode, session, classify, snapshot=None, audio_of=None, spaces=None,
              set_lang=None):
    """session(sock)：run.py 传入的会话循环（llm_tts / realtime）。classify(query)：给脑图生长用。
    snapshot()：库里已有的记忆，前端打开页面时先把脑图铺满。
    spaces=(list_fn, create_fn, use_fn, active_fn)：Memory Space 的增/查/切换。
    set_lang(lang)：界面切语言时同步给助手（回复语言 + 抽取语言）。"""
    app = FastAPI()
    pet, pet_hub = PetSupervisor(), PetHub()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await sock.send_json({"type": "session_ready", "mode": mode})
        try:
            # 包一层再交给会话循环：它感知不到区别，只是播放状态会顺手抄给桌面小人。
            await session(TeeSocket(sock, pet_hub))
        except WebSocketDisconnect:
            pass          # 关页面/刷新是正常结束，别刷一屏 traceback

    @app.websocket("/ws-pet")                        # 桌面小人的只读广播口
    async def ws_pet(sock: WebSocket):
        """跟 /ws 完全分开：连这里不会开新会话，只是听播放状态。

        小人连 /ws 的话，每连一次就凭空多开一路对话（那边在等麦克风上行）。
        """
        await sock.accept()
        pet_hub.add(sock)
        try:
            while True:                              # 只收不发，等着对面断开
                await sock.receive()
        except WebSocketDisconnect:
            pass
        finally:
            pet_hub.discard(sock)

    @app.on_event("shutdown")
    def _stop_pet():
        pet.shutdown()

    class Q(BaseModel):
        query: str

    @app.post("/api/classify")                       # 脑图 html 用它把左脑按 slot 生长
    def api_classify(body: Q) -> dict:
        c = classify(body.query)
        return {"slots": list(c.slots), "entities": list(c.entities)}

    class T(BaseModel):
        text: str

    @app.post("/api/title")                          # 给 session 起个概括性的名字
    async def api_title(body: T) -> dict:
        """用一句话概括这轮对话，给 sidebar 当标题。

        只在第一轮之后叫一次，用最小的模型、限死 16 token——不能为了一个标题
        拖慢对话，也不该花明显的钱。失败就返回空串，前端回落到用户说的第一句。
        """
        try:
            r = await _openai_client().chat.completions.create(
                model=CHAT_MODEL, max_tokens=16, temperature=0,
                messages=[
                    {"role": "system", "content":
                     "用不超过 12 个字概括这段对话在说什么，做标题用。"
                     "只输出标题本身，不要引号、不要标点、不要「关于」这类开头。"
                     # 开场常是"喂喂喂""测试一下""你好"，照实概括就成了"语音测试"，
                     # 而这段对话后面聊的可能是完全另一回事。
                     "忽略开头的寒暄、试麦、确认能不能听见这类内容，"
                     "抓真正聊到的事情。整段都只是打招呼时，才叫「随便聊聊」。"},
                    {"role": "user", "content": body.text[:600]},
                ],
            )
            return {"title": (r.choices[0].message.content or "").strip()}
        except Exception as e:
            print(f"[web] 生成标题失败：{e}", flush=True)
            return {"title": ""}

    @app.get("/api/memories")                        # 打开页面时先铺已有记忆
    def api_memories() -> dict:
        return snapshot() if snapshot else {"left": [], "right": []}

    @app.post("/api/lang")                           # 界面切语言
    async def api_lang(req: Request) -> dict:
        """返回 reply_lang：**回复实际用的语言**。

        它可能跟 lang 不一样——当前空间已经有记忆时语言不能改（混语存会让一半
        记忆检索不到）。前端据此提示用户，而不是让"界面中文、回答英文"默默发生。
        """
        lang = (await req.json()).get("lang", "zh")
        reply_lang = set_lang(lang) if set_lang else lang
        return {"lang": lang, "reply_lang": reply_lang or lang}

    if spaces:
        _list_spaces, _create_space, _use_space, _active_space = spaces

        @app.get("/api/spaces")                      # 磁盘上有哪些空间
        def api_spaces() -> dict:
            return {"spaces": _list_spaces(), "active": _active_space()}

        @app.post("/api/spaces")                     # 新建一个空的
        async def api_space_new(req: Request) -> dict:
            body = await req.json()
            name, lang = body.get("name", ""), body.get("language", "")
            try:
                return _create_space(name, lang)
            except FileExistsError as e:
                raise HTTPException(409, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))

        @app.post("/api/spaces/{name}/use")           # 切过去
        def api_space_use(name: str) -> dict:
            try:
                return {"active": _use_space(name)}
            except Exception as e:
                raise HTTPException(400, f"切不过去：{e}")

    @app.get("/api/audio/{memory_id}")               # 把当时那段原声放回来
    def api_audio(memory_id: str):
        path = audio_of(memory_id) if audio_of else None
        if not path or not Path(path).exists():
            raise HTTPException(404, "这条记忆没有存档音频")
        return FileResponse(path, media_type="audio/wav")

    (HERE / "images").mkdir(exist_ok=True)
    app.mount("/images", StaticFiles(directory=HERE / "images"), name="images")

    # no-store：demo_local 也占 8787，同源缓存会让浏览器端出上一个 demo 的旧页面
    _NOCACHE = {"Cache-Control": "no-store"}

    @app.get("/pcm-player-worklet.js")
    def pcm_player_worklet():
        return FileResponse(HERE / "pcm-player-worklet.js", headers=_NOCACHE,
                            media_type="application/javascript")

    @app.get("/mic-capture-worklet.js")
    def mic_capture_worklet():
        return FileResponse(HERE / "mic-capture-worklet.js", headers=_NOCACHE,
                            media_type="application/javascript")

    @app.get("/")
    def index(request: Request, pet_on: bool = Query(False, alias="pet")):
        """?pet=1 顺带把桌面小人拉起来；不带参数就是原来的页面，一点没变。

        只在这里拉起、不在这里关：小人的生死跟后端进程绑定，不跟标签页绑定，
        所以刷新、关页面、开很多个标签页都不会动它（见 pet_bridge 模块注释）。
        """
        if pet_on:
            pet.ensure_running(loopback_ws_url(request))
        return FileResponse(HERE / "voicemem.html", headers=_NOCACHE)

    @app.get("/classic")                             # 上一版页面，留着对照
    def classic():
        return FileResponse(HERE / "index.html", headers=_NOCACHE)

    return app
