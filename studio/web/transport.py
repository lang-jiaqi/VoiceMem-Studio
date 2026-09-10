"""Studio transport implementation."""
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from studio.web.pet_bridge import PetHub, PetSupervisor, TeeSocket, loopback_ws_url

from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder, shared_e5  # noqa: F401
from voicemem.leftbrain.local_embedder import (  # noqa: F401
    LocalEmbedder, resolve, resolve_path, shared_model,
)

def shared_embed_model():
    """Return the local embedding model shared with memory retrieval."""
    return shared_model(resolve_path(resolve()))
from voicemem.llm_config import resolve_model

HERE = Path(__file__).resolve().parent

CHAT_MODEL = resolve_model(role="reply", default="deepseek-v4-flash")
if not CHAT_MODEL or CHAT_MODEL.startswith("gpt-"):
    CHAT_MODEL = "deepseek-v4-flash"
RT_MODEL = resolve_model(role="realtime")

#:

#:

#:

from studio.core.utils.tts.providers import TTS_VOICE as _TTS_VOICE  # noqa: E402

RT_VOICE = _TTS_VOICE
client = None

def _openai_client():

    global client
    if client is None:
        client = AsyncOpenAI()
    return client

from voicemem.utils.audio.stream_io import resample  # noqa: E402,F401

from studio.core.utils.tts.providers import (  # noqa: E402,F401
    TTS_BACKEND, TTS_MODEL, TTS_VOICE, cut_point, tts_stream)

# "tts": {"provider": "openai|local", "config": {"model": ...}},

def _reply_seg(reply, name):
    seg = (reply or {}).get(name) or {}
    return seg.get("provider"), (seg.get("config") or {})

_TITLE_SYSTEM = (
    "用不超过 12 个字概括这段对话在说什么，做标题用。"
    "只输出标题本身，不要引号、不要标点、不要「关于」这类开头。"
    "忽略开头的寒暄、试麦、确认能不能听见这类内容，"
    "抓真正聊到的事情。整段都只是打招呼时，才叫「随便聊聊」。"
)

def make_title_generator(reply=None):
    """Build a small title call using the configured Studio reply provider."""
    if isinstance(reply, dict):
        segment = reply.get("llm", reply) or {}
        provider = segment.get("provider")
        cfg = segment.get("config") or {}
    else:
        provider, cfg = None, {}

    deepseek = str(provider or "").lower() == "deepseek"
    model = (cfg.get("model") or
             ('deepseek-v4-flash'
              if deepseek else CHAT_MODEL))
    client = None

    async def generate(text: str) -> str:
        nonlocal client
        if client is None:
            if deepseek or provider == "qwen":
                key = cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY" if provider == "qwen" else "DEEPSEEK_API_KEY")
                if not key:
                    raise ValueError("生成标题需要 DEEPSEEK_API_KEY")
                client = AsyncOpenAI(
                    api_key=key,
                    base_url=(cfg.get("base_url") or "https://api.deepseek.com").rstrip("/"),
                )
            else:
                client = _openai_client()
        request = {
            "model": model,
            "max_tokens": 16,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _TITLE_SYSTEM},
                {"role": "user", "content": text[:600]},
            ],
        }
        if deepseek:
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        elif provider == "qwen":
            request["extra_body"] = {"enable_thinking": False}
        response = await client.chat.completions.create(**request)
        return (response.choices[0].message.content or "").strip()

    return generate

def realtime_connect(reply=None):
    """Open a streaming realtime provider connection from the reply configuration."""
    _, cfg = _reply_seg(reply, "realtime")
    return _openai_client().realtime.connect(model=resolve_model(cfg.get("model"), "realtime"))

def _emotion_of(rb_hits) -> str:

    #

    for h in (rb_hits or []):
        if getattr(h, "source", "") != "current_signal":
            continue
        emo = ((getattr(h, "metadata", None) or {}).get("emotion") or "").strip()
        if emo:
            return emo
    return ""

_RB_PREFIX = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?(?:[⚠✓✱*]\s*)?(?:[^：:\s]{2,8}[：:]\s*)?")

_RB_SUFFIX = re.compile(
    r"[（(]\s*(?:下次|next time|内心OS|inner note)\s*[：:].*$", re.S | re.I)

def clean_rb(content: str) -> str:
    t = _RB_PREFIX.sub("", str(content or ""))
    t = _RB_SUFFIX.sub("", t)
    return t.strip()

def hits_payload(result, has_audio=None, cluster_of=None):
    """Convert normalized memory hits into the browser display payload."""
    rb = getattr(result, "rb_hits", None) or []
    cls = getattr(result, "classification", None)
    return {

        "slots": list(getattr(cls, "slots", []) or []),
        "entities": list(getattr(cls, "entities", []) or []),
        "emotion": _emotion_of(rb),
        "left_brain": [{"text": h.text, "score": h.score, "attributed_to": h.attributed_to,
                        "memory_id": h.memory_id,
                        "has_audio": bool(has_audio and has_audio(h.memory_id))}
                       for h in result.hits],

        "right_brain_hits": [{"content": clean_rb(h.content), "raw": h.content,
                              "internal": h.source == "response_experience",

                              "slot": ((getattr(h, "metadata", None) or {}).get("slot_name") or ""),

                              "claim": ((getattr(h, "metadata", None) or {}).get("claim") or ""),
                              "source": h.source, "priority": h.priority,
                              "cluster": cluster_of(h.content, h.source) if cluster_of else ""}
                             for h in (getattr(result, "rb_hits", None) or [])],
        "current_scene": getattr(result, 'current_scene', None),
        "related_summaries": getattr(result, "related_summaries", None) or {},
    }

def build_app(mode, session, classify, snapshot=None, audio_of=None, spaces=None,
              set_lang=None, title=None, pet_port=8787):
    """Build the browser API and WebSocket routes using injected session callbacks."""
    app = FastAPI()
    title = title or make_title_generator()
    pet, pet_hub = PetSupervisor(), PetHub()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await sock.send_json({"type": "session_ready", "mode": mode})
        await pet_hub.broadcast({"type": "conversation_started"})
        try:

            await session(TeeSocket(sock, pet_hub))
        except WebSocketDisconnect:
            pass
        finally:
            await pet_hub.broadcast({"type": "conversation_ended"})

    @app.websocket("/ws-pet")
    async def ws_pet(sock: WebSocket):
        await sock.accept()
        pet_hub.add(sock)
        try:
            while True:
                await sock.receive()
        except WebSocketDisconnect:
            pass
        finally:
            pet_hub.discard(sock)

    @app.on_event("startup")
    def _start_pet():
        pet.ensure_running(f"ws://127.0.0.1:{pet_port}/ws-pet")

    @app.on_event("shutdown")
    def _stop_pet():
        pet.shutdown()

    class Q(BaseModel):
        query: str

    @app.post("/api/classify")
    def api_classify(body: Q) -> dict:
        c = classify(body.query)
        return {"slots": list(c.slots), "entities": list(c.entities)}

    class T(BaseModel):
        text: str

    @app.post("/api/title")
    async def api_title(body: T) -> dict:
        try:
            return {"title": await title(body.text)}
        except Exception as e:
            print(f"[web] 生成标题失败：{e}", flush=True)
            return {"title": ""}

    @app.get("/api/memories")
    def api_memories() -> dict:
        return snapshot() if snapshot else {"left": [], "right": []}

    @app.post("/api/lang")
    async def api_lang(req: Request) -> dict:
        lang = (await req.json()).get("lang", "zh")
        reply_lang = set_lang(lang) if set_lang else lang
        return {"lang": lang, "reply_lang": reply_lang or lang}

    if spaces:
        _list_spaces, _create_space, _use_space, _active_space = spaces

        @app.get("/api/spaces")
        def api_spaces() -> dict:
            return {"spaces": _list_spaces(), "active": _active_space()}

        @app.post("/api/spaces")
        async def api_space_new(req: Request) -> dict:
            body = await req.json()
            name, lang = body.get("name", ""), body.get("language", "")
            try:
                return _create_space(name, lang)
            except FileExistsError as e:
                raise HTTPException(409, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))

        @app.post("/api/spaces/{name}/use")
        def api_space_use(name: str) -> dict:
            try:
                return {"active": _use_space(name)}
            except Exception as e:
                raise HTTPException(400, f"切不过去：{e}")

    @app.get("/api/audio/{memory_id}")
    def api_audio(memory_id: str):
        path = audio_of(memory_id) if audio_of else None
        if not path or not Path(path).exists():
            raise HTTPException(404, "这条记忆没有存档音频")
        return FileResponse(path, media_type="audio/wav")

    (HERE / "images").mkdir(exist_ok=True)
    app.mount("/images", StaticFiles(directory=HERE / "images"), name="images")

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
        if pet_on:
            pet.ensure_running(loopback_ws_url(request))
        return FileResponse(HERE / "voicemem.html", headers=_NOCACHE)

    @app.get("/classic")
    def classic():
        return FileResponse(HERE / "index.html", headers=_NOCACHE)

    return app
