"""Studio transport implementation."""
import asyncio
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from studio.web.pet_bridge import PetHub, PetSupervisor, TeeSocket, loopback_ws_url
from studio.core.voicemem import shared_embed_model

from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder, shared_e5  # noqa: F401
from voicemem.leftbrain.local_embedder import (  # noqa: F401
    LocalEmbedder, resolve, resolve_path, shared_model,
)
from voicemem.llm_config import resolve_model

HERE = Path(__file__).resolve().parent

CHAT_MODEL = resolve_model(role="reply", default="deepseek-v4-flash")
if not CHAT_MODEL or CHAT_MODEL.startswith("gpt-"):
    CHAT_MODEL = "deepseek-v4-flash"
RT_MODEL = resolve_model(role="realtime")

#:

#:

#:

from voicemem.tts import TTS_VOICE as _TTS_VOICE  # noqa: E402

RT_VOICE = _TTS_VOICE
client = None


class NoCacheStaticFiles(StaticFiles):
    """Serve the mutable Studio UI without reusing stale Electron assets."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response

def _openai_client():

    global client
    if client is None:
        key = os.environ.get("VOICEMEM_STUDIO_API_KEY")
        base_url = (os.environ.get("VOICEMEM_STUDIO_BASE_URL") or
                    "https://api.openai.com/v1") if key else None
        client = AsyncOpenAI(
            api_key=key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
    return client

from voicemem.utils.audio.stream_io import resample  # noqa: E402,F401

from voicemem.tts import (  # noqa: E402,F401
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

def make_title_generator(reply=None, fallback=None):
    """Build a title call from the reply provider or its remote fallback."""
    if isinstance(reply, dict):
        segment = reply.get("llm", reply) or {}
        provider = segment.get("provider")
        cfg = segment.get("config") or {}
    else:
        provider, cfg = None, {}
    if provider == "local" and isinstance(fallback, dict):
        segment = fallback.get("llm", fallback) or {}
        provider = segment.get("provider")
        cfg = segment.get("config") or {}

    deepseek = str(provider or "").lower() == "deepseek"
    model = (cfg.get("model") or
             ('deepseek-v4-flash'
              if deepseek else CHAT_MODEL))
    client = None

    async def generate(text: str) -> str:
        nonlocal client
        if client is None:
            if provider in {"deepseek", "qwen", "openai"}:
                key = (cfg.get("api_key") or os.environ.get("VOICEMEM_STUDIO_API_KEY") or
                       os.environ.get("DASHSCOPE_API_KEY" if provider == "qwen" else
                                      "DEEPSEEK_API_KEY" if deepseek else "OPENAI_API_KEY"))
                if not key:
                    raise ValueError("生成标题需要 Studio 回复 API Key")
                base_url = cfg.get("base_url") or {
                    "deepseek": "https://api.deepseek.com",
                    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                    "openai": "https://api.openai.com/v1",
                }[provider]
                client = AsyncOpenAI(
                    api_key=key,
                    base_url=base_url.rstrip("/"),
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
              set_lang=None, title=None, components=None, pet_port=8787,
              demo_accounts=None, demo_session=None, demo_components=None):
    """Build the browser API and WebSocket routes using injected session callbacks."""
    app = FastAPI()
    desktop_instance = os.environ.get('VOICEMEM_DESKTOP_INSTANCE', '')
    title = title or make_title_generator()
    pet, pet_hub = PetSupervisor(), PetHub()

    if demo_accounts:
        from studio.web.demo_accounts import COOKIE, SESSION_SECONDS
        app.state.account_jobs = set()

        def track_account_job(task):
            app.state.account_jobs.add(task)
            task.add_done_callback(app.state.account_jobs.discard)
            return task

        async def acquire_demo_agent(user_id):
            work = track_account_job(asyncio.create_task(
                asyncio.to_thread(demo_accounts.acquire_agent, user_id)))
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                # Native construction can finish after its requester disconnects.
                def release_abandoned(done):
                    if done.cancelled() or done.exception() is not None:
                        return
                    release = track_account_job(asyncio.create_task(asyncio.to_thread(
                        demo_accounts.release_agent, user_id, done.result())))
                    release.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                work.add_done_callback(release_abandoned)
                raise

        def same_origin(headers):
            origin = headers.get("origin", "")
            host = headers.get("host", "")
            return bool(origin and host and urlsplit(origin).netloc == host and
                        urlsplit(origin).scheme in {"http", "https"})

        @app.middleware("http")
        async def require_demo_login(request: Request, call_next):
            path = request.url.path
            if request.method not in {"GET", "HEAD", "OPTIONS"} and not same_origin(request.headers):
                return JSONResponse({"detail": "来源不受信任"}, status_code=403, headers=_NOCACHE)
            if path.startswith("/auth/") or path == "/ui/login.html":
                response = await call_next(request)
                response.headers["Cache-Control"] = "no-store"
                return response
            user = await asyncio.to_thread(demo_accounts.user, request.cookies.get(COOKIE, ""))
            if not user:
                if path == "/" or path.startswith("/ui/"):
                    return RedirectResponse("/ui/login.html", status_code=303, headers=_NOCACHE)
                return JSONResponse({"detail": "请先登录"}, status_code=401, headers=_NOCACHE)
            request.state.demo_user = user
            if path in {"/ui/technical.html", "/ui/digital.html", "/legacy", "/classic"} and re.search(
                    r"iPhone|Android|Mobile", request.headers.get("user-agent", ""), re.I):
                return RedirectResponse("/ui/pet-mobile.html", status_code=303, headers=_NOCACHE)
            request.state.demo_agent_lease = None
            try:
                response = await call_next(request)
                if path.startswith("/api/") or response.status_code >= 400:
                    response.headers["Cache-Control"] = "no-store"
                return response
            finally:
                if request.state.demo_agent_lease is not None:
                    await asyncio.to_thread(demo_accounts.release_agent, user[0],
                                            request.state.demo_agent_lease)

        class Credentials(BaseModel):
            name: str
            password: str

        async def authenticate(request: Request, body: Credentials, register: bool):
            source = request.client.host if request.client else "unknown"
            try:
                await asyncio.to_thread(demo_accounts.limit_attempt, source)
                action = demo_accounts.register if register else demo_accounts.login
                name, token = await asyncio.to_thread(action, body.name, body.password)
            except ValueError as exc:
                raise HTTPException(400 if register else 401, str(exc)) from None
            response = JSONResponse({"name": name})
            response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS,
                                httponly=True, secure=True, samesite="lax")
            return response

        @app.post("/auth/register")
        async def register(request: Request, body: Credentials):
            return await authenticate(request, body, True)

        @app.post("/auth/login")
        async def login(request: Request, body: Credentials):
            return await authenticate(request, body, False)

        @app.post("/auth/logout")
        async def logout(request: Request):
            await asyncio.to_thread(demo_accounts.logout, request.cookies.get(COOKIE, ""))
            response = JSONResponse({"ok": True})
            response.delete_cookie(COOKIE)
            return response

        @app.get("/auth/me")
        async def me(request: Request):
            user = await asyncio.to_thread(demo_accounts.user, request.cookies.get(COOKIE, ""))
            if not user:
                raise HTTPException(401, "请先登录")
            return {"name": user[1]}

        async def demo_agent(request: Request):
            if request.state.demo_agent_lease is None:
                request.state.demo_agent_lease = await acquire_demo_agent(request.state.demo_user[0])
            return request.state.demo_agent_lease

        @app.on_event("startup")
        async def start_account_cleanup():
            async def sweep():
                while True:
                    await asyncio.sleep(60)
                    await asyncio.to_thread(demo_accounts.prune_idle)
            app.state.account_cleanup = asyncio.create_task(sweep())

        @app.on_event("shutdown")
        async def stop_account_cleanup():
            task = getattr(app.state, 'account_cleanup', None)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            while app.state.account_jobs:
                await asyncio.gather(*tuple(app.state.account_jobs), return_exceptions=True)
                await asyncio.sleep(0)
            await asyncio.to_thread(demo_accounts.prune_idle, idle_seconds=0, max_idle_agents=0)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        active_agent = None
        if demo_accounts:
            if not same_origin(sock.headers):
                await sock.close(code=1008)
                return
            user = await asyncio.to_thread(demo_accounts.user, sock.cookies.get(COOKIE, ""))
            if not user:
                await sock.close(code=1008)
                return
            active_agent = await acquire_demo_agent(user[0])
        tee = TeeSocket(sock, pet_hub)
        try:
            await sock.accept()
            await sock.send_json({"type": "session_ready", "mode": mode})
            await pet_hub.broadcast({"type": "conversation_started",
                                     "session_id": tee.session_id})
            if demo_accounts:
                await demo_session(active_agent, tee)
            else:
                await session(tee)
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await pet_hub.broadcast({"type": "conversation_ended",
                                         "session_id": tee.session_id})
            finally:
                if demo_accounts and active_agent is not None:
                    await asyncio.to_thread(demo_accounts.release_agent, user[0], active_agent)

    @app.websocket("/ws-pet")
    async def ws_pet(sock: WebSocket):
        if demo_accounts:
            await sock.close(code=1008)
            return
        await sock.accept()
        pet_hub.add(sock)
        try:
            while True:
                if (await sock.receive())["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            pet_hub.discard(sock)

    @app.on_event("startup")
    def _start_pet():
        if not demo_accounts:
            pet.ensure_running(f"ws://127.0.0.1:{pet_port}/ws-pet")

    @app.on_event("shutdown")
    def _stop_pet():
        pet.shutdown()

    class Q(BaseModel):
        query: str

    @app.post("/api/classify")
    async def api_classify(body: Q, request: Request) -> dict:
        if demo_accounts:
            agent = await demo_agent(request)
            c = await asyncio.to_thread(agent.vm.classify, body.query)
        else:
            c = await asyncio.to_thread(classify, body.query)
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
    async def api_memories(request: Request) -> dict:
        if demo_accounts:
            agent = await demo_agent(request)
            return await asyncio.to_thread(agent.memory_snapshot)
        return await asyncio.to_thread(snapshot) if snapshot else {"left": [], "right": []}

    @app.get("/api/components")
    async def api_components(request: Request) -> dict:
        """Expose non-secret runtime labels for the component settings canvas."""
        if demo_accounts:
            agent = await demo_agent(request)
            return await asyncio.to_thread(demo_components, agent)
        return await asyncio.to_thread(components) if components else {}

    @app.post("/api/lang")
    async def api_lang(req: Request) -> dict:
        lang = (await req.json()).get("lang", "zh")
        if demo_accounts:
            agent = await demo_agent(req)
            ui_lang = await asyncio.to_thread(agent.set_lang, lang)
        else:
            ui_lang = await asyncio.to_thread(set_lang, lang) if set_lang else lang
        return {"lang": ui_lang, "reply_lang": "auto"}

    if spaces:
        _list_spaces, _create_space, _use_space, _active_space = spaces

        @app.get("/api/spaces")
        async def api_spaces(request: Request) -> dict:
            if demo_accounts:
                agent = await demo_agent(request)
                listed = await asyncio.to_thread(agent.list_spaces)
                return {"spaces": [item for item in listed if item["id"] == "default"],
                        "active": "default", "demo": True}
            return {"spaces": await asyncio.to_thread(_list_spaces), "active": _active_space()}

        @app.post("/api/spaces")
        async def api_space_new(req: Request) -> dict:
            body = await req.json()
            name, lang = body.get("name", ""), body.get("language", "")
            try:
                if demo_accounts:
                    raise HTTPException(403, "体验账号使用一个独立记忆空间")
                return await asyncio.to_thread(_create_space, name, lang)
            except FileExistsError as e:
                raise HTTPException(409, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))

        @app.post("/api/spaces/{name}/use")
        async def api_space_use(name: str, request: Request) -> dict:
            try:
                if demo_accounts:
                    if name != "default":
                        raise HTTPException(403, "体验账号使用一个独立记忆空间")
                    return {"active": "default"}
                return {"active": await asyncio.to_thread(_use_space, name)}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(400, f"切不过去：{e}")

    @app.get("/api/audio/{memory_id}")
    async def api_audio(memory_id: str, request: Request):
        if demo_accounts:
            agent = await demo_agent(request)
            path = await asyncio.to_thread(agent.audio_of, memory_id)
        else:
            path = await asyncio.to_thread(audio_of, memory_id) if audio_of else None
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
        if demo_accounts and re.search(r"iPhone|Android|Mobile", request.headers.get("user-agent", ""), re.I):
            return RedirectResponse("/ui/pet-mobile.html", status_code=303, headers=_NOCACHE)
        if pet_on and not demo_accounts:
            pet.ensure_running(loopback_ws_url(request))
        headers = {**_NOCACHE}
        if desktop_instance:
            headers['X-VoiceMem-Desktop-Instance'] = desktop_instance
        return FileResponse(HERE.parent / "apps" / "ui" / "index.html", headers=headers)

    if demo_accounts:
        from studio.web.pet_assets import (
            CACHE_CONTROL, MOBILE_FILES, PET_SCRIPTS, PET_VENDORS,
            CachedStaticFiles, MobilePetAssets, accepts_gzip, cached_response,
        )
        pet_root = HERE.parent / "pet"
        ui_root = HERE.parent / "apps" / "ui"
        vendor_root = HERE.parent / "apps" / "node_modules"
        bundle = MobilePetAssets(pet_root, ui_root, vendor_root)
        cached_vendor = CachedStaticFiles(vendor_root, files=PET_VENDORS, check_dir=False,
                                         gzip_assets={PET_VENDORS[name]: bundle.gzip_assets[f"vendor/{name}"]
                                                      for name in PET_VENDORS if f"vendor/{name}" in bundle.gzip_assets})

        @app.api_route("/ui/pet-mobile.html", methods=["GET", "HEAD"])
        def mobile_pet_index(request: Request):
            html = bundle.html
            if request.query_params.get("model") and bundle.model_preloads:
                html = html.replace(bundle.model_preloads, "")
            return HTMLResponse(html, headers=_NOCACHE)

        @app.api_route(f"{bundle.prefix}/ui/pet-mobile.css", methods=["GET", "HEAD"])
        def mobile_pet_css(request: Request):
            headers = {"Cache-Control": CACHE_CONTROL, "ETag": bundle.css_etag,
                       "Content-Type": "text/css; charset=utf-8", "Vary": "Accept-Encoding"}
            if bundle.css_gzip and accepts_gzip(request.headers) and "range" not in request.headers:
                return bundle.css_gzip.response(headers, request.scope)
            return cached_response(bundle.css, headers, request.scope)

        @app.api_route(f"{bundle.prefix}/vendor/{{name}}", methods=["GET", "HEAD"])
        async def mobile_pet_vendor(name: str, request: Request):
            if name not in PET_VENDORS:
                raise HTTPException(404)
            if not (vendor_root / PET_VENDORS[name]).is_file():
                raise HTTPException(503, "请先安装 studio/apps 的 npm 依赖")
            return await cached_vendor.get_response(name, request.scope)

        def gzip_files(category):
            return {name.removeprefix(f"{category}/"): asset for name, asset in bundle.gzip_assets.items()
                    if name.startswith(f"{category}/")}

        app.mount(f"{bundle.prefix}/assets", CachedStaticFiles(pet_root / "assets", gzip_assets=gzip_files("assets")),
                  name="versioned-pet-assets")
        app.mount(f"{bundle.prefix}/ui", CachedStaticFiles(
            ui_root, files={name: name for name in MOBILE_FILES}, gzip_assets=gzip_files("ui")), name="versioned-pet-ui")
        app.mount(bundle.prefix, CachedStaticFiles(
            pet_root, files={name: name for name in PET_SCRIPTS}, gzip_assets=gzip_files("pet")), name="versioned-pet-scripts")
        app.mount("/pet/assets", StaticFiles(directory=pet_root / "assets"), name="demo-pet-assets")

        @app.get("/pet/{name}")
        def demo_pet_script(name: str):
            if name not in PET_SCRIPTS:
                raise HTTPException(404)
            return FileResponse(pet_root / name, media_type="application/javascript", headers=_NOCACHE)

        @app.get("/pet/vendor/{name}")
        def demo_pet_vendor(name: str):
            packages = PET_VENDORS
            if name not in packages:
                raise HTTPException(404)
            path = HERE.parent / "apps" / "node_modules" / packages[name]
            if not path.is_file():
                raise HTTPException(503, "请先安装 studio/apps 的 npm 依赖")
            return FileResponse(path, media_type="application/javascript", headers=_NOCACHE)

    @app.get("/legacy")
    def legacy():
        return FileResponse(HERE / "voicemem.html", headers=_NOCACHE)

    @app.get("/classic")
    def classic():
        return FileResponse(HERE / "index.html", headers=_NOCACHE)

    app.mount(
        "/ui",
        NoCacheStaticFiles(directory=HERE.parent / "apps" / "ui", html=True),
        name="studio-ui",
    )
    return app
