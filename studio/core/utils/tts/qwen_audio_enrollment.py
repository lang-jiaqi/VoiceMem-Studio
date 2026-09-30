"""One-time Qwen-Audio voice enrollment for the optional Studio provider."""
from __future__ import annotations

import hashlib
import os
import time
from urllib.parse import urlparse

import httpx

from studio.paths import VOICE
from .qwen_audio_api import MODEL, VOICE_ID_FILE, settings

REFERENCE_AUDIO = VOICE / "qwen_audio_reference.wav"
REFERENCE_URL = (
    "https://raw.githubusercontent.com/lang-jiaqi/VoiceMem-Studio/main/"
    "studio/resources/voice/qwen_audio_reference.wav"
)


def payload(url: str, prefix: str = "noctelle") -> dict:
    """Build the model-bound enrollment request for a public HTTPS recording."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("参考录音需要公开可访问的 HTTPS URL")
    if not prefix.isascii() or not prefix.isalnum() or len(prefix) > 10:
        raise ValueError("音色前缀只能用最多 10 个 ASCII 字母或数字")
    return {"model": "voice-enrollment", "input": {
        "action": "create_voice", "target_model": MODEL,
        "prefix": prefix, "url": url, "language_hints": ["zh"],
        "max_prompt_audio_length": 30.0,
    }}


def _endpoint(workspace: str) -> str:
    return (f"https://{workspace}.ap-southeast-1.maas.aliyuncs.com"
            "/api/v1/services/audio/tts/customization")


def _post(client, endpoint, key, request):
    try:
        response = client.post(endpoint, headers={"Authorization": f"Bearer {key}"},
                               json=request)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Qwen TTS API 请求失败：{type(exc).__name__}: {exc}") from None
    result = response.json()
    if result.get("code"):
        raise RuntimeError(f"Qwen TTS {result['code']}: {result.get('message', '')}")
    return result.get("output") or {}


def _wait_ready(client, endpoint, key, voice):
    if not voice.startswith(MODEL + "-"):
        return
    deadline = time.monotonic() + 120
    while True:
        output = _post(client, endpoint, key, {"model": "voice-enrollment", "input": {
            "action": "query_voice", "voice_id": voice,
        }})
        if output.get("target_model") != MODEL:
            raise RuntimeError("Qwen TTS 音色绑定的模型不匹配")
        status = output.get("status")
        if status == "OK":
            return
        if status != "DEPLOYING":
            raise RuntimeError(f"Qwen TTS 音色不可用：{status or 'unknown'}")
        if time.monotonic() >= deadline:
            raise RuntimeError("Qwen TTS 音色仍在处理，请稍后重启 Studio")
        time.sleep(2)


def ensure_voice(*, url: str | None = None, prefix: str = "noctelle", force: bool = False):
    """Reuse an existing voice or enroll the checked-in reference once."""
    key, workspace, voice = settings(require_voice=False)
    endpoint = _endpoint(workspace)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        if voice and not force:
            _wait_ready(client, endpoint, key, voice)
            return voice
        if force and os.environ.get("STUDIO_QWEN_TTS_VOICE_ID"):
            raise ValueError("要重新注册音色，请先移除 STUDIO_QWEN_TTS_VOICE_ID")
        reference_url = url or os.environ.get("STUDIO_QWEN_TTS_REFERENCE_URL") or REFERENCE_URL
        request = payload(reference_url, prefix)
        if reference_url == REFERENCE_URL:
            if not REFERENCE_AUDIO.is_file():
                raise RuntimeError(f"Qwen TTS 缺少参考录音：{REFERENCE_AUDIO}")
            try:
                response = client.get(reference_url)
                response.raise_for_status()
            except httpx.HTTPError:
                raise RuntimeError("GitHub 参考录音不可访问；请先把 qwen_audio_reference.wav 推送到 main") from None
            if hashlib.sha256(response.content).digest() != hashlib.sha256(
                    REFERENCE_AUDIO.read_bytes()).digest():
                raise RuntimeError("GitHub 参考录音尚未与本机一致；请先提交并推送 qwen_audio_reference.wav")
        output = _post(client, endpoint, key, request)
        voice = output.get("voice_id")
        if not voice:
            raise RuntimeError("Qwen TTS 未返回音色 ID")
        VOICE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        VOICE_ID_FILE.write_text(voice + "\n", encoding="utf-8")
        print(f"[tts] Qwen 音色 ID 已保存：{VOICE_ID_FILE}", flush=True)
        _wait_ready(client, endpoint, key, voice)
    return voice
