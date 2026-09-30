"""Qwen-Audio-TTS adapter for Studio's streaming PCM speech contract."""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid

MODEL = "qwen-audio-3.0-tts-flash"
DEFAULT_VOICE = "longanlingxi"
SYSTEM_VOICES = frozenset({
    "longanfengyue", "longanyuanfei", "longanlingxi", "longanxiaoxin",
    "longanhuan_v3.6", "longjielidou_v3.6", "longpaopao_v3.6",
    "longhuohuo_v3.6", "longchuanshu_v3.6", "loongmary",
    "loongeva_v3.6", "loongjohn",
})
_WORKSPACE = re.compile(r"^[A-Za-z0-9_-]+$")


def selected() -> bool:
    """Select Qwen explicitly, or for a configured public demo only."""
    choice = os.environ.get("STUDIO_TTS_PROVIDER", "").strip().lower()
    if choice:
        if choice not in {"breeze", "qwen_audio_api"}:
            raise ValueError("STUDIO_TTS_PROVIDER 只能是 breeze 或 qwen_audio_api")
        return choice == "qwen_audio_api"
    return (os.environ.get("STUDIO_PUBLIC_DEMO") == "1" and
            bool(os.environ.get("STUDIO_QWEN_TTS_API_KEY") or
                 os.environ.get("STUDIO_QWEN_TTS_WORKSPACE_ID")))


def settings() -> tuple[str, str, str]:
    """Return key, Singapore workspace, and the selected system voice."""
    key = os.environ.get("STUDIO_QWEN_TTS_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    workspace = (os.environ.get("STUDIO_QWEN_TTS_WORKSPACE_ID") or "").strip()
    voice = (os.environ.get("STUDIO_QWEN_TTS_VOICE") or DEFAULT_VOICE).strip()
    if not key:
        raise ValueError("Qwen TTS 缺少 STUDIO_QWEN_TTS_API_KEY 或 DASHSCOPE_API_KEY")
    if not _WORKSPACE.fullmatch(workspace):
        raise ValueError("Qwen TTS 缺少有效的 STUDIO_QWEN_TTS_WORKSPACE_ID")
    if not voice or any(char.isspace() for char in voice):
        raise ValueError("Qwen TTS 系统音色格式无效；请检查 STUDIO_QWEN_TTS_VOICE")
    if voice not in SYSTEM_VOICES:
        raise ValueError(f"Qwen TTS 不支持该 {MODEL} 系统音色：{voice}")
    return key, workspace, voice


class QwenAudioAPI:
    """Serialize speech tasks on one WebSocket, closing it after interruption."""

    SERIAL = True
    sample_rate = 24000

    def __init__(self):
        self.api_key, self.workspace, self.voice = settings()
        self._ws = None
        self._lock = asyncio.Lock()

    async def _connect(self):
        if self._ws is not None and getattr(self._ws, "close_code", None) is not None:
            self._ws = None
        if self._ws is not None:
            return self._ws
        import websockets
        url = f"wss://{self.workspace}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference"
        self._ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Bearer {self.api_key}"},
            open_timeout=10, close_timeout=2, max_size=8 * 1024 * 1024)
        return self._ws

    @staticmethod
    async def _send(ws, action, task_id, *, payload):
        await ws.send(json.dumps({
            "header": {"action": action, "task_id": task_id, "streaming": "duplex"},
            "payload": payload,
        }))

    @staticmethod
    async def _receive(ws, task_id):
        message = await asyncio.wait_for(ws.recv(), timeout=45)
        if isinstance(message, bytes):
            return message
        event = json.loads(message)
        header = event.get("header") or {}
        if header.get("task_id") != task_id:
            raise RuntimeError("Qwen TTS returned a mismatched task ID")
        if header.get("event") == "task-failed":
            raise RuntimeError(f"Qwen TTS {header.get('error_code', 'error')}: "
                               f"{header.get('error_message', '')}")
        return header.get("event")

    async def _discard(self):
        ws, self._ws = self._ws, None
        if ws is not None:
            await ws.close()

    async def preconnect(self):
        """Establish the Singapore connection before the first reply."""
        async with self._lock:
            await self._connect()

    async def stream(self, text: str, instruction: str | None = None):
        """Yield 24 kHz mono PCM16 and apply Studio's instruction per segment."""
        if not text.strip():
            return
        async with self._lock:
            complete = False
            emitted = False
            try:
                ws = await self._connect()
                task_id = str(uuid.uuid4())
                parameters = {
                    "text_type": "PlainText", "voice": self.voice,
                    "format": "pcm", "sample_rate": self.sample_rate,
                }
                if instruction:
                    parameters["instruction"] = instruction
                await self._send(ws, "run-task", task_id, payload={
                    "task_group": "audio", "task": "tts", "function": "SpeechSynthesizer",
                    "model": MODEL, "parameters": parameters, "input": {},
                })
                started = await self._receive(ws, task_id)
                if started != "task-started":
                    raise RuntimeError(f"Qwen TTS expected task-started, got {started}")
                await self._send(ws, "continue-task", task_id, payload={"input": {"text": text}})
                await self._send(ws, "finish-task", task_id, payload={"input": {}})
                while True:
                    event = await self._receive(ws, task_id)
                    if isinstance(event, bytes):
                        if len(event) % 2:
                            raise RuntimeError("Qwen TTS returned an incomplete PCM sample")
                        if event:
                            emitted = True
                            yield event
                    elif event == "task-finished":
                        if not emitted:
                            raise RuntimeError("Qwen TTS task completed without audio")
                        complete = True
                        break
            finally:
                if not complete:
                    await self._discard()

    async def aclose(self):
        """Close the reusable vendor connection."""
        async with self._lock:
            await self._discard()
