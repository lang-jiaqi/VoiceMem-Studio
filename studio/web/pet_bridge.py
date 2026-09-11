"""Studio pet bridge implementation."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

PET_DIR = Path(__file__).resolve().parents[2] / "pet"

# VOICEMEM_PET_CMD="open -a /Applications/Noctelle.app --args"。
DEFAULT_CMD = "node launch.cjs"

class PetSupervisor:
    """Own the optional pet process and stop it when the server shuts down."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._warned = False

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_running(self, ws_url: str) -> None:
        """Start the pet process once and reuse it on later requests."""
        if os.environ.get('STUDIO_DESKTOP_PET', '1').strip().lower() in {'0', 'false', 'off', 'no'}:
            return
        with self._lock:
            if self.alive():
                return
            if not (PET_DIR / "main.cjs").exists():
                self._warn(f"找不到 {PET_DIR}/main.cjs，跳过桌面小人")
                return
            command = DEFAULT_CMD.split()
            if not shutil.which(command[0]):
                self._warn(f"命令 {command[0]} 不在 PATH 上，跳过桌面小人")
                return
            argv = [*command, "--expanded", f"--ws={ws_url}"]
            try:

                self._process = subprocess.Popen(
                    argv, cwd=PET_DIR, start_new_session=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:                      # noqa: BLE001
                self._warn(f"拉起桌面小人失败：{e}")
                return
            self._warned = False
            print(f"[pet] 已启动：{' '.join(argv)}", flush=True)

    def shutdown(self) -> None:
        with self._lock:
            if not self.alive():
                return
            print("[pet] 后端退出，关掉小人", flush=True)
            self._signal_group(signal.SIGTERM)
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)

    def _signal_group(self, sig) -> None:
        try:
            os.killpg(os.getpgid(self._process.pid), sig)
        except (ProcessLookupError, PermissionError):
            self._process.kill()

    def _warn(self, message: str) -> None:

        if not self._warned:
            print(f"[pet] {message}", file=sys.stderr, flush=True)
            self._warned = True

class PetHub:
    """Broadcast assistant output to connected pet observers."""

    def __init__(self):
        self._clients: set = set()

    def add(self, sock) -> None:
        self._clients.add(sock)

    def discard(self, sock) -> None:
        self._clients.discard(sock)

    async def broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        text = json.dumps(payload, ensure_ascii=False)
        for sock in list(self._clients):
            try:
                await sock.send_text(text)
            except Exception:                           # noqa: BLE001

                self._clients.discard(sock)

class TeeSocket:
    """Forward browser output to pet observers without creating another conversation."""

    _TEE_OUT = {"answer_interrupt", "backchannel"}

    def __init__(self, sock, hub: PetHub):
        self._sock = sock
        self._hub = hub
        self._voice_active = False

    def __getattr__(self, name):
        return getattr(self._sock, name)

    async def voice_activity(self, active: bool) -> None:
        """Notify pet observers on VAD transitions without forwarding microphone audio."""
        if active != self._voice_active:
            self._voice_active = active
            await self._hub.broadcast({"type": "user_voice", "active": active})

    async def send_json(self, data, *args, **kwargs):
        if isinstance(data, dict) and data.get("type") in self._TEE_OUT:

            await self._hub.broadcast({"type": data["type"]})
        return await self._sock.send_json(data, *args, **kwargs)

    async def receive(self, *args, **kwargs):
        message = await self._sock.receive(*args, **kwargs)

        #

        text = message.get("text") if isinstance(message, dict) else None
        if text:
            try:
                payload = json.loads(text)
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict) and payload.get("type") == "playback_checkpoint":
                await self._hub.broadcast(payload)
        return message

def loopback_ws_url(request) -> str:
    """Build a loopback WebSocket URL for the pet connection."""
    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    return f"ws://127.0.0.1:{port}/ws-pet"

__all__ = ["PetSupervisor", "PetHub", "TeeSocket", "loopback_ws_url"]
