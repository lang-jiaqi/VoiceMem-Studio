"""桌面小人（pet/）的进程管理和播放状态转播。

小人是一个独立的 Electron 进程，不是页面的一部分——网页里的 canvas 盖不到别的
应用上面去，要飘在桌面最上层只能是系统级窗口。所以它跟浏览器之间没有从属关系：

    第一个带 ?pet=1 的请求把它拉起来，之后关标签页、刷新、开十个标签页都不管它，
    只有后端进程退出时它才跟着关。

这样"小人在不在"这件事只由后端在不在决定，不需要心跳、宽限期、引用计数那一套。

嘴型走的是 TeeSocket：会话 socket 外面包一层，把跟"在不在出声"有关的消息抄一份
给小人。run.py 的会话逻辑照旧只跟 sock 打交道，一行都不用改。
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

PET_DIR = Path(__file__).resolve().parent.parent / "pet"

# 默认用 pet/ 里装好的 electron 跑源码。打包成 .app 之后可以换成
# VOICEMEM_PET_CMD="open -a /Applications/Noctelle.app --args"。
DEFAULT_CMD = "node launch.cjs"


class PetSupervisor:
    """保证同一时刻最多只有一只小人。"""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._warned = False

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_running(self, ws_url: str) -> None:
        """拉起小人；已经在跑就什么都不做。

        起不来不能影响页面——没有小人只是少个装饰，页面本身照常用，
        所以这里只打日志，不抛异常。
        """
        with self._lock:
            if self.alive():
                return
            if not (PET_DIR / "main.cjs").exists():
                self._warn(f"找不到 {PET_DIR}/main.cjs，跳过桌面小人")
                return
            command = os.environ.get("VOICEMEM_PET_CMD", DEFAULT_CMD).split()
            if not shutil.which(command[0]):
                self._warn(f"命令 {command[0]} 不在 PATH 上，跳过桌面小人")
                return
            argv = [*command, "--expanded", f"--ws={ws_url}"]
            try:
                # 单独开一个进程组：launch.cjs 只是个壳，真正的 Electron 是它的子进程，
                # 只 terminate 这个壳的话小人会活下来变成孤儿窗口，关都关不掉。
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
        """整组一起收：壳进程和它底下的 Electron 都在同一个进程组里。"""
        try:
            os.killpg(os.getpgid(self._process.pid), sig)
        except (ProcessLookupError, PermissionError):
            self._process.kill()                        # 组没了就退回单进程

    def _warn(self, message: str) -> None:
        # 每次刷新页面都刷同一行没意义，起不来的原因通常不会自己变。
        if not self._warned:
            print(f"[pet] {message}", file=sys.stderr, flush=True)
            self._warned = True


class PetHub:
    """连上来的小人（通常就一只）。断了直接丢掉，不重试——它自己会重连。"""

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
                # 小人挂了/关了：把它摘掉就行，绝不能让它影响正在进行的对话。
                self._clients.discard(sock)


class TeeSocket:
    """会话 socket 的透明代理，顺手把嘴型要用的消息抄给小人。

    没被覆盖的属性（send_bytes、client_state 等）全部透传给真 socket，
    所以 run.py 里 `sock.send_json` 被当成回调到处传也没问题——拿到的是
    这个类的绑定方法，抄送照样生效。
    """

    #: 抄给小人的出站消息。
    #: answer_interrupt 是兜底——被打断时页面不一定还来得及回 checkpoint；
    #: backchannel 是"嗯"一声，小人拿它点头。
    _TEE_OUT = {"answer_interrupt", "backchannel"}

    def __init__(self, sock, hub: PetHub):
        self._sock = sock
        self._hub = hub

    def __getattr__(self, name):
        return getattr(self._sock, name)

    async def send_json(self, data, *args, **kwargs):
        if isinstance(data, dict) and data.get("type") in self._TEE_OUT:
            # 只抄类型，不抄内容：backchannel 里带着一整条 base64 音频，
            # 而小人不放声音，把它转发过去纯属白占带宽。
            await self._hub.broadcast({"type": data["type"]})
        return await self._sock.send_json(data, *args, **kwargs)

    async def receive(self, *args, **kwargs):
        message = await self._sock.receive(*args, **kwargs)
        # playback_checkpoint 是页面**发回来**的，来源是音频 worklet 的真实播放状态，
        # 比后端自己的 answer_start/answer_done 贴近"耳朵听到的"：answer_done 发出去
        # 的时候，页面那边往往还有一截音频排在缓冲里没播完。
        #
        # 只看文本帧——上行的麦克风音频是二进制，每帧都试着 json.loads 纯属浪费。
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
    """小人永远跟后端在同一台机器上，所以连回环地址，不跟着浏览器用的域名走。

    （也跟 pet/index.html 里 CSP 放行的 ws://127.0.0.1:* 对得上。）
    """
    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    return f"ws://127.0.0.1:{port}/ws-pet"


__all__ = ["PetSupervisor", "PetHub", "TeeSocket", "loopback_ws_url"]
