"""Studio memory implementation."""
import asyncio
import re
import time
import uuid

class Memory:
    def save_turn_audio(self, pcm16k) -> str:
        """Archive turn PCM and return its path, or an empty string on failure."""
        if pcm16k is None or not len(pcm16k):
            return ""
        try:
            import numpy as np
            import soundfile as sf
            self.TURN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            path = self.TURN_AUDIO_DIR / f"turn_{uuid.uuid4().hex[:12]}.wav"
            sf.write(path, np.asarray(pcm16k, dtype="float32"), 16000)
            return str(path)
        except Exception as e:
            print(f"[web] 存本轮音频失败（不影响对话）：{e}", flush=True)
            return ""

    def remember_turn(self, pending, reply: str, owner: dict, history_turn_id: str = "",
                      memory_vm=None) -> None:
        """Persist the captured turn and heard reply to the originating memory instance."""

        text = pending.text
        meaningful = re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")
        cjk = len(re.findall(r"[\u4e00-\u9fff]", meaningful))
        if pending.audio_path and cjk < 2 and len(meaningful) < 3:
            from voicemem.stream import SOUND_ONLY_TEXT
            text = SOUND_ONLY_TEXT

        try:
            target_vm = memory_vm or self.vm
            r = target_vm.ingest(
                text, agent_reply=reply, async_facts=True,
                audio=pending.audio_path,
                on_complete=lambda result: self._finish_history_turn(history_turn_id, result),
            ) or {}
        except Exception as e:
            print(f"[web] 存这一轮失败：{type(e).__name__}: {e}", flush=True)
            return

        affect = r.get("affect")
        if isinstance(affect, dict):
            affect = affect.get("emotion") or affect.get("label") or ""
        owner["emotion"] = str(affect or "").strip()

        if r.get("recognized_tune") and pending.audio_path:
            self._remember_tune(pending.audio_path)
        elif self.BARGE_DEBUG and self._wants_sound(pending.text or ""):
            print(f"  [replay] 这一轮没记住音乐："
                  f"tune={bool(r.get('recognized_tune'))} audio={bool(pending.audio_path)}",
                  flush=True)

        sid = r.get("speaker_id") or ""
        if not sid:

            return
        if not owner["id"]:
            owner["id"] = sid
        owner["last"] = sid
        owner["miss"] = 0 if sid == owner["id"] else owner.get("miss", 0) + 1

    def hot_path_enter(self) -> dict:
        """Mark audible reply work active and return a token for balanced release."""
        token = {"open": True}
        self._HOT["n"] += 1
        self._IDLE.clear()
        try:
            asyncio.current_task().add_done_callback(lambda _: self.hot_path_exit(token))
        except Exception:
            pass
        return token

    def hot_path_exit(self, token: dict) -> None:
        if not token.get("open"):
            return
        token["open"] = False
        self._HOT["n"] = max(0, self._HOT["n"] - 1)
        if self._HOT["n"] == 0:
            self._IDLE.set()

    async def wait_idle(self, what: str = "") -> None:
        """Wait for the reply hot path to clear, with a bounded starvation fallback."""
        if self._IDLE.is_set():
            return
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(self._IDLE.wait(), timeout=self._IDLE_MAX_WAIT_S)
        except asyncio.TimeoutError:
            pass
        if self.BARGE_DEBUG and what:
            print(f"  [idle] {what} 让路 {(time.monotonic()-t0)*1000:.0f}ms"
                  f"（热路径计数 {self._HOT['n']}）", flush=True)

    async def _remember_background(self, pending, reply: str, owner: dict,
                                   history_turn_id: str, memory_vm) -> None:
        queued_at = time.monotonic()
        async with self._REMEMBER_LOCK:
            waited = time.monotonic() - queued_at
            if waited > 0.05 and self.BARGE_DEBUG:
                print(f"[memory] 入库排队 {waited:.2f}s", flush=True)
            await self.wait_idle("入库")
            started = time.monotonic()
            await asyncio.to_thread(
                self.remember_turn, pending, reply, owner, history_turn_id, memory_vm)
            if self.BARGE_DEBUG:
                print(f"[memory] 入库主流程 {time.monotonic()-started:.2f}s", flush=True)

    def queue_remember_turn(self, pending, reply: str, owner: dict,
                            history_turn_id: str = "", memory_vm=None) -> None:
        memory_vm = memory_vm or self.vm
        task = asyncio.create_task(
            self._remember_background(pending, reply, owner, history_turn_id, memory_vm))
        self._REMEMBER_TASKS.add(task)

        def done(t: asyncio.Task) -> None:
            self._REMEMBER_TASKS.discard(t)
            try:
                t.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[web] 后台记忆任务失败：{type(e).__name__}: {e}", flush=True)

        task.add_done_callback(done)
