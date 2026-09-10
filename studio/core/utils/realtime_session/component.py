"""Studio realtime session implementation."""
import asyncio
import base64
import time
import uuid
from studio.web import transport as utils
from studio.core.utils.audio_timeline.component import AudioTimeline, SpeechRateEstimator
from voicemem import gate

class RealtimeSession:
    def _turn_detection(self) -> dict:

        base = {"type": self.TURN_DETECTION, "create_response": False, "interrupt_response": False}
        if self.TURN_DETECTION == "semantic_vad":
            return {**base, "eagerness": self.VAD_EAGERNESS}

        return {**base, "threshold": self.BARGE_THRESHOLD,
                "prefix_padding_ms": 200, "silence_duration_ms": 320}

    async def start_realtime_turn(self, pending, conn, send, timeline,
                                  context_session="", context_space=""):
        """Send the confirmed turn context and start realtime generation."""
        await send({"type": "user_transcript", "text": pending.text})
        if gate.needs_memory(pending.route):
            self.note_hits(pending.result)

        await send({"type": "memory_hits", "route": pending.route,
                    **self.fill_tags(utils.hits_payload(pending.result, has_audio=self.audio_of,
                                                  cluster_of=self.hit_cluster),
                                pending.text, pending.audio_path or "", acoustic=False)})
        self._kick_acoustic(send, pending.audio_path or "")
        if pending.replay:

            self._note_replay(pending.replay)
            await send({"type": "play_memory", "memory_id": pending.replay})
        if pending.spoken:
            await conn.input_audio_buffer.commit()
        else:
            await conn.conversation.item.create(item={"type": "message", "role": "user",
                                                      "content": [{"type": "input_text", "text": pending.text}]})

        print(f"[lat] 本地判完说完 → 发 response.create", flush=True)
        await conn.response.create(response={
            "instructions": self._realtime_instructions(
                pending.memory_context,
                pending.stranger,
                replay=bool(pending.replay),
                emotion=pending.emotion,
                text=pending.text,
                context_session=context_session,
                context_space=context_space),
        })
        await send({"type": "answer_start", "output_id": timeline.output_id,
                    "sample_rate": timeline.sample_rate})

    async def truncate_provider_output(self, conn, provider_item_id: str,
                                       timeline: AudioTimeline) -> None:
        truncate = getattr(conn, "truncate_output", None)
        if callable(truncate):
            await truncate(
                provider_output_id=provider_item_id,
                media_output_id=timeline.output_id,
                audio_end_samples=timeline.rendered_cutoff_samples(),
                sample_rate=timeline.sample_rate,
            )
            return
        conversation = getattr(conn, "conversation", None)
        item_api = getattr(conversation, "item", None)
        truncate = getattr(item_api, "truncate", None)
        if callable(truncate):
            await truncate(
                item_id=provider_item_id, content_index=0,
                audio_end_ms=timeline.rendered_ms())

    async def _no_realtime(self, sock, err):
        name, text = type(err).__name__, str(err)
        network = (isinstance(err, (OSError, TimeoutError, ConnectionError))
                   or "gaierror" in name.lower()
                   or any(k in text.lower() for k in ("nodename", "temporary failure",
                                                      "name or service", "getaddrinfo",
                                                      "connection refused", "timed out")))
        if network:
            why = ("网络连不上 api.openai.com（DNS/代理/VPN 的问题，跟 key 无关）。"
                   "确认能上网后重开；离线环境用 `--mode llm_tts` 也一样连不上，"
                   "两条路都要访问 OpenAI。")
        elif any(k in text for k in ("401", "403", "invalid_api_key", "insufficient", "model_not_found")):
            why = ("这个 key 没有 Realtime 权限或模型不可用——改用 "
                   "`python web/run.py --mode llm_tts`，那条路只要普通 chat + TTS。")
        else:
            why = ("先看这条报错本身；如果只是 Realtime 用不了，可以改用 "
                   "`python web/run.py --mode llm_tts`（普通 chat + TTS）。")
        msg = f"连不上 OpenAI Realtime（{name}: {text}）。{why}"
        print(f"[web] {msg}", flush=True)
        try:
            await sock.send_json({"type": "error", "message": msg})
        except Exception:
            pass

    async def realtime_session(self, sock):
        """Run Realtime speech while local ASR/VAD owns turn confirmation."""
        connected = False
        context_session = uuid.uuid4().hex
        try:
            async with utils.realtime_connect(self.REPLY) as conn:

                await conn.session.update(session={
                    "type": "realtime",
                    "audio": {
                        "input": {"turn_detection": self._turn_detection()},
                        "output": {"voice": utils.RT_VOICE},
                    },
                })
                connected = True

                turn = {"live": False, "reply": "", "pending": None,
                        "t0": 0.0, "until": 0.0, "first": False,
                        "timeline": None, "response_done": False,
                        "provider_item_id": "", "space": "", "memory_vm": None}
                owner = {"id": "", "last": "", "miss": 0}
                speech_rate = SpeechRateEstimator()
                timelines: dict[str, AudioTimeline] = {}
                playback_tasks: set[asyncio.Task] = set()
                candidate_paused = False
                candidate_paused_at = 0.0

                response_idle = asyncio.Event()
                response_idle.set()

                def hearing() -> bool:
                    """Report active generation or remaining playback, including continuation waiting."""
                    timeline = turn["timeline"]
                    buffered = bool(timeline and not timeline.playback_done
                                    and time.monotonic() < turn["until"])
                    return (candidate_paused or turn["live"] or buffered
                            or time.monotonic() < turn["until"])

                async def pause_candidate():
                    nonlocal candidate_paused, candidate_paused_at
                    if hearing() and not candidate_paused:
                        candidate_paused = True
                        candidate_paused_at = time.monotonic()
                        await sock.send_json({"type": "answer_pause"})

                async def resume_candidate():
                    nonlocal candidate_paused, candidate_paused_at
                    if candidate_paused:
                        candidate_paused = False
                        if turn["until"]:
                            turn["until"] += max(0.0, time.monotonic() - candidate_paused_at)
                        candidate_paused_at = 0.0
                        await sock.send_json({"type": "answer_resume"})

                def close_turn(interrupted=False):
                    p, reply = turn["pending"], turn["reply"]
                    timeline = turn["timeline"]
                    space = turn["space"] or self.ACTIVE_SPACE
                    memory_vm = turn["memory_vm"] or self.vm
                    turn.update(live=False, reply="", pending=None, timeline=None,
                                response_done=False, provider_item_id="",
                                space="", memory_vm=None)
                    if p is None:
                        return
                    if interrupted:
                        reply = timeline.heard_text() if timeline else ""
                        if self.BARGE_DEBUG and timeline:
                            print(f"[context] 打断于 {timeline.rendered_ms()}ms，保留回复 "
                                  f"{reply!r}", flush=True)
                    history_turn_id = self._push_history(
                        context_session, space, p.text, reply,
                        interrupted=interrupted)
                    self.queue_remember_turn(
                        p, reply, owner, history_turn_id, memory_vm=memory_vm)
                    if timeline:
                        timelines.pop(timeline.output_id, None)

                async def playback_checkpoint(data):
                    timeline = timelines.get(str(data.get("output_id") or ""))
                    if timeline is None:
                        return
                    timeline.update_checkpoint(
                        data.get("rendered_samples", 0),
                        data.get("sample_rate", self.MIC_RATE),
                        data.get("state", "playing"))
                    if turn["timeline"] is timeline and timeline.playback_done:
                        turn["until"] = 0.0
                        if turn["response_done"]:
                            close_turn(interrupted=False)

                async def playback_fallback(timeline):
                    delay = max(0.0, turn["until"] - time.monotonic()) + 0.5
                    await asyncio.sleep(delay)
                    if turn["timeline"] is timeline and turn["response_done"]:
                        timeline.assume_drained()
                        turn["until"] = 0.0
                        close_turn(interrupted=False)

                def schedule_playback_fallback(timeline):
                    task = asyncio.create_task(playback_fallback(timeline))
                    playback_tasks.add(task)
                    task.add_done_callback(playback_tasks.discard)

                async def pump():
                    async for ev in conn:
                        t = getattr(ev, "type", "")
                        if t.endswith("output_audio.delta"):
                            if turn["live"]:
                                if not turn["first"]:

                                    turn["first"] = True
                                    print(f"[lat] realtime 首帧 "
                                          f"{(time.monotonic()-turn['t0'])*1000:.0f}ms", flush=True)
                                pcm = base64.b64decode(ev.delta)
                                timeline = turn["timeline"]
                                if timeline:
                                    timeline.append_audio(pcm)
                                    timestamps = getattr(ev, "timestamps", ()) or ()
                                    if timestamps:
                                        timeline.add_timestamps(tuple(timestamps))
                                turn["provider_item_id"] = (
                                    getattr(ev, "item_id", "") or turn["provider_item_id"])

                                turn["until"] = (max(turn["until"], time.monotonic())
                                                 + len(pcm) / 2 / 24000)
                                await sock.send_bytes(pcm)
                        elif t.endswith("output_audio_transcript.delta"):
                            if turn["live"]:
                                turn["reply"] += ev.delta
                                timeline = turn["timeline"]
                                if timeline:
                                    timeline.append_text(ev.delta)
                                    timestamps = getattr(ev, "timestamps", ()) or ()
                                    if timestamps:
                                        timeline.add_timestamps(tuple(timestamps))
                                turn["provider_item_id"] = (
                                    getattr(ev, "item_id", "") or turn["provider_item_id"])
                                await sock.send_json({"type": "answer_delta", "text": ev.delta})
                        elif t == "error" or t.endswith(".error"):
                            err = getattr(ev, "error", None)
                            code = getattr(err, "code", "")

                            if code not in (
                                    "input_audio_buffer_commit_empty",
                                    "response_cancel_not_active"):
                                print(f"[web] realtime 事件错误：{err or ev}", flush=True)
                        elif t.endswith("input_audio_buffer.speech_stopped"):

                            turn["stopped"] = time.monotonic()
                            if self.BARGE_DEBUG:
                                print("[lat] OpenAI 判说完", flush=True)
                        elif t.endswith("input_audio_buffer.speech_started"):

                            since = (time.monotonic() - turn["t0"]) * 1000
                            if self.BARGE_DEBUG:
                                print(f"[barge] OpenAI VAD 听到人声 (live={turn['live']}, "
                                      f"还在播={hearing()}, {since:.0f}ms)", flush=True)

                        elif t.endswith("response.done") or t.endswith("response.cancelled"):

                            response_idle.set()
                            if self.BARGE_DEBUG and not turn["live"]:
                                print("[barge] 旧 Realtime response 已退出", flush=True)
                            if turn["live"]:
                                timeline = turn["timeline"]
                                if timeline:
                                    timeline.mark_generation_complete()
                                if t.endswith("response.cancelled"):
                                    response_idle.clear()
                                    heard = timeline.heard_text() if timeline else ""
                                    provider_item_id = turn["provider_item_id"]
                                    if timeline:
                                        timeline.mark_interrupted()
                                    await sock.send_json({
                                        "type": "answer_interrupt",
                                        "output_id": timeline.output_id if timeline else "",
                                        "heard_text": heard,
                                    })
                                    close_turn(interrupted=True)
                                    if provider_item_id and timeline:
                                        try:
                                            await self.truncate_provider_output(
                                                conn, provider_item_id, timeline)
                                        except Exception as e:
                                            if self.BARGE_DEBUG:
                                                print(f"[barge] Provider 上下文截断失败：{e}",
                                                      flush=True)
                                    response_idle.set()
                                else:
                                    turn["live"] = False
                                    turn["response_done"] = True
                                    await sock.send_json({
                                        "type": "answer_done",
                                        "output_id": timeline.output_id if timeline else "",
                                    })
                                    if timeline and timeline.playback_done:
                                        close_turn(interrupted=False)
                                    elif timeline:
                                        schedule_playback_fallback(timeline)

                async def on_frame(raw):
                    await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

                async def on_speech():
                    nonlocal candidate_paused, candidate_paused_at
                    if not hearing():
                        if self.BARGE_DEBUG:
                            print("[barge] 有人声但助手没在说，忽略", flush=True)
                        return

                    since = (time.monotonic() - turn["t0"]) * 1000
                    if since < self.BARGE_GRACE_MS:
                        if self.BARGE_DEBUG:
                            print(f"[barge] 才说了 {since:.0f}ms，还在宽限期内，不打断", flush=True)
                        return
                    if self.BARGE_DEBUG:
                        left = max(0.0, turn["until"] - time.monotonic()) * 1000
                        print(f"[barge] ★ 打断：转写触发（前端还剩 {left:.0f}ms 没播完）",
                              flush=True)
                    active_response = not response_idle.is_set()
                    timeline = turn["timeline"]
                    heard_text = timeline.heard_text() if timeline else ""
                    output_id = timeline.output_id if timeline else ""
                    provider_item_id = turn["provider_item_id"]
                    if timeline:
                        timeline.mark_interrupted()
                    turn["live"], turn["until"] = False, 0.0
                    candidate_paused = False
                    candidate_paused_at = 0.0

                    await sock.send_json({"type": "answer_interrupt",
                                          "output_id": output_id,
                                          "heard_text": heard_text})
                    close_turn(interrupted=True)
                    if active_response:
                        await conn.response.cancel()
                    if provider_item_id and timeline:
                        try:
                            await self.truncate_provider_output(
                                conn, provider_item_id, timeline)
                        except Exception as e:
                            if self.BARGE_DEBUG:
                                print(f"[barge] Provider 上下文截断失败：{e}", flush=True)
                    if active_response:
                        try:
                            await asyncio.wait_for(response_idle.wait(), timeout=2.0)
                        except asyncio.TimeoutError:

                            print("[barge] 等待 Realtime 取消确认超时，下一轮暂缓创建", flush=True)

                pump_task = asyncio.create_task(pump())
                try:
                    async for pending in self.anticipate(sock, on_frame=on_frame,
                                                    on_speech=on_speech, owner=owner,

                                                    is_busy=hearing,
                                                    said=lambda: turn["reply"],
                                                    on_candidate=pause_candidate,
                                                    on_candidate_reject=resume_candidate,
                                                    on_playback_checkpoint=playback_checkpoint):

                        if self._is_backchannel(pending.text) and hearing():
                            if self.BARGE_DEBUG:
                                print(f"[barge] 整轮都是附和 {pending.text!r}，不算一轮，继续说",
                                      flush=True)
                            continue
                        if hearing():
                            await on_speech()
                        if not response_idle.is_set():
                            if self.BARGE_DEBUG:
                                print("[barge] 等旧 response 退出后再创建下一轮", flush=True)
                            await response_idle.wait()
                        context_space = self.ACTIVE_SPACE
                        memory_vm = self.vm
                        timeline = AudioTimeline(
                            prebuffer_seconds=0.08, rate_estimator=speech_rate)
                        timelines[timeline.output_id] = timeline
                        turn.update(
                            live=True, reply="", pending=pending,
                            t0=time.monotonic(), until=0.0, first=False,
                            timeline=timeline, response_done=False,
                            provider_item_id="", space=context_space,
                            memory_vm=memory_vm)
                        response_idle.clear()
                        try:
                            await self.start_realtime_turn(
                                pending, conn, sock.send_json, timeline,
                                context_session=context_session,
                                context_space=context_space)
                        except Exception:
                            response_idle.set()
                            raise
                finally:
                    pump_task.cancel()
                    try:
                        await pump_task
                    except asyncio.CancelledError:
                        pass
                    if turn["pending"] is not None:
                        timeline = turn["timeline"]
                        fully_played = bool(
                            turn["response_done"] and timeline and timeline.playback_done)
                        close_turn(interrupted=not fully_played)
                    for task in playback_tasks:
                        task.cancel()
                    if playback_tasks:
                        await asyncio.gather(
                            *list(playback_tasks), return_exceptions=True)
        except Exception as e:
            if connected:
                raise
            await self._no_realtime(sock, e)
        finally:
            self._SESSION_CONTEXT.clear_session(context_session)
