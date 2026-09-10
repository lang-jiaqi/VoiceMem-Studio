"""Studio capture implementation."""
from studio.core.voicemem import open_stream
import asyncio
import base64
import json
import time
from studio.core.utils.echo_guard.component import UtteranceGuard
from studio.core.utils.dialogue.component import PauseGate, backchannel_policy, is_unfinished
from studio.core.utils.reply_modes.initialize import DIRECT, MEMORY
from studio.core.utils.turn_taking.initialize import Backchannel, TurnTakingStateMachine
from voicemem import gate
from studio.core.utils.contracts.component import Pending

class Capture:
    async def anticipate(self, sock, on_frame=None, on_speech=None, owner=None, is_busy=None,
                         said=None, on_candidate=None, on_candidate_reject=None,
                         on_playback_checkpoint=None, on_filler_done=None, on_early=None,
                         on_early_cancel=None,
                         on_speech_start=None, textless_confirm_s=None,
                         turn_taking=None):
        """Yield confirmed turns while forwarding audio, playback, and cancellation events."""
        pause_gate = PauseGate()
        stream = open_stream(self.vm, spec_min_chars=self.SPEC_MIN_CHARS, gamble_s=self.GAMBLE_S,
                           confirm_s=self.CONFIRM_S, eot=self._eot(), textless_confirm_s=textless_confirm_s,
                           turn_end_guard=pause_gate.allow_end)
        utterance = UtteranceGuard()
        turn_finished = False
        last_partial = ""
        if owner is None:
            owner = {"id": "", "last": "", "miss": 0}
        from studio.core.utils.turn_taking.initialize import backchannel as _bc_mod
        turn_taking = turn_taking or TurnTakingStateMachine(
            backchannel=Backchannel(policy=backchannel_policy()),
            echo_window_s=self.BC_ECHO_WINDOW_S)
        bc = turn_taking.backchannel
        bc_speech_t0 = 0.0
        prewarm_idle = True
        prewarm_mem = None
        last_speak_t = 0.0
        bc_rms_fast = 0.0
        bc_rms_slow = 0.0
        bc_gap = 0.0
        emotion_task = None

        #

        #

        early_text = ""
        speak_run = 0.0
        eot_peak = 0.0
        early_at = 0.0
        early_paused = False                  # A real pause has followed the EOT bet.
        if _bc_mod.emitting():

            self._backchannel_voice()
        else:
            print("[backchannel] 未开启（--backchannel 打开）", flush=True)

        def _reset_early():
            nonlocal early_at, early_paused, early_text, eot_peak, bc_speech_t0
            nonlocal prewarm_idle, prewarm_mem
            early_at, early_paused, early_text = 0.0, False, ""
            eot_peak = 0.0
            bc_speech_t0 = 0.0
            prewarm_idle = True
            prewarm_mem = None
            bc.reset_turn()
            pause_gate.reset()

        def live_agent_text():
            """Return the active output text used for echo rejection."""

            return (said() if said else "") + turn_taking.recent_agent_text()

        def heard_from_agent():
            return utterance.reference or live_agent_text()
        barge_base = 0
        barged = False
        candidate = False
        candidate_updates = 0
        candidate_text = ""
        candidate_silence = 0.0
        candidate_age = 0.0
        discard_candidate_turn = False
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            if msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "playback_checkpoint":
                    if on_playback_checkpoint:
                        await on_playback_checkpoint(data)
                    continue
                if data.get("type") == "filler_done":
                    if on_filler_done:
                        on_filler_done(str(data.get("filler_id") or ""))
                    continue
                if data.get("type") == "user_text" and data.get("text", "").strip():
                    turn_taking.begin_user_turn()
                    stream.emotion = owner.get('emotion')
                    turn = await stream.feed_text(data["text"])
                    turn_taking.commit_user_turn()
                    yield Pending(turn.text, turn.memory_context, turn.result, spoken=False,
                                  replay=self._replay_id(turn.text, turn.result),
                                  emotion=owner.get("emotion", ""),
                                  route=turn.route,
                                  reply_mode=MEMORY if gate.needs_memory(turn.route) else DIRECT)
                continue
            if msg.get("bytes") is None:
                continue
            raw = msg["bytes"]
            if turn_finished:
                utterance = UtteranceGuard()
                pause_gate.reset()
                bc.reset_turn()
                bc_gap = bc_rms_fast = bc_rms_slow = 0.0
                turn_finished = False
                barge_base = 0
                barged = candidate = discard_candidate_turn = False
                candidate_updates = 0
                candidate_text = ""
                candidate_silence = candidate_age = 0.0
            # Snapshot before feed: final ASR awaits may outlive assistant playback.
            busy_at_capture = bool(is_busy and is_busy())
            reference_at_capture = live_agent_text()
            if on_frame:
                await on_frame(raw)
            stream.emotion = owner.get('emotion')
            st = await stream.feed(raw)
            if (emotion_task is None and st.spoke and st.state == "<silence>"
                    and st.silence >= 0.08):
                snapshot = st
                emotion_task = asyncio.create_task(
                    asyncio.to_thread(lambda: snapshot.emotion))
            turn_finished = bool(st.turn)
            utterance.observe(active=st.spoke or bool(st.turn) or st.state == "<speak>",
                              busy=busy_at_capture, reference=reference_at_capture)
            utterance.observe(active=False, busy=False, reference=live_agent_text())
            if getattr(st, "speech_end", 0):
                last_speak_t = st.speech_end
            cur = st.text.strip()
            busy = bool(is_busy and is_busy())
            input_echo = bool(heard_from_agent() and self._is_echo(cur, heard_from_agent()))
            if st.state == "<speak>" and not getattr(st, "speech_end", 0):
                last_speak_t = time.monotonic()

            # Early generation is speculative until the user turn is confirmed.
            # ASR may revise text during silence, but actual speech after a pause
            # means the user continued the same sentence and invalidates the bet.
            resumed_after_early = False
            if early_at:
                if st.state == "<silence>" and st.silence > 0:
                    early_paused = True
                elif early_paused and st.state == "<speak>":
                    resumed_after_early = True
                    early_at, early_paused, early_text = 0.0, False, ""
                    eot_peak = 0.0
                    prewarm_idle = True
                    if on_early_cancel:
                        await on_early_cancel("用户停顿后继续说")

            #

            if (prewarm_idle and on_speech_start and not busy
                    and st.state != "<speak>"):
                if on_speech_start():
                    prewarm_idle = False

            elif (on_speech_start and st.state == "<speak>"
                  and st.memory is not None and st.memory is not prewarm_mem):
                if on_speech_start(st.memory, st.text):
                    prewarm_mem = st.memory

            # EOT freezes an audio snapshot for the background reply path.  The UI
            # keeps receiving streaming ASR, while final ASR reviews only the frozen
            # audio before the LLM sees this turn.
            if st.eot_score > eot_peak:
                eot_peak = st.eot_score
            # EOT starts buffered work, but confirmation remains the commit point.
            # Resumed speech cancels this bet before any transcript or audio is sent.
            if (on_early and not busy and not input_echo and st.spoke and cur
                    and not is_unfinished(cur) and not pause_gate.hold_until
                    and not (utterance.started_busy and self._is_backchannel(cur))
                    and st.eot_score >= self.EARLY_EOT and not early_at
                    and not resumed_after_early):
                early_text, early_at = cur, time.monotonic()
                early_paused = st.state == "<silence>"
                refined_text = stream.refine_current_snapshot()
                await on_early(cur, st, refined_text)
                # Generation is buffered; normal turn confirmation still owns when
                # assistant audio may be released.

            #

            #

            import numpy as _np
            from voicemem.utils.audio.stream_io import resample as _resample
            frame = _np.frombuffer(raw, _np.int16).astype(_np.float32) / 32768.0
            rms = float(_np.sqrt(_np.mean(frame * frame))) if len(frame) else 0.0
            frame_s = len(frame) / self.MIC_RATE
            if st.state == "<speak>":
                bc_rms_fast = rms
                bc_rms_slow = (0.9 * bc_rms_slow + 0.1 * rms) if bc_rms_slow else rms
                if not bc_speech_t0:
                    bc_speech_t0 = time.monotonic()
                    turn_taking.begin_user_turn()
            quiet = rms < max(0.008, self.BC_QUIET_RATIO * bc_rms_slow)
            bc_gap = bc_gap + frame_s if quiet else 0.0

            unfinished = is_unfinished(cur)
            bc_ok = (unfinished or not early_at
                     or (time.monotonic() - early_at >= self.BC_AFTER_EARLY_S
                         and cur != early_text and st.state == "<speak>"))
            if not st.turn and not busy and not input_echo and not utterance.started_busy and bc_speech_t0 and bc_ok:
                voice = self._backchannel_voice() if _bc_mod.emitting() else None
                token = turn_taking.offer_backchannel(
                    text=cur, silence=bc_gap, spoke=st.spoke,
                    speech_s=time.monotonic() - bc_speech_t0,
                    emotion=owner.get("emotion", ""),
                    tail_rms=bc_rms_fast, prev_rms=bc_rms_slow,
                    lang=self.space_language(self.ACTIVE_SPACE),
                    available=voice.available if voice else set(),
                    unfinished=unfinished)
                if token:
                    voice = self._backchannel_voice()
                    pcm = voice.get(token, bc.rng) if voice else None
                    if pcm:

                        await sock.send_json({
                            "type": "backchannel", "token": token,
                            "sample_rate": 24000,
                            "pcm": base64.b64encode(pcm).decode()})
                        pause_gate.emitted(len(pcm) / (48000))
                        turn_taking.record_emission(token)
                        if self.BARGE_DEBUG:
                            print(f"[backchannel] {token!r}", flush=True)
                    elif self.BARGE_DEBUG:
                        v = self._BC_VOICE["obj"]
                        why = ("音色对不上，这条路不附和" if v is False
                               else "预合成还没好" if v is not None else "拿不到 TTS")
                        print(f"[backchannel] 判到该说 {token!r}，但{why} → 跳过", flush=True)
            frame_s = len(raw) / 2 / self.MIC_RATE

            #

            speak_run = speak_run + frame_s if st.state == "<speak>" else 0.0

            if (busy and st.state == "<speak>" and speak_run >= self.CANDIDATE_MIN_SPEECH_S
                    and not candidate and not barged
                    and cur and not self._is_echo(cur, heard_from_agent())
                    and self._has_barge_content(cur)):
                candidate = True
                discard_candidate_turn = False
                candidate_updates = 0
                candidate_text = ""
                candidate_silence = 0.0
                candidate_age = 0.0
                if self.BARGE_DEBUG:
                    print("[barge] 疑似插话 → 暂停播放，等待 ASR 确认", flush=True)
                if on_candidate:
                    await on_candidate()

            if candidate and not barged:
                candidate_age += frame_s
                candidate_silence = (candidate_silence + frame_s
                                     if st.state == "<silence>" else 0.0)
                looks_echo = bool(said is not None and cur and self._is_echo(cur, heard_from_agent()))
                if looks_echo:
                    candidate_updates = 0
                    candidate_text = ""
                if cur and not looks_echo and not self._is_backchannel(cur) and self._has_barge_content(cur):
                    normalized = self._barge_text(cur)
                    if normalized != candidate_text:
                        candidate_text = normalized
                        candidate_updates += 1

                confirmed = (self._is_explicit_interrupt(cur) and not looks_echo)
                confirmed = confirmed or (not looks_echo and candidate_updates >= self.BARGE_STABLE_UPDATES)
                if confirmed:
                    candidate = False
                    barged = True
                    barge_base = len(cur)
                    if self.BARGE_DEBUG:
                        why = "明确停止指令" if self._is_explicit_interrupt(cur) else "转写连续稳定增长"
                        print(f"[barge] {why} → 确认打断：{cur[-16:]!r}", flush=True)
                    if on_speech:
                        await on_speech()
                elif (looks_echo or (candidate_silence * 1000 >= self.BARGE_REJECT_SILENCE_MS and not cur)
                      or (candidate_age * 1000 >= self.BARGE_CANDIDATE_TIMEOUT_MS
                          and candidate_updates == 0)):
                    candidate = False
                    discard_candidate_turn = True
                    if self.BARGE_DEBUG:
                        print("[barge] 回声/无有效插话 → 恢复播放", flush=True)
                    if on_candidate_reject:
                        await on_candidate_reject()
            # Keep filtering after playback ends AND after a confirmed interrupt:
            # both can happen before delayed ASR publishes the assistant's audio.
            echo = input_echo
            show_partial = utterance.allow_partial(
                cur, echo=echo, final=bool(st.turn), backchannel=self._is_backchannel(cur),
                explicit=self._is_explicit_interrupt(cur), confirmed=barged)
            if echo and last_partial:
                last_partial = ""
                await sock.send_json({"type": "partial_transcript", "text": "", "replace": True})
            if st.text.strip() and st.text != last_partial and show_partial:
                last_partial = st.text
                await sock.send_json({"type": "partial_transcript", "text": st.text, "replace": True,
                                      "non_interrupting": utterance.started_busy and self._is_backchannel(cur)})
            if st.turn:
                # A late ASR result may arrive after playback drains or a candidate
                # is rejected. Still discard the actual emitted phrase.
                if (heard_from_agent() and self._is_echo(st.turn.text, heard_from_agent())):
                    if self.BARGE_DEBUG:
                        print(f"[echo] 丢弃助手回声：{st.turn.text!r}", flush=True)
                    if candidate and on_candidate_reject:
                        await on_candidate_reject()
                    candidate = False
                    candidate_updates = 0
                    candidate_text = last_partial = ""
                    _reset_early()
                    continue
                if utterance.started_busy and self._is_backchannel(st.turn.text):
                    # Display acknowledgement, but do not create a reply, ingest it,
                    # change speaker identity, or wait for hearing() to still be true.
                    if candidate and on_candidate_reject:
                        await on_candidate_reject()
                    await sock.send_json({"type": "user_backchannel", "text": st.turn.text})
                    if self.BARGE_DEBUG:
                        print(f"[barge] 开口时助手在说话，附和只显示不回复：{st.turn.text!r}", flush=True)
                    candidate = barged = discard_candidate_turn = False
                    candidate_updates = 0
                    candidate_text = last_partial = ""
                    _reset_early()
                    continue

                #

                if discard_candidate_turn:
                    discard_candidate_turn = False
                    last_partial = ""
                    barge_base = 0
                    _t = (st.turn.text or "").strip()
                    if self._has_barge_content(_t) and not self._is_echo(_t, heard_from_agent()):
                        if self.BARGE_DEBUG:
                            print(f"[barge] 候选虽被否决，但复核出完整内容 → 照常成一轮："
                                  f"{_t!r}", flush=True)
                    else:
                        if self.BARGE_DEBUG:
                            print(f"[barge] 丢弃未确认的声音回合：{st.turn.text!r}", flush=True)
                        _reset_early()
                        continue

                if candidate and not barged:
                    final_text = st.turn.text.strip()
                    looks_echo = bool(said is not None and final_text
                                      and self._is_echo(final_text, heard_from_agent()))
                    confirmed = (
                        not looks_echo
                        and not self._is_backchannel(final_text)
                        and (self._is_explicit_interrupt(final_text)
                             or candidate_updates >= self.BARGE_STABLE_UPDATES
                             or (self._has_barge_content(final_text)
                                 and self._has_strong_final_barge(final_text)))
                    )
                    candidate = False
                    if confirmed:
                        barged = True
                        if self.BARGE_DEBUG:
                            print(f"[barge] 完整回合确认插话：{final_text!r}", flush=True)
                        if on_speech:
                            await on_speech()
                    else:
                        if self.BARGE_DEBUG:
                            print(f"[barge] 完整回合判为附和/回声/噪声 → 恢复：{final_text!r}",
                                  flush=True)
                        if on_candidate_reject:
                            await on_candidate_reject()
                        last_partial = ""
                        barge_base = 0
                        candidate_updates = 0
                        candidate_text = ""
                        candidate_silence = 0.0
                        candidate_age = 0.0
                        continue

                if self._replaying_now() and not (st.turn.text or "").strip():
                    if self.BARGE_DEBUG:
                        print("[replay] 回放期间的空白一轮，是自己的回声，丢掉", flush=True)
                    last_partial = ""
                    barge_base = 0
                    barged = False
                    candidate = False
                    continue
                last_partial = ""
                barge_base = 0
                barged = False
                candidate = False
                candidate_updates = 0
                candidate_text = ""
                candidate_silence = 0.0
                candidate_age = 0.0

                #

                if self.BARGE_DEBUG:
                    if self._eot() is None:
                        print("[early] EOT 没启用 → 提前生成整个不工作。"
                              "多半是缺 onnxruntime：pip install onnxruntime", flush=True)
                    else:
                        print(f"[early] 本轮 EOT 最高 {eot_peak:.2f}（下注线 {self.EARLY_EOT}）"
                              f"{'，下过注' if early_at else '，没下注'}", flush=True)
                eot_peak = 0.0
                recent_emission = turn_taking.recent_agent_text()
                _bc_echo = bool(recent_emission) and self._is_echo(
                    st.turn.text, recent_emission)
                if _bc_echo:
                    if self.BARGE_DEBUG:
                        print(f"[backchannel] 这一轮是自己那声附和的回声 → 丢弃："
                              f"{st.turn.text!r}", flush=True)
                    _reset_early()
                    continue
                # Only a bet that survived the continuation window can be released.
                # ASR-only revisions during silence keep the same frozen prompt;
                # actual resumed speech already cancelled it above.
                _early_ok = bool(early_at)
                early_at, early_paused, early_text = 0.0, False, ""

                bc_speech_t0 = 0.0
                prewarm_idle = True
                prewarm_mem = None

                stranger = bool(self.SPEAKER_GATE and owner["id"]
                                and owner.get("miss", 0) >= self.STRANGER_MIN_TURNS)
                if stranger or self.SPEAKER_DEBUG:

                    print(f"[speaker] owner={owner['id'] or '-'} last={owner['last'] or '-'}"
                          f" miss={owner.get('miss', 0)} stranger={stranger}", flush=True)
                turn_taking.commit_user_turn()
                current_emotion = owner.get("emotion", "")
                if emotion_task is not None:
                    try:
                        current_emotion = (await emotion_task) or current_emotion
                    except Exception as exc:
                        if self.BARGE_DEBUG:
                            print(f"[emotion] 提前分析失败：{type(exc).__name__}: {exc}", flush=True)
                emotion_task = None
                yield Pending(st.turn.text,
                              "" if stranger else st.turn.memory_context,
                              st.turn.result, spoken=True,
                              audio_path=await asyncio.to_thread(
                                  self.save_turn_audio, getattr(st, "_pcm", None)),
                              stranger=stranger,
                              replay="" if stranger else self._replay_id(st.turn.text, st.turn.result),
                              emotion=current_emotion,
                              route=st.turn.route,
                              reply_mode=(MEMORY if gate.needs_memory(st.turn.route)
                                          else DIRECT),
                              early_ok=_early_ok,
                              speech_end=last_speak_t)

    async def _session_anticipate(self, session_id: str, sock, on_close=None, **kwargs):
        try:
            async for pending in self.anticipate(sock, **kwargs):
                yield pending
        finally:
            if on_close:
                await on_close()
            self._SESSION_CONTEXT.clear_session(session_id)
