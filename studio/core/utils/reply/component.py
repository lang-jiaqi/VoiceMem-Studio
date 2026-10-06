from studio.harness.reply_modes.policy import MEMORY_COT
"""Studio reply implementation."""
import asyncio
import json
import threading
import time
from contextlib import aclosing
from studio.web import transport as utils
from voicemem.audio_timing import TimedAudioChunk
from voicemem import gate
from studio.core.utils.tts import control as tts_control

class Reply:
    def _early_reply_compatible(self, early_text: str, final_text: str) -> bool:
        """Reuse an EOT reply only across minor ASR revisions, not new content."""
        normalize = lambda value: "".join(
            char for char in (value or "").casefold() if char.isalnum())
        early, final = normalize(early_text), normalize(final_text)
        if not early or not final:
            return False
        if early == final:
            return True

        # A suffix after a complete-looking EOT snapshot is usually continued
        # speech. Only harmless spoken fillers may be appended or removed.
        if early.startswith(final) or final.startswith(early):
            suffix = final[len(early):] if final.startswith(early) else early[len(final):]
            return bool(suffix) and all(char in "啊呀哦嗯呐哈" for char in suffix)

        longest = max(len(early), len(final))
        if abs(len(early) - len(final)) > max(2, round(longest * 0.2)):
            return False
        from difflib import SequenceMatcher
        return SequenceMatcher(None, early, final, autojunk=False).ratio() >= 0.82

    async def _send_reply_display(self, pending, send, ready, output_id, space, memory_vm):
        if getattr(pending, "opening", False):
            return
        await ready.wait()
        if self.ACTIVE_SPACE != space or self.vm is not memory_vm:
            return
        started = time.monotonic()
        cancelled = threading.Event()

        def build():
            with self._REPLY_DISPLAY_LOCK:
                if cancelled.is_set() or self.ACTIVE_SPACE != space or self.vm is not memory_vm:
                    return None
                return self.fill_tags(
                    utils.hits_payload(pending.result, has_audio=self.audio_of,
                                       cluster_of=self.hit_cluster),
                    pending.text, pending.audio_path or "", acoustic=False)

        try:
            payload = await asyncio.to_thread(build)
            if payload is None or self.ACTIVE_SPACE != space or self.vm is not memory_vm:
                return
            if gate.needs_memory(pending.route):
                self.note_hits(pending.result)
            await send({"type": "memory_hits", "route": pending.route,
                        "output_id": output_id, **payload})
            print(f"[lat-ui] 展示标签后台耗时 {(time.monotonic()-started)*1000:.0f}ms"
                  f"（已移出首音频关键路径，output={output_id}）", flush=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[web] 回复展示跳过：{type(e).__name__}: {e}", flush=True)
        finally:
            cancelled.set()

    async def voicemem_llm_tts(self, pending, send, send_audio, owner, timeline,
                               said=None, context_session="", context_space="",
                               memory_vm=None, self_harness_profile=None,
                               on_self_harness_update=None, speech_provider=None):
        """Stream with an optional conversation-owned TTS instead of the shared factory."""
        memory_vm = memory_vm or self.vm
        context_space = context_space or self.ACTIVE_SPACE
        ready = asyncio.Event()
        display = asyncio.create_task(self._send_reply_display(
            pending, send, ready, timeline.output_id, context_space, memory_vm))
        from voicemem.utils.common.prompt_trace import prompt_scope
        try:
            with prompt_scope(output_id=timeline.output_id, session=context_session,
                              space=context_space, early=bool(getattr(pending, "early_ok", False))):
                return await self._voicemem_llm_tts(
                    pending, send, send_audio, owner, timeline, said=said,
                    context_session=context_session, context_space=context_space,
                    memory_vm=memory_vm, first_audio_ready=ready,
                    self_harness_profile=self_harness_profile,
                    on_self_harness_update=on_self_harness_update,
                    speech_provider=speech_provider)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Surface provider failure without exposing exception details or credentials.
            from studio.core.utils.contracts.component import SpeechSynthesisError
            english = getattr(pending, "language", "") == "en"
            speech_failed = isinstance(exc, SpeechSynthesisError)
            message = (("Speech synthesis failed. Please try again." if english else
                        "语音合成失败，请重试。") if speech_failed else
                       ("The reply could not finish. Please try again." if english else
                        "回复未能完成，请重试。"))
            try:
                await send({"type": "error", "output_id": timeline.output_id,
                            "code": "speech_failed" if speech_failed else "reply_failed",
                            "message": message})
            except Exception:
                pass
            raise
        finally:
            display.cancel()
            await asyncio.gather(display, return_exceptions=True)

    async def _voicemem_llm_tts(self, pending, send, send_audio, owner, timeline,
                               said=None, context_session="", context_space="",
                               memory_vm=None, first_audio_ready=None,
                               self_harness_profile=None,
                               on_self_harness_update=None, speech_provider=None):
        memory_vm = memory_vm or self.vm
        context_space = context_space or self.ACTIVE_SPACE
        _entry = time.monotonic()
        if not getattr(pending, "transcript_managed", False):
            from studio.core.utils.contracts.component import input_transcript_event
            await send(input_transcript_event(pending))

        if pending.replay:
            self._note_replay(pending.replay)
            await send({"type": "play_memory", "memory_id": pending.replay})
        await send({"type": "answer_start", "output_id": timeline.output_id,
                    "sample_rate": timeline.sample_rate})

        _t0 = time.monotonic()
        _lat = {"llm": 0.0, "seg": 0.0, "audio": 0.0, "pre": 0.0}

        from studio.core.utils.speaking_style.component import (
            is_qwen36, qwen_segment_instruction,
        )
        from studio.core.utils.self_harness.component import (
            PrivateControlFilter, fixed_tone, merge_update, normalize_snapshot, profile_context,
            speech_rate_instruction, split_control_prefix,
        )
        control_enabled = self_harness_profile is not None
        active_self_harness = normalize_snapshot(self_harness_profile)

        queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        tts = speech_provider if speech_provider is not None else memory_vm.utils.get("tts")
        from voicemem.lang import detect_language
        language = getattr(pending, "language", "") or getattr(self, "SPACE_LANG", "zh")
        control = {"head": True, "buf": ""}
        private_control = PrivateControlFilter()
        tone = {"tag": "", "head": True, "buf": ""}
        from studio.core.utils.tts.markdown import MarkdownSpeech
        from studio.core.utils.contracts.component import SpeechSynthesisError
        speech_markdown = MarkdownSpeech(
            language, language_resolver=lambda: detect_language(reply, language, keep_short=False))
        logged_instruction = object()

        qwen_voice = is_qwen36(getattr(self, "REPLY", None))

        def _synth_one(seg, text_start):
            nonlocal logged_instruction
            import json
            segment_language = detect_language(seg, speech_markdown.language, keep_short=False)
            segment_base = self._speak_base_env or self._by_lang(self._SPEAK_BASE, segment_language)
            selected_tone = fixed_tone(active_self_harness) or tone["tag"]
            effective = (tts_control.instruction(selected_tone, segment_base, segment_language)
                         if selected_tone else self._speak_instruction(pending.emotion, segment_language))
            effective = effective or getattr(tts, "instruction", "") or getattr(tts, "instructions", "")
            # An explicit conversation preference wins over the model-specific
            # automatic intro arc; the general Qwen delivery hint still applies.
            if fixed_tone(active_self_harness):
                effective = qwen_segment_instruction(
                    effective, "", "", qwen=qwen_voice, language=segment_language)
            else:
                effective = qwen_segment_instruction(
                    effective, pending.text, timeline.generated_text[:text_start + len(seg)],
                    qwen=qwen_voice, language=segment_language)
            effective = speech_rate_instruction(
                effective, active_self_harness, language=segment_language)
            if effective != logged_instruction:
                logged_instruction = effective
                print(f"[tts-prompt] {timeline.output_id[:8]} "
                      f"{json.dumps(effective, ensure_ascii=False)}", flush=True)
            try:
                return tts.stream(seg, effective)
            except TypeError:
                return tts.stream(seg)

        synths: list[asyncio.Task] = []
        streams: asyncio.Queue = asyncio.Queue()

        _serial = asyncio.Semaphore(1) if getattr(tts, "SERIAL", False) else None

        async def synth():
            while (spec := await queue.get()) is not None:
                seg, text_start, text_end = spec
                chunks: asyncio.Queue = asyncio.Queue()
                state = {"complete": False, "queued": time.monotonic(), "first": False}

                async def run(seg=seg, text_start=text_start, chunks=chunks, state=state):

                    try:
                        if _serial is not None:
                            await _serial.acquire()
                        started = time.monotonic()
                        try:
                            async with aclosing(_synth_one(seg, text_start)) as speech:
                                async for chunk in speech:
                                    if not getattr(chunk, "pcm", chunk):
                                        continue
                                    if not state["first"]:
                                        state["first"] = True
                                        if self.BARGE_DEBUG:
                                            print(f"[tts-segment] chars={len(seg)} queue_ms={(started-state['queued'])*1000:.0f} first_pcm_ms={(time.monotonic()-started)*1000:.0f}", flush=True)
                                    await chunks.put(chunk)
                        finally:
                            if _serial is not None:
                                _serial.release()
                        if not state["first"]:
                            raise SpeechSynthesisError("TTS segment returned no audio")
                        state["complete"] = True
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        print(f"[web] 合成失败：{type(e).__name__}", flush=True)
                        failure = SpeechSynthesisError("TTS segment failed")
                        failure.__cause__ = e
                        await chunks.put(failure)
                    finally:
                        await chunks.put(None)

                synths.append(asyncio.create_task(run()))
                await streams.put((seg, text_start, text_end, chunks, state))
            await streams.put(None)

        async def _mark_first_audio():
            if not _lat["audio"]:
                _lat["audio"] = (time.monotonic() - _t0) * 1000

                _vad = ((time.monotonic() - pending.speech_end) * 1000
                        if pending.speech_end else 0.0)
                _head = _vad - _lat["audio"] if _vad else 0.0
                total_label = (f"{_vad:.0f}ms" if pending.speech_end else
                               f"N/A（{'文字轮' if not pending.spoken else '提前生成那份' if pending.early_ok else '无VAD起点'}）")
                print(f"[lat] 闭嘴→首帧 {total_label}；生成前总等待 {_head:.0f}"
                      f" + LLM首字 {_lat['llm']:.0f}"
                      f" + 攒第一段 {_lat['seg'] - _lat['llm']:.0f}"
                      f" + TTS首帧 {_lat['audio'] - _lat['seg']:.0f}"
                      f"（回合交出→出声 {_lat['audio']:.0f}）｜{self._lat_note(_lat['audio'])}",
                      flush=True)
                print(f"[mem] {self._mem_line()}", flush=True)
                print(f"[lat-pre] 回复入口前 {max(0, (_entry-pending.speech_end)*1000) if pending.speech_end else 0:.0f}ms"
                      f" · 控制消息 {(_t0-_entry)*1000:.0f}ms"
                      f" · prompt准备 {_lat['pre']:.0f}ms"
                      f" · 等首个文字 {max(0, _lat['llm']-_lat['pre']):.0f}ms"
                      f" · VAD起点={'有' if pending.speech_end else '无'}", flush=True)

        async def speak():
            while (item := await streams.get()) is not None:
                seg, text_start, text_end, chunks, state = item
                segment_id = timeline.begin_segment(text_start, text_end)

                if said is not None:
                    said["text"] = (said.get("text") or "") + seg
                try:
                    while (chunk := await chunks.get()) is not None:
                        if isinstance(chunk, SpeechSynthesisError):
                            raise chunk
                        if isinstance(chunk, TimedAudioChunk):
                            if chunk.sample_rate != timeline.sample_rate:
                                raise ValueError(
                                    f"TTS 输出采样率应为 {timeline.sample_rate}Hz，"
                                    f"实际为 {chunk.sample_rate}Hz")
                            pcm = chunk.pcm
                            timeline.add_segment_timestamps(
                                segment_id, chunk.timestamps)
                        else:
                            pcm = chunk
                        timeline.append_audio(pcm)
                        await _mark_first_audio()
                        await send_audio(pcm)
                        if first_audio_ready is not None:
                            first_audio_ready.set()
                except asyncio.CancelledError:
                    timeline.finish_segment(segment_id, complete=False)
                    raise
                except Exception as e:
                    timeline.finish_segment(segment_id, complete=False)
                    print(f"[web] 语音发送中断：{type(e).__name__}", flush=True)
                    raise
                else:
                    timeline.finish_segment(
                        segment_id, complete=state["complete"])

        async def segment_text():
            from studio.core.utils.tts.segmentation import SpeechBuffer
            buffer = SpeechBuffer()

            def audio_ahead():
                if not timeline.checkpoint_seen or timeline.playback_state != "playing":
                    return 0.0
                return max(0.0, (timeline.sent_samples - timeline.rendered_cutoff_samples())
                           / timeline.sample_rate)

            try:
                while True:
                    remaining = buffer.remaining_wait(time.monotonic(), audio_ahead())
                    final = False
                    try:
                        try:
                            delta = text_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            if remaining is None:
                                delta = await text_queue.get()
                            else:
                                # Recheck playback headroom while the LLM is stalled.
                                delta = await asyncio.wait_for(text_queue.get(), min(remaining, 0.1))
                    except asyncio.TimeoutError:
                        pass
                    else:
                        final = delta is None
                        if not final:
                            buffer.append(delta, time.monotonic())
                    for segment in buffer.ready(time.monotonic(), audio_ahead(), final=final):
                        if not _lat["seg"]:
                            _lat["seg"] = (time.monotonic() - _t0) * 1000
                            if self.BARGE_DEBUG:
                                print(f"[seg] 第一段 {len(segment.text)} 字 → {segment.text!r}", flush=True)
                        queue.put_nowait((segment.text, segment.start, segment.end))
                    if final:
                        break
            finally:
                queue.put_nowait(None)

        segmenter = asyncio.create_task(segment_text())
        synther = asyncio.create_task(synth())
        speaker = asyncio.create_task(speak())
        reply = ""
        interrupted = False
        generation = None

        async def _drop_pipeline():
            tasks = [segmenter, speaker, synther, *synths]
            if generation is not None:
                tasks.append(generation)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        _hot = self.hot_path_enter()
        try:

            ctx = self.build_reply_context(
                pending.memory_context, stranger=pending.stranger,
                route=pending.route, replay=pending.replay, text=pending.text,
                emotion=pending.emotion,
                continuation=getattr(pending, "continuation_prompt", False), language=language)
            if control_enabled:
                ctx = f"{ctx}\n\n{profile_context(active_self_harness)}"

            hist = ([] if pending.stranger else self._SESSION_CONTEXT.messages(
                context_session, context_space, window=self.HISTORY_TURNS))
            _lat["pre"] = (time.monotonic() - _t0) * 1000
            from voicemem.reply import apply_reply_request_options
            reply_mode = getattr(pending, "reply_mode", "")
            deltas = apply_reply_request_options(
                memory_vm.reply_stream(pending.text, ctx, hist),
                reasoning_effort=("high" if reply_mode == MEMORY_COT
                                  else "none"),
            )

            def apply_self_harness_update(update):
                if not control_enabled:
                    return
                effective = update
                if on_self_harness_update is not None:
                    try:
                        result = on_self_harness_update({
                            domain: dict(fields)
                            for domain, fields in update.items()
                        })
                        # Stateful callbacks return the subset accepted by the
                        # stability window. Stateless callbacks such as
                        # ``dict.update`` return None and preserve the old API.
                        if isinstance(result, dict):
                            effective = result
                    except Exception as exc:
                        effective = {}
                        print(f"[self-harness] 保存会话设置失败：{type(exc).__name__}: {exc}",
                              flush=True)
                merge_update(active_self_harness["profile"], effective)

            def emit_speech(delta, *, final=False):
                spoken = speech_markdown.feed(delta, final=final)
                timeline.append_speech_text(spoken.text, spoken.source_ends, spoken.consumed,
                                            previous_end=spoken.previous_end)
                if spoken.text:
                    text_queue.put_nowait(spoken.text)

            async def emit_visible(delta, *, final=False):
                nonlocal reply
                delta = private_control.feed(delta, final=final)
                if not delta:
                    return
                reply += delta
                timeline.append_source_text(delta)
                await send({"type": "answer_delta", "text": delta})
                emit_speech(delta)

            async def consume_tone(delta, *, final=False):
                if tone["head"]:
                    tone["buf"] += delta
                    tag, rest = tts_control.split(tone["buf"])
                    if tag:
                        selected = fixed_tone(active_self_harness)
                        opening = getattr(pending, "opening", False)
                        tag = selected or (tag if opening else tts_control.smooth(
                            owner.get("tone", ""), tag))
                        tone["tag"], tone["head"], tone["buf"] = (
                            tag, False, "")
                        if not opening:
                            timeline.reply_tone = tag
                        delta = rest
                        if self.BARGE_DEBUG:
                            print(f"[tone] 模型标的语气：{tag}", flush=True)
                    elif not final and len(tone["buf"]) < 26 and not any(
                            char in tone["buf"] for char in "]】"):
                        return
                    else:
                        tone["head"] = False
                        delta, tone["buf"] = tone["buf"], ""
                        if final and not delta.strip():
                            delta = ""
                await emit_visible(delta, final=final)

            async def consume_delta(delta):
                if control["head"]:
                    control["buf"] += delta
                    parsed = split_control_prefix(control["buf"])
                    if not parsed.resolved:
                        return
                    control["head"], control["buf"] = False, ""
                    if parsed.error:
                        print(f"[self-harness] 忽略模型控制头：{parsed.error}",
                              flush=True)
                    apply_self_harness_update(parsed.update)
                    delta = parsed.rest
                await consume_tone(delta)

            async def generate_text():
                try:
                    async with aclosing(deltas):
                        async for d in deltas:
                            if not _lat["llm"]:
                                _lat["llm"] = (time.monotonic() - _t0) * 1000
                            await consume_delta(d)
                    # Only normal EOF may resolve buffered prefixes. Cancellation and
                    # provider failures must never leak a private control header.
                    if control["head"]:
                        parsed = split_control_prefix(control["buf"], final=True)
                        control["head"], control["buf"] = False, ""
                        if parsed.error:
                            print(f"[self-harness] 忽略模型控制头：{parsed.error}",
                                  flush=True)
                        apply_self_harness_update(parsed.update)
                        await consume_tone(parsed.rest, final=True)
                    else:
                        await consume_tone("", final=True)
                    if timeline.source_text is not None:
                        emit_speech("", final=True)
                finally:
                    text_queue.put_nowait(None)

            generation = asyncio.create_task(generate_text())
            # Any failed stage stops the reply, even while the LLM is awaiting tokens.
            await asyncio.gather(generation, segmenter, synther, speaker)
        except asyncio.CancelledError:
            interrupted = True
        except Exception:
            await _drop_pipeline()
            timeline.mark_interrupted()
            self.hot_path_exit(_hot)
            raise

        if interrupted:
            await _drop_pipeline()
        else:

            self.hot_path_exit(_hot)
            acoustic = None
            if not getattr(timeline, "context_managed", False):
                acoustic = self._kick_acoustic(
                    send, pending.audio_path or "", output_id=timeline.output_id,
                    is_current=lambda: (not timeline.interrupted
                                        and self.ACTIVE_SPACE == context_space and self.vm is memory_vm))
            timeline.mark_generation_complete()
            try:
                await send({"type": "answer_done", "output_id": timeline.output_id})
                if getattr(timeline, "context_managed", False):
                    return
                timeout = max(2.0, min(
                    60.0, timeline.sent_samples / timeline.sample_rate + 2.0))
                await asyncio.wait_for(timeline.wait_playback_done(), timeout=timeout)
            except asyncio.TimeoutError:
                timeline.assume_drained()
            except asyncio.CancelledError:
                interrupted = True
                await _drop_pipeline()
            finally:
                if acoustic is not None:
                    acoustic.cancel()
                    await asyncio.gather(acoustic, return_exceptions=True)

        if getattr(timeline, "context_managed", False):
            if interrupted:
                timeline.mark_interrupted()
            return
        context_reply = timeline.heard_text() if interrupted else reply
        if context_reply and timeline.reply_tone:
            owner["tone"] = timeline.reply_tone
        if interrupted and self.BARGE_DEBUG:
            print(f"[context] 打断于 {timeline.rendered_ms()}ms，保留回复 "
                  f"{context_reply!r}", flush=True)
        history_turn_id = self._push_history(
            context_session, context_space, pending.text, context_reply,
            interrupted=interrupted)
        self.hot_path_exit(_hot)
        self.queue_remember_turn(
            pending, context_reply, owner, history_turn_id, memory_vm=memory_vm)
        timeline.context_saved = True
