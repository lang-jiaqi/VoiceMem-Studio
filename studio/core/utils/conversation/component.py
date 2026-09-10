from studio.core.utils.turn_taking.pause import needs_continuation
import asyncio
import base64
import json
import threading
import time
import uuid
from dataclasses import replace
from studio.core.utils.dialogue.component import CONTROLS, backchannel_policy, is_unfinished
from studio.core.utils.reply_modes.initialize import DIRECT, MEMORY
from studio.core.utils.turn_taking.initialize import Backchannel, HandoffKind, TurnTakingStateMachine, generate_filler, wait_for_filler_and_output
from studio.core.utils.audio_timeline.component import AudioTimeline, SpeechRateEstimator
from studio.core.utils.tts.audio_timing import TimedAudioChunk
from voicemem import gate
from voicemem.memory_api import build_memory_context
from studio.core.utils.contracts.component import Pending, ReplySink
from .initialize import initialize

class Conversation:

    def __init__(self, agent, sock):
        initialize(self, agent, sock)

    def filler_done(self, filler_id: str) -> None:
        event = self.filler_waiters.get(filler_id)
        if event is not None:
            event.set()

    async def pause_candidate(self):
        if self.hearing() and (not self.candidate_paused):
            self.candidate_paused = True
            self.candidate_paused_at = time.monotonic()
            await self.sock.send_json({'type': 'answer_pause'})

    async def resume_candidate(self):
        if self.candidate_paused:
            self.candidate_paused = False
            if self.turn['until']:
                self.turn['until'] += max(0.0, time.monotonic() - self.candidate_paused_at)
            self.candidate_paused_at = 0.0
            await self.sock.send_json({'type': 'answer_resume'})

    def hearing(self) -> bool:
        """Report active generation or remaining playback, including continuation waiting."""
        t = self.turn['task']
        continuation_task = self.turn.get('continuation_task')
        timeline = self.turn['timeline']
        task_active = t is not None and (not t.done()) and (not (timeline and timeline.playback_done))
        continuation_active = continuation_task is not None and (not continuation_task.done())
        return self.candidate_paused or task_active or continuation_active or (time.monotonic() < self.turn['until'])

    async def playback_checkpoint(self, data):
        timeline = self.turn['timeline']
        if timeline is None or data.get('output_id') != timeline.output_id:
            return
        timeline.update_checkpoint(data.get('rendered_samples', 0), data.get('sample_rate', self.agent.MIC_RATE), data.get('state', 'playing'))
        if data.get('event') == 'started' and (not self.turn['play_started']):
            self.turn['play_started'] = True
            if self.turn['speech_end']:
                elapsed = (time.monotonic() - self.turn['speech_end']) * 1000
                print(f'[lat] 闭嘴→浏览器开始播放（含回报传输） {elapsed:.0f}ms', flush=True)
        if data.get('state') == 'stalled' and self.agent.BARGE_DEBUG:
            from voicemem.utils.gpu_loop import gpu_loop as _gl
            print(f"[play] 缓冲见底：已播 {timeline.rendered_ms():.0f}ms，GPU 线程活跃任务 {getattr(_gl(), 'active_count', '?')}", flush=True)
        if timeline.playback_done:
            self.turn['until'] = 0.0

    async def send_audio(self, pcm: bytes):
        """Send PCM and update the output duration and first-audio observation."""
        self.turn['until'] = max(self.turn['until'], time.monotonic()) + len(pcm) / 2 / self.agent.MIC_RATE
        self.turn['echo_until'] = self.turn['until'] + 2.0
        if not self.turn['t0']:
            self.turn['t0'] = time.monotonic()
        if self.turn['measure_started'] and (not self.turn['measure_recorded']):
            self.turn['measure_recorded'] = True
            self.turn_taking.observe_first_audio(time.monotonic() - self.turn['measure_started'])
        await self.sock.send_bytes(pcm)

    async def stop_reply(self, force: bool=False):
        continuation_task = self.turn.get('continuation_task')
        if continuation_task is not None and (not continuation_task.done()) and (self.turn['task'] is None):
            self.turn['continuation_task'] = None
            continuation_task.cancel()
            await asyncio.gather(continuation_task, return_exceptions=True)
            self.candidate_paused = False
            self.candidate_paused_at = 0.0
            if self.agent.BARGE_DEBUG:
                print('[unfinished] 用户继续说，取消4秒后的追问', flush=True)
            return
        if not self.hearing():
            self.turn['task'] = None
            return
        since = (time.monotonic() - self.turn['t0']) * 1000 if self.turn['t0'] else 0.0
        if not force and self.turn['t0'] and (since < self.agent.BARGE_GRACE_MS):
            if self.agent.BARGE_DEBUG:
                print(f'[barge] 才说了 {since:.0f}ms，还在宽限期内，不打断', flush=True)
            return
        if self.agent.BARGE_DEBUG:
            left = max(0.0, self.turn['until'] - time.monotonic()) * 1000
            print(f'[barge] ★ 打断：转写触发（前端还剩 {left:.0f}ms 没播完）', flush=True)
        task = self.turn['task']
        timeline = self.turn['timeline']
        heard_text = timeline.heard_text() if timeline else ''
        output_id = timeline.output_id if timeline else ''
        if timeline:
            timeline.mark_interrupted()
        if task is not None and (not task.done()):
            task.cancel()
        (self.turn['task'], self.turn['until']) = (None, 0.0)
        self.candidate_paused = False
        self.candidate_paused_at = 0.0
        try:
            await self.sock.send_json({'type': 'answer_interrupt', 'output_id': output_id, 'heard_text': heard_text})
        except Exception:
            pass
        if task is not None and (not task.done()):
            try:
                await task
            except asyncio.CancelledError:
                pass

    def reset_output_state(self, pending, timeline, reply_state) -> None:
        self.turn.update(t0=0.0, until=0.0, speech_end=pending.speech_end, play_started=False, reply=reply_state, timeline=timeline, measure_started=0.0, measure_recorded=False)

    def cached_ack(self, pending):
        bc = self.turn_taking.backchannel
        unfinished = needs_continuation(pending.text)
        if not pending.spoken or not unfinished:
            return None
        voice = self.agent._backchannel_voice()
        if not voice or not voice.ready:
            return None
        from studio.core.utils.turn_taking.backchannel import emitting, pick_token
        if not emitting():
            return None
        token = pick_token(pending.text, pending.emotion, self.agent.space_language(self.agent.ACTIVE_SPACE), bc.rng, bc._recent, voice.available)
        pcm = voice.get(token, self.turn_taking.backchannel.rng) if token else None
        return (token, pcm) if pcm else None

    async def emit_filler(self, token: str, pcm: bytes, *, filler_id: str='') -> float:
        """Send a non-interruptible filler and return its PCM duration."""
        duration = len(pcm) / (self.agent.MIC_RATE * 2)
        message = {'type': 'backchannel', 'token': token, 'sample_rate': self.agent.MIC_RATE, 'pcm': base64.b64encode(pcm).decode(), 'interruptible': False}
        if filler_id:
            message['filler_id'] = filler_id
        await self.sock.send_json(message)
        self.turn_taking.record_emission(token, committed=True)
        self.turn['until'] = max(self.turn['until'], time.monotonic() + duration)
        self.turn['echo_until'] = max(self.turn['echo_until'], self.turn['until'] + 2.0)
        return duration

    async def wait_for_filler_end(self, filler_id: str, duration: float) -> None:
        """Wait for browser playback completion, with a bounded legacy fallback."""
        event = self.filler_waiters[filler_id]
        try:
            await asyncio.wait_for(event.wait(), timeout=max(1.0, duration + 1.0))
        except asyncio.TimeoutError:
            print(f'[handoff] 垫话播放完成回报超时（{duration:.2f}s），继续正文', flush=True)
        else:
            print('[handoff] 浏览器确认垫话播放完成，放行正文', flush=True)
        finally:
            self.filler_waiters.pop(filler_id, None)

    async def synthesize_work_filler(self, pending, memory_vm, context_space):
        """Generate and synthesize a bridge without delaying the main work."""
        history = self.agent._SESSION_CONTEXT.messages(self.context_session, context_space, window=self.agent.HISTORY_TURNS)
        text = await generate_filler(memory_vm.reply_stream, pending.text, history=history, lang=self.agent.space_language(context_space))
        if not text:
            return ('', b'')
        tts = memory_vm.utils.get('tts')
        instruction = self.agent._speak_instruction(pending.emotion)
        print(f'[tts-prompt] filler {json.dumps(instruction, ensure_ascii=False)}', flush=True)
        try:
            stream = tts.stream(text, instruction)
        except TypeError:
            stream = tts.stream(text)
        chunks = bytearray()
        async for chunk in stream:
            chunks.extend(chunk.pcm if isinstance(chunk, TimedAudioChunk) else chunk)
        return (text, bytes(chunks))

    async def release_buffered_reply(self, sink, decision, ack, pending, memory_vm, context_space) -> None:
        """Bridge into an already-running buffered reply, then release it."""
        filler_task = None
        ready_task = None
        try:
            if decision.kind is HandoffKind.CACHED_ACK and ack:
                # The acknowledgement is fire-and-forget from the reply
                # scheduler's perspective. Never wait for it or delay the
                # already-running TTS generation.
                await self.emit_filler(*ack)
            elif decision.kind is HandoffKind.LLM_FILLER:
                filler_task = asyncio.create_task(self.synthesize_work_filler(pending, memory_vm, context_space))
                ready_task = asyncio.create_task(sink.wait_for_output())
                (done, _) = await asyncio.wait((filler_task, ready_task), return_when=asyncio.FIRST_COMPLETED)
                if ready_task in done or sink.buffered_ms > 0:
                    filler_task.cancel()
                else:
                    try:
                        (text, pcm) = await filler_task
                    except Exception as e:
                        print(f'[web] 垫话生成失败，直接回复：{type(e).__name__}: {e}', flush=True)
                        (text, pcm) = ('', b'')
                    if text and pcm and (sink.buffered_ms <= 0):
                        filler_id = uuid.uuid4().hex
                        self.filler_waiters[filler_id] = asyncio.Event()
                        try:
                            duration = await self.emit_filler(text, pcm, filler_id=filler_id)
                        except BaseException:
                            self.filler_waiters.pop(filler_id, None)
                            raise
                        await wait_for_filler_and_output(self.wait_for_filler_end(filler_id, duration), ready_task)
            self.turn['until'] = 0.0
            self.turn_taking.start_reply()
            if sink.first_audio_at and self.turn['measure_started']:
                self.turn['measure_recorded'] = True
                self.turn_taking.observe_first_audio(sink.first_audio_at - self.turn['measure_started'])
            await sink.commit(final_user_text=pending.text)
        finally:
            for task in (filler_task, ready_task):
                if task is not None and (not task.done()):
                    task.cancel()
            await asyncio.gather(*(task for task in (filler_task, ready_task) if task is not None), return_exceptions=True)

    async def drop_early(self, why: str='') -> None:
        t = self.early['task']
        self.early.update(text='', task=None, sink=None, timeline=None, pending=None, said=None, space='', memory_vm=None, started=0.0)
        if t is not None and (not t.done()):
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        if why and self.agent.BARGE_DEBUG:
            print(f'[early] 丢弃提前生成：{why}', flush=True)

    async def start_early(self, text, st, refined_text=None):
        """Run final ASR on the EOT snapshot, then buffer the early reply."""
        self.stop_prewarm()
        await self.drop_early('换了新的赌注')
        if refined_text is None:
            refined_text = asyncio.create_task(asyncio.sleep(0, result=text))
        from voicemem.stream import empty_result
        result = st.memory if st.memory is not None else empty_result()
        emotion = self.owner.get('emotion', '')
        route = st.route
        context_space = self.agent.ACTIVE_SPACE
        memory_vm = self.agent.vm
        routing_history = self.agent._SESSION_CONTEXT.messages(self.context_session, context_space, window=self.agent.HISTORY_TURNS)
        sink = ReplySink(self.sock.send_json, self.send_audio)
        timeline = AudioTimeline(prebuffer_seconds=0.16, rate_estimator=self.speech_rate)
        said_state = {'text': ''}
        self.early.update(text=text, sink=sink, timeline=timeline, pending=None, said=said_state, space=context_space, memory_vm=memory_vm, started=0.0)

        async def run():
            try:
                committed_text = (await refined_text).strip() or text
                pending = Pending(committed_text, build_memory_context(result), result, spoken=True, emotion=emotion, route=route, reply_mode=MEMORY if gate.needs_memory(route) else DIRECT)
                await self.agent.route_pending_thinking(pending, memory_vm, history=routing_history)
                self.early['text'] = committed_text
                self.early['pending'] = pending
                if self.agent.BARGE_DEBUG:
                    print(f'[early] EOT {st.eot_score:.2f} · offline ASR committed: {committed_text[-20:]!r}', flush=True)
                self.early['started'] = time.monotonic()
                await self.agent.voicemem_llm_tts(pending, sink.send, sink.send_audio, self.owner, timeline, said=said_state, context_session=self.context_session, context_space=context_space, memory_vm=memory_vm)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f'[early] 提前生成失败：{type(e).__name__}: {e}', flush=True)
        self.early['task'] = asyncio.create_task(run())

    def stop_prewarm(self):
        cancelled = self.prewarm['cancelled']
        if cancelled is not None:
            cancelled.set()
        task = self.prewarm['task']
        if task is not None and (not task.done()):
            task.cancel()

    def prewarm_local(self, result=None, text=''):
        """Schedule cancellable prefix warming only while reply work is idle."""
        if self.agent._LOCAL_LLM is None or self.prewarm['closed']:
            return False
        if self.prewarm['task'] and (not self.prewarm['task'].done()):
            return False
        if self.early['task'] is not None or self.hearing():
            return False
        hist = self.agent._SESSION_CONTEXT.messages(self.context_session, self.agent.ACTIVE_SPACE, window=self.agent.HISTORY_TURNS)
        ctx = '' if result is None else self.agent.build_reply_context(build_memory_context(result), route=gate.DEEP, replay=self.agent._replay_id(text, result), text=text, emotion=self.owner.get('emotion', ''))
        cancelled = threading.Event()
        self.prewarm['cancelled'] = cancelled
        model = self.agent._LOCAL_LLM

        async def run_prewarm():
            try:
                await asyncio.to_thread(model.prewarm, hist, ctx, cancelled=cancelled)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            except Exception as e:
                print(f'[llm] 后台预热失败：{type(e).__name__}: {e}', flush=True)
        self.prewarm['task'] = asyncio.create_task(run_prewarm())
        return True

    async def close_session(self):
        self.prewarm['closed'] = True
        self.stop_prewarm()
        await self.drop_early()
        continuation_task = self.turn.get('continuation_task')
        self.turn['continuation_task'] = None
        if continuation_task is not None and (not continuation_task.done()):
            continuation_task.cancel()
            await asyncio.gather(continuation_task, return_exceptions=True)
        task = self.turn['task']
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f'[web] 回复收尾失败：{type(e).__name__}: {e}', flush=True)

    def reply_done(self, done_task):
        try:
            done_task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f'[web] 回复任务失败：{type(e).__name__}: {e}', flush=True)

    async def run_unfinished_followup(self, base_pending, memory_vm, context_space):
        """Wait for a stalled clause, then invite the user to finish it."""
        current = asyncio.current_task()
        started_reply = False
        try:
            elapsed = max(0.0, time.monotonic() - base_pending.speech_end) if base_pending.speech_end else 0.0
            await asyncio.sleep(max(0.0, float(CONTROLS['unfinished_followup_s']) - elapsed))
            if self.unfinished_wait['task'] is not current:
                return
            self.unfinished_wait['pending'] = None
            if self.agent.vm is not memory_vm or self.agent.ACTIVE_SPACE != context_space:
                return
            pending = replace(base_pending, continuation_prompt=True, early_ok=False)
            reply_state = {'text': ''}
            timeline = AudioTimeline(prebuffer_seconds=0.16, rate_estimator=self.speech_rate)
            self.reset_output_state(pending, timeline, reply_state)
            self.turn['task'] = current
            self.turn['continuation_task'] = None
            self.turn['measure_started'] = time.monotonic()
            self.turn_taking.start_reply()
            started_reply = True
            await self.agent.voicemem_llm_tts(pending, self.sock.send_json, self.send_audio, self.owner, timeline, said=reply_state, context_session=self.context_session, context_space=context_space, memory_vm=memory_vm)
            reply = timeline.heard_text()
            history_turn_id = self.agent._push_history(self.context_session, context_space, pending.text, reply)
            self.agent.queue_remember_turn(pending, reply, self.owner, history_turn_id, memory_vm=memory_vm)
        except asyncio.CancelledError:
            raise
        finally:
            if self.turn['task'] is current:
                self.turn['task'] = None
            if self.unfinished_wait['task'] is current:
                self.unfinished_wait['task'] = None
            if started_reply:
                self.turn_taking.finish_reply()

    async def defer_unfinished_reply(self, pending):
        """Emit a short acknowledgement and arm the delayed continuation prompt."""
        ack = self.cached_ack(pending)
        if ack:
            await self.emit_filler(*ack)
        task = asyncio.create_task(self.run_unfinished_followup(pending, self.agent.vm, self.agent.ACTIVE_SPACE))
        self.unfinished_wait['pending'] = pending
        self.unfinished_wait['space'] = self.agent.ACTIVE_SPACE
        self.unfinished_wait['memory_vm'] = self.agent.vm
        self.unfinished_wait['task'] = task
        self.turn['continuation_task'] = task
        task.add_done_callback(self.reply_done)
        if self.agent.BARGE_DEBUG:
            print(f"[unfinished] {pending.text!r} 未说完，{CONTROLS['unfinished_followup_s']:.1f}s后追问", flush=True)

    def ignore(self, pending):
        return self.agent._is_backchannel(pending.text) and self.hearing()

    async def merge_continuation(self, pending):
        if self.unfinished_wait['pending'] is not None:
            base = self.unfinished_wait['pending']
            wait_task = self.unfinished_wait['task']
            self.unfinished_wait['pending'] = None
            self.unfinished_wait['task'] = None
            self.turn['continuation_task'] = None
            if wait_task is not None and (not wait_task.done()):
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
            if self.unfinished_wait.get('space') != self.agent.ACTIVE_SPACE or self.unfinished_wait.get('memory_vm') is not self.agent.vm:
                return pending
            joiner = '' if self.agent.space_language(self.agent.ACTIVE_SPACE) == 'zh' else ' '
            pending = replace(pending, text=f'{base.text}{joiner}{pending.text}'.strip(), continuation_prompt=False, early_ok=False)
            if self.agent.BARGE_DEBUG:
                print(f'[unfinished] 用户续说，合并为 {pending.text!r}', flush=True)
        return pending

    async def route(self, pending):
        if self.early['task'] is not None and (self.early.get('space') != self.agent.ACTIVE_SPACE or self.early.get('memory_vm') is not self.agent.vm):
            await self.drop_early('记忆空间已改变')
        if self.early['task'] is not None and pending.early_ok and (not self.agent._early_reply_compatible(self.early['text'], pending.text)):
            await self.drop_early('EOT 快照后还有实质续话，改用完整文本回复')
        routing_history = self.agent._SESSION_CONTEXT.messages(self.context_session, self.agent.ACTIVE_SPACE, window=self.agent.HISTORY_TURNS)
        thinking_task = asyncio.create_task(self.agent.route_pending_thinking(pending, self.agent.vm, history=routing_history))
        self.stop_prewarm()
        try:
            if self.early['task'] is not None and pending.early_ok:
                await thinking_task
                early_pending = self.early.get('pending')
                if early_pending is None or early_pending.reply_mode != pending.reply_mode:
                    await self.drop_early('最终 ASR 的回复路由与提前生成不一致')
            return thinking_task
        except BaseException:
            thinking_task.cancel()
            await asyncio.gather(thinking_task, return_exceptions=True)
            raise

    async def commit_early(self, pending, thinking_task):
        if self.early['task'] is None or not pending.early_ok:
            return False
        await self.stop_reply(force=True)
        await thinking_task
        reply_state = self.early['said'] or {'text': ''}
        timeline = self.early['timeline']
        early_task = self.early['task']
        context_space = self.early['space'] or self.agent.ACTIVE_SPACE
        memory_vm = self.early['memory_vm'] or self.agent.vm
        generation_started = self.early['started'] or time.monotonic()
        (sink, ms) = (self.early['sink'], self.early['sink'].buffered_ms)
        self.reset_output_state(pending, timeline, reply_state)
        self.early.update(text='', task=None, sink=None, timeline=None, pending=None, said=None, space='', memory_vm=None, started=0.0)
        ack = self.cached_ack(pending)
        decision = self.turn_taking.decide_handoff(main_audio_ready=ms > 0, reply_mode=pending.reply_mode, cached_ack_available=bool(ack), spoken=pending.spoken)

        async def accept_early():
            try:
                self.turn_taking.start_handoff(decision)
                self.turn['measure_started'] = generation_started
                await self.release_buffered_reply(sink, decision, ack, pending, memory_vm, context_space)
                await early_task
            except asyncio.CancelledError:
                if not early_task.done():
                    early_task.cancel()
                await asyncio.gather(early_task, return_exceptions=True)
                raise
            finally:
                self.turn_taking.finish_reply()
        task = asyncio.create_task(accept_early())
        self.turn['task'] = task
        task.add_done_callback(self.reply_done)
        if self.agent.BARGE_DEBUG:
            state = '释放现成音频' if ms > 0 else '尚无音频，继续等待生成'
            print(f'[early] ★ 文本命中，已缓冲 {ms:.0f}ms 音频，{state}；handoff={decision.kind.value}', flush=True)
        return True

    async def start_reply(self, pending, thinking_task):
        await self.drop_early('这一轮没赌成' if self.early['task'] else '')
        await thinking_task
        reply_state = {'text': ''}
        context_space = self.agent.ACTIVE_SPACE
        memory_vm = self.agent.vm
        timeline = AudioTimeline(prebuffer_seconds=0.16, rate_estimator=self.speech_rate)
        self.reset_output_state(pending, timeline, reply_state)

        def save_interrupted_context() -> None:
            if timeline.context_saved:
                return
            reply = timeline.heard_text()
            history_turn_id = self.agent._push_history(self.context_session, context_space, pending.text, reply, interrupted=True)
            self.agent.queue_remember_turn(pending, reply, self.owner, history_turn_id, memory_vm=memory_vm)
            timeline.context_saved = True

        async def run_reply(send_json=self.sock.send_json, send_pcm=self.send_audio, pending=pending, timeline=timeline, context_space=context_space, memory_vm=memory_vm, reply_state=reply_state):
            try:
                await self.agent.voicemem_llm_tts(pending, send_json, send_pcm, self.owner, timeline, said=reply_state, context_session=self.context_session, context_space=context_space, memory_vm=memory_vm)
            finally:
                save_interrupted_context()
        ack = self.cached_ack(pending)
        decision = self.turn_taking.decide_handoff(main_audio_ready=False, reply_mode=pending.reply_mode, cached_ack_available=bool(ack), spoken=pending.spoken)

        async def coordinate_reply():
            child_tasks = []
            try:
                self.turn_taking.start_handoff(decision)
                if decision.kind is not HandoffKind.DIRECT:
                    sink = ReplySink(self.sock.send_json, self.send_audio)
                    self.turn['measure_started'] = time.monotonic()
                    main = asyncio.create_task(run_reply(sink.send, sink.send_audio))
                    child_tasks.append(main)
                    await self.release_buffered_reply(sink, decision, ack, pending, memory_vm, context_space)
                    await main
                    return
                self.turn_taking.start_reply()
                self.turn['measure_started'] = time.monotonic()
                await run_reply()
            finally:
                for child in child_tasks:
                    if not child.done():
                        child.cancel()
                if child_tasks:
                    await asyncio.gather(*child_tasks, return_exceptions=True)
                save_interrupted_context()
                self.turn_taking.finish_reply()
        task = asyncio.create_task(coordinate_reply())
        self.turn['task'] = task
        task.add_done_callback(self.reply_done)

    def listen(self):
        return self.agent._session_anticipate(self.context_session, self.sock, on_speech=self.stop_reply, owner=self.owner, is_busy=self.hearing, said=lambda : self.turn['reply']['text'] if self.hearing() or time.monotonic() < self.turn['echo_until'] else '', on_candidate=self.pause_candidate, on_candidate_reject=self.resume_candidate, on_playback_checkpoint=self.playback_checkpoint, on_filler_done=self.filler_done, on_close=self.close_session, on_early=self.start_early, on_early_cancel=self.drop_early, on_speech_start=self.prewarm_local, textless_confirm_s=0.2, turn_taking=self.turn_taking)
