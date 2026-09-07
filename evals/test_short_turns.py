"""CPU-only regressions for short speech, late acknowledgements and echo UI.

Execute the actual anticipate coroutine without importing the model-loading demo.
"""
import ast
import asyncio
import base64
import dataclasses
import json
from pathlib import Path
import re
import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "web")]
from echo_guard import UtteranceGuard, is_echo
from voicemem import gate
from voicemem.stream import VoiceStream, StreamState


class ShortSpeechTests(unittest.IsolatedAsyncioTestCase):
    def make_stream(self, result="ok.", enabled=True):
        calls = []
        def transcribe(pcm):
            calls.append(pcm.copy())
            return result
        stream = VoiceStream(types.SimpleNamespace(), src_rate=16000,
                             gate=lambda _: "backchannel", confirm_s=.2,
                             textless_confirm_s=.2 if enabled else None)
        stream._final_asr = types.SimpleNamespace(transcribe=transcribe)
        stream._asr_w = types.SimpleNamespace(
            push=lambda _: "", reset=lambda: None, report=lambda: "test")
        stream._vad = types.SimpleNamespace(is_speech=lambda _: self.speaking)
        self.speaking = True
        return stream, calls

    async def feed_for(self, stream, seconds, *, speaking, energy=0):
        self.speaking = speaking
        # 20ms frames: measured in audio duration, not sleep/wall-clock timing.
        raw = np.full(320, energy, np.int16).tobytes()
        states = []
        for _ in range(round(seconds / .02)):
            states.append(await stream.feed(raw))
        return states

    async def test_empty_streaming_ok_finishes_after_200ms_even_with_residual_energy(self):
        stream, calls = self.make_stream()
        await self.feed_for(stream, .16, speaking=True, energy=1000)
        states = await self.feed_for(stream, .22, speaking=False, energy=1000)
        turns = [s.turn for s in states if s.turn]
        self.assertEqual([t.text for t in turns], ["ok."])
        self.assertEqual(len(calls), 1)  # no duplicate final decode / stream flush
        self.assertLessEqual(len(calls[0]) / 16000, .38)
        self.assertEqual(stream._voiced_s, 0)

    async def test_empty_probe_is_not_a_reply_and_does_not_repeat_in_silence(self):
        stream, calls = self.make_stream(result="")
        await self.feed_for(stream, .16, speaking=True)
        states = await self.feed_for(stream, .8, speaking=False)
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(s.turn is None for s in states))
        self.assertGreater(stream._pcm_len, 0)  # preserve sound-only input
        await self.feed_for(stream, .16, speaking=True)
        await self.feed_for(stream, .22, speaking=False)
        self.assertEqual(len(calls), 2)  # new speech can try again

    async def test_noise_without_voice_and_library_default_do_not_probe(self):
        for enabled, voiced in [(True, 0), (True, .04), (False, .16)]:
            stream, calls = self.make_stream(enabled=enabled)
            await self.feed_for(stream, voiced, speaking=True)
            states = await self.feed_for(stream, .8, speaking=False, energy=1000)
            self.assertEqual(calls, [])
            self.assertTrue(all(s.turn is None for s in states))

    async def test_continuing_speech_and_subthreshold_pause_do_not_cut(self):
        stream, calls = self.make_stream(result="ok但是我还想问一个问题")
        await self.feed_for(stream, .4, speaking=True)
        await self.feed_for(stream, .1, speaking=False)
        await self.feed_for(stream, .4, speaking=True)
        self.assertEqual(calls, [])
        states = await self.feed_for(stream, .22, speaking=False)
        self.assertEqual([s.turn.text for s in states if s.turn], ["ok但是我还想问一个问题"])


def anticipate_namespace():
    tree = ast.parse((ROOT / "web/run.py").read_text())
    names = {"anticipate", "Pending", "_is_echo", "_is_backchannel", "_barge_text",
             "_is_explicit_interrupt", "_has_barge_content", "_has_strong_final_barge", "_lcs_len"}
    ns = dict(asyncio=asyncio, base64=base64, json=json, time=time,
              dataclass=dataclasses.dataclass, gate=gate, UtteranceGuard=UtteranceGuard,
              BACKCHANNEL_ON=True, ECHO_WINDOW=300, ECHO_RATIO=.6, ECHO_FUZZY_MIN=4,
              _bc_norm=gate.norm, _FILLER_PREFIX=re.compile(r"^[嗯呃啊哦噢喔欸诶唉哈哼]+"),
              _INTERRUPT_PREFIXES=("停", "等一下", "stop", "wait"), BARGE_MIN_CHARS=2,
              SPEC_MIN_CHARS=6, GAMBLE_S=.2, CONFIRM_S=.2, MIC_RATE=24000,
              BC_ECHO_WINDOW_S=3, BC_QUIET_RATIO=.25, BC_AFTER_EARLY_S=1,
              EARLY_EOT=.5, CONFIRM_READY_S=.1, EARLY_MIN_COVER=.7,
              CANDIDATE_MIN_SPEECH_S=.04, BARGE_STABLE_UPDATES=2,
              BARGE_REJECT_SILENCE_MS=200, BARGE_CANDIDATE_TIMEOUT_MS=1200,
              BARGE_DEBUG=False, SPEAKER_GATE=False, SPEAKER_DEBUG=False,
              STRANGER_MIN_TURNS=1, ACTIVE_SPACE="test", _eot=lambda: None,
              space_language=lambda _: "zh", _replaying_now=lambda: False,
              _replay_id=lambda *a: "", save_turn_audio=lambda *a: "")
    exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, "name", "") in names],
                            type_ignores=[]), str(ROOT / "web/run.py"), "exec"), ns)
    return ns


def state(text="", *, final=False):
    turn = types.SimpleNamespace(text=text, raw_text=text, result=object(),
                                 memory_context="", route="shallow") if final else None
    return StreamState("turn_over" if final else "<speak>", text, None, turn,
                       spoke=not final, speech_end=time.monotonic())


class AnticipateTests(unittest.IsolatedAsyncioTestCase):
    async def run_frames(self, frames, on_early=None):
        ns, sent, yielded, interrupted = anticipate_namespace(), [], [], []
        playback = {"busy": False, "text": ""}
        frames = iter(frames)
        current = None
        class Sock:
            async def receive(self):
                nonlocal current
                current = next(frames, None)
                if current is None:
                    return {"type": "websocket.disconnect"}
                playback.update(busy=current[0], text=current[1])
                return {"bytes": np.zeros(960, np.int16).tobytes()}
            async def send_json(self, message):
                sent.append(message)
        async def feed(_):
            if len(current) > 3:  # playback can drain inside final ASR's await
                playback.update(busy=current[3], text="")
            return current[2]
        async def stop():
            interrupted.append(True)
        stream = types.SimpleNamespace(feed=feed, confirm_s=.2)
        ns["vm"] = types.SimpleNamespace(stream=lambda **kw: stream)
        with patch("harness.backchannel.emitting", return_value=False), \
             patch("harness.backchannel.Backchannel.offer", return_value=None):
            async for pending in ns["anticipate"](
                    Sock(), is_busy=lambda: playback["busy"],
                    said=lambda: playback["text"], on_speech=stop, on_early=on_early):
                yielded.append(pending)
        return sent, yielded, interrupted

    async def test_late_ok_is_display_only_even_when_playback_drains_during_final(self):
        sent, turns, interrupted = await self.run_frames([
            (True, "我是语音助手", state()),
            (True, "我是语音助手", state("ok.", final=True), False),
        ])
        self.assertEqual(turns, [])
        self.assertEqual(interrupted, [])
        self.assertEqual([m["text"] for m in sent if m["type"] == "user_backchannel"], ["ok."])
        partial = next(m for m in sent if m["type"] == "partial_transcript")
        self.assertTrue(partial["non_interrupting"])

    async def test_idle_ok_and_ok_with_content_are_real_turns(self):
        for busy, text in [(False, "ok."), (True, "ok但是我想换个话题")]:
            sent, turns, _ = await self.run_frames([
                (busy, "我是语音助手" if busy else "", state()),
                (False, "", state(text, final=True)),
            ])
            self.assertEqual([p.text for p in turns], [text])
            self.assertFalse(any(m["type"] == "user_backchannel" for m in sent))

    async def test_echo_is_hidden_during_playback_and_after_reference_expires(self):
        sent, turns, _ = await self.run_frames([
            (True, "今天天气非常好适合出去走走", state("今天天气非常好")),
            (False, "", state("今天天气非常好适合出去走走")),
            (False, "", state("今天天气非常好适合出去走走", final=True)),
        ])
        self.assertEqual(turns, [])
        self.assertFalse(any(m.get("text") for m in sent))

    async def test_guard_does_not_leak_into_next_user_turn(self):
        sent, turns, _ = await self.run_frames([
            (True, "我是语音助手", state()),
            (False, "", state("ok.", final=True)),
            (False, "", state("ok.")),
            (False, "", state("ok.", final=True)),
        ])
        self.assertEqual(len([m for m in sent if m["type"] == "user_backchannel"]), 1)
        self.assertEqual([p.text for p in turns], ["ok."])

    async def test_real_interrupt_survives_and_late_echo_is_still_hidden(self):
        sent, turns, interrupted = await self.run_frames([
            (True, "我是语音助手今天很高兴认识你", state("停一下")),
            (False, "", state("今天很高兴认识你")),
            (False, "", state("停一下我想换个话题", final=True)),
        ])
        self.assertTrue(interrupted)
        self.assertEqual([p.text for p in turns], ["停一下我想换个话题"])
        self.assertNotIn("今天很高兴认识你", [m.get("text") for m in sent])

    async def test_late_ack_or_echo_cannot_start_early_generation(self):
        calls = []
        async def early(*args):
            calls.append(args)
        for text in ("ok.", "我是语音助手"):
            partial = state(text)
            partial.eot_score = .99
            await self.run_frames([
                (True, "我是语音助手", state()),
                (False, "", partial),
                (False, "", state(text, final=True)),
            ], on_early=early)
        self.assertEqual(calls, [])

    async def test_new_idle_speech_can_still_start_early_generation(self):
        calls = []
        async def early(*args):
            calls.append(args)
        partial = state("今天天气怎么样")
        partial.eot_score = .99
        await self.run_frames([(False, "", partial)], on_early=early)
        self.assertEqual(len(calls), 1)


class GuardTests(unittest.TestCase):
    def test_fragment_is_held_until_growth_but_ack_and_explicit_stop_are_immediate(self):
        guard = UtteranceGuard()
        guard.observe(active=True, busy=True, reference="我是语音助手")
        self.assertFalse(guard.allow_partial("我想", echo=False))
        self.assertFalse(guard.allow_partial("我想", echo=False))
        self.assertTrue(guard.allow_partial("我想问你", echo=False))
        self.assertTrue(guard.allow_partial("ok", echo=False, backchannel=True))
        self.assertTrue(guard.allow_partial("停", echo=False, explicit=True))
        self.assertFalse(guard.allow_partial("语音助手", echo=True, confirmed=True))

    def test_echo_normalization_and_non_echo_topic_overlap(self):
        self.assertTrue(is_echo("在呢，刚刚在整理", "在呢刚在整理"))
        self.assertFalse(is_echo("项目其实上周就交了", "是不是最近那个项目压得慌"))


if __name__ == "__main__":
    unittest.main()
