"""CPU-only dialogue timing regressions; no live ASR, TTS or API required."""
import asyncio
import os
import random
import types
import unittest
from unittest.mock import patch

import numpy as np

from harness.backchannel import Backchannel
from voicemem.stream import VoiceStream
from web.harness import PauseGate, backchannel_policy, is_unfinished, system_prompt
from evals.test_short_turns import anticipate_namespace


class BackchannelTests(unittest.TestCase):
    def offer(self, bc, now, text="我就是觉得", available=None):
        bc.offer(text=text, silence=0, spoke=True, speech_s=1, now=now)
        return bc.offer(text=text, silence=.1, spoke=True, speech_s=1,
                        now=now, unfinished=is_unfinished(text), available=available)

    @patch.dict(os.environ, {"VOICEMEM_BACKCHANNEL_EMIT": "1"})
    def test_first_opportunity_and_cross_turn_three_second_cooldown(self):
        bc = Backchannel(policy=backchannel_policy(), rng=random.Random(2))
        self.assertIn(self.offer(bc, 0), {"嗯", "嗯哼", "嗯？"})
        bc.reset_turn()
        self.assertIsNone(self.offer(bc, 2.999))
        self.assertIn(self.offer(bc, 3), {"嗯", "嗯哼", "嗯？"})

    @patch.dict(os.environ, {"VOICEMEM_BACKCHANNEL_EMIT": "1"})
    def test_one_offer_per_pause_and_only_playable_continuers(self):
        bc = Backchannel(policy=backchannel_policy(), rng=random.Random(1))
        self.assertEqual(self.offer(bc, 0, available={"嗯", "对"}), "嗯")
        self.assertIsNone(bc.offer(text="我就是觉得", silence=.15, spoke=True,
                                  speech_s=1, now=5, unfinished=True))
        self.assertIsNone(self.offer(bc, 6, available={"太好了", "对"}))

    def test_opening_probability_is_high_and_long_speech_is_quieter(self):
        bc = Backchannel(policy=backchannel_policy())
        probabilities = [bc.probability(text="今天我们聊了不少东西", speech_s=s, now=0)[0]
                         for s in (1, 3, 10, 30)]
        self.assertAlmostEqual(probabilities[0], .8)
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertLess(probabilities[-1], .2)

    def test_unfinished_detection_tolerates_asr_punctuation(self):
        for text in ("我经常就", "我就是觉得。", "我就是觉得这种", "因为，", "I feel..."):
            self.assertTrue(is_unfinished(text), text)
        for text in ("我觉得今天很好。", "为什么？", "我经常就这样结束。", "ok", "我想你了",
                     "我也是这么觉得", "你怎么想？", "你觉得？", "我不想将就"):
            self.assertFalse(is_unfinished(text), text)

    def test_realtime_never_gets_spoken_control_tags(self):
        self.assertNotIn("语音控制协议", system_prompt("zh"))
        self.assertIn("温和|", system_prompt("zh", tagged=True))
        self.assertIn("English", system_prompt("en"))


class PauseStreamTests(unittest.IsolatedAsyncioTestCase):
    def make_stream(self, text="我就是觉得", *, final_text=None, textless=False):
        self.text = "" if textless else text
        self.speaking = True
        self.calls = []
        gate = self.pause_gate = PauseGate()
        stream = VoiceStream(types.SimpleNamespace(), src_rate=16000,
                             gate=lambda _: "backchannel", confirm_s=.2,
                             textless_confirm_s=.2, spec_min_chars=1000,
                             gamble_s=999, turn_end_guard=gate.allow_end,
                             eot=types.SimpleNamespace(score=lambda _: .99, threshold=.5))
        def transcribe(pcm):
            self.calls.append(pcm.copy())
            return final_text if final_text is not None else text
        async def flush():
            return self.text
        stream._final_asr = types.SimpleNamespace(transcribe=transcribe)
        stream._asr_w = types.SimpleNamespace(push=lambda _: self.text, reset=lambda: None,
                                              report=lambda: "test", flush=flush)
        stream._vad = types.SimpleNamespace(is_speech=lambda _: self.speaking)
        return stream

    async def feed(self, stream, seconds, *, speaking=False, quiet=False):
        self.speaking = speaking
        raw = np.full(320, 1200 if speaking and not quiet else 0, np.int16).tobytes()
        return [await stream.feed(raw) for _ in range(round(seconds / .02))]

    async def test_incomplete_wait_blocks_both_high_eot_and_timeout(self):
        stream = self.make_stream()
        await self.feed(stream, .2, speaking=True)
        states = await self.feed(stream, .38)
        self.assertFalse(any(s.turn for s in states))
        states = await self.feed(stream, .02)
        self.assertEqual(states[-1].turn.text, "我就是觉得")

    async def test_clip_then_300ms_blank_before_confirm(self):
        stream = self.make_stream()
        await self.feed(stream, .2, speaking=True)
        await self.feed(stream, .1)
        self.pause_gate.emitted(.2)
        self.assertFalse(any(s.turn for s in await self.feed(stream, .48)))
        self.assertIsNotNone((await self.feed(stream, .02))[-1].turn)

    async def test_vad_bridging_quiet_does_not_cancel_clip_wait(self):
        stream = self.make_stream()
        await self.feed(stream, .2, speaking=True)
        await self.feed(stream, .1, speaking=True, quiet=True)
        self.pause_gate.emitted(.2)
        await self.feed(stream, .2, speaking=True, quiet=True)
        self.assertGreater(self.pause_gate.hold_until, 0)
        self.assertFalse(any(s.turn for s in await self.feed(stream, .28)))
        self.assertIsNotNone((await self.feed(stream, .02))[-1].turn)

    async def test_resuming_speech_preserves_turn_and_cancels_wait(self):
        stream = self.make_stream(final_text="我就是觉得今天很好")
        await self.feed(stream, .2, speaking=True)
        await self.feed(stream, .1)
        self.pause_gate.emitted(.2)
        await self.feed(stream, .16)
        self.text = "我就是觉得今天很好"
        states = await self.feed(stream, .2, speaking=True)
        self.assertFalse(any(s.turn for s in states))
        self.assertEqual(self.pause_gate.hold_until, 0)
        turns = [s.turn for s in await self.feed(stream, .2) if s.turn]
        self.assertEqual([t.text for t in turns], [self.text])

    async def test_late_offline_incomplete_text_is_held_without_redecoding(self):
        for textless in (False, True):
            stream = self.make_stream(text="今天", final_text="我就是觉得", textless=textless)
            await self.feed(stream, .2, speaking=True)
            # Complete-looking partial or no partial at all: offline ASR discovers a half-sentence.
            states = await self.feed(stream, .2)
            self.assertFalse(any(s.turn for s in states))
            turns = [s.turn for s in await self.feed(stream, .6) if s.turn]
            self.assertEqual([t.text for t in turns], ["我就是觉得"])
            self.assertEqual(len(self.calls), 1)

    async def test_web_emits_short_audio_before_turn_and_keeps_cooldown_next_turn(self):
        stream = self.make_stream()
        stream.src_rate = 24000
        ns = anticipate_namespace()
        sent, turns, early = [], [], []
        # 200ms voiced + 800ms silence, twice: the next opening is within 3s.
        frames = iter(([True] * 10 + [False] * 40) * 2)
        captured = 0
        def stream_factory(**kwargs):
            stream.turn_end_guard = kwargs['turn_end_guard']
            return stream
        ns['vm'] = types.SimpleNamespace(stream=stream_factory)
        ns['_backchannel_voice'] = lambda: types.SimpleNamespace(
            available={'嗯', '嗯哼', '嗯？'}, get=lambda *a: bytes(9600))
        test = self
        class Sock:
            async def receive(self):
                nonlocal captured
                speaking = next(frames, None)
                if speaking is None:
                    return {'type': 'websocket.disconnect'}
                captured += 1
                test.speaking = speaking
                return {'bytes': np.full(480, 1200 if speaking else 0, np.int16).tobytes()}
            async def send_json(self, message):
                sent.append((captured, message))
        async def on_early(*args):
            early.append(args)
        with patch.dict(os.environ, {'VOICEMEM_BACKCHANNEL_EMIT': '1'}):
            async for turn in ns['anticipate'](Sock(), is_busy=lambda: False, on_early=on_early):
                turns.append((captured, turn))
        clips = [(frame, msg) for frame, msg in sent if msg['type'] == 'backchannel']
        self.assertEqual(len(clips), 1)
        self.assertEqual([turn.text for _, turn in turns], ['我就是觉得', '我就是觉得'])
        # One 200ms clip followed by 300ms silence, measured in captured audio.
        self.assertGreaterEqual((turns[0][0] - clips[0][0]) * .02, .5)
        self.assertEqual(early, [])


if __name__ == "__main__":
    unittest.main()
