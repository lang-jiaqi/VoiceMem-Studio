"""CPU-only dialogue timing regressions; no live ASR, TTS or API required."""
import asyncio
import os
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from studio.core.utils.turn_taking.initialize import backchannel as backchannel_module
from studio.core.utils.speaking_style.component import (
    content_emotion_note, prompt_rule,
)
from studio.core.utils.self_harness.component import (
    SelfHarnessState,
    StagedSelfHarnessUpdate,
    default_profile,
    profile_context,
    public_backchannel_curve_schema,
    public_prompt_schema,
    public_schema,
    speech_rate_instruction,
    split_control_prefix,
    validate_update,
    validate_backchannel_curve,
)
from studio.core.utils.turn_taking.initialize import (
    Backchannel,
    FillerPlan,
    HandoffKind,
    SessionFrequencyCurve,
    TurnPhase,
    TurnTakingStateMachine,
    generate_filler,
    run_overlapped_handoff,
    short_ack_plan,
    wait_for_filler_and_output,
)
from voicemem.stream import VoiceStream
from web.harness import (
    PauseGate,
    backchannel_policy,
    backchannel_policy_summary,
    is_unfinished,
    system_prompt,
)
from evals.test_short_turns import anticipate_namespace


class BackchannelTests(unittest.TestCase):
    def test_pause_ack_accepts_incomplete_continuation_points(self):
        with patch('random.Random.random', return_value=0):
            for text in ("你好", "你好！", "今天天气怎么样", "你能帮我吗？"):
                bc = Backchannel(policy=backchannel_policy())
                self.assertIsNone(self.offer(bc, 0, text=text))
            for text in ("今天真的特别开心", "这件事情说来话长", "其实我觉得", "今天", "因为", "我现在是这么想的"):
                bc = Backchannel(policy=backchannel_policy())
                self.assertIsNotNone(self.offer(bc, 0, text=text))

    def offer(self, bc, now, text="这件事情说来话长", available=None):
        bc.offer(text=text, silence=0, spoke=True, speech_s=2.1, now=now)
        return bc.offer(text=text, silence=.1, spoke=True, speech_s=2.1,
                        now=now, unfinished=is_unfinished(text), available=available)

    @patch.dict(os.environ, {"VOICEMEM_BACKCHANNEL_EMIT": "1"})
    def test_first_opportunity_and_cross_turn_cooldown(self):
        bc = Backchannel(policy=backchannel_policy(), rng=random.Random(1))
        with patch('random.Random.random', return_value=0):
            self.assertIsNotNone(self.offer(bc, 0))
            bc.reset_turn()
            self.assertIsNone(self.offer(bc, 1.999))
            self.assertIsNotNone(self.offer(bc, 2.0))

    @patch.dict(os.environ, {"VOICEMEM_BACKCHANNEL_EMIT": "1"})
    def test_one_offer_per_pause_and_only_playable_continuers(self):
        bc = Backchannel(policy=backchannel_policy(), rng=random.Random(1))
        self.assertEqual(self.offer(bc, 0, available={"嗯"}), "嗯")
        self.assertEqual(self.offer(bc, 2, text="我就是觉得", available={"嗯"}), "嗯")
        self.assertIsNone(self.offer(bc, 6, available=set()))

    def test_chinese_bank_has_only_the_reviewed_ten_tokens(self):
        affirm = backchannel_module.ZH_AFFIRMATIVE_TOKENS
        questions = backchannel_module.ZH_QUESTION_TOKENS
        self.assertEqual(
            (*affirm, *questions),
            ("哦", "哦哦", "嗯嗯", "嗯", "啊", "对", "明白", "哦？", "嗯？", "是吗？"),
        )
        active = {token for group in backchannel_module._TOKENS["zh"].values()
                  for token in group}
        self.assertEqual(active, set((*affirm, *questions)))

    def test_natural_continuers_have_extra_audio_variants(self):
        voice = backchannel_module.BackchannelVoice(object(), lang="zh")
        for token in ("哦", "嗯嗯", "嗯", "啊"):
            self.assertEqual(list(voice._style_indices(token)), list(range(5)))
        self.assertEqual(list(voice._style_indices("对")), [0, 1])
        self.assertEqual(list(voice._style_indices("嗯？")), [0])

    def test_clip_trim_preserves_tail_beyond_the_old_700ms_limit(self):
        audio = np.concatenate((
            np.full(round(24000 * .9), 4000, dtype=np.int16),
            np.zeros(round(24000 * .08), dtype=np.int16),
        ))
        pcm = backchannel_module.BackchannelVoice._trim(audio.tobytes())
        self.assertGreater(len(pcm) / 48, 900)
        tail = np.frombuffer(pcm, np.int16)[-round(24000 * .08):]
        self.assertLess(np.max(np.abs(tail)), 100)

    def test_clip_trim_rejects_overlong_audio_instead_of_hard_cutting_it(self):
        audio = np.concatenate((
            np.full(round(24000 * 3.1), 4000, dtype=np.int16),
            np.zeros(round(24000 * .08), dtype=np.int16),
        ))
        self.assertEqual(backchannel_module.BackchannelVoice._trim(audio.tobytes()), b"")

    def test_chinese_synthesis_adds_natural_sentence_punctuation(self):
        voice = backchannel_module.BackchannelVoice(object(), lang="zh")
        self.assertEqual(voice._spoken_text("嗯"), "嗯。")
        self.assertEqual(voice._spoken_text("哦哦"), "哦哦。")
        self.assertEqual(voice._spoken_text("是吗？"), "是吗？")

    def test_chinese_bank_has_exactly_seventeen_recorded_variants(self):
        voice = backchannel_module.BackchannelVoice(object(), lang="zh")
        tokens = (*backchannel_module.ZH_AFFIRMATIVE_TOKENS,
                  *backchannel_module.ZH_QUESTION_TOKENS)
        jobs = [(token, i) for token in tokens
                for i in voice._style_indices(token, variants=2)]
        self.assertEqual(len(jobs), 17)
        self.assertEqual(len({voice._bundled_filename(*job) for job in jobs}), 17)

    def test_clip_trim_rejects_a_loud_hard_boundary(self):
        tone = np.full(round(24000 * .9), 4000, dtype=np.int16)
        self.assertEqual(backchannel_module.BackchannelVoice._trim(tone.tobytes()), b"")

    def test_reviewed_wav_selection_ignores_deleted_variant_cache(self):
        voice = backchannel_module.BackchannelVoice(object(), lang="zh")
        current = np.concatenate((
            np.full(4800, 3000, dtype=np.int16),
            np.zeros(1920, dtype=np.int16),
        )).tobytes()
        with tempfile.TemporaryDirectory() as root, \
             patch.object(backchannel_module, "_cache_root", return_value=Path(root)):
            voice._uses_bundled_voice = lambda: True
            voice._bundled_pcm = lambda token, i: current if i == 1 else b""
            voice._path("嗯", 0).write_bytes(b"stale deleted variant")
            count = asyncio.run(voice.prime(
                tokens=["嗯"], variants=2, cache_only=True))
        self.assertEqual(count, 1)
        self.assertEqual(len(voice._clips["嗯"]), 1)

    def test_bundled_voice_resolves_studio_audio_directory(self):
        from studio.paths import VOICE
        tts = types.SimpleNamespace(
            ref_audio=str(VOICE / "noctelle_ref_short.wav"))
        voice = backchannel_module.BackchannelVoice(tts, lang="zh")
        self.assertTrue(voice._uses_bundled_voice())
        self.assertTrue(voice._bundled_pcm("哦", 1))
        expected = list((VOICE / "backchannel").glob("OK_*.wav"))
        with tempfile.TemporaryDirectory() as cache, patch(
                "studio.core.utils.turn_taking.backchannel._cache_root",
                return_value=Path(cache)):
            count = asyncio.run(voice.prime(cache_only=True))
        self.assertEqual(count, len(expected))
        self.assertEqual(sum(map(len, voice._clips.values())), len(expected))

    def test_bundled_filename_parser_accepts_new_tokens_and_variants(self):
        parse = backchannel_module.BackchannelVoice._bundled_job
        self.assertEqual(parse("OK_好呀.wav"), ("好呀", 0))
        self.assertEqual(parse("OK_好呀_7.wav"), ("好呀", 6))

    def test_new_reviewed_token_is_available_before_and_after_six_seconds(self):
        with patch('random.Random.random', return_value=0):
            for speech_s in (2.1, 6.1):
                policy = backchannel_policy()
                policy.refractory_s = 0
                bc = Backchannel(policy=policy)
                bc.offer(text="我还在继续说", silence=0, spoke=True,
                         speech_s=speech_s, now=0)
                self.assertEqual(bc.offer(
                    text="我还在继续说", silence=.15, spoke=True,
                    speech_s=speech_s, now=.1, unfinished=True,
                    available={"新附和"}), "新附和")

    @patch.dict(os.environ, {"VOICEMEM_BACKCHANNEL_EMIT": "1"})
    def test_end_ack_and_in_speech_backchannel_share_cooldown(self):
        bc = Backchannel(policy=backchannel_policy(), rng=random.Random(1))
        token = bc.choose(text="我说完了", available={"嗯"}, now=0)
        self.assertEqual(token, "嗯")
        bc.mark_emitted(token, now=0)
        self.assertIsNone(bc.choose(text="下一句", available={"嗯"}, now=1.999))
        self.assertEqual(bc.choose(text="下一句", available={"嗯"}, now=2.0), "嗯")

    def test_count_targets_change_within_each_turn(self):
        curve = SessionFrequencyCurve()
        self.assertEqual([curve.draw_target(0, roll) for roll in (.0, .79, .80)],
                         [2, 2, 1])
        self.assertEqual([curve.draw_target(3, roll) for roll in (.0, .79, .80)],
                         [1, 1, 0])
        self.assertEqual([curve.draw_target(6, roll) for roll in (
            .0, .29, .30, .59, .60, .89, .90)], [1, 1, 2, 2, 3, 3, 0])
        self.assertIn("单轮次数=前段2次80%/1次20%",
                      backchannel_policy_summary())
        for curve in (curve, backchannel_policy().session_curve):
            self.assertEqual([curve.phase(t) for t in (5.999, 6, 9.999, 10, 30)],
                             [1, 2, 2, 3, 3])
            self.assertEqual([curve.draw_target(10, roll) for roll in (0, .799, .8, .99)],
                             [2, 2, 0, 0])

    def test_custom_curve_uses_each_node_as_the_expected_phase_quota(self):
        policy = backchannel_policy()
        policy.refractory_s = 0
        bc = Backchannel(policy=policy, custom_phase_quotas=(0.0, 0.0, 1.5, 0.0))

        def offer(roll):
            bc.reset_turn()
            bc.offer(text="我还在继续说", silence=0, spoke=True,
                     speech_s=6.1, now=0)
            with patch('random.Random.random', return_value=roll):
                bc.offer(text="我还在继续说", silence=.15, spoke=True,
                         speech_s=6.1, now=.1, unfinished=True,
                         available={"嗯"})
            return bc._phase_targets[2]

        self.assertEqual(offer(.49), 2)
        self.assertEqual(offer(.50), 1)

    def test_first_six_seconds_have_a_shared_two_clip_cap(self):
        policy = backchannel_policy()
        policy.refractory_s = 0
        bc = Backchannel(policy=policy)

        def offer(now, speech_s):
            bc.offer(text="我就是觉得", silence=0, spoke=True,
                     speech_s=speech_s, now=now)
            return bc.offer(text="我就是觉得", silence=.15, spoke=True,
                            speech_s=speech_s, now=now, unfinished=True,
                            available={"嗯"})

        with patch('random.Random.random', return_value=0):
            self.assertEqual(offer(0, 2.1), "嗯")
            self.assertEqual(offer(.1, 2.2), "嗯")
            self.assertIsNone(offer(.2, 3.1))
            self.assertEqual(offer(.3, 6), "嗯")

    def test_late_phase_has_its_own_quota_and_resets(self):
        bc = Backchannel(policy=backchannel_policy())
        def offer(now, speech_s):
            bc.offer(text="我就是觉得", silence=0, spoke=True, speech_s=speech_s, now=now)
            return bc.offer(text="我就是觉得", silence=.15, spoke=True,
                            speech_s=speech_s, now=now, unfinished=True, available={"嗯"})
        with patch('random.Random.random', return_value=0):
            self.assertEqual(offer(6, 6), "嗯")
            self.assertIsNone(offer(8, 8))
            self.assertEqual(offer(10, 10), "嗯")
            self.assertEqual(offer(12, 12), "嗯")
            self.assertIsNone(offer(13, 13))
            bc.complete_turn()
            self.assertEqual(offer(20, 10), "嗯")

    def test_opening_quota_limits_emissions_and_resets_each_turn(self):
        bc = Backchannel(policy=backchannel_policy())
        with patch('random.Random.random', return_value=0):
            self.assertIsNotNone(self.offer(bc, 0))
            self.assertIsNotNone(self.offer(bc, 2.0))
            self.assertIsNone(self.offer(bc, 4.0))
            bc.complete_turn()
            self.assertIsNotNone(self.offer(bc, 6.0))
        self.assertIn("后段1/2/3次各30%/0次10%",
                      backchannel_policy_summary())

    def test_unfinished_detection_tolerates_asr_punctuation(self):
        for text in (
            "我经常就", "我就是觉得。", "我就是觉得这种", "因为，", "I feel...",
            "我想问一下。", "我今天", "好你先跟我", "我想打断一", "我有一个问题",
            "我", "我是说", "这个",
        ):
            self.assertTrue(is_unfinished(text), text)
        for text in ("我觉得今天很好。", "为什么？", "我经常就这样结束。", "ok", "我想你了",
                     "我也是这么觉得", "你怎么想？", "你觉得？", "我不想将就",
                     "你还能更快吗？", "今天下雨了", "主", "不", "不是", "不对",
                     "这是我的梦想", "我不想", "你觉得"):
            self.assertFalse(is_unfinished(text), text)

    def test_web_eot_can_end_the_turn_immediately(self):
        from evals.studio_helpers import studio_source
        source = studio_source()
        self.assertNotIn("eot_ends_turn=False", source)

    def test_unfinished_voice_turn_arms_a_delayed_followup(self):
        root = Path(__file__).resolve().parents[1]
        from evals.studio_helpers import studio_source
        run_source = studio_source()
        harness_source = (root / "studio/harness/turn_taking/policy.py").read_text()
        self.assertIn("run_unfinished_followup", run_source)
        self.assertIn("continuation_prompt", run_source)
        self.assertIn('"unfinished_followup_s": 2.5', harness_source)

    def test_realtime_never_gets_spoken_control_tags(self):
        self.assertNotIn("语音控制协议", system_prompt("zh"))
        self.assertNotIn("<self_harness>", system_prompt("zh"))
        self.assertIn("温和|", system_prompt("zh", tagged=True))
        self.assertIn("<self_harness>{}</self_harness>",
                      system_prompt("zh", tagged=True))
        self.assertEqual(system_prompt("en"), system_prompt("zh"))


class PauseStreamTests(unittest.IsolatedAsyncioTestCase):
    def make_stream(self, text="我就是觉得", *, final_text=None, textless=False,
                    eot_score=.99):
        self.text = "" if textless else text
        self.speaking = True
        self.calls = []
        gate = self.pause_gate = PauseGate()
        stream = VoiceStream(types.SimpleNamespace(), src_rate=16000,
                             gate=lambda _: "backchannel", confirm_s=.2,
                             textless_confirm_s=.2, spec_min_chars=1000,
                             gamble_s=999, turn_end_guard=gate.allow_end,
                             eot=types.SimpleNamespace(
                                 score=lambda _: eot_score, threshold=.5))
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
        states = await self.feed(stream, 1.16)
        self.assertFalse(any(s.turn for s in states))
        turns = [s.turn for s in await self.feed(stream, .08) if s.turn]
        self.assertEqual([turn.text for turn in turns], ["我就是觉得"])

    async def test_complete_turn_keeps_the_configured_200ms_fallback(self):
        stream = self.make_stream(text="你还能更快吗", eot_score=0)
        await self.feed(stream, .2, speaking=True)
        states = await self.feed(stream, .18)
        self.assertFalse(any(s.turn for s in states))
        turns = [s.turn for s in await self.feed(stream, .04) if s.turn]
        self.assertEqual([turn.text for turn in turns], ["你还能更快吗"])

    async def test_high_eot_ends_a_complete_turn_without_confirm_timeout(self):
        stream = self.make_stream(text="今天下雨了", eot_score=.99)
        await self.feed(stream, .2, speaking=True)
        turns = [state.turn for state in await self.feed(stream, .04) if state.turn]
        self.assertEqual([turn.text for turn in turns], ["今天下雨了"])

    async def test_question_preface_and_question_remain_one_turn(self):
        stream = self.make_stream(
            text="我想问一下", final_text="我想问一下你还能更快吗")
        await self.feed(stream, .2, speaking=True)
        self.assertFalse(any(s.turn for s in await self.feed(stream, 1.0)))
        self.text = "我想问一下你还能更快吗"
        await self.feed(stream, .2, speaking=True)
        turns = [s.turn for s in await self.feed(stream, .04) if s.turn]
        self.assertEqual(
            [turn.text for turn in turns], ["我想问一下你还能更快吗"])

    async def test_clip_then_300ms_blank_before_confirm(self):
        stream = self.make_stream(text="我觉得今天很好", eot_score=0)
        await self.feed(stream, .2, speaking=True)
        self.pause_gate.emitted(.2)
        self.assertFalse(any(s.turn for s in await self.feed(stream, .48)))
        self.assertTrue(any(s.turn for s in await self.feed(stream, .04)))

    async def test_vad_bridging_quiet_does_not_cancel_clip_wait(self):
        stream = self.make_stream(text="我觉得今天很好")
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
        states = await self.feed(stream, 1.0)
        self.assertFalse(any(s.turn for s in states))
        self.text = "我就是觉得今天很好"
        states = await self.feed(stream, .2, speaking=True)
        self.assertFalse(any(s.turn for s in states))
        self.assertEqual(self.pause_gate.hold_until, 0)
        turns = [s.turn for s in await self.feed(stream, .04) if s.turn]
        self.assertEqual([t.text for t in turns], [self.text])

    async def test_late_offline_incomplete_text_is_held_without_redecoding(self):
        for textless in (False, True):
            stream = self.make_stream(text="今天", final_text="我就是觉得", textless=textless)
            await self.feed(stream, .2, speaking=True)
            # Complete-looking partial or no partial at all: offline ASR discovers a half-sentence.
            states = await self.feed(stream, .5)
            self.assertFalse(any(s.turn for s in states))
            remaining = self.pause_gate.unfinished_until - self.pause_gate.silence
            self.assertFalse(any(s.turn for s in await self.feed(stream, remaining - .04)))
            turns = [s.turn for s in await self.feed(stream, .08) if s.turn]
            self.assertEqual([t.text for t in turns], ["我就是觉得"])
            self.assertEqual(len(self.calls), 1)

    async def test_web_requires_sustained_speech_before_emitting_and_keeps_cooldown_next_turn(self):
        stream = self.make_stream(text="我就是觉得")
        stream.src_rate = 24000
        ns = anticipate_namespace()
        sent, turns = [], []
        # More than two seconds of voiced audio + enough silence for the unfinished fallback, twice.
        # Synthetic frames run instantly, so the second turn remains in cooldown.
        frames = iter(([True] * 105 + [False] * 65) * 2)
        captured = 0
        ns['time'] = types.SimpleNamespace(monotonic=lambda: captured * .02)
        def stream_factory(**kwargs):
            stream.turn_end_guard = kwargs['turn_end_guard']
            return stream
        ns['vm'] = types.SimpleNamespace(stream=stream_factory)
        ns['_backchannel_voice'] = lambda: types.SimpleNamespace(
            available={'嗯', '嗯嗯'}, get=lambda *a: bytes(9600))
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
        with patch.dict(os.environ, {'VOICEMEM_BACKCHANNEL_EMIT': '1'}), \
             patch('random.Random.random', return_value=.39):
            async for turn in ns['anticipate'](Sock(), is_busy=lambda: False):
                turns.append((captured, turn))
        clips = [(frame, msg) for frame, msg in sent if msg['type'] == 'backchannel']
        self.assertEqual(len(clips), 1)
        self.assertEqual([turn.text for _, turn in turns],
                         ['我就是觉得', '我就是觉得'])
        # One 200ms clip followed by 300ms silence, measured in captured audio.


class SelfHarnessTests(unittest.TestCase):
    def test_public_schema_contains_only_safe_enum_metadata(self):
        schema = public_schema()
        speed = schema["speaking_style"]["speech_rate"]
        self.assertEqual(speed["default"], "normal")
        self.assertEqual(speed["labels"]["slow"], "稍慢")
        self.assertNotIn("tts", speed)
        self.assertNotIn(
            "prompt", schema["persona"]["interaction_style"])
        self.assertEqual(public_prompt_schema(), {
            "persona": {"default": "", "max_length": 2000}})
        curve = public_backchannel_curve_schema()
        self.assertEqual(curve["profiles"]["auto"], [1.8, 0.8, 1.8, 1.6])
        self.assertEqual([phase["label"] for phase in curve["phases"]],
                         ["0–3s", "3–6s", "6–10s", "10s+"])

    def test_original_prompts_are_exposed_as_immutable_defaults(self):
        from studio.harness.persona.policy import DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPT
        from studio.harness.reply_modes.policy import (
            DEFAULT_EXAMPLES, DEFAULT_SYSTEM, EXAMPLES, SYSTEM,
        )
        from studio.harness.speaking_style.policy import (
            DEFAULT_PROMPT, DEFAULT_QWEN36_PROMPT, DEFAULT_TONE_RULE,
            PROMPT, QWEN36_PROMPT, TONE_RULE,
        )
        from studio.harness.turn_taking.policy import (
            DEFAULT_FILLER_INPUT_PROMPT, DEFAULT_FILLER_PROMPT,
            FILLER_INPUT_PROMPT, FILLER_PROMPT,
        )
        self.assertEqual(SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(SYSTEM, DEFAULT_SYSTEM)
        self.assertEqual(tuple(EXAMPLES), DEFAULT_EXAMPLES)
        self.assertEqual(QWEN36_PROMPT, DEFAULT_QWEN36_PROMPT)
        self.assertEqual((PROMPT, TONE_RULE), (DEFAULT_PROMPT, DEFAULT_TONE_RULE))
        self.assertEqual(
            (FILLER_PROMPT, FILLER_INPUT_PROMPT),
            (DEFAULT_FILLER_PROMPT, DEFAULT_FILLER_INPUT_PROMPT))

    def test_private_control_prefix_is_stream_safe_and_strict(self):
        self.assertFalse(split_control_prefix("<self").resolved)
        self.assertFalse(split_control_prefix("  \n").resolved)
        parsed = split_control_prefix(
            '<self_harness>{"speaking_style":{"speech_rate":"slow",'
            '"tone":"鼓励"}}</self_harness>'
            '认真|好，我们慢一点。')
        self.assertTrue(parsed.resolved)
        self.assertEqual(parsed.update, {"speaking_style": {
            "speech_rate": "slow", "tone": "鼓励"}})
        self.assertEqual(parsed.rest, "认真|好，我们慢一点。")

        malformed = split_control_prefix(
            '<self_harness>{"provider":"evil"}</self_harness>温和|继续')
        self.assertTrue(malformed.resolved)
        self.assertEqual(malformed.update, {})
        self.assertEqual(malformed.rest, "温和|继续")
        self.assertIn("unknown Self Harness domain", malformed.error)
        self.assertEqual(split_control_prefix("普通正文").rest, "普通正文")
        incomplete = split_control_prefix("<self_", final=True)
        self.assertEqual(incomplete.rest, "")
        self.assertIn("unterminated", incomplete.error)

    def test_profile_renders_all_modules_and_request_scoped_tts_instruction(self):
        profile = default_profile()
        profile["persona"]["interaction_style"] = "listener"
        profile["speaking_style"].update(
            speech_rate="slow", reply_length="concise")
        profile["reply_modes"]["reasoning_depth"] = "deep"
        profile["turn_taking"]["backchannel"] = "off"
        note = profile_context(profile)
        self.assertIn("persona.interaction_style: listener", note)
        self.assertIn("speaking_style.speech_rate: slow (稍慢)", note)
        self.assertIn("reply_modes.reasoning_depth: deep", note)
        self.assertIn("turn_taking.backchannel: off", note)
        self.assertIn("后续回复尽量简短", note)
        self.assertEqual(speech_rate_instruction("基础。", profile),
                         "基础。语速稍慢，咬字清楚，句尾自然收住。")

    def test_speculative_update_commits_only_after_acceptance(self):
        target = SelfHarnessState()
        staged = StagedSelfHarnessUpdate(target)
        staged.stage({"speaking_style": {"tone": "轻快"}})
        self.assertEqual(target.profile["speaking_style"]["tone"], "auto")
        staged.commit()
        self.assertEqual(target.profile["speaking_style"]["tone"], "轻快")

    def test_recent_changes_expire_and_large_updates_are_rejected(self):
        state = SelfHarnessState()
        state.apply({"persona": {"interaction_style": "listener"}})
        self.assertIn("persona.interaction_style", state.snapshot()["recent"])
        state.apply({})
        self.assertIn("persona.interaction_style", state.snapshot()["recent"])
        state.apply({})
        self.assertNotIn("persona.interaction_style", state.snapshot()["recent"])
        with self.assertRaisesRegex(ValueError, "at most 2"):
            validate_update({
                "persona": {"interaction_style": "coach"},
                "speaking_style": {"tone": "轻快"},
                "reply_modes": {"reasoning_depth": "deep"},
            })

    def test_rapid_conflict_is_staged_until_the_same_value_is_repeated(self):
        state = SelfHarnessState()
        first = {"speaking_style": {"tone": "轻快"}}
        conflict = {"speaking_style": {"tone": "认真"}}

        self.assertEqual(state.apply(first), first)
        self.assertEqual(state.apply(conflict), {})
        self.assertEqual(state.profile["speaking_style"]["tone"], "轻快")
        self.assertEqual(
            state.snapshot()["pending"],
            {"speaking_style.tone": "认真"})
        replacement = {"speaking_style": {"tone": "温和"}}
        self.assertEqual(state.apply(replacement), {})
        self.assertEqual(state.profile["speaking_style"]["tone"], "轻快")
        self.assertEqual(state.snapshot()["pending"], {
            "speaking_style.tone": "温和"})
        self.assertEqual(state.apply(replacement), replacement)
        self.assertEqual(state.profile["speaking_style"]["tone"], "温和")
        self.assertEqual(state.snapshot()["pending"], {})

    def test_explicit_settings_choice_applies_immediately_and_notifies(self):
        snapshots = []
        state = SelfHarnessState(on_change=snapshots.append)
        state.apply({"speaking_style": {"speech_rate": "slow"}})
        state.apply({"speaking_style": {"speech_rate": "fast"}})
        self.assertEqual(state.profile["speaking_style"]["speech_rate"], "slow")
        self.assertTrue(state.snapshot()["pending"])

        state.set_explicit({"speaking_style": {"speech_rate": "very_fast"}})
        self.assertEqual(
            state.profile["speaking_style"]["speech_rate"], "very_fast")
        self.assertEqual(state.snapshot()["pending"], {})
        self.assertEqual(
            snapshots[-1]["profile"]["speaking_style"]["speech_rate"],
            "very_fast")

        state.set_prompt("persona", "像熟人一样聊，但不要过度热情。")
        self.assertIn("像熟人一样聊", profile_context(state.snapshot()))
        self.assertEqual(
            snapshots[-1]["prompts"]["persona"],
            "像熟人一样聊，但不要过度热情。")
        with self.assertRaisesRegex(ValueError, "unknown Harness prompt"):
            state.set_prompt("tone", "任意语气")

        state.set_backchannel_curve([1.2, 0.4, 2.4, 1.0])
        self.assertEqual(
            snapshots[-1]["backchannel_curve"], [1.2, 0.4, 2.4, 1.0])
        self.assertIn(
            "1.2 / 0.4 / 2.4 / 1.0", profile_context(state.snapshot()))
        self.assertEqual(
            state.profile["turn_taking"]["backchannel"], "auto")
        state.apply({"turn_taking": {"backchannel": "more"}})
        self.assertIsNotNone(state.snapshot()["backchannel_curve"])
        state.apply({"turn_taking": {"backchannel": "more"}})
        self.assertIsNone(state.snapshot()["backchannel_curve"])
        with self.assertRaisesRegex(ValueError, "four phase"):
            validate_backchannel_curve([1, 2])


class SpeakingStyleTests(unittest.TestCase):

    def test_intro_arc_is_shared_but_general_boost_is_qwen_only(self):
        from studio.core.utils.speaking_style.component import qwen_segment_instruction
        for qwen in (False, True):
            bright = qwen_segment_instruction("base", "请介绍一下你自己", "我是超级智能！", qwen)
            sad = qwen_segment_instruction("base", "请介绍一下你自己", "我是超级智能！如果说有什么我做不到，就是没有身体。", qwen)
            self.assertIn("明亮", bright)
            self.assertIn("明显悲伤", sad)
            self.assertIn("语速明显放慢", sad)
        self.assertEqual(qwen_segment_instruction("base", "今天如何", "今天不错", False), "base")

    def test_prompt_selects_depth_from_context_and_places_emotion_in_content(self):
        rule = prompt_rule("zh")
        self.assertIn("根据上下文决定", rule)
        self.assertIn("情绪不是只交给 TTS", rule)
        note = content_emotion_note("委屈", "zh")
        self.assertIn("委屈", note)
        self.assertIn("弱信号", note)
        self.assertIn("绝不说出这个标签", note)


class TurnTakingTimingTests(unittest.IsolatedAsyncioTestCase):
    def test_first_slow_turn_can_offer_work_filler_after_fast_smalltalk(self):
        machine = TurnTakingStateMachine(initial_wait_s=.2, work_filler_probability=1.0)
        self.assertIs(machine.decide_handoff(main_audio_ready=False, reply_mode='memory_cot',
                                             cached_ack_available=False).kind, HandoffKind.LLM_FILLER)
        for mode in ('direct', 'memory'):
            self.assertIs(machine.decide_handoff(main_audio_ready=False, reply_mode=mode,
                                                 cached_ack_available=False).kind, HandoffKind.DIRECT)
        self.assertIs(machine.decide_handoff(main_audio_ready=True, reply_mode='memory_cot',
                                             cached_ack_available=False).kind, HandoffKind.DIRECT)

    def test_state_machine_selects_handoff_from_readiness_and_reply_mode(self):
        machine = TurnTakingStateMachine(initial_wait_s=2.0, work_filler_probability=1.0)
        machine.commit_user_turn()

        self.assertIs(
            machine.decide_handoff(
                main_audio_ready=True,
                reply_mode="memory_cot",
                cached_ack_available=True,
            ).kind,
            HandoffKind.DIRECT,
        )
        self.assertEqual(
            machine.decide_handoff(
                main_audio_ready=False,
                reply_mode="memory_cot",
                cached_ack_available=True,
                spoken=False,
            ).reason,
            "text_turn",
        )
        self.assertIs(
            machine.decide_handoff(
                main_audio_ready=False,
                reply_mode="memory",
                cached_ack_available=True,
            ).kind,
            HandoffKind.CACHED_ACK,
        )
        decision = machine.decide_handoff(
            main_audio_ready=False,
            reply_mode="memory_cot",
            cached_ack_available=True,
        )
        self.assertIs(decision.kind, HandoffKind.LLM_FILLER)
        machine.start_handoff(decision)
        self.assertIs(machine.phase, TurnPhase.FILLING)

    def test_state_machine_tracks_session_echo_and_latency(self):
        machine = TurnTakingStateMachine(
            initial_wait_s=1.0, estimate_weight=0.5, echo_window_s=4.0)
        machine.begin_user_turn()
        machine.commit_user_turn()
        machine.record_emission("嗯嗯", now=10.0)
        machine.observe_first_audio(0.6)
        machine.start_reply()
        machine.finish_reply()

        self.assertEqual(machine.completed_turns, 1)
        self.assertEqual(machine.recent_agent_text(now=12.0), "嗯嗯")
        self.assertEqual(machine.recent_agent_text(now=15.0), "")
        self.assertAlmostEqual(machine.expected_wait_s, 0.8)
        self.assertIs(machine.phase, TurnPhase.LISTENING)

    def test_frequency_curve_boundaries(self):
        curve = SessionFrequencyCurve()
        self.assertEqual([curve.phase(i) for i in range(8)],
                         [0, 0, 0, 1, 1, 1, 2, 2])

    def test_short_ack_uses_actual_clip_length_without_cutting_its_tail(self):
        self.assertAlmostEqual(short_ack_plan(.72).main_start_seconds, .72)

    async def test_main_work_starts_before_filler_finishes(self):
        events = []

        async def filler():
            events.append("filler-start")
            await asyncio.sleep(.03)
            events.append("filler-end")

        async def main():
            events.append("main-start")

        await run_overlapped_handoff(
            filler, main, FillerPlan(.03, .01, "test"))
        self.assertEqual(events, ["filler-start", "main-start", "filler-end"])

    async def test_zero_lead_plan_finishes_filler_before_main(self):
        events = []

        async def filler():
            events.append("filler-start")
            await asyncio.sleep(0)
            events.append("filler-end")

        async def main():
            events.append("main-start")

        await run_overlapped_handoff(
            filler, main, FillerPlan(.03, 0, "test"))
        self.assertEqual(events, ["filler-start", "filler-end", "main-start"])

    async def test_spoken_filler_and_main_output_must_both_finish_before_release(self):
        filler_done = asyncio.Event()
        output_ready = asyncio.Event()
        handoff = asyncio.create_task(wait_for_filler_and_output(
            filler_done.wait(), output_ready.wait()))

        output_ready.set()
        await asyncio.sleep(0)
        self.assertFalse(handoff.done())
        filler_done.set()
        await handoff

        filler_done.clear()
        output_ready.clear()
        handoff = asyncio.create_task(wait_for_filler_and_output(
            filler_done.wait(), output_ready.wait()))
        filler_done.set()
        await asyncio.sleep(0)
        self.assertFalse(handoff.done())
        output_ready.set()
        await handoff

    async def test_reply_model_generates_only_one_short_filler(self):
        calls = []

        async def reply_stream(text, context, history):
            calls.append((text, context, history))
            for chunk in ("温和|稍等呀，", "我帮你看一下。", "这句不该出现"):
                yield chunk

        filler = await generate_filler(
            reply_stream, "查询今天的天气", history=[{"role": "user", "content": "你好"}])
        self.assertEqual(filler, "稍等呀，我帮你看一下。")
        self.assertIn("约四秒", calls[0][0])
        self.assertIn("查询今天的天气", calls[0][0])


if __name__ == "__main__":
    unittest.main()
