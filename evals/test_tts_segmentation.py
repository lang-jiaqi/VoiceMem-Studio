"""Offline text-boundary and reply-pipeline regressions; no models or services."""
import asyncio
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from studio.core.utils.tts.segmentation import (
    SpeechBuffer, cut_point, FIRST_WAIT_S, NEXT_WAIT_S, BUFFERED_WAIT_S,
    FIRST_MAX_CHARS, NEXT_MAX_CHARS,
)


class SpeechBufferTests(unittest.TestCase):
    def test_short_comma_phrases_form_one_complete_first_sentence(self):
        buffer = SpeechBuffer()
        parts = ['你好呀，', '我会用自然的语气回答你的问题，', '接下来我们检查语音是否能够连续播放。']
        for part in parts[:-1]:
            buffer.append(part, 0.0)
            self.assertEqual(buffer.ready(0.0), [])
        buffer.append(parts[-1], 0.0)
        self.assertEqual([s.text for s in buffer.ready(0.0)], [''.join(parts)])

    def test_following_sentence_keeps_short_comma_clauses_together(self):
        buffer = SpeechBuffer()
        buffer.append('好的。', 0.0)
        self.assertEqual([s.text for s in buffer.ready(0.0)], ['好的。'])
        buffer.append('我们先把这一句话完整地说出来，', 0.0)
        self.assertEqual(buffer.ready(0.0), [])
        buffer.append('再接着说后面的内容。', 0.0)
        self.assertEqual(len(buffer.ready(0.0)), 1)

    def test_arriving_tokens_do_not_restart_the_first_deadline(self):
        buffer = SpeechBuffer()
        buffer.append('你好呀，', 0.0)
        buffer.append('我们先讨论', FIRST_WAIT_S / 2)
        self.assertAlmostEqual(buffer.remaining_wait(FIRST_WAIT_S / 2), FIRST_WAIT_S / 2)
        self.assertEqual(buffer.ready(FIRST_WAIT_S - 0.001), [])
        self.assertEqual([s.text for s in buffer.ready(FIRST_WAIT_S)], ['你好呀，我们先讨论'])

    def test_playback_headroom_only_extends_the_rest_wait_and_is_bounded(self):
        buffer = SpeechBuffer()
        buffer.append('先说第一句话', 0.0)
        self.assertAlmostEqual(buffer.remaining_wait(0.0, 100), FIRST_WAIT_S)
        buffer.ready(0.0, final=True)
        buffer.append('后面的完整意思还在生成', 1.0)
        self.assertEqual(buffer.ready(1.0 + NEXT_WAIT_S, 100), [])
        self.assertAlmostEqual(buffer.remaining_wait(1.0, 100), BUFFERED_WAIT_S)
        self.assertTrue(buffer.ready(1.0 + BUFFERED_WAIT_S + 0.001, 100))

    def test_reduced_playback_headroom_releases_pending_text(self):
        buffer = SpeechBuffer()
        buffer.append('开始。', 0.0)
        buffer.ready(0.0)
        buffer.append('后面这句话没有结束', 0.0)
        self.assertEqual(buffer.ready(NEXT_WAIT_S + 0.01, 2.0), [])
        self.assertTrue(buffer.ready(NEXT_WAIT_S + 0.01, 0.0))

    def test_end_of_stream_flushes_short_reply_without_waiting(self):
        buffer = SpeechBuffer()
        buffer.append('好', 0.0)
        self.assertEqual([s.text for s in buffer.ready(0.0, final=True)], ['好'])
        self.assertIsNone(buffer.remaining_wait(0.0))

    def test_batched_deltas_keep_exact_text_offsets_and_closing_quotes(self):
        text = '  你好呀，我们一起看看。  “这句也要完整说完！”\n最后没有句号'
        buffer = SpeechBuffer()
        buffer.append(text, 0.0)
        segments = buffer.ready(0.0, final=True)
        self.assertEqual([s.text for s in segments],
                         ['你好呀，我们一起看看。', '“这句也要完整说完！”', '最后没有句号'])
        for segment in segments:
            self.assertEqual(text[segment.start:segment.end], segment.text)
        self.assertEqual(buffer.offset, len(text))

    def test_long_unpunctuated_text_is_bounded_and_never_dropped(self):
        text = '连续没有标点的文字' * 40
        buffer = SpeechBuffer()
        buffer.append(text, 0.0)
        segments = buffer.ready(0.0, final=True)
        self.assertEqual(''.join(s.text for s in segments), text)
        self.assertLessEqual(len(segments[0].text), FIRST_MAX_CHARS)
        self.assertTrue(all(len(s.text) <= NEXT_MAX_CHARS for s in segments[1:]))

    def test_long_phrases_can_still_use_comma_fallback(self):
        buffer = SpeechBuffer()
        first = '这' * 28 + '，'
        buffer.append(first, 0.0)
        self.assertEqual([s.text for s in buffer.ready(0.0)], [first])
        rest = '那' * 60 + '，'
        buffer.append(rest, 0.0)
        self.assertEqual([s.text for s in buffer.ready(0.0)], [rest])

    def test_english_sentence_and_decimal_split_across_deltas(self):
        buffer = SpeechBuffer()
        buffer.append('The value is 3.', 0.0)
        self.assertEqual(buffer.ready(0.0), [])
        buffer.append('14, which is useful. Next sentence!', 0.0)
        self.assertEqual([s.text for s in buffer.ready(0.0)],
                         ['The value is 3.14, which is useful.', 'Next sentence!'])

    def test_length_fallback_uses_word_boundaries(self):
        text = 'one two three four five six seven eight nine ten eleven twelve'
        buffer = SpeechBuffer()
        buffer.append(text, 0.0)
        segments = buffer.ready(0.0, final=True)
        self.assertEqual(' '.join(s.text for s in segments), text)

    def test_whitespace_and_punctuation_do_not_make_empty_synthesis_requests(self):
        buffer = SpeechBuffer()
        buffer.append('   ', 0.0)
        self.assertIsNone(buffer.remaining_wait(0.0))
        buffer.append('？！', 0.0)
        self.assertEqual(buffer.ready(0.0, final=True), [])
        self.assertFalse(cut_point('你好呀，', True))
        self.assertTrue(cut_point('你好呀。', True))


class ReplySegmentationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from studio.core.utils.reply.component import Reply
        from studio.core.utils.audio_timeline.component import AudioTimeline
        self.reply = Reply()
        self.timeline = AudioTimeline()
        self.requests = []
        self.messages = []
        self.audio = []
        self.synthesized = asyncio.Event()
        self.received_audio = asyncio.Event()
        self.remembered = []
        self.pending = SimpleNamespace(
            text='测试正常回答', memory_context='', result=None, spoken=False,
            replay='', emotion='', stranger=False, audio_path='', speech_end=0.0,
            reply_mode='memory_cot', route='deep')
        self.reply.__dict__.update(
            ACTIVE_SPACE='segmentation-test', _LAST_TONE={'tag': ''},
            _speak_base_env='', _SPEAK_BASE={}, _by_lang=lambda _: '',
            _speak_instruction=lambda _: '', BARGE_DEBUG=False,
            _SESSION_CONTEXT=SimpleNamespace(messages=lambda *a, **kw: []), HISTORY_TURNS=6,
            build_reply_context=lambda *a, **kw: '', hot_path_enter=lambda: None,
            hot_path_exit=lambda _: None, _lat_note=lambda _: 'test', _mem_line=lambda: 'test',
            _kick_acoustic=lambda *a: None, _push_history=lambda *a, **kw: 'test-turn',
            queue_remember_turn=lambda *a, **kw: self.remembered.append(a))
        self.enterContext(patch('studio.core.utils.tts.segmentation.FIRST_WAIT_S', 0.03))
        self.enterContext(patch('studio.core.utils.tts.segmentation.NEXT_WAIT_S', 0.05))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    async def synthesize(self, text, instruction=None):
        self.requests.append(text)
        self.synthesized.set()
        yield b'\x01\x00' * 480

    async def send(self, message):
        self.messages.append(message)
        if message['type'] == 'answer_done':
            self.timeline.update_checkpoint(self.timeline.sent_samples, 24000, 'drained')

    async def send_audio(self, pcm):
        self.audio.append(pcm)
        self.received_audio.set()

    def start(self, stream, synthesize=None):
        tts = SimpleNamespace(SERIAL=True, stream=synthesize or self.synthesize)
        self.reply.vm = SimpleNamespace(
            reply_stream=stream, utils=SimpleNamespace(get=lambda _: tts))
        task = asyncio.create_task(self.reply._voicemem_llm_tts(
            self.pending, self.send, self.send_audio, {}, self.timeline))

        async def cleanup():
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.addAsyncCleanup(cleanup)
        return task

    async def test_real_pipeline_merges_clauses_and_preserves_subtitles_and_offsets(self):
        parts = ['你好呀，', '我会用自然的语气回答你的问题，', '接下来我们检查语音是否能够连续播放。']

        async def stream(*args):
            yield '温和|'
            for part in parts:
                yield part

        await asyncio.wait_for(self.start(stream), 1)
        self.assertEqual(self.requests, [''.join(parts)])
        self.assertEqual([m['text'] for m in self.messages if m['type'] == 'answer_delta'], parts)
        self.assertEqual(self.timeline.generated_text, ''.join(parts))
        self.assertEqual(self.timeline.heard_text(), ''.join(parts))
        self.assertTrue(self.timeline.context_saved)

    async def test_stalled_llm_flushes_text_without_cancelling_or_restarting_stream(self):
        from voicemem.reply import _REQUEST_OPTIONS
        release = asyncio.Event()
        options = []

        async def stream(*args):
            options.append(_REQUEST_OPTIONS.get().reasoning_effort)
            yield '温和|你好呀，我们先说这一部分'
            await release.wait()
            options.append(_REQUEST_OPTIONS.get().reasoning_effort)
            yield '然后再接着往下说。'

        task = self.start(stream)
        await asyncio.wait_for(self.synthesized.wait(), 1)
        self.assertFalse(task.done())
        self.assertEqual(self.requests, ['你好呀，我们先说这一部分'])
        release.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(options, ['high', 'high'])
        self.assertEqual(self.requests[-1], '然后再接着往下说。')

    async def test_cancel_before_deadline_discards_text_and_closes_llm(self):
        entered, closed = asyncio.Event(), asyncio.Event()

        async def stream(*args):
            try:
                yield '温和|你好呀，'
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        task = self.start(stream)
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(task, 1)
        self.assertTrue(closed.is_set())
        await asyncio.sleep(0.06)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.audio, [])

    async def test_cancel_during_synthesis_reaps_all_reply_workers(self):
        closed = asyncio.Event()
        baseline = asyncio.all_tasks()

        async def stream(*args):
            yield '温和|第一句完整结束。第二句也完整结束。'
            await asyncio.Event().wait()

        async def speech(text, instruction=None):
            try:
                self.requests.append(text)
                yield b'\x01\x00' * 480
                await asyncio.Event().wait()
            finally:
                closed.set()

        task = self.start(stream, speech)
        await asyncio.wait_for(self.received_audio.wait(), 1)
        task.cancel()
        await asyncio.wait_for(task, 1)
        self.assertTrue(closed.is_set())
        self.assertEqual(self.requests, ['第一句完整结束。'])
        self.assertFalse(asyncio.all_tasks() - baseline)

    async def test_llm_error_discards_pending_text_and_does_not_leave_workers(self):
        baseline = asyncio.all_tasks()

        async def stream(*args):
            yield '温和|你好呀，'
            raise RuntimeError('test stream failure')

        with self.assertRaisesRegex(RuntimeError, 'test stream failure'):
            await self.start(stream)
        self.assertEqual(self.requests, [])
        self.assertFalse(asyncio.all_tasks() - baseline)


if __name__ == '__main__':
    unittest.main()
