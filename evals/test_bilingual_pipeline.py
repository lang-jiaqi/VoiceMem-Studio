"""Bilingual ownership regressions without APIs, private audio or active Spaces."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import Mock, patch

from voicemem.lang import (
    contextualize, detect_language, language_scope, memory_language,
    resolve_for_space, scoped_operation,
)
from voicemem.orchestrator import Orchestrator
from studio.core.utils.spaces.component import Spaces
from studio.core.utils.contracts.component import Pending


class LanguageDetectionTests(unittest.TestCase):
    def test_prose_selects_language_and_technical_words_do_not_flip_chinese(self):
        for text, expected in (
            ('解释一下 Transformer 和 attention mechanism', 'zh'),
            ('用 FastAPI 和 PostgreSQL 写 API', 'zh'),
            ('解释 Scala and Kotlin', 'zh'),
            ('请用英文解释注意力机制', 'zh'),
            ('Can you explain attention?', 'en'),
            ("I'm sad.", 'en'),
            ('What about my brother?', 'en'),
            ('I work in Singapore，偶尔出差。', 'en'),
            ('Hello!', 'en'),
        ):
            with self.subTest(text=text):
                self.assertEqual(detect_language(text, 'zh'), expected)

    def test_short_ambiguous_input_and_structured_text_retain_context(self):
        for language in ('zh', 'en'):
            for text in ('OK', '嗯', 'Alex', 'Python', '3.14',
                         '```python\nprint("this is a test")\n```', '$x^2 + y^2$',
                         r'\[\mathrm{Attention}(Q,K,V)\]', r'\(QK^T\)'):
                with self.subTest(language=language, text=text):
                    self.assertEqual(detect_language(text, language), language)


class LanguageOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_tasks_and_workers_keep_independent_language_and_restore_on_cancel(self):
        baseline = memory_language()
        ready = asyncio.Event()
        started = asyncio.Event()

        async def worker(language):
            with language_scope(language):
                started.set()
                await ready.wait()
                self.assertEqual(memory_language(), language)
                return await asyncio.to_thread(memory_language)

        en = asyncio.create_task(worker('en'))
        zh = asyncio.create_task(worker('zh'))
        await started.wait()
        cancelled = asyncio.create_task(worker('en'))
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        ready.set()
        self.assertEqual(await asyncio.gather(en, zh), ['en', 'zh'])
        self.assertEqual(memory_language(), baseline)

    async def test_delayed_workers_capture_the_originating_language(self):
        with language_scope('en'):
            english = contextualize(memory_language)
        with language_scope('zh'):
            chinese = contextualize(memory_language)
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(pool.submit(english).result(), 'en')
            self.assertEqual(pool.submit(chinese).result(), 'zh')

    async def test_session_and_space_fallbacks_do_not_cross(self):
        from studio.core.utils.capture.component import Capture
        from studio.core.utils.session_context.component import SessionBuffer
        capture = Capture()
        capture.ACTIVE_SPACE = 'a'
        capture.space_language = lambda _: 'zh'
        capture._SESSION_CONTEXT = SessionBuffer()

        async def fake_capture(*args, **kwargs):
            for space, text in [('a', 'Can you explain this?'), ('a', 'OK'),
                                ('b', 'OK'), ('a', '解释一下 attention'), ('a', 'OK')]:
                capture.ACTIVE_SPACE = space
                yield Pending(text, '', None)
        capture.anticipate = fake_capture
        result = [p.language async for p in capture._session_anticipate('one', object())]
        self.assertEqual(result, ['en', 'en', 'zh', 'zh', 'zh'])

        async def short_input(*args, **kwargs):
            yield Pending('OK', '', None)
        capture.anticipate = short_input
        result = [p.language async for p in capture._session_anticipate('two', object())]
        self.assertEqual(result, ['zh'])


class MemoryLanguageTests(unittest.TestCase):
    def test_resolving_space_defaults_does_not_change_another_request(self):
        with tempfile.TemporaryDirectory() as directory, language_scope('en'):
            root = Path(directory)
            self.assertEqual(resolve_for_space(root / 'zh', 'zh'), 'zh')
            self.assertEqual(memory_language(), 'en')
            self.assertEqual(resolve_for_space(root / 'en', 'en'), 'en')
            self.assertEqual(resolve_for_space(root / 'zh'), 'zh')
            self.assertEqual(memory_language(), 'en')

    def test_library_default_and_explicit_input_scope_are_both_preserved(self):
        class Operation:
            memory_language = 'zh'
            follow_input_language = False

            @scoped_operation
            def run(self, text):
                return memory_language()
        operation = Operation()
        self.assertEqual(operation.run('I work in Singapore.'), 'zh')
        operation.follow_input_language = True
        self.assertEqual(operation.run('I work in Singapore.'), 'en')
        with language_scope('en'):
            self.assertEqual(operation.run('OK'), 'en')

    def test_actual_async_ingest_keeps_language_after_call_scope_ends(self):
        memory = Orchestrator.__new__(Orchestrator)
        memory.memory_language = 'zh'
        memory.follow_input_language = True
        memory.remember_reply = Mock()
        memory.last_agent_reply = lambda **kw: ''
        memory._clap_memory_enabled = lambda: False
        values = dict(speaker='user', emotion='', environment='', environment_hint='',
                      scene_tag='', scene_raw_labels=[], person_id='', tune_result=None,
                      abnormal_hits=[], detection=None)
        memory.preprocess = lambda *a: SimpleNamespace(**values)
        release, finished = threading.Event(), threading.Event()
        observed = []

        def finish(context):
            if not release.wait(2):
                raise TimeoutError('test worker not released')
            observed.append((context['text'], memory_language()))
            return {'persistent_memory_created': False}
        memory._finish_ingest = finish
        try:
            memory.Ingest('OK', agent_reply='', async_facts=True, language='en',
                          on_complete=lambda _: finished.set())
            with language_scope('zh'):
                release.set()
                self.assertTrue(finished.wait(2))
                self.assertEqual(memory_language(), 'zh')
            self.assertEqual(observed, [('OK', 'en')])
        finally:
            release.set()

    def test_fact_extraction_keeps_mixed_source_text_and_scopes_generated_labels(self):
        from voicemem.leftbrain.merged_extraction import prompt_addendum
        for language, label in (('zh', 'Chinese'), ('en', 'English')):
            with self.subTest(language=language), language_scope(language):
                prompt = prompt_addendum()
                self.assertIn('Preserve natural Chinese/English mixing', prompt)
                self.assertIn(f'Write every label in {label}', prompt)
                self.assertIn('not to factual memory texts or entity names', prompt)

    def test_ui_language_never_reopens_memory_or_changes_its_default(self):
        spaces = Spaces()
        spaces.SPACE_LANG = 'zh'
        spaces.vm = original = object()
        spaces.PUBLIC_DEMO = True
        spaces._write_space_language = Mock()
        spaces.get_space = Mock()
        self.assertEqual(spaces.set_lang('en'), 'en')
        self.assertIs(spaces.vm, original)
        self.assertEqual(spaces.SPACE_LANG, 'zh')
        spaces._write_space_language.assert_not_called()
        spaces.get_space.assert_not_called()


class BilingualAdapterTests(unittest.TestCase):
    def test_studio_memory_bridge_always_injects_bilingual_asr_and_input_language(self):
        from studio.core.voicemem import open_memory
        from studio.core.utils.asr.initialize import streaming
        with patch('studio.core.voicemem.VoiceMem') as create, \
                patch('studio.core.utils.llm.initialize.create', return_value=object()), \
                patch('studio.core.utils.llm.initialize.credential', return_value='fixture'):
            open_memory({'mode': 'text_mode', 'memory_language': 'zh',
                         'reply': {'llm': {'provider': 'deepseek', 'config': {'system': 'shared'}}}})
        self.assertIs(create.call_args.kwargs['asr'], streaming)
        self.assertIs(create.call_args.kwargs['follow_input_language'], True)

    def test_stream_flush_is_idempotent_and_can_resume(self):
        import numpy as np
        from voicemem.utils.audio.asr import StreamingASR
        native = SimpleNamespace(accept_waveform=Mock())
        recognizer = SimpleNamespace(create_stream=Mock(return_value=native),
                                     is_ready=lambda _: False, get_result=lambda _: 'HELLO')
        stream = StreamingASR.from_recognizer(recognizer)
        self.assertEqual(stream.flush(), 'Hello')
        self.assertEqual(stream.flush(), 'Hello')
        self.assertEqual(native.accept_waveform.call_count, 1)
        self.assertEqual(len(native.accept_waveform.call_args.args[1]), stream.FINAL_PAD_SAMPLES)
        stream.feed(np.ones(160, dtype=np.float32))
        stream.flush()
        self.assertEqual(native.accept_waveform.call_count, 3)

    def test_bilingual_model_preparation_is_independent_of_ui_default(self):
        from studio.core.utils.models.initialize import models
        common = dict(eot=False, mode='realtime', backend='mlx', llm='deepseek')
        selected_models = []
        for language in ('zh', 'en'):
            selected = [m for m in models(SimpleNamespace(lang=language, **common))
                        if '流式 ASR' in m.name]
            self.assertEqual(len(selected), 1)
            selected_models.append(selected[0])
        self.assertEqual(selected_models[0], selected_models[1])
        model = selected_models[0]
        self.assertEqual(model.directory, 'asr/funasr-paraformer-zh-streaming')
        self.assertEqual(model.repository, 'funasr/paraformer-zh-streaming')
        self.assertEqual(model.required,
                         ('config.yaml', 'model.pt', 'tokens.json', 'am.mvn', 'seg_dict'))

    def test_default_streaming_uses_prepared_funasr_weights_on_both_backends(self):
        from studio.core.utils.asr import initialize
        model_path = str(initialize.MODELS / 'asr/funasr-paraformer-zh-streaming')
        self.addCleanup(initialize._streaming_model.cache_clear)
        for backend, device in (('mlx', 'cpu'), ('cuda', 'cuda:0')):
            with self.subTest(backend=backend), \
                    patch.dict('os.environ', {'STUDIO_BACKEND': backend, 'STUDIO_DEVICE': 'cuda:0'}), \
                    patch('voicemem.utils.audio.asr.pick_device', return_value='cpu'), \
                    patch.object(initialize, 'FunASRStreamingASR') as funasr, \
                    patch.object(initialize, 'StreamingASR') as sherpa:
                initialize._streaming_model.cache_clear()
                self.assertIs(initialize.streaming(), funasr.return_value.new_stream.return_value)
                funasr.assert_called_once_with(model=model_path, device=device)
                sherpa.assert_not_called()

    def test_shared_recognizer_keeps_decoding_streams_private(self):
        from voicemem.utils.audio.asr import StreamingASR
        recognizer = SimpleNamespace(create_stream=Mock(side_effect=[object(), object(), object()]))
        a = StreamingASR.from_recognizer(recognizer)
        b = StreamingASR.from_recognizer(recognizer)
        self.assertIs(a.rec, b.rec)
        before = b.stream
        self.assertIsNot(a.stream, b.stream)
        a.reset()
        self.assertIs(b.stream, before)

    def test_tone_keys_remain_canonical_with_english_speech_instructions(self):
        from studio.core.utils.tts.control import instruction, TONES
        from studio.core.utils.self_harness.component import speech_rate_instruction
        for tag in TONES:
            self.assertTrue(any('\u3400' <= char <= '\u9fff' for char in instruction(tag)))
            self.assertFalse(any('\u3400' <= char <= '\u9fff' for char in instruction(tag, language='en')))
        for rate in ('very_slow', 'slow', 'fast', 'very_fast'):
            text = speech_rate_instruction('Base.', {'speaking_style': {'speech_rate': rate}}, language='en')
            self.assertNotEqual(text, 'Base.')
            self.assertFalse(any('\u3400' <= char <= '\u9fff' for char in text))

    def test_english_followup_can_carry_personal_context(self):
        from studio.core.utils.routing.component import contextual_memory_query
        history = [{'role': 'user', 'content': 'Where does my sister work?'}]
        self.assertEqual(contextual_memory_query('What about my brother?', history),
                         'Where does my sister work? What about my brother?')

    def test_main_prompt_is_shared_and_has_no_turn_language_template(self):
        from studio.core.utils.prompts.component import system_prompt
        self.assertEqual(system_prompt('zh', tagged=True), system_prompt('en', tagged=True))
        self.assertIn('自然跟随用户表达和对话上下文', system_prompt('zh'))

    def test_memory_display_rewriting_preserves_the_claim_language(self):
        from studio.harness.speaking_style.policy import RB_HUMANIZE_PROMPT, RB_HUMANIZE_EXAMPLES
        self.assertIn('保留原判断的语言和自然混用', RB_HUMANIZE_PROMPT)
        self.assertIn('A little recognition makes them light up.', RB_HUMANIZE_EXAMPLES)

    def test_background_memory_uses_captured_instance_and_input_language(self):
        from studio.core.utils.memory.component import Memory
        memory = Memory()
        memory.vm = current = SimpleNamespace(ingest=Mock())
        memory._finish_history_turn = Mock()
        memory.BARGE_DEBUG = False
        target = SimpleNamespace(ingest=Mock(return_value={}), memory_language='zh')
        pending = Pending('I work in Singapore.', '', None, language='en')
        memory.remember_turn(pending, '好的。', {'id': '', 'last': '', 'miss': 0}, memory_vm=target)
        target.ingest.assert_called_once()
        self.assertEqual(target.ingest.call_args.args[0], pending.text)
        self.assertEqual(target.ingest.call_args.kwargs['language'], 'en')
        current.ingest.assert_not_called()


class SpeechLanguageTests(unittest.IsolatedAsyncioTestCase):
    async def test_tts_follows_actual_reply_and_keeps_raw_text_and_input_language(self):
        from evals.test_reply_text import ShortReplyTextTests
        from studio.core.utils.context.component import Context
        from studio.prompt_config import tts_prompts
        cases = [
            ('zh', '请用英文解释这个公式。',
             '认真|Here is the main idea.\n$$\\frac{QK^T}{\\sqrt{d_k}}$$\nThe weights select relevant information.', 'en'),
            ('en', 'Please explain this in Chinese.',
             '认真|我们先看这个公式。\n$$\\frac{QK^T}{\\sqrt{d_k}}$$\n权重决定哪些信息更相关。', 'zh'),
        ]
        for input_language, question, output, spoken_language in cases:
            with self.subTest(input_language=input_language):
                fixture = ShortReplyTextTests()
                fixture.setUp()
                agent = fixture.agent
                agent.SPACE_LANG = input_language
                agent._SPEAK_BASE = tts_prompts()['base']
                agent._TONE = tts_prompts()['fallback_by_user_emotion']
                for name in ('_by_lang', '_tone_note', '_speak_instruction'):
                    setattr(agent, name, MethodType(getattr(Context, name), agent))
                fixture.pending.text = question
                fixture.pending.language = input_language
                requests = []

                async def model(text, context, history):
                    requests.append((text, context))
                    yield output
                await fixture.pipeline(model)
                self.assertEqual(requests, [(question, '')])
                self.assertEqual(fixture.pending.language, input_language)
                visible = ''.join(m['text'] for m in fixture.messages if m['type'] == 'answer_delta')
                self.assertEqual(visible, output.split('|', 1)[1])
                self.assertNotIn('\\frac', ''.join(fixture.tts_text))
                self.assertNotIn('$$', ''.join(fixture.tts_text))
                self.assertTrue(fixture.tts_instructions)
                for instruction in fixture.tts_instructions:
                    has_chinese = any('\u3400' <= c <= '\u9fff' for c in instruction)
                    self.assertEqual(has_chinese, spoken_language == 'zh')


if __name__ == '__main__':
    unittest.main()
