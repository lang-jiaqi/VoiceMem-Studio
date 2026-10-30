"""Startup must exercise inference without persisting synthetic input."""
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from studio.core.utils.startup.audio import warmup_audio
from studio.core.utils.startup.component import Startup
from studio.core.voicemem import memory_warmups


class ProbeTests(unittest.TestCase):
    def test_non_silent_probe_is_disposable_even_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                try:
                    with warmup_audio() as (pcm, path):
                        recorded, rate = sf.read(path, dtype='float32')
                        self.assertEqual(rate, 16000)
                        self.assertEqual(pcm.shape, (48000,))
                        self.assertGreater(float(np.sqrt(np.mean(pcm ** 2))), .005)
                        np.testing.assert_allclose(recorded, pcm, atol=1 / 32768)
                        if fail:
                            raise ValueError('synthetic failure')
                except ValueError:
                    self.assertTrue(fail)
                self.assertFalse(path.parent.exists())


class MemoryWarmupTests(unittest.TestCase):
    def setUp(self):
        self.asr = Mock()
        self.scene = Mock()
        self.scene.detect_full.return_value = {'embedding': np.ones(3)}
        self.speaker = Mock()
        self.speaker.embed.return_value = np.ones(3)
        self.emotion = Mock()
        self.memory = SimpleNamespace(
            memory_language='en', classify=Mock(),
            utils={'asr': self.asr, 'vad': Mock()},
            _o=SimpleNamespace(
                _get_env_detector=lambda: self.scene,
                _get_speaker_encoder=lambda: self.speaker,
                _get_emotion_detector=lambda: self.emotion))

    def steps(self, **kwargs):
        with patch('voicemem.gate.ON', False):
            return dict(memory_warmups(self.memory, **kwargs))

    def test_inference_runs_on_shared_probe_without_ingestion(self):
        with warmup_audio() as (_, path):
            for action in self.steps(audio_path=path).values():
                action()
            self.scene.detect_full.assert_called_once_with(path)
            self.speaker.embed.assert_called_once_with(path)
            self.emotion.detect.assert_called_once_with(path)
            self.emotion.warmup.assert_called_once_with()
            self.scene._load.assert_not_called()
            self.speaker._ensure_started.assert_not_called()
            self.assertEqual([c[0] for c in self.asr.method_calls], ['feed', 'flush', 'reset'])

    def test_standalone_audio_check_cleans_up_after_provider_failure(self):
        paths = []

        def infer(path):
            paths.append(path)
            self.assertTrue(path.is_file())
            raise RuntimeError('synthetic inference failure')

        self.scene.detect_full.side_effect = infer
        with self.assertRaisesRegex(RuntimeError, 'synthetic inference'):
            self.steps()['AST 场景']()
        self.assertFalse(paths[0].parent.exists())

    def test_failed_asr_always_resets_decoder_state(self):
        for method in ('feed', 'flush'):
            with self.subTest(method=method):
                self.asr.reset_mock(side_effect=True)
                getattr(self.asr, method).side_effect = RuntimeError('synthetic ASR failure')
                with self.assertRaisesRegex(RuntimeError, 'synthetic ASR'):
                    self.steps()['流式 ASR']()
                self.asr.reset.assert_called_once_with()

    def test_scene_and_speaker_fallbacks_are_not_ready(self):
        for vector in (None, [], [0, 0], [float('nan')], [[1, 2]]):
            with self.subTest(vector=vector):
                self.scene.detect_full.return_value = {'embedding': vector}
                self.speaker.embed.return_value = vector
                steps = self.steps(audio_path=Path('synthetic.wav'))
                for name in ('AST 场景', '声纹'):
                    with self.assertRaises(RuntimeError):
                        steps[name]()

    def test_semantic_gate_is_warmed_only_when_enabled(self):
        with patch('voicemem.gate.ON', True), \
                patch('voicemem.gate.semantic_margin') as margin:
            steps = dict(memory_warmups(self.memory))
            steps['Turn Gate 语义路由']()
            self.assertEqual(margin.call_args.args[1], 'en')
            margin.side_effect = RuntimeError('synthetic gate failure')
            with self.assertRaisesRegex(RuntimeError, 'synthetic gate'):
                steps['Turn Gate 语义路由']()
        self.assertNotIn('Turn Gate 语义路由', self.steps())


class StartupInferenceTests(unittest.TestCase):
    def test_eot_executes_scoring_and_rejects_invalid_results(self):
        agent = Startup()
        model = Mock()
        agent._eot = lambda: model
        pcm = np.ones(16000, dtype=np.float32)
        for score in (.2, float('nan'), -1, 2):
            model.score.return_value = score
            if score == .2:
                agent._warm_eot(pcm)
            else:
                with self.assertRaises(RuntimeError):
                    agent._warm_eot(pcm)
        self.assertIs(model.score.call_args.args[0], pcm)
        agent._eot = lambda: None
        with self.assertRaisesRegex(RuntimeError, 'EOT 未初始化'):
            agent._warm_eot(pcm)

    def test_emotion2vec_runs_loaded_model_and_validates_output(self):
        agent = Startup()
        model = Mock()
        agent._E2V = {'m': model}
        auto_model = Mock()
        with patch.dict('sys.modules', {'funasr': SimpleNamespace(AutoModel=auto_model)}):
            model.generate.return_value = [{'labels': ['neutral'], 'scores': np.array([.9])}]
            agent._warm_emotion2vec(Path('synthetic.wav'))
            model.generate.assert_called_once_with(
                'synthetic.wav', granularity='utterance', extract_embedding=False)
            auto_model.assert_not_called()
            for result in ([], [{}], [{'labels': ['neutral'], 'scores': []}],
                           [{'labels': ['neutral'], 'scores': [float('nan')]}]):
                model.generate.return_value = result
                with self.assertRaises(RuntimeError):
                    agent._warm_emotion2vec(Path('synthetic.wav'))

    def test_failure_prevents_ready_but_finishes_remaining_checks_and_cleanup(self):
        for failed_step in ('scene', 'speech', None):
            with self.subTest(failed_step=failed_step):
                events, paths = [], []
                agent = Startup()
                agent.MODE, agent.ACTIVE_SPACE = 'llm_tts', 'synthetic-space'
                agent.ARGS = SimpleNamespace(backend='cuda', backchannel=False, eot=True)
                agent._LOCAL_LLM = None
                agent.space_language = lambda _: 'zh'
                agent._speak_instruction = lambda emotion, language: 'Synthetic speech style'

                def scene():
                    events.append('scene')
                    if failed_step == 'scene':
                        raise RuntimeError('synthetic scene failure')

                def memory_checks(memory, *, audio_path):
                    paths.append(audio_path)
                    return [('scene', scene)]

                def emotion(path):
                    self.assertTrue(path.is_file())
                    events.append('emotion')

                async def speech(text, instruction=None):
                    events.append('speech')
                    if failed_step != 'speech':
                        yield b'\x01\x00' * 16
                    events.append('speech-complete')

                agent.vm = SimpleNamespace(utils={'tts': SimpleNamespace(stream=speech)})
                agent._warm_final_asr = lambda _: events.append('final-asr')
                agent._warm_eot = lambda _: events.append('eot')
                agent._sensevoice = lambda: SimpleNamespace(
                    run_with_emotion=lambda path: events.append('sensevoice'))
                agent._warm_emotion2vec = emotion
                agent._emotion_by_meaning = lambda _: events.append('semantic-emotion')
                agent._print_backchannel_status = lambda: events.append('ready')
                agent._mem_line = lambda: 'synthetic'
                output = io.StringIO()
                with patch('studio.core.voicemem.memory_warmups', side_effect=memory_checks), \
                        patch('studio.core.utils.reply_modes.initialize.thinking_router'), \
                        redirect_stdout(output):
                    if failed_step:
                        with self.assertRaises(RuntimeError):
                            agent.warmup()
                        self.assertNotIn('ready', events)
                        self.assertNotIn('模型已就绪', output.getvalue())
                    else:
                        agent.warmup()
                        self.assertIn('ready', events)
                self.assertFalse(paths[0].parent.exists())
                self.assertLess(events.index('emotion'), events.index('speech'))
                self.assertIn('eot', events)
                self.assertIn('semantic-emotion', events)
                self.assertIn('speech-complete', events)


if __name__ == '__main__':
    unittest.main()
