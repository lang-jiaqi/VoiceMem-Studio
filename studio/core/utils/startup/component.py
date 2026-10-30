"""Load selected providers before accepting browser sessions."""
import asyncio
import time
from contextlib import aclosing


class Startup:
    def _warm_final_asr(self, memory_vm):
        """Decode silence so final recognition cannot fail silently on first input."""
        import numpy as np
        model = memory_vm.utils.get('asr_final')
        if model is None:
            raise RuntimeError('最终 ASR 未初始化')
        model.transcribe(np.zeros(16000, dtype=np.float32))


    def _warm_emotion2vec(self, audio_path):
        """Run emotion2vec inference and reject its empty-result fallback."""
        import numpy as np
        from studio.core.utils.perception.component import emotion_inference
        result = emotion_inference(self._E2V, audio_path)
        labels = result[0].get('labels', []) if result else []
        scores = np.asarray(result[0].get('scores', []) if result else [], dtype=float)
        if (not len(labels) or scores.ndim != 1 or len(scores) != len(labels)
                or not np.isfinite(scores).all()):
            raise RuntimeError('emotion2vec 未输出分类结果')

    def _warm_eot(self, pcm):
        """Exercise EOT features and inference before capture can request a score."""
        import math

        model = self._eot()
        if model is None:
            raise RuntimeError('EOT 未初始化')
        score = model.score(pcm)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError('EOT 输出无效分数')

    async def _warm_speech(self):
        """Validate speech and its voice cache, then release loop-bound API clients."""
        from studio.core.utils.component_config.component import selection
        provider = selection('speech')['provider']
        tts = self.vm.utils.get('tts')
        language = self.space_language(self.ACTIVE_SPACE)
        try:
            from studio.core.utils.turn_taking.filler_voice import WorkFillerVoice, instruction
            from studio.core.utils.self_harness.component import SelfHarnessState
            snapshot = SelfHarnessState().snapshot()
            probes = [(language, '你好，我在这里。' if language == 'zh' else 'Hello, I am here.')]
            if provider == 'recommended':
                # Exercise both supported languages with the configured speech
                # instructions, leaving the active language warm last.
                texts = {'zh': '你好，我在这里，准备好听你说话了。',
                         'en': 'Hello, I am here and ready to listen.'}
                order = ('en', 'zh') if language == 'zh' else ('zh', 'en')
                probes = [(lang, texts[lang]) for lang in order]
            for lang, text in probes:
                started = time.monotonic()
                first = None
                samples = 0
                source = (tts.stream(text, instruction(self, lang, snapshot))
                          if provider == 'recommended' else tts.stream(text))
                async with aclosing(source) as chunks:
                    async for chunk in chunks:
                        pcm = getattr(chunk, 'pcm', chunk)
                        if pcm:
                            if provider == 'recommended' and (not isinstance(pcm, bytes) or len(pcm) % 2):
                                raise ValueError('TTS 预热需要完整的 PCM16 音频')
                            if first is None:
                                first = time.monotonic() - started
                            samples += len(pcm) // 2
                if first is None:
                    raise RuntimeError(f'TTS 未输出音频 ({lang})')
                if provider == 'recommended':
                    elapsed = time.monotonic() - started
                    audio_s = samples / 24000
                    print(f'[startup-tts] lang={lang} first_pcm_ms={first*1000:.0f}'
                          f' audio_ms={audio_s*1000:.0f} rtf={elapsed/audio_s:.3f}', flush=True)
            if self.ARGS.backchannel:
                from studio.core.utils.turn_taking.initialize import BackchannelVoice
                voice = BackchannelVoice(tts, lang=language)
                if provider == 'recommended':
                    await voice.prime(cache_only=True)
                else:
                    tokens = ('嗯', '明白') if language == 'zh' else ('mm-hmm', 'okay')
                    await voice.prime(tokens=tokens, variants=1)
                    if not voice.ready:
                        raise RuntimeError('自定义音色未生成可用附和音频')
                self._BC_VOICE['obj'] = voice
            if self.MODE == 'llm_tts':
                banks = []
                for lang in ('zh', 'en'):
                    bank = WorkFillerVoice(tts, lang, instruction(self, lang, snapshot))
                    await bank.prime()
                    banks.append(bank)
                self._WORK_FILLER_VOICES = banks
                print('[startup] 等待垫话缓存已就绪 · 中文 4 条 / 英文 4 条', flush=True)
        finally:
            if provider == 'openai':
                await tts.aclose()

    def warmup(self):
        """Report every failed warmup and refuse to serve a degraded pipeline."""
        from studio.core.utils.component_config.probe import validate_changed_llms
        validate_changed_llms(self.ARGS)
        from studio.core.voicemem import memory_warmups
        from studio.core.utils.startup.audio import warmup_audio
        from studio.core.utils.reply_modes.initialize import thinking_router
        failures = []

        def step(name, action):
            started = time.monotonic()
            try:
                action()
                print(f'[startup] {name} 已预热 · {time.monotonic() - started:.1f}s', flush=True)
            except Exception as exc:
                failures.append(name)
                print(f'[startup] {name} 失败：{type(exc).__name__}: {exc}', flush=True)

        async def warm_reply():
            self._LOCAL_LLM.prewarm()
            async with aclosing(self._LOCAL_LLM('你好', '', [])) as tokens:
                async for _ in tokens:
                    break

        if self.MODE == 'llm_tts':
            step('三级回复路由', lambda: thinking_router().warmup())
        with warmup_audio() as (pcm, audio_path):
            for name, action in memory_warmups(self.vm, audio_path=audio_path):
                step(name, action)
            step('最终 ASR', lambda: self._warm_final_asr(self.vm))
            if self.ARGS.eot:
                step('EOT', lambda: self._warm_eot(pcm))
            step('SenseVoice', lambda: self._sensevoice().run_with_emotion(str(audio_path)))
            step('emotion2vec', lambda: self._warm_emotion2vec(audio_path))
        step('语义情绪', lambda: self._emotion_by_meaning('这是一条启动检查消息。'))
        if self._LOCAL_LLM is not None:
            step('本地回复前缀和首字', lambda: asyncio.run(warm_reply()))
        if self.MODE == 'llm_tts':
            # Warm complete speech after other providers have loaded their weights.
            step('TTS / 附和缓存', lambda: asyncio.run(self._warm_speech()))
        if failures:
            raise RuntimeError('模型预热失败：' + '、'.join(failures))
        self._print_backchannel_status()
        print(f'[mem] 模型已就绪 · {self._mem_line()}', flush=True)
