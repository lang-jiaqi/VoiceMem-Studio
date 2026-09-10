"""Load selected providers before accepting browser sessions."""
import asyncio
import time
from contextlib import aclosing
from pathlib import Path
from tempfile import TemporaryDirectory


class Startup:
    def _warm_final_asr(self, memory_vm):
        """Decode silence so final recognition cannot fail silently on first input."""
        import numpy as np
        model = memory_vm.utils.get('asr_final')
        if model is None:
            raise RuntimeError('最终 ASR 未初始化')
        model.transcribe(np.zeros(16000, dtype=np.float32))


    def _warm_emotion2vec(self):
        """Load emotion2vec without classifying synthetic silence."""
        from funasr import AutoModel
        from studio.paths import MODELS
        if "m" not in self._E2V:
            self._E2V["m"] = AutoModel(model=str(MODELS / "emotion2vec"),
                                         hub="hf", disable_update=True)

    def warmup(self):
        """Report every failed warmup and refuse to serve a degraded pipeline."""
        import numpy as np
        import soundfile as sf
        from studio.core.voicemem import memory_warmups
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

        async def warm_speech():
            tts = self.vm.utils.get('tts')
            async with aclosing(tts.stream('你好')) as chunks:
                async for _ in chunks:
                    break
            if self.ARGS.backchannel:
                from studio.core.utils.turn_taking.initialize import BackchannelVoice
                voice = BackchannelVoice(tts, lang=self.space_language(self.ACTIVE_SPACE))
                await voice.prime(cache_only=True)
                self._BC_VOICE['obj'] = voice

        async def warm_reply():
            self._LOCAL_LLM.prewarm()
            async with aclosing(self._LOCAL_LLM('你好', '', [])) as tokens:
                async for _ in tokens:
                    break

        if self.MODE == 'llm_tts':
            if self.ARGS.backend == 'mlx':
                import mlx.core as mx
                from voicemem.utils.gpu_loop import gpu_loop
                gpu_loop().call(lambda: mx.set_cache_limit(536870912))
            step('Breeze TTS / 附和缓存', lambda: asyncio.run(warm_speech()))
            step('三级回复路由', lambda: thinking_router().warmup())
        for name, action in memory_warmups(self.vm):
            step(name, action)
        step('最终 ASR', lambda: self._warm_final_asr(self.vm))
        if self.ARGS.eot:
            step('EOT', self._eot)
        step('SenseVoice', self._sensevoice)
        with TemporaryDirectory(prefix='studio-warmup-') as temporary:
            silence = Path(temporary) / 'silence.wav'
            sf.write(silence, np.zeros(16000, dtype=np.float32), 16000)
            step('emotion2vec', self._warm_emotion2vec)
        if self._LOCAL_LLM is not None:
            step('本地回复前缀和首字', lambda: asyncio.run(warm_reply()))
        if failures:
            raise RuntimeError('模型预热失败：' + '、'.join(failures))
        self._print_backchannel_status()
        print(f'[mem] 模型已就绪 · {self._mem_line()}', flush=True)
