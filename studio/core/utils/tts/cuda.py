"""Local CUDA Breeze streaming using the existing Breeze inference checkout."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import aclosing
import hashlib
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

import numpy as np

from studio.paths import MODELS, ROOT
from voicemem.prompt_config import tts_prompts


def source_directory() -> Path:
    """Locate inference code independently of model weights."""
    return Path(os.environ.get('BREEZE_CODE_DIR') or ROOT.parent / 'breeze-tts').expanduser().resolve()


def model_directory() -> Path:
    return Path(os.environ.get('BREEZE_MODEL_DIR') or MODELS / 'tts/breeze-tts-2').expanduser().resolve()


class BreezeCUDATTS:
    """Share one local model and serialize synthesis on its owning worker thread."""

    SERIAL = True
    sample_rate = 24000

    def __init__(self, model=None, ref_audio=None, ref_text=None, instruction=None,
                 device='cuda:0', cfg_scale=1.0, seed=42, chunk_frames=2,
                 first_frames=2, max_tokens=750, depth_mode='compiled'):
        self.model_name = str(model or model_directory())
        self.device = device
        self.ref_audio = str(Path(ref_audio).resolve()) if ref_audio else None
        self.ref_text = ref_text
        if self.ref_audio:
            ref = Path(self.ref_audio)
            if not ref.is_file():
                raise ValueError(f'Breeze reference audio not found: {ref}')
            if not self.ref_text and ref.with_suffix('.txt').is_file():
                self.ref_text = ref.with_suffix('.txt').read_text(encoding='utf-8').strip()
            if not self.ref_text:
                raise ValueError('Breeze reference audio requires its transcript')
        self.instruction = instruction or tts_prompts()['breeze_default_instruction']
        self.cfg_scale, self.seed = float(cfg_scale), int(seed)
        self.chunk_frames, self.first_frames = int(chunk_frames), int(first_frames)
        self.max_tokens = int(max_tokens)
        if depth_mode not in {'cached', 'compiled'}:
            raise ValueError('Breeze CUDA depth_mode must be cached or compiled')
        self.depth_mode = depth_mode
        if min(self.chunk_frames, self.first_frames, self.max_tokens) < 1:
            raise ValueError('Breeze chunk sizes and max_tokens must be positive')
        self.voice = 'breeze:' + (Path(self.ref_audio).stem if self.ref_audio else 'female-design')
        digest = hashlib.sha256(f'{self.model_name}|{self.ref_text}|{self.instruction}|{self.cfg_scale}|{self.seed}'.encode())
        if self.ref_audio:
            digest.update(Path(self.ref_audio).read_bytes())
        self.cache_voice_id = self.voice + ':' + digest.hexdigest()[:16]
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='breeze-cuda')
        self._runtime = None
        self._closed = False
        self._requests: dict[threading.Event, Future] = {}

    def _load(self):
        """Load models only on the synthesis worker; preserve streaming codec state."""
        if self._runtime is not None:
            return
        source = source_directory()
        if not (source / 'models/fast_streaming.py').is_file():
            raise FileNotFoundError(f'BREEZE_CODE_DIR lacks models/fast_streaming.py: {source}')
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
        from breeze_infer import runtime as runtime_module
        if not Path(runtime_module.__file__).resolve().is_relative_to(source):
            raise RuntimeError('Another Breeze checkout is already imported in this process')
        self._tokenizer, model, codec = load_runtime(
            Path(self.model_name), device=self.device, attn_implementation='eager')
        update_generation_config_for_breeze(model)
        runtime = FastBreezeStreamingRuntime(
            model, codec,
            FastStreamingConfig(max_new_tokens=self.max_tokens, max_seq_len=2048,
                                fast_all=None, fast_depth_decoder=self.depth_mode == 'compiled',
                                repetition_penalty=1.1),
            tokenizer=self._tokenizer)
        if runtime.sample_rate != self.sample_rate:
            raise ValueError(f'Breeze output must be {self.sample_rate}Hz, got {runtime.sample_rate}')
        if self.depth_mode == 'compiled':
            from models.warmup_profile import load_warmup_profile
            print('[tts] 正在预热 Breeze CUDA depth graph，首次编译需要等待…', flush=True)
            runtime.warmup_from_profile(load_warmup_profile(source / 'configs/fast.json'))
        self._model, self._codec, self._runtime = model, codec, runtime
        print(f'[tts] Breeze CUDA ready: {self.device}, {self.depth_mode} depth, local streaming', flush=True)

    def _segments(self, text, instruction, cancelled):
        if cancelled.is_set():
            return
        self._load()
        if cancelled.is_set():
            return
        import torch
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name

        request = {'id': uuid.uuid4().hex, 'text': text, 'speaker': 'S0',
                   'instruction': instruction or self.instruction}
        if self.ref_audio:
            request.update(ref_audio_path=self.ref_audio, ref_text=self.ref_text)
        with torch.inference_mode(), torch.cuda.device(self.device):
            set_all_seeds(self.seed)
            inputs = prepare_inputs(self._tokenizer, self._codec, self._model, [request],
                                    get_template(select_template_name(request)),
                                    guidance_scale=self.cfg_scale,
                                    guidance_scale_ref=None, guidance_scale_ins=None)
            if cancelled.is_set():
                return
            set_all_seeds(self.seed)
            chunks = self._runtime.iter_audio_chunks(inputs, request_id=request['id'], seed=self.seed)
            pending, frames, target = [], 0, self.first_frames
            try:
                for chunk in chunks:
                    if cancelled.is_set():
                        return
                    audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
                    if chunk.sample_rate != self.sample_rate or not np.isfinite(audio).all():
                        raise ValueError('Invalid Breeze CUDA PCM format')
                    if not audio.size:
                        continue
                    pending.append((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
                    frames += int(chunk.codec_frames)
                    if frames >= target:
                        yield b''.join(pending)
                        pending, frames, target = [], 0, self.chunk_frames
                if pending and not cancelled.is_set():
                    yield b''.join(pending)
            finally:
                chunks.close()

    async def stream(self, text, instruction=None):
        """Yield PCM16 with bounded buffering and cancellation at acoustic frames."""
        async with aclosing(self._stream(text, instruction)) as output:
            async for pcm in output:
                yield pcm

    async def _stream(self, text, instruction):
        if not text.strip():
            return
        if self._closed:
            raise RuntimeError('Breeze CUDA provider is closed')
        from voicemem.prompt_trace import record_request
        record_request('tts', 'breeze_cuda', {
            'model': self.model_name, 'text': text,
            'instruct': instruction or self.instruction,
            'ref_audio': self.ref_audio, 'ref_text': self.ref_text,
            'cfg_scale': self.cfg_scale, 'seed': self.seed,
            'max_tokens': self.max_tokens,
        })
        cancelled = threading.Event()
        output = queue.Queue(maxsize=4)

        def emit(value):
            while not cancelled.is_set():
                try:
                    output.put(value, timeout=0.05)
                    return
                except queue.Full:
                    continue

        def work():
            chunks = self._segments(text, instruction, cancelled)
            try:
                for pcm in chunks:
                    if cancelled.is_set():
                        break
                    emit(('pcm', pcm))
            except Exception as exc:
                emit(('error', exc))
            finally:
                try:
                    chunks.close()
                except Exception as exc:
                    emit(('error', exc))
                finally:
                    # Release a pending consumer even when it was cancelled while
                    # its executor thread was waiting on the bounded queue.
                    while True:
                        try:
                            output.put(('done', None), timeout=0.05)
                            break
                        except queue.Full:
                            if cancelled.is_set():
                                try:
                                    output.get_nowait()
                                except queue.Empty:
                                    pass

        future = self._worker.submit(work)
        self._requests[cancelled] = future
        started = time.monotonic()
        byte_count = 0
        first_at = None
        completed = False
        try:
            while True:
                kind, value = await asyncio.to_thread(output.get)
                if kind == 'done':
                    completed = True
                    return
                if kind == 'error':
                    raise value
                if first_at is None:
                    first_at = time.monotonic()
                byte_count += len(value)
                yield value
        finally:
            cancelled.set()
            # Pending work must still run its tiny cancellation path to wake any
            # consumer thread. It will not enter model inference.
            self._requests.pop(cancelled, None)
            if byte_count and completed:
                elapsed = time.monotonic() - started
                duration = byte_count / (self.sample_rate * 2)
                print(f'[tts-cuda] depth={self.depth_mode} · 音频 {duration:.2f}s'
                      f' · 耗时 {elapsed:.2f}s · RTF={elapsed / duration:.2f}'
                      f' · 首帧 {(first_at - started) * 1000:.0f}ms', flush=True)

    async def aclose(self):
        self._closed = True
        for cancelled in list(self._requests):
            cancelled.set()
        await asyncio.to_thread(self._worker.shutdown, wait=True)
