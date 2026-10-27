"""Opt-in local depth classification with separate current/context encoding."""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import threading

import numpy as np

from .component import FAST, SLOW, ThinkingDecision
from .decision import LocalDecisionRouter
from .decision_features import VERSION, audio_vector, texts, decision_state


class EncoderDepthRouter(LocalDecisionRouter):
    """Retain shared cancellation and independent filler generation boundaries."""

    def __init__(self, model, *, device='cpu', backend='eager', allow_prototype=False):
        self.path = Path(model).expanduser().resolve()
        self.metadata = json.loads((self.path/'depth_encoder.json').read_text())
        if (self.metadata.get('version') != 1 or self.metadata.get('task') != 'voice-depth-encoder'
                or self.metadata.get('features_version') != VERSION
                or self.metadata.get('labels') != ['ordinary', 'deep']):
            raise ValueError('Unsupported depth encoder checkpoint')
        if self.metadata.get('prototype', True) and not allow_prototype:
            raise ValueError('Prototype checkpoint requires allow_prototype=true')
        if backend not in {'eager', 'onnx'} or (backend == 'onnx' and device != 'cpu'):
            raise ValueError('Depth encoder ONNX requires device=cpu')
        self.temperature = float(self.metadata['temperature'])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Invalid depth encoder calibration')
        self.accepts_audio_state = self.metadata['report'].get('trained_audio_state', False)
        if type(self.accepts_audio_state) is not bool:
            raise ValueError('Invalid audio-state training flag')
        if type(self.metadata.get('max_length')) is not int or not 8 <= self.metadata['max_length'] <= 512:
            raise ValueError('Invalid depth encoder token budget')
        filename = 'model.onnx' if backend == 'onnx' else 'model.safetensors'
        with (self.path/filename).open('rb') as handle:
            handle.read(1)
        self.device, self.backend = device, backend
        self._model = self._tokenizer = None
        self._model_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cache, self._jobs = OrderedDict(), {}

    def _input_key(self, text, history, audio_state):
        return (json.dumps(decision_state(text, history,
                       audio_state if self.accepts_audio_state else None), ensure_ascii=False),)

    def _load(self):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.path/'tokenizer', local_files_only=True,
                                                       trust_remote_code=False)
        if self.backend == 'onnx':
            import onnxruntime as ort
            from voicemem.engine import cpu_session_options
            self._model = ort.InferenceSession(str(self.path/'model.onnx'),
                                               sess_options=cpu_session_options(ort),
                                               providers=['CPUExecutionProvider'])
        else:
            from transformers import AutoConfig
            from safetensors.torch import load_file
            from .depth_network import DepthEncoder
            config = AutoConfig.from_pretrained(self.path/'encoder', local_files_only=True,
                                                trust_remote_code=False)
            self._model = DepthEncoder(config)
            self._model.load_state_dict(load_file(str(self.path/'model.safetensors')), strict=True)
            self._model.to(self.device).eval()

    def classify(self, text, memory_prefetch_hint=False, history=None, *, audio_state=None):
        from voicemem.engine.scheduling.cancellation import check_cancelled
        from voicemem.engine import device_lock, inference_call
        current, recent = texts(text, history)
        if not current:
            return ThinkingDecision(FAST, 'encoder:empty')
        observed = audio_state if self.accepts_audio_state else None
        vector = np.concatenate((audio_vector(observed), np.asarray([bool(recent)], np.float32)))
        key = self._input_key(text, history, observed)
        check_cancelled()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        with self._model_lock:
            check_cancelled()
            with inference_call('reply_router', 'onnx' if self.backend == 'onnx' else 'torch',
                                lock=device_lock(self.device)):
                if self._model is None:
                    self._load()
                encoded = self._tokenizer([current, recent], padding=True, truncation=True,
                                          max_length=self.metadata['max_length'], return_tensors='np')
                inputs = {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'],
                          'observed': vector[None].astype(np.float32)}
                if self.backend == 'onnx':
                    logits = self._model.run(['logits'], inputs)[0][0]
                else:
                    import torch
                    with torch.inference_mode():
                        logits = self._model(**{name: torch.from_numpy(value).to(self.device)
                                               for name, value in inputs.items()})[0].float().cpu().numpy()
        check_cancelled()
        scaled = np.asarray(logits, np.float64)/self.temperature
        if scaled.shape != (2,) or not np.isfinite(scaled).all():
            raise ValueError('Invalid depth encoder logits')
        values = np.exp(scaled-scaled.max())
        probability = float(values[1]/values.sum())
        result = ThinkingDecision(SLOW if probability >= .5 else FAST, f'encoder:p_deep={probability:.6f}')
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
        return result


def create(*, model, base_url='', api_key='', options=None):
    """Construct an explicitly selected local encoder and runtime adapter."""
    options = dict(options or {})
    if base_url or api_key or set(options)-{'device', 'backend', 'allow_prototype'}:
        raise ValueError('Depth encoder accepts only local checkpoint/runtime options')
    if 'allow_prototype' in options and type(options['allow_prototype']) is not bool:
        raise ValueError('allow_prototype must be a boolean')
    if 'device' in options and (not isinstance(options['device'], str) or not options['device']):
        raise ValueError('device must be a nonempty device name')
    return EncoderDepthRouter(model, **options)
