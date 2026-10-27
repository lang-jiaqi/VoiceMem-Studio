"""Opt-in local Laya depth decisions with the checkpoint's trained question."""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import threading

from .component import FAST, SLOW, ThinkingDecision
from .decision import LocalDecisionRouter
from .decision_features import decision_state, texts


class LayaDecisionRouter(LocalDecisionRouter):
    """Use upstream typed inference; retain shared cancellation and filler delegation."""

    def __init__(self, model, *, device='cpu', allow_prototype=False,
                 backend='eager', onnx_path=None):
        self.path = Path(model).expanduser().resolve()
        self.metadata = json.loads((self.path / 'voice_decision.json').read_text(encoding='utf-8'))
        if self.metadata.get('version') != 1 or self.metadata.get('task') != 'voice-dialogue-depth':
            raise ValueError('Unsupported Laya voice decision checkpoint')
        if self.metadata.get('prototype', True) and not allow_prototype:
            raise ValueError('Prototype checkpoint requires allow_prototype=true')
        question = self.metadata.get('question', {})
        if (question.get('type') != 'choice'
                or set(question.get('criteria', {})) != {'ordinary', 'deep'}):
            raise ValueError('Checkpoint must contain the trained depth question')
        if backend not in {'eager', 'onnx'}:
            raise ValueError('Laya backend must be eager or onnx')
        if backend == 'onnx':
            if device != 'cpu':
                raise ValueError('The portable ONNX decision adapter uses device=cpu')
            graph = Path(onnx_path).expanduser().resolve() if onnx_path else self.path / 'laya.onnx'
            if not graph.is_file():
                raise ValueError('Local Laya ONNX graph is missing')
            with graph.open('rb') as handle:
                handle.read(1)
            self.onnx_path = str(graph)
        elif not (self.path / 'model.safetensors').is_file():
            raise ValueError('Local Laya checkpoint weights are missing')
        else:
            with (self.path / 'model.safetensors').open('rb') as handle:
                handle.read(1)
        self.backend = backend
        self.question = question
        report = self.metadata.get('report', {})
        self.accepts_audio_state = report.get(
            'trained_audio_state', report.get('observed_audio_rows', 0) > 0)
        if type(self.accepts_audio_state) is not bool:
            raise ValueError('Invalid trained audio-state flag')
        self.device = device
        self._agent = None
        self._model_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self._jobs = {}

    def classify(self, text, memory_prefetch_hint=False, history=None, *, audio_state=None):
        """Read one typed distribution off-loop; memory eligibility remains external."""
        from voicemem.engine.scheduling.cancellation import check_cancelled
        from voicemem.engine import device_lock, inference_call
        if not self.accepts_audio_state:
            audio_state = None
        current, recent = texts(text, history)
        if not current:
            return ThinkingDecision(FAST, 'laya:empty')
        key = self._input_key(text, history, audio_state)
        check_cancelled()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        with self._model_lock:
            check_cancelled()
            with inference_call('reply_router', 'onnx' if self.backend == 'onnx' else 'torch',
                                lock=device_lock(self.device)):
                if self._agent is None:
                    if self.backend == 'onnx':
                        from .onnx_depth import OnnxDepthRuntime
                        self._agent = OnnxDepthRuntime(self.path, self.onnx_path, self.question)
                    else:
                        import laya
                        self._agent = laya.load(str(self.path), device=self.device)
                answer = self._agent.predict(
                    decision_state(text, history, audio_state), {'depth': self.question})['answers']['depth']
        check_cancelled()
        probabilities = answer.get('probabilities', {})
        if set(probabilities) != {'ordinary', 'deep'} or answer.get('choice') not in probabilities:
            raise ValueError('Invalid Laya depth decision')
        values = [float(probabilities[name]) for name in ('ordinary', 'deep')]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values) or abs(sum(values)-1) > .01:
            raise ValueError('Invalid Laya depth probabilities')
        result = ThinkingDecision(SLOW if answer['choice'] == 'deep' else FAST,
                                  f'laya:p_deep={values[1]:.6f}')
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
        return result

    def _input_key(self, text, history, audio_state):
        observed = audio_state if self.accepts_audio_state else None
        return (json.dumps(decision_state(text, history, observed), ensure_ascii=False),)


def create(*, model, base_url='', api_key='', options=None):
    """Load an explicitly selected local Laya checkpoint through Studio adapters."""
    options = dict(options or {})
    if base_url or api_key or set(options) - {'allow_prototype', 'device', 'backend', 'onnx_path'}:
        raise ValueError('Laya router accepts only local checkpoint/runtime options')
    if 'allow_prototype' in options and type(options['allow_prototype']) is not bool:
        raise ValueError('allow_prototype must be a boolean')
    if 'device' in options and (not isinstance(options['device'], str) or not options['device']):
        raise ValueError('device must be a nonempty device name')
    if 'onnx_path' in options and not isinstance(options['onnx_path'], str):
        raise ValueError('onnx_path must be text')
    return LayaDecisionRouter(model, **options)
