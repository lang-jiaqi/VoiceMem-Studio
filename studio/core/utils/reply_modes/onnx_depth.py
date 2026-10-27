"""Bounded CPU inference for a Laya depth graph using upstream token construction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class OnnxDepthRuntime:
    """Load only tokenizer/config/graph, with the Studio CPU thread budget."""

    def __init__(self, model, graph, question):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from laya.agent import Agent
        from voicemem.engine import cpu_session_options
        path = Path(model)
        self.cfg = json.loads((path/'rl_agent_config.json').read_text())
        if self.cfg.get('option_layout', 'sequential') != 'sequential':
            raise ValueError('The depth ONNX graph requires sequential option layout')
        self.tokenizer = AutoTokenizer.from_pretrained(path/'tokenizer', local_files_only=True)
        Agent._check_question('depth', question)
        self.question = question
        self.internal = Agent._to_internal(question)
        self.session = ort.InferenceSession(str(graph), sess_options=cpu_session_options(ort),
                                           providers=['CPUExecutionProvider'])

    def predict(self, state, questions):
        """Return one typed distribution with the checkpoint's fitted temperature."""
        from laya.common import build_sequence, temp_bucket
        if questions != {'depth': self.question}:
            raise ValueError('Depth runtime received a different question')
        ids, markers = build_sequence(self.tokenizer, state, self.internal,
                                      self.cfg.get('max_len', 512), self.cfg.get('head_max_len', 256))
        labels = list(self.question['criteria'])
        if len(markers) != len(labels):
            raise ValueError('Depth option markers were truncated')
        feeds = {'input_ids': np.asarray([ids], np.int64),
                 'attention_mask': np.ones((1, len(ids)), np.int64),
                 'marker_pos': np.asarray([markers], np.int64),
                 'marker_mask': np.ones((1, len(markers)), bool),
                 'qtype': np.zeros(1, np.int64)}
        feeds = {item.name: feeds[item.name] for item in self.session.get_inputs()}
        logits = np.asarray(self.session.run(['logits'], feeds)[0][0, :len(labels)], np.float64)
        if not np.isfinite(logits).all():
            raise ValueError('Depth graph returned nonfinite logits')
        temperature = self.cfg.get('temperature_by_options', {}).get(
            temp_bucket(0, len(labels)), self.cfg.get('temperature', [1, 1, 1])[0])
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid graph calibration')
        values = (logits-logits.max())/temperature
        probabilities = np.exp(values)/np.exp(values).sum()
        return {'answers': {'depth': {'choice': labels[int(probabilities.argmax())],
                                     'probabilities': dict(zip(labels, probabilities.tolist()))}}}
