"""Shared task-specific encoder network for offline training and local inference."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from .decision_features import audio_vector, texts


def features(text, history=None, audio_state=None):
    """Return bounded semantic inputs and observed features with missingness."""
    current, recent = texts(text, history)
    values = np.concatenate((audio_vector(audio_state), np.asarray([bool(recent)], np.float32)))
    return current, recent, values.astype(np.float32)


class DepthEncoder(nn.Module):
    """Encode current text and history separately using one shared forward batch."""

    def __init__(self, config, *, pretrained=None, dropout=.15):
        super().__init__()
        if config.model_type != 'bert':
            raise ValueError('Unsupported depth encoder architecture')
        options = {'add_pooling_layer': False, 'attn_implementation': 'eager', 'trust_remote_code': False}
        self.encoder = (AutoModel.from_pretrained(str(pretrained), local_files_only=True,
                                                 weights_only=True, **options)
                        if pretrained else AutoModel.from_config(config, **options))
        self.hidden_size = config.hidden_size
        self.classifier = nn.Sequential(nn.LayerNorm(3*self.hidden_size+7),
                                        nn.Linear(3*self.hidden_size+7, 256), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(256, 2))

    def forward(self, input_ids, attention_mask, observed):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        pairs = states.reshape(observed.shape[0], 2, self.hidden_size)
        current = pairs[:, 0]
        recent = pairs[:, 1]*observed[:, -1:]
        joined = torch.cat((current, recent, current*recent, observed), dim=-1)
        return self.classifier(joined)
