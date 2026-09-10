"""Depth cache optimization from the user-supplied breezetts2_mac_fast.py.

Source implementation is preserved in tools/breezetts2_mac_fast.py.
Requires mlx-audio==0.5.1 and mlx==0.32.2.
"""
from __future__ import annotations

class ReferencePromptCache:
    """Immutable reference prefix, matching mlx-audio 0.5.1's segment order.

    Two levels of caching, both keyed on (ref_audio, ref_text, voice):

    1. **Prefix embeddings** (text encoder over ref_text + codec over ref_audio).
    2. **Prefix backbone KV** — the 3B backbone's K/V for those prefix positions.
       Release ``generate()`` builds a fresh ``KVCache`` and prefills
       [ref_text | ref_codes | eos | instruct+text] every call, so every segment
       pays for the reference again although it never changes. With the KV
       snapshot, ``make_cache()`` hands back caches already holding the prefix
       and the backbone only prefills the instruct+text suffix. RoPE offsets and
       the causal mask come from ``KVCache.offset``, so positions are identical
       to a full prefill; ``verify()`` checks that numerically at load time.

    Sharing the snapshot arrays across calls is safe: ``KVCache.update_and_fetch``
    always takes the concatenate path on the first append (the snapshot is
    trimmed exactly to ``offset``), so it never writes into the shared array.
    """
    def __init__(self, model, prefix_kv=True):
        self.model = model
        self.key = self.prefix = None
        self.prefix_kv = bool(prefix_kv)
        self.kv = None            # [(keys, values)] per layer, trimmed to prefix_len
        self.prefix_len = 0
        self._pending = 0         # make_cache() calls that should be prefix-seeded
        self._installed = False

    def _build_prefix(self, obj, voice, ref_audio, ref_text):
        import mlx.core as mx
        key = (ref_audio, ref_text, voice)
        if self.key == key:
            return
        ids = obj._text_ids(f"{obj._speaker(voice)}{ref_text}")
        hidden = obj.text_encoder_proj(obj.text_encoder(ids[None, :]))
        codes = obj._encode_reference(ref_audio)
        eos = mx.full((1, 1, obj.num_codebooks), obj.config.codebook_eos_token_id)
        prefix = mx.concatenate((hidden, obj.backbone_model.embed_tokens(codes),
                                 obj.backbone_model.embed_tokens(eos)), axis=1)
        mx.eval(prefix)
        self.prefix, self.key = prefix, key
        self.kv, self.prefix_len = None, 0
        if self.prefix_kv:
            caches = self._original_make_cache()
            obj.backbone_model(input_embeddings=prefix, cache=caches)
            kv = []
            for c in caches:
                k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
                mx.eval(k, v)
                kv.append((k, v))
            self.kv, self.prefix_len = kv, int(caches[0].offset)

    def _seeded_caches(self):
        caches = self._original_make_cache()
        for c, (k, v) in zip(caches, self.kv):
            c.keys, c.values, c.offset = k, v, self.prefix_len
        return caches

    def install(self):
        target, cls = self.model, type(self.model)
        original = self.original = cls._prompt_embeddings
        self.owned = "_prompt_embeddings" in cls.__dict__
        self._original_make_cache = target.backbone_model.make_cache
        self._pending = 0
        self._installed = True

        def cached(obj, text, *, voice, instruct, ref_audio, ref_text):
            if obj is not target or ref_audio is None or not isinstance(ref_audio, str):
                return original(obj, text, voice=voice, instruct=instruct,
                                ref_audio=ref_audio, ref_text=ref_text)
            if not ref_text:
                raise ValueError("Breeze voice cloning requires ref_text")
            self._build_prefix(obj, voice, ref_audio, ref_text)
            suffix = original(obj, text, voice=voice, instruct=instruct,
                              ref_audio=None, ref_text=None)
            if self.kv is not None:
                self._pending += 1        # the next make_cache() gets the prefix KV
                return suffix
            import mlx.core as mx
            return mx.concatenate((self.prefix, suffix), axis=1)
        cls._prompt_embeddings = cached

        def make_cache():
            if self._pending > 0 and self.kv is not None:
                self._pending -= 1
                return self._seeded_caches()
            return self._original_make_cache()
        target.backbone_model.make_cache = make_cache

    def close(self):
        if self.owned:
            type(self.model)._prompt_embeddings = self.original
        else:
            delattr(type(self.model), "_prompt_embeddings")
        try:
            del self.model.backbone_model.make_cache      # back to the class method
        except AttributeError:
            pass
        self._pending = 0
        self._installed = False

    def verify(self, text, *, voice, instruct, ref_audio, ref_text, tol=5e-2):
        """Compare the last hidden state of a full prefill against prefix-KV +
        suffix prefill. Returns (ok, rel_err). On mismatch, disables prefix KV so
        generation silently falls back to the release path."""
        import mlx.core as mx
        if not self.prefix_kv:
            return True, 0.0
        self.install()
        try:
            obj = self.model
            self._build_prefix(obj, voice, ref_audio, ref_text)
            suffix = self.original(obj, text, voice=voice, instruct=instruct,
                                   ref_audio=None, ref_text=None)
            full = mx.concatenate((self.prefix, suffix), axis=1)
            h_full = obj.backbone_model(input_embeddings=full,
                                        cache=self._original_make_cache())[:, -1, :]
            h_fast = obj.backbone_model(input_embeddings=suffix,
                                        cache=self._seeded_caches())[:, -1, :]
            a, b = h_full.astype(mx.float32), h_fast.astype(mx.float32)
            err = float(mx.abs(a - b).max() / (mx.abs(a).max() + 1e-6))
            ok = err <= tol
            if not ok:
                self.prefix_kv, self.kv, self.prefix_len = False, None, 0
            return ok, err
        finally:
            self.close()

class FrameKVCache:
    """Small append-only cache, newly allocated for EACH acoustic frame.

    At most num_codebooks entries; no 256-token capacity allocation is needed.
    Used with the pinned MLX-Audio Llama attention's offset/update_and_fetch API.
    """
    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        import mlx.core as mx
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate((self.keys, keys), axis=2)
            self.values = mx.concatenate((self.values, values), axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

class FastDepth:
    """Cache depth attention and defer host reads to the end of each frame.

    This leaves the text encoder, backbone generation, EOS, and streaming audio
    codec in the release implementation. Sampling uses the release's sampler.
    Optional compilation captures random state explicitly. Cached is the default:
    compiled full-frame graphs may take time to compile on a particular Mac.
    """
    def __init__(self, model, mode="cached"):
        self.model, self.mode = model, mode
        self.functions = {}
        self.original = None

    def initial(self, first, hidden):
        import mlx.core as mx
        m = self.model.depth_decoder.model
        if m.backbone_hidden_state_projector is not None:
            hidden = m.backbone_hidden_state_projector(hidden)
        ids = mx.broadcast_to(first.reshape(1, 1), (hidden.shape[0], 1))
        embeds = mx.concatenate((hidden[:, None, :], m.embed_tokens(ids)), axis=1)
        x = m.inputs_embeds_projector(embeds)
        caches = [FrameKVCache() for _ in m.layers]
        for layer, cache in zip(m.layers, caches):
            x = layer(x, "causal", cache)
        return x, caches

    def advance(self, token, codebook_index, caches, batch):
        import mlx.core as mx
        m = self.model.depth_decoder.model
        ids = mx.broadcast_to(token.reshape(1, 1), (batch, 1))
        # At absolute depth position p>=1, upstream uses offset (p-1)*vocab.
        embeds = m.embed_tokens(ids + codebook_index * m.vocab_size)
        x = m.inputs_embeds_projector(embeds)
        for layer, cache in zip(m.layers, caches):
            x = layer(x, None, cache)
        return x

    def logits(self, x, head):
        d = self.model.depth_decoder
        return d.model.norm(x[:, -1, :]) @ d.codebooks_head.weight[head]

    def _function(self, temperature, top_p, top_k, cfg_scale, use_cfg):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_audio.lm.sample_utils import make_sampler

        key = (temperature, top_p, top_k, cfg_scale, use_cfg)
        if key in self.functions:
            return self.functions[key]
        valid = self.model.vocab_size
        effective_k = min(top_k, valid) if top_k else 0
        if effective_k == valid:
            effective_k = 0
        sample = make_sampler(temp=temperature, top_p=top_p, top_k=effective_k)

        def frame(first, hidden):
            x, caches = self.initial(first, hidden)
            tokens = [first.reshape(1)]
            for head in range(self.model.num_codebooks - 1):
                scores = self.logits(x, head)
                if use_cfg:
                    scores = scores[1:2] + cfg_scale * (scores[:1] - scores[1:2])
                scores = self.model._mask_reserved_codec_logits(scores)[..., :valid]
                token = sample(nn.log_softmax(scores, axis=-1)).astype(mx.int32).reshape(1)
                tokens.append(token)
                if head + 1 < self.model.num_codebooks - 1:
                    x = self.advance(token, head + 1, caches, hidden.shape[0])
            return mx.concatenate(tokens)

        fn = (mx.compile(frame, inputs=mx.random.state, outputs=mx.random.state)
              if self.mode == "compiled" else frame)
        self.functions[key] = fn
        return fn

    def generate_frame(self, first_codebook, conditional_hidden, *,
                       unconditional_hidden, cfg_scale, temperature, top_p, top_k):
        import mlx.core as mx
        if conditional_hidden.shape[0] != 1:
            raise ValueError("FastDepth currently supports one utterance at a time.")
        if self.model.num_codebooks != self.model.depth_decoder.model.num_codebooks:
            raise ValueError("Wrapper/depth codebook counts do not match.")
        hidden = conditional_hidden
        use_cfg = unconditional_hidden is not None
        if use_cfg:
            hidden = mx.concatenate((hidden, unconditional_hidden), axis=0)
        fn = self._function(temperature, top_p, top_k, cfg_scale, use_cfg)
        codes = fn(mx.array([first_codebook], dtype=mx.int32), hidden)
        # One host read for all remaining codebooks instead of .item() per code.
        return codes.tolist()

    def install(self):
        if self.mode == "native":
            return
        target = self.model
        cls = type(target)
        original = cls._depth_tokens
        self.original = original
        self.owned_method = "_depth_tokens" in cls.__dict__

        def optimized(obj, *args, **kwargs):
            if obj is target:
                return self.generate_frame(*args, **kwargs)
            return original(obj, *args, **kwargs)

        cls._depth_tokens = optimized

    def close(self):
        if self.original is not None:
            if self.owned_method:
                type(self.model)._depth_tokens = self.original
            else:
                delattr(type(self.model), "_depth_tokens")
            self.original = None

    def warmup(self, frames, cfg_scale, instruct):
        import mlx.core as mx
        m = self.model.depth_decoder.model
        try:
            # Shape/dtype match the projected backbone output. This warms depth
            # only; it is NOT a full text/codec warmup and is reported separately.
            hidden = mx.zeros((1, m.backbone_hidden_size),
                              dtype=self.model.depth_decoder.codebooks_head.weight.dtype)
            for _ in range(frames):
                self.model._depth_tokens(
                    1, hidden, unconditional_hidden=hidden if instruct and cfg_scale != 1 else None,
                    cfg_scale=cfg_scale, temperature=0.9, top_p=1.0, top_k=50)
        finally:
            mx.random.seed(0)  # generate(seed=args.seed) reseeds each measured run.
