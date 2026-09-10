"""Reuse BPE's immutable vocabulary, never its per-generation text state."""
from copy import copy

def cache_bpe_vocabulary(tokenizer):
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

    # Do not alter custom/unknown decoders or global library classes.
    if not isinstance(tokenizer, TokenizerWrapper):
        return False
    if tokenizer._detokenizer_class is not BPEStreamingDetokenizer:
        return False
    prototype = tokenizer.detokenizer
    prototype.tokenmap = tuple(prototype.tokenmap)

    def fresh(owner):
        if owner is not tokenizer:
            return BPEStreamingDetokenizer(owner)
        decoder = copy(prototype)
        decoder.reset()  # new tokens list, offset, text and pending UTF-8 bytes
        return decoder

    tokenizer._detokenizer_class = fresh
    return True
