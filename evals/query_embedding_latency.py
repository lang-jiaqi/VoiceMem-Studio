"""CPU-only, offline E5 A/B: duplicate query encoding vs retrieval-scoped reuse.

This measures encoding only, not classification + complete memory retrieval.
Does not open/write any memory database or load a GPU model.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    from voicemem.leftbrain.query_embedding import encode_query, query_embedding_scope
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(ROOT / "models/embedding"), device="cpu", local_files_only=True)
    queries = ["我上次说的那个项目怎么样了", "我对什么过敏", "我朋友里谁是做设计的",
               "我最近是不是有点累", "我在哪家公司上班"]
    encode_query(model, "query: " + queries[0])
    old, new = [], []
    for _ in range(2):
        for query in queries:
            text = "query: " + query
            start = time.perf_counter()
            first, second = encode_query(model, text), encode_query(model, text)
            old.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            with query_embedding_scope():
                reused_first, reused_second = encode_query(model, text), encode_query(model, text)
            new.append((time.perf_counter() - start) * 1000)
            np.testing.assert_array_equal(first, reused_first)
            np.testing.assert_array_equal(second, reused_second)
    print(f"device=cpu torch_threads=2 n={len(old)} "
          f"duplicate_encoding_median_ms={statistics.median(old):.1f} "
          f"reused_encoding_median_ms={statistics.median(new):.1f} "
          "vectors=identical", flush=True)


if __name__ == "__main__":
    main()
