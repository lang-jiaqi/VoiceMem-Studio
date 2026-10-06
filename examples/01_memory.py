"""Demonstrate audio ingestion, text ingestion and memory search.

    export OPENAI_API_KEY=sk-...
    python examples/01_memory.py

Embeddings and slot classification use local models. Fact extraction during
ingestion uses the configured API key. This SDK example runs independently of Studio.
"""
import os
from pathlib import Path

from voicemem import VoiceMem

# Resolve the sample path independently of the working directory.
AUDIO = str(Path(__file__).resolve().parent.parent / "assets/input.wav")

LOCAL = {
    "embedding": {"provider": "local"},
    "slots": {"provider": "local"},
    "api_key": os.environ["OPENAI_API_KEY"],
    # Keep this local-embedding store separate from the default Space.
    # Changing embedding models requires re-embedding existing vectors.
    "memory_root": str(Path(__file__).resolve().parent / "example_memory"),
}

vm = VoiceMem.from_config({**LOCAL, "mode": "normal"})

# Load local capabilities before processing the first recording.
vm.warmup(verbose=True)

vm.ingest(audio=AUDIO)

result = vm.search("我的饮食禁忌是什么？", top_k=5)

print(result.result_leftbrain, result.result_rightbrain)


# Text-only ingestion stores facts without acoustic perception.
vm = VoiceMem.from_config({**LOCAL, "mode": "leftbrain_only"})

vm.ingest("我是素食主义者，对坚果过敏。")

result = vm.search("我的饮食禁忌是什么？", top_k=5)

print(result.result_leftbrain, result.result_rightbrain)
