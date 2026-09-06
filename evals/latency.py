"""全链路时延分解：一句话从说完到出声，时间花在哪儿。

分两类，不能混着看：

    说话时并行跑的   ASR partial / EOT / 闸门 / 检索 —— 藏在用户说话的时间里，
                     再慢也不占用户的等待（只要不超过说话时长）
    说完之后串行跑的  复核 ASR → LLM 首字 → TTS 首帧 → 前端缓冲 —— 这条链的总和
                     才是用户听到的等待

跑：python3 evals/latency.py
"""
import asyncio
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEXT = "I want to find a quiet place to study on weekends."


async def main():
    from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
    from voicemem.tts import OpenAITTS, cut_point
    from voicemem.utils.audio.eot import EndOfTurn
    from voicemem.utils.audio.stream_io import read_wav, resample
    from voicemem.utils.defaults import default_utils

    cache = Path("results/test_audio/quiet.wav")
    if not cache.is_file():
        print("先跑一次 tests/test_voice_pipeline.py 生成测试音频")
        return
    a24, _ = read_wav(cache)
    a16 = resample(a24, src=24000)
    root = "voicemem_memoryspace/demo"
    u = default_utils(None, root)
    rows = []

    def take(label, fn, n=5, group="说话时并行"):
        xs = []
        for _ in range(n):
            t = time.perf_counter(); fn(); xs.append((time.perf_counter() - t) * 1000)
        rows.append((group, label, st.median(xs)))

    asr = u["asr"]()
    step = int(0.043 * 16000)
    take("流式 ASR（每帧 43ms）", lambda: asr.feed(a16[:step]))
    fin = u["asr_final"]()
    take("离线复核 ASR（整轮）", lambda: fin.transcribe(a16), n=3, group="说完之后串行")
    eot = EndOfTurn()
    take("EOT 打分（每 40ms 一次）", lambda: eot.score(a16))
    from voicemem import gate
    take("闸门判深浅", lambda: gate.route("我上次说的那个项目呢"))
    slots = u["slots"]()
    take("slot 分类", lambda: slots.classify(TEXT))

    # 检索：用真实库
    from voicemem.core import VoiceMem
    vm = VoiceMem.from_config({"mode": "text_mode", "space": "demo",
                               "embedding": {"provider": "local"},
                               "slots": {"provider": "local"}})
    c = vm.classify(TEXT)
    take("记忆检索（左脑+右脑）", lambda: vm.search(TEXT, slots=c.slots, entities=c.entities))

    # LLM 首字 + TTS 首帧
    from openai import AsyncOpenAI
    from voicemem.llm_config import resolve_model
    cli = AsyncOpenAI()
    llm, tts_ms = [], []
    for _ in range(3):
        t = time.perf_counter(); buf = ""; seg = None
        s = await cli.chat.completions.create(
            model=resolve_model(role="reply", default="gpt-4o"), stream=True,
            messages=[{"role": "system", "content": "你是语音助手，简短自然地回答。"},
                      {"role": "user", "content": TEXT}])
        async for ch in s:
            d = ch.choices[0].delta.content
            if not d:
                continue
            if not buf:
                llm.append((time.perf_counter() - t) * 1000)
            buf += d
            if cut_point(buf, first=True):
                seg = buf.strip(); break
        t = time.perf_counter()
        async for _c in OpenAITTS(voice="alloy").stream(seg or "好的"):
            tts_ms.append((time.perf_counter() - t) * 1000); break
    rows.append(("说完之后串行", "LLM 首字", st.median(llm)))
    rows.append(("说完之后串行", "TTS 首帧", st.median(tts_ms)))
    rows.append(("说完之后串行", "前端预缓冲", 160.0))

    for g in ("说话时并行", "说完之后串行"):
        print(f"\n【{g}】")
        tot = 0.0
        for gg, label, ms in rows:
            if gg != g:
                continue
            tot += ms
            print(f"  {label:24}{ms:8.1f} ms")
        if g == "说完之后串行":
            print(f"  {'—' * 24}{'—' * 11}\n  {'合计（用户听到的等待）':24}{tot:8.1f} ms")

if __name__ == "__main__":
    asyncio.run(main())
