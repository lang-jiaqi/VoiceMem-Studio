import os
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
)

PROMPT = "用两句话介绍杭州。"
ROUNDS = 5

def one_round(verbose=True):
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model="qwen3.6-flash",
        messages=[
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": PROMPT},
        ],
        extra_body={"enable_thinking": False},
        stream=True,
    )

    ttft = None
    chars = 0
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if not piece:
            continue
        if ttft is None:
            ttft = time.perf_counter() - t0
        chars += len(piece)
        if verbose:
            print(piece, end="", flush=True)

    total = time.perf_counter() - t0
    if verbose:
        print()
    return ttft, total, chars


print("--- warmup（不计入统计）---")
w_ttft, _, _ = one_round(verbose=False)
print(f"warmup TTFT: {w_ttft*1000:.0f} ms")

results = []
for i in range(ROUNDS):
    print(f"\n--- 第 {i+1} 轮 ---")
    ttft, total, chars = one_round()
    tps = chars / (total - ttft) if total > ttft else 0
    print(f"TTFT: {ttft*1000:.0f} ms | 总耗时: {total*1000:.0f} ms | {chars} 字 | 首字后 {tps:.1f} 字/秒")
    results.append(ttft)

results.sort()
n = len(results)
p50 = results[n // 2]
p90 = results[min(int(n * 0.9), n - 1)]
print(f"\n=== {ROUNDS} 轮 TTFT（已排除 warmup）===")
print(f"最快 {results[0]*1000:.0f} | P50 {p50*1000:.0f} | P90 {p90*1000:.0f} | 最慢 {results[-1]*1000:.0f} ms")
print(f"warmup 比 P50 多花了 {(w_ttft - p50)*1000:.0f} ms（建连开销）")