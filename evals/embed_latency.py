"""本地 embedding 的编码延迟：塞不塞得进投机预取那 0–300ms。

检索质量再好，编码本身超预算就没意义——那 300ms 是「用户说完到助手开口」之间的
全部时间，编码只是其中一步（还要 slot 分类 + 向量检索 + 右脑）。

每轮实际要编码两次短文本（slot 分类一次、记忆检索一次），所以看的是 **单条 query
的中位延迟 ×2**。

    python3 evals/embed_latency.py            # 默认比 e5 / bge-en / qwen
    python3 evals/embed_latency.py e5 qwen
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERIES = ["我明天有什么安排", "我上次说的那个项目怎么样了", "我对什么过敏",
           "我朋友里谁是做设计的", "我最近是不是有点累", "我在哪家公司上班"]
#: 入库那侧是批量编码，跟查询不是一回事，单独量。
DOCS = ["用户对花生过敏，吃了会起疹子"] * 64


def bench(key: str):
    from voicemem.leftbrain.local_embedder import resolve, resolve_path, shared_model
    spec = resolve(key)
    t0 = time.time()
    m = shared_model(resolve_path(spec), spec.tokenizer_kwargs)
    load = time.time() - t0

    enc = lambda xs, pre: m.encode([f"{pre}{x}" for x in xs], normalize_embeddings=True)
    enc(QUERIES[:1], spec.query_prefix)               # 预热，别把首次的图构建算进去

    lat = []
    for _ in range(3):
        for q in QUERIES:
            t = time.time(); enc([q], spec.query_prefix); lat.append((time.time() - t) * 1000)

    t = time.time(); enc(DOCS, spec.passage_prefix); batch = (time.time() - t) * 1000
    return spec, load, statistics.median(lat), batch


def main():
    keys = sys.argv[1:] or ["e5", "bge-en", "qwen"]
    print(f"{'模型':<9}{'维度':>5}{'加载':>8}{'单条 query':>12}{'每轮×2':>9}"
          f"{'64条批量':>10}   预算内?")
    print("─" * 68)
    for k in keys:
        try:
            spec, load, med, batch = bench(k)
        except Exception as e:
            print(f"{k:<9} 跑不了：{type(e).__name__}: {e}")
            continue
        per_turn = med * 2
        verdict = "✓" if per_turn < 100 else ("紧张" if per_turn < 300 else "✗ 超预算")
        print(f"{spec.key:<9}{spec.dims:>5}{load:>7.1f}s{med:>11.1f}ms"
              f"{per_turn:>8.0f}ms{batch:>9.0f}ms   {verdict}")
    print("\n判据：投机预取总共 0–300ms，编码只是其中一步（还有 slot 分类、向量检索、"
          "右脑）。\n每轮 ×2 超过 100ms 就开始挤占其他步骤，超过 300ms 是纯粹放不下。")


if __name__ == "__main__":
    main()
