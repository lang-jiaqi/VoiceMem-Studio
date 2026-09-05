"""换 embedding 之前的正式对照：LoCoMo 上比检索命中率。

``embed_ab.py`` 只能回答"两个模型给的结果一样吗"，回答不了"谁更准"——因为自己的
记忆库没有标准答案。LoCoMo 有：每道题标了 ``evidence``（哪几轮对话是答案依据）。
所以这里不需要 LLM 裁判，也不需要 API key，直接量**依据那一轮排进前几名**。

    对话的每一轮 = 一条"记忆"
    题目          = 查询
    evidence      = 标准答案（dia_id）
    指标          = recall@1 / recall@5 / MRR

category 5 是对抗题（问的东西对话里根本没有），没有 evidence，跳过。

数据：https://github.com/snap-research/locomo → data/locomo10.json

    python3 evals/embed_locomo.py <locomo10.json> [第几段] [模型a] [模型b]
    python3 evals/embed_locomo.py data/locomo10.json 0 e5 bge-en
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_conv(sample: dict):
    """一段对话 → ({dia_id: 文本}, [题目]).

    每轮拼上说话人：LoCoMo 是双人对话，"我搬去杭州了"是谁说的直接决定答案对不对。
    """
    conv = sample.get("conversation", {})
    turns: dict[str, str] = {}
    for key in sorted(k for k in conv if k.startswith("session_")
                      and not k.endswith("date_time")):
        for t in conv[key] or []:
            did = t.get("dia_id") or ""
            text = (t.get("text") or "").strip()
            if did and text:
                turns[did] = f"{t.get('speaker', '')}: {text}"
    qs = [q for q in sample.get("qa", [])
          if q.get("evidence") and q.get("category") != 5]
    return turns, qs


def evaluate(key: str, turns: dict, qs: list, lang: str = "en"):
    import numpy as np
    from voicemem.leftbrain.local_embedder import resolve, resolve_path, shared_model
    spec = resolve(key, language=lang)
    m = shared_model(resolve_path(spec))
    ids = list(turns)
    D = np.asarray(m.encode([f"{spec.passage_prefix}{turns[i]}" for i in ids],
                            normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False))
    Q = np.asarray(m.encode([f"{spec.query_prefix}{q['question']}" for q in qs],
                            normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False))
    order = np.argsort(-(Q @ D.T), axis=1)

    r1 = r5 = 0
    mrr = 0.0
    per_q = []
    for i, q in enumerate(qs):
        gold = {e for e in q["evidence"] if e in turns}
        if not gold:
            per_q.append(0)                       # evidence 指向的轮次不在这段里
            continue
        ranked = [ids[j] for j in order[i][:20]]
        hit = next((r + 1 for r, d in enumerate(ranked) if d in gold), 0)
        r1 += hit == 1
        r5 += 0 < hit <= 5
        mrr += 1.0 / hit if hit else 0.0
        per_q.append(hit)
    n = len(qs)
    return {"model": spec, "r1": r1 / n, "r5": r5 / n, "mrr": mrr / n,
            "per_q": per_q, "n": n}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/locomo10.json"
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    a = sys.argv[3] if len(sys.argv) > 3 else "e5"
    b = sys.argv[4] if len(sys.argv) > 4 else "bge-en"

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [{**v, "sample_id": k} for k, v in data.items()]
    sample = data[idx]
    turns, qs = load_conv(sample)
    print(f"{sample.get('sample_id', idx)}：{len(turns)} 轮对话，{len(qs)} 道有依据的题\n")

    ra, rb = evaluate(a, turns, qs), evaluate(b, turns, qs)
    print(f"{'模型':<10}{'仓库':<34}{'维度':>5}{'recall@1':>10}{'recall@5':>10}{'MRR':>8}")
    for r in (ra, rb):
        s = r["model"]
        print(f"{s.key:<10}{s.repo:<34}{s.dims:>5}"
              f"{r['r1']:>10.1%}{r['r5']:>10.1%}{r['mrr']:>8.3f}")

    win = sum(x and (not y or x < y) for x, y in zip(ra["per_q"], rb["per_q"]))
    lose = sum(y and (not x or y < x) for x, y in zip(ra["per_q"], rb["per_q"]))
    print(f"\n逐题比：{a} 排得更前 {win} 题，{b} 更前 {lose} 题，"
          f"其余 {ra['n'] - win - lose} 题打平")


if __name__ == "__main__":
    main()
