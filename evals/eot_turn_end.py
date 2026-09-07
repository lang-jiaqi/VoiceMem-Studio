"""EOT 能不能取代掐表：真实回合末的分数 vs 尾部静音长度。

线上 confirm_s=200ms 一直没被 EOT 取代（实测 EOT 只在兜底掐表时才结束回合）。
这个脚本量的是原因：``EndOfTurn._frame`` 把音频切到「语音结束 + 200ms」，可回合
**还没结束时缓冲区里的静音不足 200ms**，模型看到的取景跟它训练时不一样。

所以对每一段真实回合录音，量两件事：

    原样      在静音 d 毫秒处直接问（= 线上现在的做法，取景短了 200-d ms）
    补齐      把尾部补零到 200ms 再问（模型拿到规范取景）

再拿句中停顿当负样本，看补齐会不会把误切率也一起抬上去。

跑：python3 evals/eot_turn_end.py [样本数]
"""
import sys
import statistics as st
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000
OFFSETS = [0, 40, 80, 120, 160, 200]     # 静音多久时问（ms）
LEVEL = 0.01                              # 跟 eot.py 的能量阈值一致
PAUSE_MIN_S = 0.12                        # 句中停顿至少这么长才当负样本


def speech_bounds(a):
    idx = np.nonzero(np.abs(a) > LEVEL)[0]
    return (int(idx[0]), int(idx[-1])) if idx.size else (0, 0)


def pauses(a, end):
    """句中停顿的结束位置（负样本：在这里切就是把人的话切断了）。"""
    loud = np.abs(a[:end]) > LEVEL
    step = SR // 50                                   # 20ms 一格
    grid = np.array([loud[i:i + step].any() for i in range(0, len(loud) - step, step)])
    out, run = [], 0
    for i, v in enumerate(grid):
        if not v:
            run += 1
            continue
        if run * 0.02 >= PAUSE_MIN_S and i * step > SR:   # 头 1 秒不算
            out.append(i * step)
        run = 0
    return out


def main():
    from voicemem.utils.audio.eot import EndOfTurn
    from voicemem.utils.audio.stream_io import read_wav, resample

    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    files = sorted(Path("results/turn_audio").glob("turn_*.wav"))
    files = files[:: max(1, len(files) // n_want)][:n_want]
    eot = EndOfTurn()
    pad = lambda a, need: np.pad(a, (0, need)) if need > 0 else a

    raw = {d: [] for d in OFFSETS}       # 回合末，原样问
    fix = {d: [] for d in OFFSETS}       # 回合末，补齐到 200ms
    neg_raw, neg_fix, neg_gap, kept = [], [], [], 0
    for f in files:
        a, sr = read_wav(f)
        if sr != SR:
            a = resample(a, src=sr)
        if len(a) < SR * 0.5:
            continue
        s0, s1 = speech_bounds(a)
        if s1 - s0 < SR * 0.4:
            continue
        kept += 1
        for d in OFFSETS:
            n = int(d / 1000 * SR)
            seg = a[:min(len(a), s1 + n)]
            raw[d].append(eot.score(seg))
            fix[d].append(eot.score(pad(seg, int(0.2 * SR) - n)))
        for p in pauses(a, s1)[:2]:                   # 每段最多取 2 个句中停顿
            seg = a[:p]
            neg_raw.append(eot.score(seg))
            neg_fix.append(eot.score(pad(seg, int(0.2 * SR))))
            neg_gap.append((s1 - p) / SR)

    def line(tag, xs, thr):
        hit = sum(x >= thr for x in xs) / len(xs) * 100
        return f"{tag} 中位 {st.median(xs):.2f}  ≥{thr} 占 {hit:4.1f}%"

    print(f"\n真实回合 {kept} 段，句中停顿 {len(neg_raw)} 个\n")
    if neg_gap:
        print("句中停顿离回合末的距离（秒）分位：",
              [f"{np.quantile(neg_gap, q):.2f}" for q in (.1, .25, .5, .75, .9)])
        far = [(s, g) for s, g in zip(neg_raw, neg_gap) if g >= 1.0]
        if far:
            print(f"其中离末尾 ≥1s 的 {len(far)} 个：中位 {st.median([s for s,_ in far]):.2f}")
        print()
    for thr in (0.4, 0.5, 0.6):
        print(f"── 阈值 {thr} ──")
        for d in OFFSETS:
            print(f"  静音 {d:3d}ms   原样: {line('', raw[d], thr)}"
                  f"   补齐: {line('', fix[d], thr)}")
        if neg_raw:
            print(f"  句中停顿（越低越好）  原样: {line('', neg_raw, thr)}"
                  f"   补齐: {line('', neg_fix, thr)}")
        print()


if __name__ == "__main__":
    main()
