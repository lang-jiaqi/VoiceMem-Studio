# 交接：本地语音链路的延迟与稳定性

2026-09-06。开新会话时把这份给我，我能直接接上。

## 一句话现状

本地 qwen 4B + 本地 Qwen-TTS，**说完到出声 1250ms**（今天从 2300ms 压下来的）。
不再崩机。ASR 按空间语言分流后中文转写正常。

## 怎么起

```bash
cd web
TTS_BACKEND=qwen VOICEMEM_LLM_DEBUG=1 VOICEMEM_BARGE_DEBUG=1 \
VOICEMEM_BACKCHANNEL_EMIT=1 VOICEMEM_FINAL_ASR=0 VOICEMEM_EOT_ENDS_TURN=1 \
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 run.py \
  --mode llm_tts --llm local --confirm_ms 200 2>&1 | tee /tmp/vm.log
```

> 系统默认的 `python` 是 anaconda，**没有 mlx**，必须用上面那个绝对路径。
> 开服要等 6~7 秒（把人设灌进 KV 母本），看到 `[web] 就绪` 再连。
>
> **日志重定向到文件，别整段贴进对话**——emotion2vec 每次加载吐几百行
> `init param, map: blocks.7...`，贴进来会把上下文挤爆、让我变笨。只说
> "看 /tmp/vm.log"，我自己 grep `[lat]` `[llm]` `[early]` 这几行。

## 延迟分解（稳态，中文短句）

```
闭嘴→出声 1250ms
  ＝ 判说完+复核 310   VAD 等静音确认（--confirm_ms）+ 离线 ASR 复核
  +  LLM 首字   490   210 起 stream_generate 的固定开销 + 尾巴 28 token 预填
  +  攒第一段   200   解码语气标签 + 首句约 7 个 token（28ms/token）
  +  TTS 首帧   190   Qwen-TTS 出第一块音频
```

这台机器的常数（实测）：prefill **2.4ms/token**，decode **28ms/token**（≈36 tok/s），
`stream_generate` 起一次固定 **210ms**，Qwen-TTS 热首帧 **115~200ms**。

## 关键设计

### KV 母本分两级（`voicemem/local_llm.py`）
prompt 是 `[人设][历史][记忆 + 这句话]`。前缀每轮都一样，所以只算一次、常驻。

- **人设那级永不失效**（开服建，1265 token 中文 / 1569 英文）
- **历史那级**每轮空闲时续；被误打断和换说话人冲乱时，退回人设那级

生成时 `_fork` 复制母本——**只换容器不拷数组**。真拷 75MB，几轮就 Metal OOM。

填充时机（越早填越少等）：

| 时机 | 填什么 |
|---|---|
| 开服 | 人设 |
| 每轮空闲 | 上一轮的历史 |
| **说话中** | 这一轮投机检索到的记忆 |
| 说完 | 只剩那句话，28 token |

### 单条 GPU 流（`voicemem/utils/gpu_loop.py`）
**这是防崩机的，不能拆。** 两个线程并发往同一块 GPU 提交命令缓冲会触发 Apple
AGX 驱动的引用计数下溢，直接 kernel panic + 重启（2026-09-06 一天崩三次，
`completeMemory() prepare count underflow` @IOGPUMemory.cpp:492）。

MLX 的流是线程本地的，没法共享流对象，只能共享**一个拥有流的线程**。LLM 前向和
TTS 合成都排它，轮询交替推进（TTS 权重 4、LLM 权重 1）。

**取消是必须的**：赌错的提前生成被丢弃后，生成器仍会跑满 512 token，真正那轮排在
后面干等 400~1300ms。

## 已知没解决

1. **EOT 对升调疑问句判不准**——实测只有 0.38~0.48，够不到 0.6 的线，这类只能退回
   计时器。陈述句是 0.97。文本兜底（`？`/`吗` 结尾）加了但用不上：流式 ASR 要到
   flush 才补标点，下注那一刻文本里还没有。
   能试的旋钮：`VOICEMEM_EARLY_EOT=0.35`（多赌几次，赌错代价已经不大了）。

2. **缺 torchvision**，情绪归因一直没真正启用，右脑的情感记忆不可信。
   `pip install torchvision`。

3. **中英混说会翻车**。一个空间一种语言（ASR 模型只认一种）。要混说得
   `VOICEMEM_ASR=bilingual`，但那档实测会把英文吐成中文。

4. **后台让路的闸没在真环境验证过**。机制在（`wait_idle` / `hot_path_enter`），但我的
   测试客户端不回报播放进度，`wait_playback_done()` 只能等超时，读数失真。
   实机看 `[idle] … 让路 Xms（热路径计数 N）`，正常该是几百毫秒、计数 0。

## 日志怎么读

```
[llm] 母本人设底座 1265 token          开服预热
[llm] 母本续 1265+40 token → 共 1305   空闲时续历史
[llm] prompt 1792｜命中缓存 1764｜要算 28    ← 要算的越少越好
[llm] 首字 490ms（排队等 GPU 线程 1 / 渲染+分词 3 / fork 0）  ← 排队该是个位数
[seg] 第一段 24 字（词边界切）
[lat] 闭嘴→出声 1250ms ＝ 判说完+复核 310 + LLM首字 490 + 攒第一段 200 + TTS首帧 190
[early] 本轮 EOT 最高 0.97（下注线 0.5），下过注
```

出现 `没命中｜要算 1300` 就是缓存被冲垮了，去看前缀在第几个 token 分叉。

## 踩过的坑，别再踩

- **别对 run.py 做盲字符串替换**。`s[a:b]` 取到空串再 `replace('')` 会在每个字符
  之间插一遍，文件从 3900 行涨到 62 万行。
- **MLX 的英文 token 是空格开头的**（`" have"` 不是 `"have "`）。任何"结尾是不是
  空格"的判断都不会成立。
- **切段前先 `strip()` 会抹掉结尾空格**，词边界判断因此恒假。
- **模型必须在用它的那个线程里加载**，否则 `There is no Stream(gpu, 0) in current
  thread`。
- **换人设要把两级母本都丢掉**，不丢会拿旧人设的 KV 生成——那不是慢，是答错。
