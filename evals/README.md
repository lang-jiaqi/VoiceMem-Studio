# 评测脚本

这里保留手动运行的延迟、检索效果和模型质量评测。自动回归测试与运行方法见 [tests/README.md](../tests/README.md)。

| 评测范围 | 脚本 |
| --- | --- |
| 对话和语音延迟 | `latency.py`、`deepseek_latency.py`、`breeze_cuda_latency.py` |
| 本机推理调度竞争 | `engine_contention.py`（Apple Silicon，离线 ABBA 对照） |
| MLX 深度解码编译 | `mlx_tts_decode.py`（Apple Silicon，缓存/校验编译 ABBA 对照） |
| 分块预填充 | `mlx_prefill.py`（Apple Silicon，合成小模型，原生调度和缓存数值对照） |
| 稳定输入的 EOT 计算复用 | `eot_stable_input.py`（CPU，校验追加静音前后分数逐值一致） |
| 回合结束与打断 | `eot_turn_end.py`、`echo_barge.py` |
| 向量编码与检索 | `embed_latency.py`、`query_embedding_latency.py`、`embed_ab.py`、`embed_locomo.py` |
| 检索排序和相关性 | `topk_relevance.py`、`recency_ab.py`、`recency_rerank.py` |
| 路由质量 | `router_quality.py`、`turn_gate.py` |
| 标签维护对比 | `retag_llm.py` |

每个脚本文件顶部说明其参数和所需数据。它们可能需要模型、API、音频样本或专用 Memory Space，并非全部可以在干净环境中直接运行。不要使用个人正在使用的记忆库做实验；运行会写入数据的模式前，先阅读对应参数说明。

这里的评测用于观察效果与性能。自动测试通过不能替代真实设备上的语音和延迟测试。

`python -m evals.work_filler_cache --out results/work-filler-cache.json`
使用当前推荐的本地 TTS 预合成中英文等待语，检查完整音频、磁盘复用和内存命中。
该脚本会写入忽略的等待语音缓存，但不会播放、录音或调用 API。

`python evals/engine_contention.py --studio-voice --order ABBAABBA --utterances 4 --background-jobs 4`
在同一进程交替关闭（A）和开启（B）动态后台准入，保持模型和合成参数一致。
它使用临时合成输入，同时运行 ASR、EOT、TTS 和串行后台感知，输出生产间隔、
ASR 耗时、事件循环延迟、后台完成时间和真实换页速率。模型需要提前安装；
不请求 API，不读写 Memory Space，不采集或播放音频，也不保存合成语音。
机器温度和其他进程仍会影响结果；该评测不能替代浏览器播放和真人打断测试。

`python -m evals.engine_contention --capacity-crossover --background-workers 2 --order ABBAABBA --utterances 2`
单独对照后台原生容量控制：A 使用原有语音准入，B 在相同策略前增加跨模型容量约束。
两组使用相同的模型、线程设置和解码参数；每组都等待后台流水线完成再切换策略。
该模式拒绝同时修改线程数、解码方式或关闭原有语音准入，避免多个变量混在一起。
每句输出 PCM 的 SHA-256 用于校验样本一致性，不保存音频；`capacity_ms` 单独记录
容量排队时间。运行时应暂停其他评测和单测。结果只能说明原生竞争下的生产表现，
不能据此认定浏览器断音、端到端回复延迟或持续发热已经改善。

`python -m evals.engine_contention --reuse-crossover --embedding-repeats 4 --order ABBAABBA --utterances 2 --depth-a compiled --depth-b compiled`
对照单个后台任务内的精确 embedding 复用：两组使用相同语音准入、后台容量和解码配置，
A 重算同一合成批次，B 在任务内复用。报告实际 encoder 调用数、向量摘要、TTS 音频摘要、
ASR 耗时及后台完成时间。重复次数是显式测试条件，不能据此推断真实入库的命中率，
也不能替代长时间真人播放、实际记忆队列或发热测试。

`python -m evals.mlx_tts_decode --with-asr --order ABBAABBA --utterances 3`
单独对照 Breeze 深度解码：A 使用缓存路径，B 使用启动时校验过输出和随机状态的
整帧编译路径。两组保留相同模型、采样、音频批次及同一个 MLX 执行线程。
可选流式识别使用内存中生成的同一条语音，默认在 MPS 上按 600 ms 节奏输入；
`--asr-device cpu` 可独立测 CPU 场景。输出包含实际编译校验状态、PCM 与识别结果
摘要、实际编译调用数、首帧、生产间隔、RTF、CPU 累计时间、原生耗时、
MLX 活跃/缓存/峰值内存和系统热状态分类。
该脚本不读取个人录音或音色参考，不保存合成音频，也不打开记忆库；
语音样本的一致性不等于所有输入的数值证明，短时生产测试也不能证明持续降温。

环境记录包含 FunASR 的 `cpu_threads`；`--fixed-policy B --threads-a 4 --threads-b 2`
可保持语音准入不变，单独对照线程预算。每组的 `process_cpu_s` 是主进程及其线程
累计 CPU 时间，不包含独立 speaker 子进程，也不是功耗。系统摘要持续记录可用的
`thermal_state`，包括空闲期；该值是操作系统热状态分类，不是温度或实际时钟频率。

`python -m evals.mlx_prefill --order ABBAABBA` 使用固定种子的小型 Qwen3.5 架构，
交替对比整段与分块预填充，校验循环状态、注意力缓存和后续贪心 token，记录单步占用
及竞争计算的等待。无需预训练权重，不请求 API、不读写 Memory Space、不生成音频。
该结果用于验证让路机制，不代表实际回复模型的速度或真人听感。
增加 `--architecture qwen3 --step-size 128` 可验证路由同类架构的缓存一致性和分块让路。

CUDA 原生验证使用对应的 NVIDIA 环境，分别观察单独合成和同卡 ASR 竞争：

```bash
python -m evals.breeze_cuda_latency --segmented --repeats 4
python -m evals.breeze_cuda_latency --with-asr --segmented --repeats 4
```

多卡环境可增加 `--device cuda:0 --asr-device cuda:1` 检查分卡路径。输出包含首帧、
RTF、最大交付间隔、交付缺口、按模型/设备统计的原生执行与锁等待，以及缓存的
显存 allocated/reserved/peak/free/total 指标。`free/total` 是设备全局数据；
其余是当前进程 Torch 分配器的数据。脚本不请求 API、不打开 Memory Space，
合成音频仅用于计时，不播放、不保存。自动回归或 Mac 测试不能证明 CUDA 性能改善。
