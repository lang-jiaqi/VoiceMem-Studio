# VoiceMem SDK 示例

这些脚本继承自上游 VoiceMem，用于演示记忆 API 和独立命令行语音 Agent。
**它们不参与 Studio 的启动或对话流程。** 使用 Studio 请从仓库根目录运行
`python -m studio`，并参考根 README 的安装和配置说明。

示例中的模型、提示词和音频循环属于各自脚本，不代表 Studio 的默认配置。
这里不承诺所有脚本在任意机器上直接运行；需要先安装项目、准备对应模型、
配置 API Key，并在使用麦克风的示例中授予设备权限。

## 示例列表

| 文件 | 演示内容 | 主要条件 |
| --- | --- | --- |
| [01_memory.py](01_memory.py) | 音频／文本入库与记忆检索 | 本地模型、`OPENAI_API_KEY` |
| [02_streaming.py](02_streaming.py) | WAV 分块输入与轮次结果 | 本地模型、音频文件、`OPENAI_API_KEY` |
| [03_simple_agent_with_voicemem_memory.py](03_simple_agent_with_voicemem_memory.py) | 独立语音 Agent：记忆、OpenAI 回复和 OpenAI TTS | 麦克风、扬声器、`OPENAI_API_KEY` |
| [04_all_local_l40s.py](04_all_local_l40s.py) | 本地语音 Agent：vLLM 回复和 VoxCPM TTS | CUDA 环境、本机 vLLM、VoxCPM、音频设备 |
| [05_realtime_gpt_qwen.py](05_realtime_gpt_qwen.py) | 将记忆接入 OpenAI／Qwen Realtime | 音频设备、所选服务的 API Key、对应模型权限 |
| [06_mic_memory.py](06_mic_memory.py) | 麦克风转写、记忆检索和后台入库 | 本地模型、麦克风、`OPENAI_API_KEY` |
| [_audio.py](_audio.py) | 04／05 共用的音频工具 | 由对应示例导入，无需单独运行 |

`01/02/06` 主要展示记忆接口；`03/04/05` 包含独立的语音 Agent 循环。
部分脚本在模块顶层初始化或运行，不应作为工具模块导入。

## 运行方式

在仓库根目录完成项目安装后，选择需要的示例：

```bash
export OPENAI_API_KEY=sk-...
python examples/01_memory.py
python examples/02_streaming.py assets/speech.wav
python examples/03_simple_agent_with_voicemem_memory.py
python examples/06_mic_memory.py
```

上面是不同入口，不需要依次执行。`01` 包含录音入库，`06` 虽然不生成助手回复，
入库时仍可能调用 LLM 提取事实。`03` 的 OpenAI Key 还用于默认远程向量、回复和 TTS。

`04` 需要另开终端启动本机 vLLM；脚本中的 `EMPTY` 是未启用鉴权的本地服务占位值：

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.5
python examples/04_all_local_l40s.py
```

`05` 选择远程语音服务：

```bash
OPENAI_API_KEY=sk-... python examples/05_realtime_gpt_qwen.py gpt
DASHSCOPE_API_KEY=sk-... python examples/05_realtime_gpt_qwen.py qwen
```

Key 只放在环境变量或本地配置中，不写进脚本或提交到 Git。

## 配置与记忆目录

`01/02/04/05/06` 使用 `VoiceMem.from_config(...)`，显式选择本地向量和槽位分类。
`03` 使用 `VoiceMem(openai_key=...)`，保留默认构造方式，包括默认远程向量提供者。

`01/02/06` 共用 `examples/example_memory/`。`03/04/05` 未指定 `memory_root`，
会按 VoiceMem 的默认 Space 和环境配置解析目录，可能使用已有记忆。
需要隔离实验数据时，应在对应脚本的构造配置中显式指定 `memory_root`。

不同向量模型的结果不能直接混用，即使维度相同也不代表向量空间兼容。
更换向量模型时，应使用独立目录或重新生成已有记录的向量。

## 流式处理与音频

`02/06` 展示流式转写和轮次确认。VoiceMem 可以在说话期间投机检索；
确认轮次仍需检查、复用或更新相应结果，不保证每次检索都已提前完成。
预热可以把模型初始化移到录音前，但具体耗时取决于机器、模型和缓存。

`06` 将同步入库任务放到后台线程。当前终端的入库提示紧跟任务调度，
不能将该提示视为写入已经完成的确认。

`_audio.py` 使用同一个双工音频回调，将播放样本作为回声消除参考，
并在检测到持续人声时清理播放缓冲、通知调用方打断。03 保留自己的音频实现。
这些工具服务于命令行示例；Studio 的浏览器音频、会话状态和播放管理有独立实现。

`04` 使用 `voicemem.tts.speak_stream()`，按文本片段顺序合成并输出音频。
文本、回调或合成失败会向调用方抛出异常；取消或显式关闭会回收后台任务。
打断后提前结束迭代需要关闭生成器，脚本通过 `contextlib.aclosing` 完成清理。
这个示例记录的是生成文本，没有 Studio 的已听内容追踪机制。

`05` 使用不同队列消费上行音频和本地轮次处理，并通过每次
`response.create` 的 `instructions` 附带当前记忆。网络事件消费者不能等待
阻塞式播放或同步入库；服务端停止生成后，也需要清除本地已缓存的音频。
