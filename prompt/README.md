# 生效的 Prompt 配置

**直接修改本目录里的文件，重启 demo 后生效。这里就是运行时读取的原始配置，不是副本。**
不需要再到 Python 源码中修改人设或语气提示。当前内容是从原实现原样迁移的。

| 文件 | 修改内容 |
| --- | --- |
| `llm_system_zh.md` | 中文回复人设、聊天风格、记忆使用原则 |
| `llm_system_en.md` | 英文回复人设 |
| `llm_tone_rule_zh.md` / `llm_tone_rule_en.md` | 要求回复模型在开头给出语气标签的规则 |
| `tts.json` | 当前 DeepSeek＋Breeze demo 的 TTS 语气配置，见下方字段说明 |
| `llm_context.json` | 无记忆、陌生人、回复语言、录音回放等条件提示 |

## TTS 怎么改

`tts.json` 中：

- `base.zh` / `base.en`：每轮共有的发声基调。
- `tones`：八个语气标签对应的发声指令。正文回复实际使用 `base + tones[标签]`。
- `fallback_by_user_emotion`：回复模型没有输出语气标签时，按用户情绪追加的指令。
- `backchannel_styles`：合成“嗯嗯、对”等附和音频使用的提示词。修改后缓存键会改变，启动会重新合成。
- `breeze_default_instruction`：Breeze 没收到逐轮指令时的默认提示，例如预热、直接调用 TTS。

请保留 `tones` 的八个键（温和／共情／轻快／认真／鼓励／俏皮／抱歉／平静）和
`标签|正文` 协议，只改指令内容。增加标签需要同时修改解析器，不是纯 prompt 调整。
JSON 格式错误或必填键缺失会明确报错，不会悄悄退回旧提示词。

`VOICEMEM_SPEAK_BASE` 会覆盖 `base`；`VOICEMEM_BREEZE_INSTRUCTION` 会覆盖 Breeze
默认提示。想完全使用本目录配置，请取消这两个环境变量。Breeze 的逐轮指令优先于默认指令。
音色的参考音频及对应转写仍通过 `VOICEMEM_BREEZE_REF_AUDIO` / `VOICEMEM_BREEZE_REF_TEXT` 设置。

读取路径不依赖终端当前目录；源码运行默认使用仓库根目录 `prompt/`。
也可用 `VOICEMEM_PROMPT_DIR=/绝对路径` 指向另一套完整配置。修改后必须重启，
运行中的会话不会热更新；这避免了读文件占用语音关键路径。

## 实际请求记录（不是配置）

启动后每次运行在 `prompt/logs/` 创建时间戳子目录。每次回复生成使用独立的 `.jsonl` 文件；
同一文件按调用顺序记录 LLM 请求和各段 TTS 请求。编辑这些日志不会改变模型行为。

- `llm`：实际发送的模型、完整 messages（system、历史、记忆和当前问题）及生成参数。
- `tts` / `breeze_mlx`：每段实际文本、最终 instruct、参考音频路径、完整 ref_text 和生成参数。
- `output_id`：关联该次音频输出；提前生成的请求也保留，不代表最终一定播放。
- 无回复上下文的合成（例如附和预合成）单独存文件，标记 `purpose: unscoped`。

预缓存附和的播放不会重新调用 TTS，因此不会伪造新的 TTS 请求记录。
文件后台写入；正常退出会等待写完，强制杀进程可能丢失末尾尚未写入的记录。
这里只保存 prompt 和允许的生成参数，不保存 API key / Authorization headers。
日志可能包含个人记忆和完整对话，已被 `.gitignore` 排除；不要公开分享。
上面列出的配置文件会提交 Git，`logs/` 和旧版产生的时间戳日志目录不会提交。

终端默认精简；加 `--verbose` 恢复完整输出。详细运行日志仍保留在 `results/logs/`。
