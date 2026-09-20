# Studio 的默认 Prompt 配置

这个目录会随 `studio/` 一起迁移。其中 `llm_system_zh.md` 和 `llm_system_en.md`
用于旧版 `voicemem.reply` 的默认回复人设，不会覆盖 Studio Web 的实际对话人设；
后者及 Self Harness 的默认策略在 [`../harness/`](../harness/)。
`llm_context.json` 提供旧版回复所需的陌生人、无记忆、语言及录音回放提示；
`llm_tone_rule_*.md` 定义语气标签格式；`tts.json` 配置 TTS 指令和附和风格。

默认配置从本目录读取，与运行命令时所在的目录无关。如需换用自己的配置，
可将 `STUDIO_PROMPT_DIR` 设为包含全部六个文件的绝对路径。旧变量
`VOICEMEM_PROMPT_DIR` 仍受支持；两者同时设置时，优先使用 `STUDIO_PROMPT_DIR`。
六个文件都会随 Studio Python 包安装。修改后需要重启服务；配置解析失败时会
直接报错，不会悄悄切回默认值。

在 `tts.json` 中，`base.zh/en` 是基础语气，`tones` 为八个语气标签分别提供
逐轮指令；模型没有给出标签时使用 `fallback_by_user_emotion`。
`backchannel_styles` 用于合成附和片段，`breeze_default_instruction` 用于
没有逐轮指令的情况。请保留八个 `tones` 键和 `标签|正文` 格式；如需增加标签，
还要同步修改解析器。
`VOICEMEM_SPEAK_BASE` 和 `VOICEMEM_BREEZE_INSTRUCTION` 仍可覆盖对应指令。

请求跟踪日志不属于这套配置。目前它仍写入仓库根目录被忽略的 `prompt/logs/`。
日志可能包含完整对话和记忆，请勿迁移、提交或公开。
