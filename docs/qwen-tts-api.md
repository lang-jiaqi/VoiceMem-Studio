# Qwen-Audio-TTS Flash（新加坡地域）

Studio 可选使用 `qwen-audio-3.0-tts-flash` 代替 Breeze。它通过 WebSocket 流式输出 24 kHz PCM，支持为每段语音设置自然语言情绪指令。默认系统音色为 `longanlingxi`，无需参考录音或音色注册；旧的复刻音色 ID 不再被读取。桌面版默认仍使用 Breeze；公开体验模式在配置 Qwen TTS Key 或 Workspace ID 后选择 Qwen。

## 配置与启动

在仓库根目录 `.env` 填入：

```dotenv
STUDIO_QWEN_TTS_API_KEY=你的新加坡地域APIKey
STUDIO_QWEN_TTS_WORKSPACE_ID=你的新加坡地域WorkspaceId
STUDIO_TTS_PROVIDER=qwen_audio_api
```

已有新加坡地域的 `DASHSCOPE_API_KEY` 可以代替第一项。桌面 App 需要 `STUDIO_TTS_PROVIDER=qwen_audio_api`；公开体验模式自动选择 Qwen，不要求这一项。无需设置 `STUDIO_QWEN_TTS_VOICE_ID`。如果想试听另一个系统音色，可设置 `STUDIO_QWEN_TTS_VOICE=longanxiaoxin`（亲切活泼）或 `longanfengyue`（自然亲切）。可用音色以[阿里云官方列表](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-voice-list)为准。

```bash
# 静态检查，不会调用云端合成接口
.venv/bin/python -m studio --llm deepseek --memory-llm deepseek --check

# 桌面 App
npm --prefix studio/apps start
```

公开体验站按 [体验站说明](public-demo.md) 启动。`--check` 只验证配置和本地依赖；实际音色、情绪表现和延迟需要启动后试听。服务使用新加坡 Workspace 专属 WebSocket，并复用连接发送后续语音段。

若服务返回 `AllocationQuota.FreeTierOnly`，说明账号的免费额度不可用或尚未完成开通，且当前不允许按量付费。希望继续使用付费 API 时，到阿里云 Model Studio 控制台检查账号开通与“免费额度耗尽即停”设置。

阿里云文档：[音色列表](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-voice-list)、[指令控制](https://www.alibabacloud.com/help/en/model-studio/realtime-tts-user-guide)、[WebSocket 事件](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-client-events)。
