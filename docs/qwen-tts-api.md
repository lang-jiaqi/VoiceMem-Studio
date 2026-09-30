# Qwen-Audio-TTS Flash（新加坡地域）

Studio 可选使用 `qwen-audio-3.0-tts-flash` 代替 Breeze。它支持流式输出、复刻音色和每段语音的自然语言情绪指令。Studio 原有的语气与 Self Harness 指令会进入 API 请求的 `instruction` 参数。桌面版默认仍使用 Breeze；公开体验模式在配置 Qwen TTS Key 或 Workspace ID 后选择 Qwen。

## 首次启动

把本次代码和 `studio/resources/voice/qwen_audio_reference.wav` 推送到 GitHub `main`。默认注册 URL 已配置为：

```text
https://raw.githubusercontent.com/lang-jiaqi/VoiceMem-Studio/main/studio/resources/voice/qwen_audio_reference.wav
```

这是从仓库中已授权的诺可参考录音生成的约 19.6 秒语音样本。首次注册前，Studio 会下载该 URL 并与本地文件核对；如果还没推送或内容不一致，会明确报错，不会注册错误录音。

在仓库根目录 `.env` 填入：

```dotenv
STUDIO_QWEN_TTS_API_KEY=你的新加坡地域APIKey
STUDIO_QWEN_TTS_WORKSPACE_ID=你的新加坡地域WorkspaceId
```

已有新加坡地域的 `DASHSCOPE_API_KEY` 可以代替第一项。本机桌面测试还需加入 `STUDIO_TTS_PROVIDER=qwen_audio_api`；公开体验模式无需这一项。首次正式启动会自动调用一次 Qwen-Audio 音色注册接口，把返回的 ID 保存在本机的 `studio/resources/voice/qwen_audio_voice_id.txt`，并等待音色处理完成。后续启动复用这个 ID。该 ID 文件已被 Git 忽略；同一云账号与 Workspace 的另一台机器可选择在 `.env` 填 `STUDIO_QWEN_TTS_VOICE_ID`，否则会在那台机器首次启动时各自注册一次。此前为 `qwen3-tts-vc-realtime-2026-01-15` 创建的 ID 不能混用。

若使用其他录音，设置 `STUDIO_QWEN_TTS_REFERENCE_URL` 为公开可访问的 HTTPS URL。Qwen-Audio 的音色注册接口要求 URL，不能直接提交本地 WAV/MP3 的 Base64。也可以手动运行 `.venv/bin/python -m studio.tools.enroll_qwen_audio_voice` 注册或检查音色。

## 测试

```bash
# 静态检查，不会调用云端注册接口
.venv/bin/python -m studio --llm deepseek --memory-llm deepseek --check

# 首次启动自动注册音色；后续启动复用已保存的 ID
npm --prefix studio/apps start
```

公开体验站按 [体验站说明](public-demo.md) 启动。`--check` 只验证配置和本地依赖；真实音色、情绪表现和延迟需要启动后试听。服务使用新加坡 Workspace 专属 WebSocket，并复用连接发送后续语音段。

阿里云文档：[支持的模型与指令](https://www.alibabacloud.com/help/en/model-studio/tts-model/)、[WebSocket 事件](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-client-events)、[音色注册 API](https://www.alibabacloud.com/help/en/model-studio/voice-clone-design-http-api)。
