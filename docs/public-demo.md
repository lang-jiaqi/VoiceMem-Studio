# 简单体验站

访客使用名称和密码注册；每个账号有独立的 Memory Space 和录音目录。手机打开后直接进入数字人页面。数字人下方的“VoiceMem · 本轮记忆”可查看本轮召回与新入库内容。聊天列表只保存在当前页面，刷新后不保留；长期记忆会保留。

## 在提供服务的 Mac 上启动

先按仓库现有说明准备 Python 环境、模型和回复及记忆 API Key。体验模式需要 API 回复模型，并使用 `llm_tts` 模式。然后在仓库根目录运行：

```bash
STUDIO_PUBLIC_DEMO=1 .venv/bin/python -m studio --host 127.0.0.1 --port 8790 --mode llm_tts --llm deepseek --memory-llm deepseek --lang zh
```

服务启动后，在另一个终端运行：

```bash
tailscale funnel --bg --https=443 8790
tailscale funnel status
```

如果 8790 已被占用，把两条命令里的端口一起换成同一个空闲端口。

把 `tailscale funnel status` 给出的 HTTPS 地址发给体验者。该地址会公开给知道链接的人；访客无需安装 Tailscale。结束体验时运行 `tailscale funnel --https=443 off`，并停止 Studio 进程。

账号数据默认在 `~/.local/share/voicemem-studio/demo/`，也可在启动前设置 `STUDIO_DEMO_DATA_DIR`。不要把此目录加入 Git。建议只短期开放体验链接，并留意 API 用量。

本模式只监听本机回环地址。普通桌面 App 启动不启用账号系统；也无需运行 Funnel。
