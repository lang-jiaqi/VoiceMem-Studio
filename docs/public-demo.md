# 简单体验站

访客使用名称和密码注册；每个账号有一个独立的 Memory Space 和录音目录。手机打开后进入白藤宠物页面，默认只显示人物与简短状态。底部“对话”按钮打开文字记录和输入框；VoiceMem 按钮查看本轮召回与新入库内容。右上角设置可调整对话偏好，选择会保存在该账号的空间中。设置中没有 API Key 和组件配置。文字记录刷新后不保留；长期记忆会保留。

## 在提供服务的 Mac 上启动

先运行 `npm --prefix studio/apps run ensure:deps`，让手机宠物页面能加载仓库中的 PixiJS 依赖。

白藤还需要 Live2D Cubism Core。若官网 CDN 无法访问，从 Live2D 官方 SDK 取得 `live2dcubismcore.min.js` 并放到 `studio/pet/assets/live2d/vendor/`。白藤模型的公开使用与再分发范围也应先向权利人确认；见 `studio/pet/assets/live2d/rattan/README-SOURCE.md`。

先按仓库现有说明准备 Python 环境和模型。桌面 App 保存的 API Key 不会自动传给直接运行的 Python 服务。在仓库根目录执行 `nano .env`，加入一行 `DEEPSEEK_API_KEY=你的密钥`，保存后运行 `chmod 600 .env`。`.env` 已被 Git 忽略，不要把密钥贴到命令行或提交到 Git。一个 DeepSeek Key 即可用于下方命令的记忆和回复服务。

可先运行 `STUDIO_PUBLIC_DEMO=1 .venv/bin/python -m studio --llm deepseek --memory-llm deepseek --check`，确认两项 API Key 均显示“已找到”。体验模式需要 API 回复模型，并使用 `llm_tts` 模式。然后在仓库根目录运行：

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
