# 简单体验站

首次准备 Apple Silicon / Python 3.12 环境时运行 `./studio/scripts/setup_mlx.sh`。
安装脚本使用仓库固定的 Mac 依赖版本，并检查依赖一致性；CUDA 部署继续使用独立的依赖配置。

访客使用名称和密码注册；每个账号有一个独立的 Memory Space 和录音目录。手机打开后进入白藤宠物页面，默认只显示人物与简短状态。底部“对话”按钮打开文字记录和输入框；VoiceMem 按钮查看本轮召回与新入库内容。右上角设置可调整对话偏好，选择会保存在该账号的空间中。设置中没有 API Key 和组件配置。文字记录刷新后不保留；长期记忆会保留。

## 在提供服务的 Mac 上启动

同一个账号空间支持中英文自然混用；回复跟随用户表达和上下文。界面语言只改变按钮和提示，不会切换、翻译或清空记忆。实时转写使用支持中英文的 FunASR Paraformer；启动时复用本机已有权重，缺失时自动下载。英文短附和需要对应音色的已缓存片段，缺少时该次附和保持安静；长垫话仍按本轮语言生成。

先运行 `npm --prefix studio/apps run ensure:deps`，让手机宠物页面能加载仓库中的 PixiJS 依赖。

白藤还需要 Live2D Cubism Core。建议在部署机器上，从 Live2D 官方 SDK 取得 `live2dcubismcore.min.js` 并放到 `studio/pet/assets/live2d/vendor/`，让手机直接从体验站加载，避免等待外部官网 CDN。缺少本地文件时仍使用官方 CDN。白藤模型的公开使用与再分发范围也应先向权利人确认；见 `studio/pet/assets/live2d/rattan/README-SOURCE.md`。

先按仓库现有说明准备 Python 环境和模型。桌面 App 保存的 API Key 不会自动传给直接运行的 Python 服务。在仓库根目录执行 `nano .env`，加入一行 `DEEPSEEK_API_KEY=你的密钥`，保存后运行 `chmod 600 .env`。`.env` 已被 Git 忽略，不要把密钥贴到命令行或提交到 Git。一个 DeepSeek Key 即可用于下方命令的记忆和回复服务。

若体验站改用 Qwen 系统音色 TTS，按 [Qwen TTS API 配置](qwen-tts-api.md) 填新加坡地域的 Key 与 Workspace ID。默认使用 `longanlingxi`，无需注册录音；未配置时仍使用 Breeze。

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

## 手机资源缓存

同一账号可以打开多个页面。每个连接有独立的实时识别和 VAD 状态；模型权重在进程内复用。
断开连接会关闭识别工作线程。账号的空闲运行实例会自动回收，长期记忆、录音和已保存的设置保留在磁盘。
手机的入库列表在后端实际完成写入后更新，不会在长回复播放时轮询超时。

手机页面的模型、贴图、背景、脚本和本地运行库使用带内容版本的地址，允许浏览器缓存。
首次打开仍需要下载并初始化人物；再次打开时，浏览器保留的资源可以直接复用。
账号、记忆接口和页面入口不缓存。

对话面板支持 Markdown 预览，包括标题、列表、引用、链接、表格和代码块。
正文和标题沿用当前字号，标题通过字重区分；复制和记录保留原文。
行内、独立公式和 `math` / `latex` / `tex` 代码块使用随仓库提供的 KaTeX 排版，
运行库和字体从部署机器加载并缓存，无需访问外部公式 CDN。公式基准字号跟随正文，
较长的独立公式可以横向滚动。流式输出等待公式完整后排版，格式有误时保留原文。
TTS 使用独立的朗读文本，过滤常见排版标记。
成段代码、复杂公式会用简短的查看提示代替逐字朗读，完整内容仍保留在对话里。
提示有多种措辞，连续的结构化内容会减少重复提示；短变量名保留，简单的数字算式转为
“加、减、乘以、除以、等于”等口语。公式应使用 `$...$`、`$$...$$`、`\(...\)`、
`\[...\]`，代码使用 Markdown 代码块；未标记的普通算式不会被一律略去。
回复规则要求模型在公式、代码前后解释关键含义；语音规则本身不总结或推断内容，也不额外
调用模型。打断时根据实际播放进度映射回对应原文，截去后面尚未说出的解释。

首次打开会提前下载人物模型、原尺寸贴图和运行库，并行加载 Core 与 Pixi。
后端启动时会为适合压缩的模型、脚本等资源生成无损 gzip 版本；支持 gzip 的浏览器
自动解压，画质、模型内容和动画帧率不变。压缩不会在每位访客打开页面时重复执行。
首次速度仍取决于体验站网络、手机网络和人物初始化耗时。

更新部署机器上的代码或模型资源后，重启 Studio 后端，再刷新手机页面即可获取新版本，
无需手动清理浏览器缓存。加入本地 Cubism Core 后也需要重启。
