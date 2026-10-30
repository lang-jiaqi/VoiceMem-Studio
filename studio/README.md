# 使用 VoiceMem Studio

VoiceMem Studio 是展示 VoiceMem 记忆能力的语音对话应用。主要入口是桌面 App，提供科技风、数字人两种对话界面和桌宠，并可管理本机后端或连接已有服务。Studio 负责语音输入、对话、语音合成和界面；VoiceMem 在 Studio 后端中处理记忆。

以下启动方式需要完整的仓库源码。VoiceMem Python wheel 提供 SDK 及其依赖的 Python 辅助代码，不包含完整 Studio 界面、数字人资源或模型权重。

如果你使用 Apple Silicon Mac，可以直接从 [本机启动](#apple-silicon-mac-本机启动)开始。已有 Linux/Windows 后端，或使用 Intel Mac 的用户，可按[连接已有服务](#连接已有服务)只运行桌面界面。

## Apple Silicon Mac 本机启动

请先安装 Node.js 22.12 或更新版本，以及原生 ARM64 的 Python 3.12。首次运行还需要网络连接，以便安装依赖和下载模型。

在仓库根目录执行：

```bash
npm --prefix studio/apps start
```

首次启动，App 会引导你完成以下步骤：

1. 选择 VoiceMem 记忆处理使用的模型服务（DeepSeek、Qwen 或 OpenAI），填写对应的 API Key。
2. 等待本机环境和记忆、语音相关模型准备完成。首次下载约需 8–12 GB，取决于是否使用本地回复模型；中断后可继续。
3. 选择 Studio 对话使用的模型服务（DeepSeek、Qwen、OpenAI 或本地 MLX）。如果选同一家服务，可复用前面输入的 Key。
4. 后端启动并预热完成后，选择对话界面，开始使用。

VoiceMem 在 Studio 后端进程内运行，不用另开一个 VoiceMem 服务。之后再运行同一命令，会复用已有环境和保存的配置；App 管理的本机后端会随 App 退出而停止。

不想在本机运行后端？第 3 步也可以选择“只启动 UI 界面（后台已有完整 Studio 服务）”。这时需要先输入服务地址并连接成功，才会进入界面。若一开始就打算连接已有服务，使用下一节的命令更直接，也不会在本机准备 Python 或模型。

## 连接已有服务

适用于 Intel Mac、已有 Linux/Windows 后端，或者只想在当前电脑运行桌面界面的情况。先确认另一台机器上的**完整 Studio 服务**已经启动，再在仓库根目录执行：

```bash
npm --prefix studio/apps run start:remote
```

连接页必须手动输入这次要连接的地址，例如同一台电脑上的 `http://127.0.0.1:8787`，或服务器提供的 HTTPS 地址。远程服务器也可通过 SSH 把其 8787 端口转发到本机，再连接 `http://127.0.0.1:8787`。不要填写 `0.0.0.0`，它是服务端的监听地址，不是客户端的连接地址。非回环地址不能使用明文 HTTP。

只启动界面不会安装本机 Python、下载推理模型或启动服务器，也不会在 App 中修改或重启远端后端。需要更换远端模型服务时，请在服务器端修改配置并重启 Studio。

## Linux / NVIDIA 后端

Linux 可以通过 Docker Compose 运行 Studio 服务，无需安装桌面 App。先按[部署说明](deploy/README.md)准备 Docker、NVIDIA 驱动及 Container Toolkit，然后在仓库根目录创建配置：

```bash
cp -n studio/.env.example .env
```

编辑 `.env`，填写所选模型服务的 API Key；默认服务使用 `DEEPSEEK_API_KEY`。随后运行：

```bash
docker compose up -d --build
docker compose logs -f studio
```

看到启动成功日志后，本机浏览器可打开 `http://localhost:8787`。如果要从其他电脑连接，请先配置 HTTPS 和访问控制，或使用 SSH 端口转发；默认服务只绑定在服务器的 `127.0.0.1`，不会直接对公网开放。首次构建、模型下载和预热可能需要较长时间，具体要求和数据卷说明见[部署说明](deploy/README.md)。

Windows 用户先安装 Node.js 22.12 或更新版本、Docker Desktop（WSL2 后端、Linux containers）和兼容的 NVIDIA 驱动，再从 Windows PowerShell 执行 `npm.cmd --prefix studio/apps start`。入口会引导 API 配置、自动构建缺失的后端镜像、启动容器并等待模型准备，然后打开 App；无需手动准备 Ubuntu Python 或 Breeze。Windows 全流程尚需实机验收，详见[桌面 App 说明](apps/README.md)。已有服务器仍可使用 `start:remote`。

## 配置与常见问题

- **如何指定已有记忆库？** 用 `--memory_root /path/to/memory --space myspace` 指定一个具体库目录。Studio 保留该库的位置和数据，并固定使用启动时的 Space；界面不能新建或切换其他 Space。不指定 `--memory_root` 时仍可管理多个 Space。

- **两个 API 要怎么选？** VoiceMem API 用于记忆处理，Studio API 用于对话生成；语音输入、语音合成和界面仍属于同一个 Studio 应用。两处可以选同一家服务，也可以分别选择。
- **从哪里更换模型服务？** App 管理的本机后端可在“设置 → 组件”填写模型名称、服务地址和 Key，支持多项一起应用并重启，启动失败时回到原配置。当前链路保留为推荐默认；常用接入包括记忆/回复的 Chat API、Deepgram 流式识别和 OpenAI-compatible 语音合成。轮次控制等组件保留高级适配器入口，具体协议见[组件接入说明](core/utils/component_config/README.md)。连接已有或 Docker 服务时这里是只读的，需要在服务器端设置后重启。
- **启动前想检查环境？** 已装好对应 Python 环境后，可在仓库根目录运行 `python -m studio --check`。它只做检查，不下载模型，也不打开记忆空间。
- **默认配置文件在哪里？** 样例在 `studio/.env.example`，实际配置是仓库根目录的 `.env` 或进程环境变量。不要提交含真实 Key 的 `.env`。命令行参数优先于环境变量；已导出的变量优先于 `.env`。
- **流式 ASR 用哪个模型？** Studio 仅内置 [X-ASR-zh-en](https://huggingface.co/GilgameshWind/X-ASR-zh-en) 480ms，INT8、CPU 单线程。中文、英文和中英混说共用一套权重，不随界面语言切换。首次启动下载约 614 MB 官方 ONNX 权重，在本机准备约 168 MB INT8 缓存，后续启动直接复用；最终转写仍使用 SenseVoice。
- **英文界面还能说中文吗？** 可以。界面分类和画像展示跟随界面语言，原始发言、事实记忆和自定义 Space 名称保留原文。新记忆的生成描述跟随当轮输入语言，不会因切换界面而改写旧记忆。
- **浏览器能用吗？** 浏览器是补充访问方式。独立启动的后端默认可通过 `http://localhost:8787` 访问同一套对话界面；App 管理的后端使用其实际监听端口。浏览器在远程地址使用麦克风时需要 HTTPS，或通过 SSH 转发后访问本机地址。

更多平台细节见[桌面 App 说明](apps/README.md)和[后端部署说明](deploy/README.md)。

Studio 的 Harness 使用共享的 [`voicemem/engine/`](../voicemem/engine/README.md) 作为推理基础设施：MLX 使用统一执行线程；CUDA 按设备协调语音合成、ASR、路由和后台模型，并统一处理输出背压、取消、音频缓冲反馈及资源观测。模型适配与对话策略由各自模块负责。平台性能需要对应硬件验证，Mac 上的延迟结果不能代表 NVIDIA 后端。
