# 使用 VoiceMem Studio

VoiceMem Studio 是展示 VoiceMem 记忆能力的语音对话应用。Studio 负责语音输入、对话、语音合成和界面；VoiceMem 在 Studio 后端中处理记忆。它们不是两个需要分别启动的对话服务。

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

Windows 用户可在 WSL2/NVIDIA 环境中运行后端，再使用桌面 App；也可以直接连接一台已运行 Studio 的 Linux 服务器。Windows 本机路径尚需实机验证，相关要求见[桌面 App 说明](apps/README.md)；Linux/Docker 部署见[部署说明](deploy/README.md)。App 不会替你安装 WSL2、显卡驱动或 Python 解释器。

## 配置与常见问题

- **两个 API 要怎么选？** VoiceMem API 用于记忆处理，Studio API 用于对话生成；语音输入、语音合成和界面仍属于同一个 Studio 应用。两处可以选同一家服务，也可以分别选择。
- **从哪里更换模型服务？** 本机 App 的“设置 → 组件”可修改记忆或回复模型服务，并由 App 重启它管理的本机后端。连接已有服务时这里是只读的，需要到服务器修改 `.env` 后重启。
- **启动前想检查环境？** 已装好对应 Python 环境后，可在仓库根目录运行 `python -m studio --check`。它只做检查，不下载模型，也不打开记忆空间。
- **默认配置文件在哪里？** 样例在 `studio/.env.example`，实际配置是仓库根目录的 `.env` 或进程环境变量。不要提交含真实 Key 的 `.env`。命令行参数优先于环境变量；已导出的变量优先于 `.env`。
- **浏览器能用吗？** 后端就绪后，本机访问 `http://localhost:8787`。浏览器在远程地址使用麦克风时需要 HTTPS，或通过 SSH 转发后访问本机地址。

更多平台细节见[桌面 App 说明](apps/README.md)和[后端部署说明](deploy/README.md)。
