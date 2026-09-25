# VoiceMem Studio 桌面 App

桌面 App 适用于 macOS 和 Windows；Linux/WSL 用于运行后端，不需要安装桌面界面。App 提供科技风、数字人两种对话界面，以及随 App 运行的桌宠。它可以在本机启动 Studio 后端，也可以只连接已经运行的 Studio 服务。

## 开始前

安装 Node.js 22.12 或更新版本，并克隆完整仓库。首次启动需要联网安装桌面依赖；这些依赖会在缺失时自动按锁定版本安装。

| 你的电脑 | 后端运行在哪里 | 推荐入口 |
| --- | --- | --- |
| Apple Silicon Mac | 本机 Python 3.12 + MLX | `npm --prefix studio/apps start` |
| Windows x64 + NVIDIA | 已准备好的 WSL2/CUDA 环境 | `npm --prefix studio/apps start` |
| Intel Mac，或不在本机运行推理的 Mac/Windows | 已运行的本机或远程 Studio 服务 | `npm --prefix studio/apps run start:remote` |

本机模式不会帮你安装 Python 解释器、WSL2 或显卡驱动。Apple Silicon Mac 需要原生 ARM64 Python 3.12；Windows 本机模式需要先在 WSL2 中准备 Python/CUDA 环境和对应依赖。Linux/NVIDIA 和 Docker 的部署要求见[后端部署说明](../deploy/README.md)。Windows 的 WSL2/CUDA 路径已有自动启动逻辑和模拟测试，但仍需在实际设备上验证；初次使用建议先确认后端能独立启动。

## 本机启动

在仓库根目录执行：

```bash
npm --prefix studio/apps start
```

首次启动会按顺序引导你：

1. 选择 VoiceMem 记忆处理的 API（DeepSeek、Qwen 或 OpenAI），填写 Key。
2. 等待本机环境、记忆、感知和转写模型准备完成。
3. 选择 Studio 对话的 API（DeepSeek、Qwen、OpenAI；Apple Silicon 还可选择本地 MLX），或选择最后一项“只启动 UI 界面（后台已有完整 Studio 服务）”。两次选择同一家服务时可以复用 Key。
4. 本机后端完成模型检查和预热后，选择对话界面；如果选择只启动 UI，必须先填写已有服务的地址并连接成功。

VoiceMem 作为记忆模块在 Studio 后端进程内运行；第 1 步和第 3 步选择的是记忆处理与对话生成各自使用的模型服务，不是启动两套回复服务。后续启动会复用环境和保存的配置。本机模式由 App 管理的后端会随 App 退出而停止；已经独立运行的服务不会被 App 关闭。

首次安装和模型下载可能花较长时间，进度显示在启动 App 的终端中。App 会在服务就绪后再显示风格选择页。如果失败，请先查看终端的具体错误；不要直接删除记忆空间或已有模型。
本机模式使用 `127.0.0.1:8787`。如果你已在同一端口单独启动 Web 后端，请关闭它后再运行本机模式，或使用下方的 `start:remote` 连接现有服务。

## 只连接已有服务

先在服务器上启动完整 Studio 后端，然后在客户端的仓库根目录执行：

```bash
npm --prefix studio/apps run start:remote
```

这个入口只启动桌面界面，不检查客户端的 Python、WSL 或 GPU，也不下载推理模型。连接页每次都需要手动填写地址，连接成功后才进入风格选择页。同机服务可填写 `http://127.0.0.1:8787`；远程服务请使用 HTTPS，或通过 SSH 端口转发后连接本机的 `127.0.0.1`。不要把 `0.0.0.0` 当成连接地址。

连接已有服务时，App 不会更改远端 API Key 或重启远端后端。Windows/Intel Mac 也可以使用这个入口连接本机已运行的 Docker 服务。

## 进入界面后

- 首次开始语音对话时，允许 App 使用麦克风。科技风和数字人使用同一个 Studio 服务；语言和文字大小在界面的“设置”中调整。
- 本机模式下，“设置 → 组件”可分别更换记忆和回复模型服务的服务商、模型名称、兼容服务地址及 Key。App 会重启它管理的后端；新配置启动失败时恢复原配置。远程模式只展示配置，修改服务请到服务器进行。
- 主窗口在前台时桌宠默认隐藏；切到后台或最小化主窗口时桌宠会出现。可从 Studio 菜单关闭本次运行的后台桌宠，或找回它的位置。桌宠的通话按钮使用当前主界面的语音会话，不会另外建立一段对话。
- 也可以在后端所在电脑的浏览器打开 `http://localhost:8787`。从其他电脑通过浏览器使用麦克风时，需要 HTTPS 或 SSH 本地转发。

## 配置与数据

项目根目录的 `.env` 用于自行运行的后端；可从 `studio/.env.example` 创建。App 管理的本机后端会安全保存成功使用的配置和 Key，下次不必重复输入。不要提交 `.env` 或向他人发送 Key。

桌面包不包含模型权重、个人记忆、录音或运行日志。远程模式的模型和 Key 保留在服务器；App 不会将它们复制到客户端。Linux/WSL 后端不自动启动桌宠。

遇到环境或模型准备问题，请查阅[Studio 使用说明](../README.md)和[后端部署说明](../deploy/README.md)。
