# 桌面端与后端的平台设计

一套 Electron App 和桌宠源代码，只发布 Windows/macOS 客户端；Linux 只运行后端。
操作系统差异放在启动适配层，不分叉 Web UI、记忆、ASR、Router 或 TTS 业务逻辑。

## 平台分工

| 用户电脑 | App / 桌宠 / 麦克风 / 播放 | 本机推理后端 | 发布物 |
| --- | --- | --- | --- |
| Windows x64 + NVIDIA | Windows 原生 Electron | WSL2 中的 Linux CUDA；优先复用 Docker 镜像 | 一个含桌宠的 EXE 安装包 |
| Apple Silicon Mac | macOS 原生 Electron | macOS 原生 Python + MLX/Metal | 一个含桌宠的 DMG |
| Intel Mac 或无兼容 GPU 的 Windows | 同一个桌面客户端 | 连接远程 Linux/NVIDIA 服务 | 对应架构的桌面包 |
| Linux / WSL | 不安装 App，不自动启动桌宠 | CUDA Docker 或已有原生 Python 环境 | 后端镜像 / 源码 |

Windows 不运行 Linux App，也不通过 WSLg 运行桌宠。客户端发送音频给后端，
桌宠只观察 `/ws-pet`；两者均使用现有协议，不要求将麦克风设备透传给 WSL 或容器。

Windows 访问 WSL 内的服务优先使用 `http://127.0.0.1:8787`。
Microsoft 支持由 Windows 通过 localhost 访问 WSL 服务，但 VPN、防火墙及 WSL 配置仍可能影响连通性；
连接失败应诊断转发，不自动开放所有网卡或改防火墙。
参见 [WSL 网络说明](https://learn.microsoft.com/en-us/windows/wsl/networking)。

Windows 的 Docker 路线需要已运行的 Docker Desktop、启用 WSL2 后端，以及支持 WSL 的 NVIDIA 驱动。
这不是为 Windows 另做一套 CUDA 镜像，仍复用 Linux 镜像。
参见 [Docker Desktop GPU 条件](https://docs.docker.com/desktop/features/gpu/)。

## 配置页面

远程连接页与对话风格选择页共用浅色卡片视觉和 680×430 紧凑窗口；菜单中统一使用“连接设置”，主对话页面继续复用 Web。
删除宣传性小字，保留会影响操作的错误、进度和必要提示。

最终建议提供两个服务选项：

- 本机：自动识别平台。Mac 使用 MLX；Windows 使用 WSL2/CUDA。只展示对应平台所需的目录或运行环境。
- 远程：只配置服务地址，不要求本机具备 CUDA、MLX、WSL 或 Docker。

其余配置继续统一：API、模型路径、语言、语音和桌宠。不要分别维护 Windows 和 Mac 两份配置页面。
模型文件存在且通过完整性检查就复用；缺失时先显示下载内容和进度，不覆盖已有模型。
本机模型路径必须属于实际运行后端的文件系统：Windows、WSL 和容器路径不是同一个命名空间，不能直接混用。
远程模式的模型和 API Key 留在服务器，不能拿客户端目录代替服务器配置。

首次 `npm start` 在终端分别选择 VoiceMem 记忆 API 与 Studio Agent 对话 API，分别读取需要的密钥。托管后端启动后，“设置 → 组件”
可分别修改记忆和回复的服务商、OpenAI-compatible 地址、模型名与 Key，并由 App 受控重启。
成功配置使用系统安全存储加密，下次启动直接复用；远程连接仍只读，由服务器管理配置和重启。

## 源码 App 启动流程

```text
npm start        → 首次选择 VoiceMem API / 输入 Key，或读取已加密配置
                 → 检查本机环境并准备记忆、感知、转写模型（非独立 VoiceMem 服务）
                 → 首次选择 Studio Agent 对话 API / 输入不同服务商的 Key
                 ├─ Apple Silicon Mac：启动受控的原生 MLX 后端
                 └─ Windows 本机：通过 wsl.exe 启动 WSL2/CUDA 后端
                 → 检查并下载回复、TTS 模型 → 严格预热
                 → 创建风格选择页 → 主界面 + 桌宠

                 → Studio API 的最后一项：仅 UI，输入已有 Studio 地址后连接

npm run start:remote
                 → 不检查或启动本机后端
                 → 打开空地址配置页 → 必须手动输入地址并连接已有服务
```

准备和等待期间只使用终端，不创建本地连接/状态窗口。启动失败时显示原生错误框并退出。
受管后端首次下载和预热默认最多等待 30 分钟；连接已有服务仍使用较短的三分钟就绪等待。

自动识别平台不等于自动安装系统组件。缺少 WSL、驱动、Docker 或 MLX 环境时应准确提示，
不擅自提权、修改系统配置或新建第二套模型目录。

只有由 App 新创建并明确拥有的原生后端进程才随 App 退出；已运行的共享服务、Docker 容器和远程服务不自动停止。
Mac 的启动适配应传入 `STUDIO_DESKTOP_PET=0`，桌宠由 App 唯一管理；Linux/WSL 后端本身禁止自动拉起桌宠。

## 当前实现与待验证边界

- 已实现：共享 App + 桌宠、配置文案、Windows/macOS 打包配置、连接现有服务、Linux/WSL 禁止自动拉起桌宠。
- 已实现代码并做模拟测试：`npm start` 的记忆/回复独立 provider 与密钥步骤、UI-only 必填地址、组件内模型服务配置、安全持久化与失败回退、VoiceMem 模型准备、Mac MLX 受控进程和 Windows `wsl.exe` CUDA 受控进程；App 退出时结束自己启动的进程。
- 已实现代码并做模拟测试：Windows Docker CLI 本机 named-pipe 检查和 Compose 启动；只启动已有镜像和配置。
  不自动启动 Docker Desktop 本身，不构建、拉取或重建容器。
- 尚未实现：安装包内置 Python/模型环境；本地 ASR、TTS、感知和声纹模型继续由后端 profile 管理。
- 尚未实机验收：Windows 安装包、WSL/CUDA 联调、Mac 打包签名和麦克风权限。
  Linux 的虚拟桌面测试只验证共享客户端代码，不等于支持 Linux 桌面发布，也不能代替 Windows/macOS 验收。
