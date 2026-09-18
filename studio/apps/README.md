# VoiceMem Studio App

## 双风格 UI 与 API 选择

源码运行只需要启动 App：

```bash
cd studio/apps
npm start
```

启动前会检查 Electron、PixiJS 和 Pixi Live2D。缺失时自动按 `package-lock.json` 执行
`npm ci --include=dev` 下载锁定版本；该项目级安装会启用 Electron 必需的安装脚本，即使全局 npm
配置关闭了脚本。依赖完整时跳过安装。首次运行需要能够访问 npm registry 和 Electron 下载源。

终端只选择一次 API 并隐藏输入一次 API Key。DeepSeek、Qwen 或 OpenAI 会同时用于 VoiceMem 记忆处理和 Studio 可见回复；Apple Silicon Mac 选择本地 MLX 回复时，只输入 VoiceMem 记忆处理与对话标题所需的 DeepSeek Key。直接回车可沿用环境变量或项目根目录的 `.env`。

App 随后自动启动 macOS MLX 或 Windows WSL2/CUDA 后端，检查并下载记忆、感知、转写、回复路由、Breeze TTS 和可选本地回复模型。准备期间只在终端显示进度，不创建连接或等待窗口；所有模型完整并预热成功后才创建并显示风格选择页。API Key 只传给本次准备和后端进程，不写入连接设置。

首页复用原版 `index.html`，分别进入 `technical.html` 科技风和 `digital.html` 数字人。语言、UI 字号和内容字号在进入后的「设置」中调整。也可单独打开已运行后端的 `http://localhost:8787`。

UI 源码在 `studio/apps/ui/`，由后端 `/ui/` 提供，App 和浏览器共用。请使用更新后的后端。原界面保留在 `/legacy`。
两种风格的文字、ASR、回复、情绪和召回面板使用现有服务事件；脑图保留视觉导航示意，不代表真实节点数量。聊天列表暂存在当前页面，刷新重置；切换历史条目后的新输入会建立新后端会话。原始 `voicemem_qa` 项目保持原样。
回复旁的小喇叭会重播后端为该条回复实际生成的语音，包括已经播放的打断前缀；音频仅限当前页面并保存在有上限的内存缓存中，刷新后清空。

「设置 → 组件」以可拖动卡片展示语音输入、记忆、回复、语音合成和桌宠。组件都是固定的，不能删除，只能调整位置；布局仅保存在浏览器本机。点击卡片可查看后端、服务商和 API 配置状态，页面只接收“是否已配置”等非敏感信息，不会读取、显示或保存 API Key。需要更换服务商或 Key 时，退出 App 后重新运行 `npm start`，仍只输入一次。

---

# VoiceMem Studio 桌面 App

主窗口直接加载 Studio 已有的 Web 页面，界面、语音协议、记忆面板与 Web 版一致。
桌面 App 和桌宠只面向 Windows/macOS；Linux 和 WSL 只运行后端。
桌面壳负责窗口、配置设置、麦克风授权、桌宠及可选的本机 Docker 自动启动，不包含推理模型或 Python 后端。
原有 `studio/apps/__init__.py` 保留；桌面包与 Python 包相互独立。

## 桌面方案

本项目通过原有 HTTP/WebSocket 协议连接，不迁移后端通信协议。
平台分工、配置边界、目标启动流程和实现状态见 [平台设计](PLATFORMS.md)。

## 直接运行

在 Windows 或 macOS 安装 Node.js 22.12 或更新版本，在本目录执行：

```bash
npm ci
npm start
```

`npm start` 先补齐缺失的 Node 运行依赖，再自动使用项目根目录已有的 Python 环境：macOS 为 `.venv`，Windows 为 WSL2 中的 `.venv-cuda`。启动流程不创建本地等待页，服务就绪后直接打开风格选择页。
服务仍在预热时最多等待三分钟；App 退出时会停止它本次启动的后端。
它不会自动安装 Python、WSL、驱动或 Python/模型依赖；这些后端环境需要事先按部署文档准备好。
通过菜单 **Studio → 配置设置**（`Ctrl/Cmd+,`）更改地址。第一次开始语音时会请求麦克风授权。

## App 内置桌宠

连接成功后默认显示雾铃，作为同一个 App 的透明置顶窗口，不再单独启动 Electron 或安装 `pet` 目录的依赖；App 自己的开发依赖包含打包所需的 Pixi 运行库。
菜单 **Studio → 显示桌宠** 控制本次运行中的显示/隐藏，不退出主窗口。
**找回桌宠位置** 将它移回主屏幕。按住人物或背景拖动窗口，
拖动任意一个角等比例缩放（40%–150%）。界面没有顶部控制条；也可使用 **Studio → 桌宠大小** 恢复原始大小。
位置与大小自动保存，打开主界面时保持显示，由你手动缩小或移开以免遮住字幕。
双击人物或背景、或按 Escape 收成小点；语音事件不会自动展开，点击小点恢复之前的大小。

- 原版 `pet/` 的渲染、动画与 `voicemem-link.js` 保留为唯一来源，不修改它的独立启动方式。
- 桌宠通过当前后端的 `/ws-pet` 只读观察真实播放进度、附和、打断和断线，不采集麦克风、不创建第二段对话。
- 切换服务会关闭旧桌宠连接；关闭 Studio 主窗口或退出 App 会关闭内置桌宠，Docker 后端继续运行。
- 打包包含 Cubism 4 Live2D 模型、Pixi 运行库、原生动作、`scene.js` 和窗帘背景，不再包含旧的 Canvas/PNG 人物。
- 保留 `sit` / `lie` 作为交互/休息状态，使用 3:4 竖屏构图；双模式页面把实际播放音量发送到 `/ws-pet`，由 RMS 驱动嘴型。
- 源码运行和打包自动执行 `prepare:pet`，按白名单生成忽略于 Git 的 `.pet-runtime/`。
  不打包 `pet/checks`、旧模型、旧第三方运行库或第二份 Electron。
- 检测到旧 Canvas 或 Live2D 构建缓存时，先移到 `.pet-runtime.previous-*/resources/` 再生成新资源；备份不进入 Git 或安装包。
  不认识的额外文件会使准备步骤报错，不会被删除或意外打包。

Linux/WSL 后端不会自动启动桌宠，Docker 也保留 `STUDIO_DESKTOP_PET=0`。
如果 App 连接同一台 Mac 的原生后端，启动后端时关闭它原来的自动桌宠，避免出现两只：

```bash
STUDIO_DESKTOP_PET=0 python -m studio
# 或给原有 scripts/run_studio_mlx.sh 加同一个环境变量
```

这只改变谁负责桌宠窗口，不关闭 `/ws-pet` 广播。App 不会终止已经独立运行的桌宠进程。

## 自动启动本机 Docker

在 Windows 配置页勾选“自动启动本机 Docker 服务”，选择已配置好的 VoiceMem-Studio 项目根目录并确认授权。
下次打开 App 会自动启动对应 `studio` 服务，并使用 Docker 实际发布的端口连接。

- 需要 Windows + NVIDIA、已运行的 Docker Desktop（WSL2 后端）、Docker Compose、已有的 Studio 镜像及项目配置。
- 这里只启动 Compose 服务，不启动 Docker Desktop，也不安装 WSL、驱动或模型环境。
- 沿用所选目录的 `compose.yaml` 和可选的 `compose.override.yaml`，不复制或修改 `.env`。
- 使用 `up -d --no-build --no-recreate --pull never studio`，不自动构建、拉取镜像或重建已有容器。
- 退出 App、取消等待均不停止容器，不删除模型、记忆或卷。
- 自动启动不操作远程 Docker context。远程 GPU 服务器请通过 HTTPS 或 SSH 转发连接。
- `npm start` 的本机启动路径不使用 Docker：Mac 启动原生 MLX，Windows 通过 `wsl.exe` 启动 WSL2/CUDA。
- 本节的 Docker 选项保持原有 Compose 行为，供已有镜像和共享容器的连接方式使用。
- Windows named-pipe 与命令参数已做模拟测试，完整 WSL/GPU 流程仍需 Windows 实机验收。

如果镜像还没构建，请先按 [Docker 部署说明](../../docker/README.md) 完成环境准备。
项目代码或 Docker 配置更新后仍需按原流程重新部署后端；App 不代替这一步。

## 打包

```bash
# 在 Windows x64 上构建含桌宠的 EXE 安装包
npm run dist:win

# 在 Apple Silicon Mac 上构建 DMG 和 ZIP
npm run dist:mac

# 在相应 Mac 环境中构建 Intel 版本
npm run dist:mac:x64
```

产物位于本目录的 `dist/`，不进入 Git。版本和依赖通过 `package.json`、`package-lock.json` 固定。
不再发布 Linux AppImage 或桌面便携包。已有 Linux 产物仅为历史开发测试文件，不代表最新发布。
Windows/macOS 安装包应在对应系统构建并验收；Windows 发布签名和 Mac 签名/公证需要发布者自己的凭据。
请用普通桌面用户运行，不要为日常使用关闭 Electron sandbox。
Mac 正式分发需开发者签名和公证；仓库提供麦克风用途说明和相应 entitlement，但不包含证书。
图标由仓库原有 `assets/logo.png` 裁切，生成脚本为 `scripts/make-icon.py`（仅重新生成图标时需要 Pillow）。

## 安全与数据

- 远程地址必须使用 HTTPS；HTTP 仅允许回环地址。不要通过关闭证书校验或 Web 安全性连接远程服务。
- 服务地址仅支持根地址，不带用户名、密码、路径或查询参数。`npm start` 输入的 LLM API Key 只通过进程环境传给本机后端；远程后端仍自行管理密钥。
- 主界面没有 Node.js 或本机 Docker IPC 权限，仅本地配置页有受限的设置接口。
- 桌宠只有受限窗口操作接口，没有连接设置、Docker 或 Node 权限。资源请求限于内置文件和当前 `/ws-pet`。
- 麦克风仅对配置的服务主页面授权，不授权摄像头、屏幕录制或第三方 iframe。
- 连接设置和页面缓存位于操作系统的 VoiceMem Studio 应用数据目录；不会导入浏览器私有缓存。
- 桌面包不含 `.env`、模型、记忆库、录音或运行日志。后端数据继续使用原位置或 Docker 卷。
- 上述模型指 ASR/TTS/LLM 等推理权重；内置桌宠的 PNG 显示素材是 App 资源。
- 桌宠原版第三方声明随包保留；角色素材沿用原有许可边界，打包不代表新增公开再分发授权。
- `VOICEMEM_DESKTOP_USER_DATA` 可以指定独立桌面配置目录，适合隔离测试。

## 验证

```bash
npm test
```

Linux 可用 `node scripts/smoke.cjs` 在 Xvfb 中验证真实窗口。它只使用临时假后端和假 Docker，
不调用真实容器、不打开个人记忆、不采集麦克风；截图留在其输出的临时目录。
它加载实际 Live2D 人物和场景，用合成的 `/ws-pet` 事件验证嘴型、原生动作、打断、暂停/断线、服务切换及隐藏窗口。
`node scripts/smoke.cjs --asar` 先按打包白名单生成临时 ASAR，再验证归档内的共享客户端和资源；不生成 Linux 安装包。
`--keep-pet-on-exit` 额外验证关闭主窗口时仍显示的桌宠一起退出。
可以用 `VOICEMEM_DESKTOP_BINARY` 指向已打包的程序，`VOICEMEM_XVFB` 指定 Xvfb。
`VOICEMEM_TEST_TMP` 指定临时目录；受控 root 测试额外要求 `VOICEMEM_SMOKE_ALLOW_ROOT=1`，
仅该测试启动器会临时使用 `--no-sandbox`。实际麦克风、音频设备和 Mac 权限仍需在目标桌面验收。
