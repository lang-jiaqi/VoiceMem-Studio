# Studio 部署

支持两种部署方式，业务、人格和 TTS 切句共用现有代码：

| 平台 | 运行方式 | 启动 |
| --- | --- | --- |
| Linux / NVIDIA | 完整 Docker 容器，GPU 0 | `docker compose up -d --build` |
| macOS / Apple Silicon | 原生 Python + MLX/Metal | `./scripts/run_studio_mlx.sh` |

Mac 使用原生 MLX/Metal 部署。普通 Docker 容器不能透传 Metal GPU；
即使是 Docker Model Runner 的 Mac GPU 后端，也是在宿主机运行推理，而不是在容器里。
这里保留现有的原生入口，不增加模型代理或独立 TTS HTTP 服务。
参见 [Docker 的执行环境说明](https://docs.docker.com/ai/model-runner/#execution-environment)。

## Linux / NVIDIA

先安装 Docker Engine、Compose 插件、NVIDIA 驱动和
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
使用 Linux Docker Engine，不使用 Docker Desktop 的 Linux VM。
镜像使用 Python 3.12、PyTorch 2.8.0 / CUDA 12.8 与 `.[studio-cuda]`，不需要宿主机 Python 环境。
Compose 默认镜像标签为 `voicemem-studio:torch2.8-cu128`，对应 TorchAudio 2.8.0 和 TorchVision 0.23.0。
宿主机仍需兼容的 NVIDIA 驱动；镜像不会安装或升级宿主机驱动。
CUDA 小版本兼容对运行时编译有限制，是否可用还需通过 Breeze 的编译和模型预热检查，
不能只以 `torch.cuda.is_available()` 为准，见 [NVIDIA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

在仓库根目录操作：

```bash
# 仅首次创建；已有 .env 不会被覆盖
cp -n .env.example .env
# 编辑 .env，填写 DEEPSEEK_API_KEY
docker compose up -d --build
docker compose logs -f studio
```

看到 `[startup] VoiceMem Studio 启动成功` 后打开 `http://localhost:8787`。初次构建需要下载依赖；
初次启动需要下载模型并编译预热，可能较久。之后复用持久化权重和编译缓存。
默认 DeepSeek、中文、`studio-zh`、详细终端日志，Breeze 在 Studio 进程内流式合成。
Compose 只分配宿主机 GPU 0，容器内部 ASR/Router/TTS 都使用 `cuda:0`。
容器默认时区为 `Asia/Shanghai`，可在 `.env` 中用 `TZ` 覆盖；Mac 保持宿主机时区。
Docker 使用 `STUDIO_DESKTOP_PET=0` 禁用本地 Electron 窗口，但保留 `/ws-pet` 事件通道。
Mac 原生启动保留上游随服务启动桌宠的行为，环境准备脚本同时安装 `pet/package-lock.json` 中的依赖。

默认使用 Debian 官方 HTTPS 软件源。网络较慢时，可选用镜像构建参数；
安全更新仍使用官方源，签名校验保持开启：

```bash
docker compose build \
  --build-arg DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
docker compose up -d
```

PyPI 的 Linux x86_64 PyTorch 2.8.0 使用 CUDA 12.8；镜像构建会检查 Torch、TorchAudio、
TorchVision 和实际 CUDA 版本，不依赖宿主机已安装的 PyTorch，也不会接受 CPU-only 版本。

默认仅绑定宿主机 `127.0.0.1`，避免把没有认证的记忆接口直接暴露到外网。
远程使用可通过 SSH 转发，浏览器仍访问本地地址：

```bash
ssh -L 8787:127.0.0.1:8787 user@server
```

如果宿主机的原生 Studio 已占用 8787，先手动停止它，或在 `.env` 设置
`STUDIO_PORT=8788` 使用不同的宿主机端口。端口不同也仍共享 GPU 0 的资源，
不建议同时运行两套推理服务。需要公开部署时，应另外配置 HTTPS 和访问控制。

### 常用命令

```bash
docker compose logs -f studio
docker compose ps
docker compose stop
docker compose up -d

# 代码更新后重建并启动
docker compose up -d --build

# 仅检查凭据存在性、依赖、GPU 和模型完整性；不下载或打开 Memory Space
docker compose run --rm --no-deps studio --check
```

模型预热成功后才监听 HTTP，所以健康检查成功表示页面服务已就绪；
它不验证远程 LLM 凭据是否有效，也不代替麦克风和真实语音链路测试。

### 数据与密钥

镜像不包含模型权重、`.env`、私人记忆、运行录音、对话日志或虚拟环境。
Breeze CUDA 源码从独立仓库按固定提交构建，权重在首次运行时下载。
Studio 参考声音和附和素材来自仓库已有的审核录音，不生成替代声音。
用户配置 `studio/harness/` 和 `prompt/tts.json` 作为只读绑定挂载；编辑后重启容器生效。
这两个配置路径需要保留在宿主机仓库中，不能用空目录替代。

| Compose 卷 | 内容 |
| --- | --- |
| `models` | 模型权重和下载元数据 |
| `memory` | Memory Spaces |
| `results` | 运行日志、会话录音和结果 |
| `prompt-logs` | 含对话上下文的请求日志，需要按敏感数据管理 |
| `cache` | 模型下载缓存、Torch/CUDA 编译缓存、附和 PCM 缓存 |

`docker compose down` 保留这些卷。**不要使用 `docker compose down -v`，它会删除卷及数据。**
默认创建独立容器数据，不自动迁移本机的模型软链接或个人记忆。
宿主机上的绝对软链接通常无法在容器内解析；如需复用模型，应挂载真实权重目录到
对应容器路径，或在独立卷中导入解引用后的文件。不要直接把现有 `studio/models`
软链接目录当作可移植模型包，也不要让原生和容器进程同时写同一记忆库。

容器以 UID/GID `10001:10001` 运行。改用宿主机绑定目录时，需要自行保证这些目录可写；
不要通过修改整个仓库或现有记忆目录的权限来规避问题。
`.env` 只在运行时注入，不作为构建参数。不要把 `docker compose config` 的完整输出
直接粘贴到公开日志，因为它可能展开凭据；验证配置可使用 `docker compose config --quiet`。

Breeze 模型权重及自托管输出受其研究/非商业许可约束；容器化不会改变模型许可。
实际使用与分发请检查 [Breeze 许可](https://github.com/breezeblue-ai/breeze-tts#license-and-responsible-use)。

## macOS / Apple Silicon

使用原生 ARM64 Python 3.12，不在 Docker 或 Rosetta 中安装 MLX。
若尚未安装 Python 和 Node.js，可使用 `brew install python@3.12 node`。

```bash
./scripts/setup_studio_mlx.sh
# 编辑 .env，填写 DEEPSEEK_API_KEY
./scripts/run_studio_mlx.sh
```

准备脚本安装 `.[studio]` 和桌宠的 Node/Electron 依赖，并在缺失时创建 `.env` 模板；不会覆盖已有 `.env`、
重建现有虚拟环境或修改记忆数据。安装后只需执行原有启动脚本。
Mac 的模型、缓存和 Memory Spaces 仍留在原来的本机位置。
