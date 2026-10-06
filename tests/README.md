# 测试

这里保存自动回归测试和人工界面检查。测试使用模拟模型、合成音频帧和临时数据库，不使用个人记忆库、录音或真实 API。它们不会在应用启动或对话时自动运行。

| 目录 | 用途 |
| --- | --- |
| `voicemem/` | 记忆、检索、ASR 生命周期和 SDK 接口回归 |
| `studio/` | 对话、播放、取消、Harness、服务配置和资源打包回归 |
| `frontend/` | 浏览器、音频 worklet、桌面启动器和宠物界面模拟测试 |
| `helpers/` | 共用的合成输入、模拟回复、会话和展示夹具 |
| `manual/` | 需要在浏览器中检查的界面页面 |

## Python

在仓库根目录使用 Python 3.12。已有开发环境可直接运行：

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

只检查一个分组或文件：

```bash
.venv/bin/python -m unittest discover -s tests/voicemem -t .
.venv/bin/python -m unittest tests.studio.test_context_commit
```

`-t .` 指定仓库根目录，避免测试目录中的 `voicemem`、`studio` 与运行时代码重名。

如只准备测试环境，可在独立虚拟环境中安装 `tests/requirements.txt`，使用对应解释器运行上述命令。该依赖清单不安装推理模型；本机 MLX 解码器检查需要已有 `mlx-lm`，未安装时跳过。Docker 配置检查在没有 Docker CLI 时跳过。

## JavaScript

在仓库根目录运行全部前端回归：

```bash
node --test tests/frontend/*.cjs
```

桌面 App 原有入口继续有效，只运行桌面相关回归：

```bash
npm --prefix studio/apps test
```

端口冲突测试需要允许临时监听本机回环端口。宠物资源打包测试需要已安装 App 的 npm 依赖；如缺少依赖，先按桌面 App 的安装说明准备环境。

## 人工检查

从仓库根目录启动静态文件服务：

```bash
python -m http.server 8000 --bind 127.0.0.1
```

在浏览器打开 `http://127.0.0.1:8000/tests/manual/markdown-typography.html`，检查不同界面与视口下的 Markdown 字体。此页面属于人工检查，不计入自动测试通过数量。

## 维护

- 新用例放到对应功能分组；共用夹具放在 `helpers/`，不从另一个测试文件导入夹具。
- 覆盖故障行为、取消、断开和状态隔离；不要仅为了固定当前源码写法而添加断言。
- 延迟、模型效果和真实音频体验评测放在仓库根目录的 `evals/`，它们需要另外准备环境或数据。
- 根目录下的本地实验测试仍由 `.gitignore` 忽略；不要把个人数据放进公开夹具。


## 体验站（demo 分支）

账号隔离、手机资源缓存和 Qwen 独立会话连接的回归同样使用临时数据与模拟 API：

```bash
.venv/bin/python -m unittest tests.studio.test_public_demo tests.studio.test_demo_lifecycle tests.studio.test_qwen_audio_api
node --test tests/frontend/test_mobile_avatar_loading.cjs tests/frontend/test_mobile_memory_status.cjs
```

手机资源接口测试需要 App 的 npm 依赖；先运行 `npm --prefix studio/apps run ensure:deps`。测试不会使用部署机器上的真实账号或 Key。
