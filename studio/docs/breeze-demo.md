# 在 Apple Silicon 上运行 Breeze Studio

使用原生 Apple Silicon macOS 和 Python 3.12，在仓库根目录执行：

```bash
python -m pip install -e '.[studio]'
python -m studio --mode llm_tts --space demo-zh --lang zh --confirm_ms 200 --verbose
```

默认对话模型服务为 DeepSeek，凭据从启动进程的环境变量读取。想先检查环境、但不打开记忆空间或下载权重，可以在启动命令后加 `--check`。其他启动方式见 [Studio 说明](../README.md)。

`studio/core/utils/tts/initialize.py` 负责选择已审核的参考音频、Breeze 权重、随机种子、首个声学批次和缓存的 depth decoder。MLX 任务共用 GPU 调度器。参考录音位于 `studio/resources/voice/`；缺少已审核的附和片段时，等待阶段保持静音，不会临时生成替代音频。

Breeze 需要 `mlx==0.32.2` 和 `mlx-audio==0.5.1`。启动时会检查版本、将缺失的权重下载到 `studio/models/`，并在接受浏览器连接前预热首段音频。首段及后续片段都会保留两个声学帧，浏览器仍使用现有的接入缓冲。

传入 `--llm local` 会使用本地对话适配器及可取消的前缀缓存。最终 ASR 修正、对话结束判断（EOT）、推测输出、打断时已听内容的处理和填充音播放结束，仍由各自的阶段控制。

确定性的回归测试不能代替实际体验检查。麦克风、Metal、音质和端到端延迟都需要在装有完整模型的原生环境中验证。

Studio 还要求 `transformers==5.16.1` 和 `huggingface-hub>=1.5,<2` 与 `mlx-audio==0.5.1` 配套安装。请安装完整的 `studio` 依赖组，不要用 `--no-deps` 跳过版本约束。
