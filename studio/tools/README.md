# Studio 开发工具

这里的脚本用于维护和排查问题，不是启动 Studio 服务或桌面 App 的必经步骤。需要运行时，请在仓库根目录使用 Studio 的 Python 环境。音频生成工具还需要 Apple Silicon、MLX，以及对应的本机语音素材和模型权重。

- `generate_backchannel_clips.py`：重新生成已审核的附和片段。它可能替换现有文件，运行前请确认参数和输出目录。
- `install_backchannel_clips.py`：将已审核片段放入本机缓存。
- `test_breeze_tts.py`：合成测试音频，输出到 `results/`。
- `breezetts2_mac_fast.py`：独立的 Breeze 优化实验，也是 `studio/core/utils/tts/cache.py` 的参考来源。
- `benchmark_qwen.py`：通过网络请求 Qwen，检查回复延迟；会使用 `DASHSCOPE_API_KEY` 发起真实模型调用。
- `seed_demo_zh.py` 和 `fixtures/demo_zh_clean10.json`：准备、验证合成的演示 Memory Space。`install` 操作会把已有的同名空间改名备份；不要对正在使用或包含个人数据的空间运行。
