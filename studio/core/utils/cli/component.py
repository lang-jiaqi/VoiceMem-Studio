"""Parse launch options before importing inference dependencies."""
import argparse
import os
import platform

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="VoiceMem Studio")
    parser.add_argument("--backend", choices=("cuda", "mlx"),
                        default=os.environ.get("STUDIO_BACKEND") or
                        ("mlx" if platform.system() == "Darwin" else "cuda"),
                        help="cuda=Linux NVIDIA 本地推理；mlx=Apple Silicon 本地推理")
    parser.add_argument("--tts-device", default=os.environ.get("STUDIO_TTS_DEVICE") or "",
                        help="Breeze CUDA 设备，如 cuda:0；默认与 --device 相同")
    parser.add_argument("--device", default=os.environ.get("STUDIO_DEVICE") or "",
                        help="Studio ASR/Router 设备，如 cuda:0；与 Breeze GPU 分开配置")
    parser.add_argument("--mode", choices=("llm_tts", "realtime"), default="llm_tts")
    parser.add_argument("--space", default="studio-zh")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh")
    parser.add_argument("--confirm_ms", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--llm", choices=("deepseek", "qwen", "openai", "local"), default="deepseek")
    parser.add_argument("--memory_root", default="")
    parser.add_argument("--spec_min_chars", type=int, default=6)
    parser.add_argument("--gamble_ms", type=int, default=200)
    parser.add_argument("--eot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backchannel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--no-file-log", action="store_true")
    parser.add_argument("--check", action="store_true", help="只检查环境、凭据和模型，不启动或下载")
    args = parser.parse_args(argv)
    args.device = args.device or ("cuda:0" if args.backend == "cuda" else "cpu")
    args.tts_device = args.tts_device or args.device
    if args.backend == "cuda" and args.llm == "local" and args.mode == "llm_tts":
        parser.error("--llm local 目前仅支持 --backend mlx；CUDA 请使用 deepseek/openai/qwen")
    if args.backend == "cuda" and not (args.device == "cuda" or
            (args.device.startswith("cuda:") and args.device[5:].isdigit())):
        parser.error("CUDA backend 的 --device 必须是 cuda 或 cuda:N")
    if args.backend == "cuda" and not (args.tts_device == "cuda" or
            (args.tts_device.startswith("cuda:") and args.tts_device[5:].isdigit())):
        parser.error("CUDA backend 的 --tts-device 必须是 cuda 或 cuda:N")
    if not 1 <= args.port <= 65535 or args.confirm_ms <= 0 or args.gamble_ms < 0 or args.spec_min_chars < 1:
        parser.error("port、confirm_ms、gamble_ms 或 spec_min_chars 超出有效范围")
    return args
