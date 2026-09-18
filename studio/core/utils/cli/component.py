"""Parse launch options before importing inference dependencies."""
import argparse
import os
import platform
import sys


def select_provider(backend, read=input):
    """Choose the provider shared by memory and visible replies."""
    providers = [("deepseek", "DeepSeek"), ("qwen", "Qwen / DashScope"),
                 ("openai", "OpenAI")]
    if backend == "mlx":
        providers.append(("local", "本地模型 / MLX"))
    print("\n选择 API（VoiceMem 记忆与 Studio 回复共用）：", flush=True)
    for index, (_, label) in enumerate(providers, 1):
        print(f"  {index}. {label}", flush=True)
    while True:
        try:
            value = read("请选择 [1]：").strip().lower() or "1"
        except EOFError:
            raise SystemExit("未选择 API；可使用 --llm deepseek 非交互启动。") from None
        for index, (provider, _) in enumerate(providers, 1):
            if value in (str(index), provider):
                print(f"已选择：{provider}\n", flush=True)
                return provider
        print("请输入有效编号或 API 名称。", flush=True)

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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--llm", choices=("deepseek", "qwen", "openai", "local"),
                        default=os.environ.get("VOICEMEM_STUDIO_PROVIDER") or None,
                        help="Studio 可见回复 API；省略时在交互终端选择")
    parser.add_argument("--memory-llm", choices=("deepseek", "qwen", "openai"),
                        default=os.environ.get("VOICEMEM_MEMORY_PROVIDER") or None,
                        help="VoiceMem 记忆整理 API；默认跟随 --llm")
    parser.add_argument("--memory_root", default="")
    parser.add_argument("--spec_min_chars", type=int, default=6)
    parser.add_argument("--gamble_ms", type=int, default=200)
    parser.add_argument("--eot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backchannel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--no-file-log", action="store_true")
    parser.add_argument("--check", action="store_true", help="只检查环境、凭据和模型，不启动或下载")
    parser.add_argument("--prepare-stage", choices=("memory",), default="",
                        help="只检查并下载 VoiceMem 基础模型，然后退出")
    args = parser.parse_args(argv)
    interactive = argv is None and sys.stdin.isatty() and not args.check and args.mode == "llm_tts"
    if args.llm is None:
        args.llm = args.memory_llm or (select_provider(args.backend) if interactive else "deepseek")
    if args.memory_llm is None:
        args.memory_llm = args.llm if args.llm != "local" else "deepseek"
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
