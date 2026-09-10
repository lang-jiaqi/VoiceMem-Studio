"""Parse launch options before importing inference dependencies."""
import argparse

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="VoiceMem Studio")
    parser.add_argument("--mode", choices=("llm_tts", "realtime"), default="llm_tts")
    parser.add_argument("--space", default="demo-zh")
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
    if not 1 <= args.port <= 65535 or args.confirm_ms <= 0 or args.gamble_ms < 0 or args.spec_min_chars < 1:
        parser.error("port、confirm_ms、gamble_ms 或 spec_min_chars 超出有效范围")
    return args
