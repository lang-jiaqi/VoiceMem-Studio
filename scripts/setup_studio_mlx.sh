#!/usr/bin/env bash
set -euo pipefail
studio_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$studio_root"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo '[setup] MLX 需要原生 Apple Silicon macOS，请勿使用 Rosetta 或 Linux 容器。' >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo '[setup] 桌宠需要 Node.js 和 npm，请先安装，例如：brew install node' >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  if ! command -v python3.12 >/dev/null 2>&1; then
    echo '[setup] 请先安装 Python 3.12，例如：brew install python@3.12' >&2
    exit 1
  fi
  python3.12 -c 'import platform; assert platform.machine() == "arm64", "请选择原生 ARM64 Python 3.12"'
  python3.12 -m venv .venv
fi

.venv/bin/python -c 'import platform, sys; assert sys.version_info[:2] == (3, 12) and platform.machine() == "arm64", "现有 .venv 不是原生 ARM64 Python 3.12，请另行处理；脚本不会覆盖它"'
.venv/bin/python -m pip install -e '.[studio]'
npm ci --prefix "$studio_root/pet" --include=dev
if [[ ! -e .env ]]; then
  cp .env.example .env
fi
echo '[setup] 环境已准备。填写 .env 中的 API Key 后，运行 ./scripts/run_studio_mlx.sh'
