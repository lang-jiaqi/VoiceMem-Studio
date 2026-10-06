#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
studio_python="${STUDIO_PYTHON:-$project_root/.venv/bin/python}"
export STUDIO_PUBLIC_DEMO=1
exec "$studio_python" -m studio --host 127.0.0.1 --port 8790 \
  --mode llm_tts --llm deepseek --memory-llm deepseek --lang zh "$@"
