#!/usr/bin/env bash
set -euo pipefail
studio_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$studio_root"
studio_python="${STUDIO_PYTHON:-$studio_root/.venv/bin/python}"
exec "$studio_python" -m studio --backend mlx --verbose "$@"
