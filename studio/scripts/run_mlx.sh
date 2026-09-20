#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
studio_python="${STUDIO_PYTHON:-$project_root/.venv/bin/python}"
exec "$studio_python" -m studio --backend mlx --verbose "$@"
