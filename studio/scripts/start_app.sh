#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
studio_npm="${STUDIO_NPM:-npm}"

if [[ "$(uname -s)" != Darwin ]]; then
  echo '[studio] 这个快捷入口用于 macOS。Windows 从 studio/apps 运行 npm start，Linux 只部署后端。' >&2
  exit 1
fi

exec "$studio_npm" --prefix "$project_root/studio/apps" start "$@"
