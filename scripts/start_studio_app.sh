#!/usr/bin/env bash
set -euo pipefail

studio_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
studio_npm="${STUDIO_NPM:-npm}"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo '[studio] 本机一键入口需要 Apple Silicon macOS。其他平台请查看 studio/apps/README.md。' >&2
  exit 1
fi

exec "$studio_npm" --prefix "$studio_root/studio/apps" start
