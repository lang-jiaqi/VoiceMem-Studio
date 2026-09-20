'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

if (process.platform === 'darwin' && process.arch === 'arm64') {
  const root = path.resolve(__dirname, '../../..');
  const marker = path.join(root, '.venv/.voicemem-studio-setup-complete');
  const setup = path.join(root, 'studio/scripts/setup_mlx.sh');
  const sources = [path.join(root, 'pyproject.toml'), setup];
  const stale = !fs.existsSync(marker) || sources.some(file => fs.statSync(file).mtimeMs > fs.statSync(marker).mtimeMs);
  if (stale) {
    const result = spawnSync('/bin/bash', [setup], { cwd: root, env: process.env, stdio: 'inherit' });
    if (result.error) throw result.error;
    if (result.status !== 0) process.exit(result.status ?? 1);
  }
}
