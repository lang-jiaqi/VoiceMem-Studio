'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const required = Object.freeze([
  ['Electron', 'node_modules/electron/path.txt'],
  ['PixiJS', 'node_modules/pixi.js/dist/browser/pixi.min.js'],
  ['Pixi Live2D', 'node_modules/pixi-live2d-display/dist/cubism4.min.js'],
]);

function missingDependencies(apps = path.resolve(__dirname, '..')) {
  return required.filter(([, relative]) => !fs.existsSync(path.join(apps, relative)));
}

function installCommand(env = process.env) {
  if (env.npm_execpath) return { file: process.execPath, args: [env.npm_execpath] };
  return { file: process.platform === 'win32' ? 'npm.cmd' : 'npm', args: [] };
}

function installEnvironment(env) {
  const child = { ...env };
  for (const key of Object.keys(child)) {
    if (['npm_config_ignore_scripts', 'npm_config_omit'].includes(key.toLowerCase())) delete child[key];
  }
  child.npm_config_ignore_scripts = 'false';
  child.npm_config_omit = '';
  return child;
}

function ensureDependencies({ apps = path.resolve(__dirname, '..'), env = process.env, run = spawnSync } = {}) {
  const missing = missingDependencies(apps);
  if (!missing.length) {
    console.log('[desktop] Node dependencies are ready.');
    return false;
  }
  if (!fs.existsSync(path.join(apps, 'package-lock.json'))) {
    throw new Error('缺少 studio/apps/package-lock.json，无法安装锁定的 App 依赖。');
  }
  console.log(`[desktop] Missing ${missing.map(([name]) => name).join(', ')}; installing locked App dependencies…`);
  const npm = installCommand(env);
  const result = run(npm.file, [...npm.args, 'ci', '--include=dev', '--ignore-scripts=false', '--no-audit', '--no-fund'], {
    cwd: apps, env: installEnvironment(env), stdio: 'inherit', shell: false,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`npm ci 安装 App 依赖失败，状态码 ${result.status ?? 'unknown'}。`);
  const remaining = missingDependencies(apps);
  if (remaining.length) throw new Error(`依赖安装完成后仍缺少：${remaining.map(([name]) => name).join(', ')}。`);
  return true;
}

if (require.main === module) {
  try { ensureDependencies(); }
  catch (error) { console.error(`[desktop] ${error.message}`); process.exitCode = 1; }
}

module.exports = { required, missingDependencies, installCommand, installEnvironment, ensureDependencies };
