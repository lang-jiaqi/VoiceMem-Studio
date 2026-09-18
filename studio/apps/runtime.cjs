'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { setTimeout: delay } = require('node:timers/promises');

const DEFAULTS = Object.freeze({ serverUrl: 'http://127.0.0.1:8787', autoStartDocker: false, projectDir: '' });
const MANAGED_PROVIDERS = new Set(['deepseek', 'qwen', 'openai', 'local']);
const MEMORY_PROVIDERS = new Set(['deepseek', 'qwen', 'openai']);
const PROVIDER_CREDENTIALS = Object.freeze({
  deepseek: 'DEEPSEEK_API_KEY', qwen: 'DASHSCOPE_API_KEY',
  openai: 'OPENAI_API_KEY', local: 'OPENAI_API_KEY',
});

function loopback(hostname) {
  return hostname === 'localhost' || hostname === '[::1]' || /^127(?:\.\d{1,3}){3}$/.test(hostname);
}

function serverUrl(value) {
  let url;
  try { url = new URL(String(value).trim()); } catch { throw new Error('请输入完整服务地址，例如 http://127.0.0.1:8787'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('服务地址只填写 HTTP/HTTPS 根地址，不包含用户名、密码、路径或查询参数。');
  }
  if (url.protocol === 'http:' && !loopback(url.hostname)) {
    if (url.hostname === '0.0.0.0') throw new Error('0.0.0.0 是监听地址，请改填 http://127.0.0.1:8787 或实际服务器地址。');
    throw new Error('远程服务请使用 HTTPS；也可以通过 SSH 转发后连接 http://127.0.0.1:8787。');
  }
  return url.origin;
}

function settings(value = DEFAULTS) {
  if (!value || typeof value !== 'object' || typeof value.autoStartDocker !== 'boolean' || typeof value.projectDir !== 'string') {
    throw new Error('连接配置格式不正确，请重新设置。');
  }
  return { serverUrl: serverUrl(value.serverUrl), autoStartDocker: value.autoStartDocker, projectDir: value.projectDir };
}

function sameOrigin(value, origin) {
  try { return new URL(value).origin === origin; } catch { return false; }
}

function audioPermission(permission, details, origin, requestingUrl) {
  if (permission !== 'media' || details.isMainFrame === false || !sameOrigin(requestingUrl, origin)) return false;
  if (Array.isArray(details.mediaTypes)) return details.mediaTypes.length === 1 && details.mediaTypes[0] === 'audio';
  return details.mediaType === 'audio';
}

async function loadSettings(file) {
  try { return settings(JSON.parse(await fs.readFile(file, 'utf8'))); }
  catch (error) {
    if (error.code === 'ENOENT') return { ...DEFAULTS };
    throw new Error('无法读取连接配置，请在连接页重新保存设置。');
  }
}

async function saveSettings(file, value) {
  const validated = settings(value);
  await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
  await fs.writeFile(`${file}.tmp`, JSON.stringify(validated, null, 2), { mode: 0o600 });
  await fs.rename(`${file}.tmp`, file);
}

async function validateProject(directory) {
  if (!directory || !path.isAbsolute(directory)) throw new Error('请选择已配置好的 VoiceMem-Studio 项目目录。');
  const root = await fs.realpath(directory);
  for (const name of ['compose.yaml', 'pyproject.toml']) {
    if (!(await fs.stat(path.join(root, name))).isFile()) throw new Error('所选目录不是完整的 VoiceMem-Studio 项目。');
  }
  return root;
}

function command(file, args, { cwd, signal, timeoutMs = 60000, spawnImpl = spawn } = {}) {
  return new Promise((resolve, reject) => {
    const control = AbortSignal.any([...(signal ? [signal] : []), AbortSignal.timeout(timeoutMs)]);
    let output = '', errors = '';
    const child = spawnImpl(file, args, { cwd, signal: control, shell: false, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    child.stdout.on('data', data => { output = (output + data.toString()).slice(-16000); });
    child.stderr.on('data', data => { errors = (errors + data.toString()).slice(-16000); });
    child.once('error', error => reject(error.code === 'ENOENT'
      ? new Error(file === 'docker' ? '找不到 Docker 命令，请先安装 Docker 和 Compose。' : `找不到 ${file} 命令。`)
      : error));
    const label = file === 'docker' ? 'Docker' : file;
    child.once('close', code => code === 0 ? resolve(output.trim()) : reject(new Error(errors.trim() || `${label} 命令退出，状态码 ${code}`)));
  });
}

function managedLaunch(env = process.env) {
  const projectDir = String(env.VOICEMEM_DESKTOP_PROJECT_ROOT || '').trim();
  const legacy = String(env.VOICEMEM_DESKTOP_MANAGED_PROVIDER || '').trim().toLowerCase();
  const memoryProvider = String(env.VOICEMEM_DESKTOP_MEMORY_PROVIDER || legacy).trim().toLowerCase();
  const replyProvider = String(env.VOICEMEM_DESKTOP_REPLY_PROVIDER || legacy).trim().toLowerCase();
  if (!projectDir && !memoryProvider && !replyProvider) return null;
  if (!path.isAbsolute(projectDir) || !MEMORY_PROVIDERS.has(memoryProvider)
      || !MANAGED_PROVIDERS.has(replyProvider)) {
    throw new Error('App 本机后端启动配置无效，请重新运行 npm start。');
  }
  return { projectDir, memoryProvider, replyProvider };
}

function appendWslEnvironment(env, names) {
  const entries = String(env.WSLENV || '').split(':').filter(Boolean);
  for (const name of names)
    if (!entries.some(entry => entry.split('/')[0] === name)) entries.push(name);
  return entries.join(':');
}

async function managedBackendCommand(directory, memoryProvider, replyProvider, {
  platform = process.platform, env = process.env, signal, run = command, prepareStage = '',
} = {}) {
  if (!MEMORY_PROVIDERS.has(memoryProvider)) throw new Error(`不支持的 VoiceMem API：${memoryProvider}`);
  if (!MANAGED_PROVIDERS.has(replyProvider)) throw new Error(`不支持的 Studio 回复 API：${replyProvider}`);
  if (platform === 'win32' && replyProvider === 'local') throw new Error('本地 MLX 模型只支持 Apple Silicon macOS。');
  if (!['darwin', 'win32'].includes(platform)) throw new Error('App 本机后端只支持 Windows 和 macOS。');
  const root = await validateProject(directory);
  const backend = platform === 'darwin' ? 'mlx' : 'cuda';
  const args = ['-m', 'studio', '--backend', backend, '--memory-llm', memoryProvider, '--llm', replyProvider,
    '--host', '127.0.0.1', '--port', '8787', '--verbose'];
  if (prepareStage) args.push('--prepare-stage', prepareStage);
  const childEnv = { ...env, STUDIO_DESKTOP_PET: '0', PYTHONUNBUFFERED: '1' };
  if (platform === 'darwin') {
    const python = path.resolve(root, env.STUDIO_PYTHON || '.venv/bin/python');
    try { await fs.access(python); }
    catch { throw new Error(`找不到 macOS Studio Python 环境：${python}。请先完成 MLX 环境安装。`); }
    return { file: python, args, cwd: root, env: childEnv, url: DEFAULTS.serverUrl };
  }
  let linuxRoot;
  try {
    linuxRoot = await run('wsl.exe', ['wslpath', '-a', root], { cwd: root, signal, timeoutMs: 10000 });
  } catch (error) {
    throw new Error(`无法进入 WSL2：${error.message}`);
  }
  if (!linuxRoot) throw new Error('WSL2 无法解析 VoiceMem-Studio 项目路径。');
  const relativePython = String(env.STUDIO_WSL_PYTHON || '.venv-cuda/bin/python').replace(/\\/g, '/');
  const python = relativePython.startsWith('/') ? relativePython : `${linuxRoot}/${relativePython}`;
  const credentials = [PROVIDER_CREDENTIALS[memoryProvider], PROVIDER_CREDENTIALS[replyProvider]].filter(Boolean);
  childEnv.WSLENV = appendWslEnvironment(childEnv,
    [...credentials, 'VOICEMEM_MEMORY_API_KEY', 'VOICEMEM_STUDIO_API_KEY',
      'STUDIO_DESKTOP_PET', 'PYTHONUNBUFFERED']);
  return {
    file: 'wsl.exe', args: ['--cd', linuxRoot, '--exec', python, ...args],
    cwd: root, env: childEnv, url: DEFAULTS.serverUrl,
  };
}

async function startManagedBackend(directory, memoryProvider, replyProvider, {
  signal, platform = process.platform, env = process.env, run = command, spawnImpl = spawn,
} = {}) {
  const specification = await managedBackendCommand(directory, memoryProvider, replyProvider,
    { signal, platform, env, run });
  signal?.throwIfAborted();
  let child;
  try {
    child = spawnImpl(specification.file, specification.args, {
      cwd: specification.cwd, env: specification.env, shell: false,
      windowsHide: true, stdio: 'inherit',
    });
  } catch (error) { throw new Error(`无法启动本机 Studio 后端：${error.message}`); }
  let settled = false;
  let finish;
  const exited = new Promise(resolve => { finish = value => { if (!settled) { settled = true; resolve(value); } }; });
  child.once('error', error => finish({ error }));
  child.once('exit', (code, exitSignal) => finish({ code, signal: exitSignal }));
  const stop = () => {
    if (!settled && !child.killed) child.kill(platform === 'win32' ? undefined : 'SIGTERM');
  };
  const abort = () => stop();
  if (signal) signal.addEventListener('abort', abort, { once: true });
  void exited.then(() => signal?.removeEventListener('abort', abort));
  return { ...specification, child, exited, stop };
}

async function prepareManagedMemory(directory, provider, {
  signal, platform = process.platform, env = process.env, run = command, spawnImpl = spawn,
} = {}) {
  const specification = await managedBackendCommand(directory, provider, provider, {
    signal, platform, env, run, prepareStage: 'memory',
  });
  signal?.throwIfAborted();
  await new Promise((resolve, reject) => {
    let settled = false;
    const finish = action => value => {
      if (settled) return;
      settled = true; signal?.removeEventListener('abort', abort); action(value);
    };
    let child;
    try {
      child = spawnImpl(specification.file, specification.args, {
        cwd: specification.cwd, env: specification.env, shell: false,
        windowsHide: true, stdio: 'inherit',
      });
    } catch (error) { reject(new Error(`无法准备 VoiceMem：${error.message}`)); return; }
    const abort = () => { if (!child.killed) child.kill(platform === 'win32' ? undefined : 'SIGTERM'); };
    signal?.addEventListener('abort', abort, { once: true });
    child.once('error', finish(error => reject(new Error(`无法准备 VoiceMem：${error.message}`))));
    child.once('exit', finish((code, exitSignal) => code === 0 ? resolve() : reject(new Error(
      `VoiceMem 准备失败${code === null ? '' : `，状态码 ${code}`}${exitSignal ? `（${exitSignal}）` : ''}。`,
    ))));
  });
}

function publishedUrl(output) {
  const line = output.trim().split('\n')[0];
  const match = /:(\d+)$/.exec(line);
  if (!match || +match[1] < 1 || +match[1] > 65535) throw new Error('无法取得 studio 容器的发布端口，请检查 Compose 配置。');
  return `http://127.0.0.1:${+match[1]}`;
}

async function startDocker(directory, { signal, run = command, platform = process.platform, env = process.env } = {}) {
  if (!['win32', 'linux'].includes(platform)) throw new Error('桌面 Docker 自动启动用于 Windows 的 WSL2/NVIDIA 后端。Mac 请连接原生 MLX 或远程服务。');
  const root = await validateProject(directory);
  const isLocal = endpoint => typeof endpoint === 'string' && (platform === 'win32'
    ? /^npipe:\/{2,4}\.\/pipe\/[A-Za-z0-9_.-]+$/.test(endpoint)
    : endpoint.startsWith('unix://'));
  if (env.DOCKER_HOST && !isLocal(env.DOCKER_HOST)) throw new Error('自动启动只支持本机 Docker；远程服务请直接填写服务地址。');
  const endpoint = JSON.parse(await run('docker', ['context', 'inspect', '--format', '{{json .Endpoints.docker.Host}}'], { cwd: root, signal }));
  if (!isLocal(endpoint)) throw new Error('当前 Docker context 不是本机，请切换到本机 context。');
  const compose = ['compose', '--project-directory', root, '-f', path.join(root, 'compose.yaml')];
  try {
    if ((await fs.stat(path.join(root, 'compose.override.yaml'))).isFile()) compose.push('-f', path.join(root, 'compose.override.yaml'));
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  await run('docker', [...compose, 'up', '-d', '--no-build', '--no-recreate', '--pull', 'never', 'studio'], { cwd: root, signal });
  return publishedUrl(await run('docker', [...compose, 'port', 'studio', '8787'], { cwd: root, signal }));
}

async function probe(url, { signal, timeoutMs = 2500, fetchImpl = fetch } = {}) {
  const response = await fetchImpl(`${serverUrl(url)}/`, {
    signal: AbortSignal.any([...(signal ? [signal] : []), AbortSignal.timeout(timeoutMs)]), redirect: 'error',
  });
  let reader;
  try {
    if (!response.ok) throw new Error(`服务返回 HTTP ${response.status}`);
    reader = response.body.getReader();
    let html = '';
    while (html.length < 16384) {
      const { value, done } = await reader.read();
      if (done) break;
      html += new TextDecoder().decode(value.subarray(0, 16384 - html.length));
      if (/<title>\s*VoiceMem(?: Studio)?\s*<\/title>/i.test(html)) return;
    }
    throw new Error('该地址没有返回 VoiceMem 页面，请检查服务地址。');
  } finally {
    try { await (reader ? reader.cancel() : response.body?.cancel()); } catch { /* The request may already have been aborted. */ }
  }
}

async function waitForStudio(url, { signal, timeoutMs = 180000, intervalMs = 1500, check = probe, onWait = () => {} } = {}) {
  const started = Date.now();
  let lastError;
  while (Date.now() - started < timeoutMs) {
    signal?.throwIfAborted();
    try { await check(url, { signal }); return; }
    catch (error) { signal?.throwIfAborted(); lastError = error; }
    onWait(Math.floor((Date.now() - started) / 1000));
    await delay(intervalMs, undefined, { signal });
  }
  throw new Error(`服务尚未就绪，请检查 Studio 后端日志或稍后重试。${lastError?.message || ''}`);
}

module.exports = { DEFAULTS, serverUrl, settings, sameOrigin, audioPermission, loadSettings, saveSettings,
  validateProject, command, managedLaunch, appendWslEnvironment, managedBackendCommand,
  startManagedBackend, prepareManagedMemory, publishedUrl, startDocker, probe, waitForStudio };
