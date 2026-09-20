'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { setTimeout: delay } = require('node:timers/promises');

const DEFAULTS = Object.freeze({ serverUrl: 'http://127.0.0.1:8787', autoStartDocker: false, projectDir: '' });
const MANAGED_STARTUP_TIMEOUT_MS = 30 * 60 * 1000;
const MANAGED_PROVIDERS = new Set(['deepseek', 'qwen', 'openai', 'local']);
const MEMORY_PROVIDERS = new Set(['deepseek', 'qwen', 'openai']);
const PROVIDER_CREDENTIALS = Object.freeze({
  deepseek: 'DEEPSEEK_API_KEY', qwen: 'DASHSCOPE_API_KEY',
  openai: 'OPENAI_API_KEY', local: 'OPENAI_API_KEY',
});
const MODEL_SERVICE_DEFAULTS = Object.freeze({
  deepseek: Object.freeze({ model: 'deepseek-v4-flash', baseUrl: 'https://api.deepseek.com' }),
  qwen: Object.freeze({ model: 'qwen3.6-flash', baseUrl: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1' }),
  openai: Object.freeze({ model: 'gpt-4o', baseUrl: 'https://api.openai.com/v1' }),
  local: Object.freeze({ model: 'mlx-community/Qwen3.5-4B-4bit', baseUrl: '' }),
});

function managedStartupTimeout(env = process.env) {
  const value = String(env.VOICEMEM_DESKTOP_STARTUP_TIMEOUT_SECONDS || '').trim();
  if (!value) return MANAGED_STARTUP_TIMEOUT_MS;
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 60) {
    throw new Error('VOICEMEM_DESKTOP_STARTUP_TIMEOUT_SECONDS 必须是不小于 60 的秒数。');
  }
  return Math.round(seconds * 1000);
}

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

async function validateBackendProject(directory) {
  if (!directory || !path.isAbsolute(directory)) throw new Error('请选择包含 Studio 后端的绝对路径。');
  const root = await fs.realpath(directory);
  for (const name of ['pyproject.toml', 'studio/__main__.py']) {
    try {
      if ((await fs.stat(path.join(root, name))).isFile()) continue;
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    throw new Error(`所选目录缺少 ${name}，无法启动 Studio Python 后端。`);
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

function modelEndpoint(value) {
  let url;
  try { url = new URL(String(value || '').trim()); }
  catch { throw new Error('模型服务地址必须是完整的 HTTP/HTTPS URL。'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('模型服务地址不能包含用户名、密码、查询参数或片段。');
  }
  if (url.protocol === 'http:' && !loopback(url.hostname)) {
    throw new Error('远程模型服务必须使用 HTTPS；本机服务可以使用 HTTP。');
  }
  return url.href.replace(/\/$/, '');
}

function modelService(role, value = {}) {
  const providers = role === 'memory' ? MEMORY_PROVIDERS : MANAGED_PROVIDERS;
  const provider = String(value.provider || '').trim().toLowerCase();
  if (!providers.has(provider)) throw new Error(`${role === 'memory' ? '记忆' : '回复'}服务商不受支持。`);
  const defaults = MODEL_SERVICE_DEFAULTS[provider];
  if (provider === 'local') return { provider, model: defaults.model, baseUrl: '', apiKey: '' };
  const model = String(value.model || defaults.model).trim();
  if (!model || model.length > 300 || /[\x00-\x1f\x7f]/.test(model)) throw new Error('模型名称格式不正确。');
  const apiKey = String(value.apiKey || '').trim();
  if (apiKey.length > 8192 || /[\r\n]/.test(apiKey)) throw new Error('API Key 格式不正确。');
  return { provider, model, baseUrl: modelEndpoint(value.baseUrl || defaults.baseUrl), apiKey };
}

function modelServices(value) {
  if (!value || typeof value !== 'object') throw new Error('模型服务配置格式不正确。');
  return { memory: modelService('memory', value.memory), reply: modelService('reply', value.reply) };
}

function modelServicesFromLaunch(launch, env = process.env) {
  const service = (role, provider) => {
    const prefix = role === 'memory' ? 'VOICEMEM_MEMORY' : 'VOICEMEM_STUDIO';
    return {
      provider,
      model: env[`${prefix}_MODEL`],
      baseUrl: env[`${prefix}_BASE_URL`],
      apiKey: env[`${prefix}_API_KEY`] || env[PROVIDER_CREDENTIALS[provider]],
    };
  };
  return modelServices({
    memory: service('memory', launch.memoryProvider),
    reply: service('reply', launch.replyProvider),
  });
}

function updateModelService(current, role, value) {
  if (!['memory', 'reply'].includes(role)) throw new Error('未知的模型服务角色。');
  const existing = modelServices(current);
  const requestedProvider = String(value?.provider || '').trim().toLowerCase();
  const previous = existing[role];
  const apiKey = String(value?.apiKey || '').trim()
    || (requestedProvider === previous.provider ? previous.apiKey : '');
  const updated = modelService(role, { ...value, apiKey });
  if (updated.provider !== 'local' && !updated.apiKey) throw new Error('更换服务商时请输入对应的 API Key。');
  return { ...existing, [role]: updated };
}

function publicModelServices(value) {
  const services = modelServices(value);
  return Object.fromEntries(Object.entries(services).map(([role, service]) => [role, {
    provider: service.provider, model: service.model, baseUrl: service.baseUrl,
    configured: service.provider === 'local' || Boolean(service.apiKey),
  }]));
}

function modelServiceEnvironment(env, value) {
  const services = modelServices(value);
  const next = { ...env };
  for (const [role, service] of Object.entries(services)) {
    const prefix = role === 'memory' ? 'VOICEMEM_MEMORY' : 'VOICEMEM_STUDIO';
    for (const suffix of ['PROVIDER', 'API_KEY', 'MODEL', 'BASE_URL']) delete next[`${prefix}_${suffix}`];
    next[`${prefix}_PROVIDER`] = service.provider;
    next[`${prefix}_MODEL`] = service.model;
    if (service.baseUrl) next[`${prefix}_BASE_URL`] = service.baseUrl;
    if (service.apiKey) next[`${prefix}_API_KEY`] = service.apiKey;
  }
  return next;
}

async function loadModelServices(file, unprotect) {
  try {
    const stored = JSON.parse(await fs.readFile(file, 'utf8'));
    if (stored.schema !== 1) throw new Error('unsupported schema');
    return modelServices({
      memory: { ...stored.memory, apiKey: unprotect(stored.memory.secret) },
      reply: { ...stored.reply, apiKey: stored.reply.provider === 'local' ? '' : unprotect(stored.reply.secret) },
    });
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw new Error('无法读取已保存的模型服务配置，请重新配置。');
  }
}

async function saveModelServices(file, value, protect) {
  const services = modelServices(value);
  const stored = { schema: 1 };
  for (const [role, service] of Object.entries(services)) {
    stored[role] = {
      provider: service.provider, model: service.model, baseUrl: service.baseUrl,
      secret: service.provider === 'local' ? '' : protect(service.apiKey),
    };
  }
  await fs.mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
  await fs.writeFile(`${file}.tmp`, JSON.stringify(stored, null, 2), { mode: 0o600 });
  await fs.chmod(`${file}.tmp`, 0o600);
  await fs.rename(`${file}.tmp`, file);
}

function appendWslEnvironment(env, names) {
  const entries = String(env.WSLENV || '').split(':').filter(Boolean);
  for (const name of names)
    if (!entries.some(entry => entry.split('/')[0] === name)) entries.push(name);
  return entries.join(':');
}

async function managedBackendCommand(directory, memoryProvider, replyProvider, {
  platform = process.platform, arch = process.arch, env = process.env, signal, run = command, prepareStage = '',
} = {}) {
  if (!MEMORY_PROVIDERS.has(memoryProvider)) throw new Error(`不支持的 VoiceMem API：${memoryProvider}`);
  if (!MANAGED_PROVIDERS.has(replyProvider)) throw new Error(`不支持的 Studio 回复 API：${replyProvider}`);
  if (platform === 'win32' && replyProvider === 'local') throw new Error('本地 MLX 模型只支持 Apple Silicon macOS。');
  if (!['darwin', 'win32'].includes(platform)) throw new Error('App 本机后端只支持 Windows 和 macOS。');
  if (platform === 'darwin' && arch !== 'arm64') {
    throw new Error('本机 MLX 后端需要 Apple Silicon；Intel Mac 请运行 npm run start:remote 连接已有服务。');
  }
  const root = await validateBackendProject(directory);
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
  try {
    await run('wsl.exe', ['--cd', linuxRoot, '--exec', 'test', '-x', python], {
      cwd: root, signal, timeoutMs: 10000,
    });
  } catch {
    throw new Error(`找不到 WSL2 Studio Python 环境：${python}。请先在 WSL2 中安装 .venv-cuda。`);
  }
  const credentials = [PROVIDER_CREDENTIALS[memoryProvider], PROVIDER_CREDENTIALS[replyProvider]].filter(Boolean);
  childEnv.WSLENV = appendWslEnvironment(childEnv,
    [...credentials, 'VOICEMEM_MEMORY_API_KEY', 'VOICEMEM_STUDIO_API_KEY',
      'VOICEMEM_MEMORY_PROVIDER', 'VOICEMEM_MEMORY_MODEL', 'VOICEMEM_MEMORY_BASE_URL',
      'VOICEMEM_STUDIO_PROVIDER', 'VOICEMEM_STUDIO_MODEL', 'VOICEMEM_STUDIO_BASE_URL',
      'STUDIO_DESKTOP_PET', 'PYTHONUNBUFFERED']);
  return {
    file: 'wsl.exe', args: ['--cd', linuxRoot, '--exec', python, ...args],
    cwd: root, env: childEnv, url: DEFAULTS.serverUrl,
  };
}

async function startManagedBackend(directory, memoryProvider, replyProvider, {
  signal, platform = process.platform, arch = process.arch, env = process.env, run = command, spawnImpl = spawn,
} = {}) {
  const specification = await managedBackendCommand(directory, memoryProvider, replyProvider,
    { signal, platform, arch, env, run });
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
  signal, platform = process.platform, arch = process.arch, env = process.env, run = command, spawnImpl = spawn,
} = {}) {
  const specification = await managedBackendCommand(directory, provider, provider, {
    signal, platform, arch, env, run, prepareStage: 'memory',
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

module.exports = { DEFAULTS, MANAGED_STARTUP_TIMEOUT_MS, MANAGED_PROVIDERS, MEMORY_PROVIDERS, managedStartupTimeout,
  serverUrl, settings, sameOrigin, audioPermission, loadSettings, saveSettings,
  validateProject, validateBackendProject, command, managedLaunch, MODEL_SERVICE_DEFAULTS, modelEndpoint,
  modelService, modelServices, modelServicesFromLaunch, updateModelService,
  publicModelServices, modelServiceEnvironment, loadModelServices, saveModelServices,
  appendWslEnvironment, managedBackendCommand,
  startManagedBackend, prepareManagedMemory, publishedUrl, startDocker, probe, waitForStudio };
