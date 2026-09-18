'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { EventEmitter } = require('node:events');
const r = require('../runtime.cjs');

async function temporary(t) {
  const directory = await fs.mkdtemp(path.join(process.env.VOICEMEM_TEST_TMP || os.tmpdir(), 'studio-desktop-test-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  return directory;
}

test('server URL allows HTTPS and local HTTP but not unsafe origins or credentials', () => {
  for (const url of ['http://localhost:8787', 'http://127.0.0.1:8787/', 'http://[::1]:8787', 'https://studio.example.com/']) assert.equal(r.serverUrl(url), new URL(url).origin);
  for (const url of ['file:///etc/passwd', 'javascript:alert(1)', 'http://studio.example.com', 'http://localhost.evil.test', 'https://user:secret@studio.example.com', 'http://0.0.0.0:8787', 'https://studio.example.com/path', 'https://studio.example.com/?key=secret']) assert.throws(() => r.serverUrl(url));
});

test('only main-frame audio from the configured origin is eligible', () => {
  const origin = 'https://studio.example.com';
  assert.equal(r.audioPermission('media', { mediaTypes: ['audio'], isMainFrame: true }, origin, `${origin}/`), true);
  assert.equal(r.audioPermission('media', { mediaType: 'audio' }, origin, origin), true);
  for (const details of [{ mediaTypes: ['video'] }, { mediaTypes: ['audio', 'video'] }, { mediaTypes: [] }, { mediaType: 'unknown' }, { mediaTypes: ['audio'], isMainFrame: false }]) assert.equal(r.audioPermission('media', details, origin, origin), false);
  assert.equal(r.audioPermission('media', { mediaTypes: ['audio'] }, origin, 'https://other.example.com'), false);
  assert.equal(r.audioPermission('display-capture', {}, origin, origin), false);
});

test('settings persist only connection fields and reject malformed data', async t => {
  const directory = await temporary(t), file = path.join(directory, 'profile', 'connection.json');
  assert.deepEqual(await r.loadSettings(file), r.DEFAULTS);
  await r.saveSettings(file, { ...r.DEFAULTS, serverUrl: 'http://localhost:9000/', unrelatedSecret: 'test-only' });
  assert.deepEqual(await r.loadSettings(file), { ...r.DEFAULTS, serverUrl: 'http://localhost:9000' });
  assert.equal((await fs.readFile(file, 'utf8')).includes('unrelatedSecret'), false);
  await fs.writeFile(file, '{broken');
  await assert.rejects(r.loadSettings(file), /重新保存/);
  assert.throws(() => r.settings({ ...r.DEFAULTS, autoStartDocker: 'true' }));
});

test('managed npm launch validates its project and provider', () => {
  assert.equal(r.managedLaunch({}), null);
  assert.deepEqual(r.managedLaunch({
    VOICEMEM_DESKTOP_PROJECT_ROOT: '/fixture/project',
    VOICEMEM_DESKTOP_MEMORY_PROVIDER: 'qwen',
    VOICEMEM_DESKTOP_REPLY_PROVIDER: 'deepseek',
  }), { projectDir: '/fixture/project', memoryProvider: 'qwen', replyProvider: 'deepseek' });
  assert.deepEqual(r.managedLaunch({
    VOICEMEM_DESKTOP_PROJECT_ROOT: '/fixture/project',
    VOICEMEM_DESKTOP_MANAGED_PROVIDER: 'qwen',
  }), { projectDir: '/fixture/project', memoryProvider: 'qwen', replyProvider: 'qwen' });
  assert.throws(() => r.managedLaunch({
    VOICEMEM_DESKTOP_PROJECT_ROOT: '/fixture/project',
    VOICEMEM_DESKTOP_MANAGED_PROVIDER: 'unknown',
  }), /配置无效/);
});

test('managed model services validate custom endpoints without exposing credentials', () => {
  const services = r.modelServicesFromLaunch({ memoryProvider: 'deepseek', replyProvider: 'openai' }, {
    DEEPSEEK_API_KEY: 'memory-secret', OPENAI_API_KEY: 'reply-secret',
    VOICEMEM_MEMORY_MODEL: 'deepseek-chat',
    VOICEMEM_STUDIO_MODEL: 'custom-chat',
    VOICEMEM_STUDIO_BASE_URL: 'https://models.example.com/v1/',
  });
  assert.equal(services.memory.model, 'deepseek-chat');
  assert.equal(services.reply.baseUrl, 'https://models.example.com/v1');
  assert.deepEqual(r.publicModelServices(services).reply, {
    provider: 'openai', model: 'custom-chat', baseUrl: 'https://models.example.com/v1', configured: true,
  });
  assert.equal(JSON.stringify(r.publicModelServices(services)).includes('secret'), false);
  assert.throws(() => r.modelEndpoint('http://models.example.com/v1'), /HTTPS/);
  assert.equal(r.modelEndpoint('http://127.0.0.1:11434/v1/'), 'http://127.0.0.1:11434/v1');
});

test('model service updates retain same-provider keys and require a key after provider changes', () => {
  const current = r.modelServices({
    memory: { provider: 'deepseek', apiKey: 'memory-secret' },
    reply: { provider: 'openai', apiKey: 'reply-secret' },
  });
  const updated = r.updateModelService(current, 'reply', {
    provider: 'openai', model: 'gpt-4.1-mini', baseUrl: 'https://api.openai.com/v1', apiKey: '',
  });
  assert.equal(updated.reply.apiKey, 'reply-secret');
  assert.equal(updated.reply.model, 'gpt-4.1-mini');
  assert.throws(() => r.updateModelService(current, 'reply', {
    provider: 'qwen', model: 'qwen-plus', baseUrl: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', apiKey: '',
  }), /API Key/);
});

test('model service persistence protects keys and restores process environment', async t => {
  const directory = await temporary(t), file = path.join(directory, 'model-services.json');
  const services = r.modelServices({
    memory: { provider: 'deepseek', apiKey: 'memory-secret' },
    reply: { provider: 'openai', model: 'custom-chat', baseUrl: 'http://localhost:11434/v1', apiKey: 'reply-secret' },
  });
  const protect = value => Buffer.from(value).toString('base64');
  const unprotect = value => Buffer.from(value, 'base64').toString();
  await r.saveModelServices(file, services, protect);
  const raw = await fs.readFile(file, 'utf8');
  assert.equal(raw.includes('memory-secret') || raw.includes('reply-secret'), false);
  assert.deepEqual(await r.loadModelServices(file, unprotect), services);
  const env = r.modelServiceEnvironment({ KEEP: 'yes', VOICEMEM_STUDIO_API_KEY: 'old' }, services);
  assert.equal(env.KEEP, 'yes');
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'memory-secret');
  assert.equal(env.VOICEMEM_MEMORY_PROVIDER, 'deepseek');
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, 'reply-secret');
  assert.equal(env.VOICEMEM_STUDIO_PROVIDER, 'openai');
  assert.equal(env.VOICEMEM_STUDIO_MODEL, 'custom-chat');
  assert.equal(env.VOICEMEM_STUDIO_BASE_URL, 'http://localhost:11434/v1');
});

test('managed startup uses a long first-run timeout with a validated override', () => {
  assert.equal(r.managedStartupTimeout({}), 30 * 60 * 1000);
  assert.equal(r.managedStartupTimeout({ VOICEMEM_DESKTOP_STARTUP_TIMEOUT_SECONDS: '600' }), 600000);
  assert.throws(() => r.managedStartupTimeout({ VOICEMEM_DESKTOP_STARTUP_TIMEOUT_SECONDS: '20' }), /不小于 60/);
});

test('managed macOS backend uses MLX and keeps credentials out of arguments', async t => {
  const directory = await temporary(t);
  for (const file of ['compose.yaml', 'pyproject.toml']) await fs.writeFile(path.join(directory, file), 'fixture');
  const python = path.join(directory, '.venv/bin/python');
  await fs.mkdir(path.dirname(python), { recursive: true });
  await fs.writeFile(python, 'fixture');
  const specification = await r.managedBackendCommand(directory, 'qwen', 'deepseek', {
    platform: 'darwin', env: { DASHSCOPE_API_KEY: 'memory-secret', DEEPSEEK_API_KEY: 'reply-secret' },
  });
  assert.equal(specification.file, path.join(await fs.realpath(directory), '.venv/bin/python'));
  assert.ok(specification.args.includes('mlx'));
  assert.deepEqual(specification.args.slice(2, 8), ['--backend', 'mlx', '--memory-llm', 'qwen', '--llm', 'deepseek']);
  assert.equal(specification.args.includes('memory-secret') || specification.args.includes('reply-secret'), false);
  assert.equal(specification.env.STUDIO_DESKTOP_PET, '0');
});

test('managed macOS backend directs Intel users to remote mode', async t => {
  const directory = await temporary(t);
  for (const file of ['compose.yaml', 'pyproject.toml']) await fs.writeFile(path.join(directory, file), 'fixture');
  await assert.rejects(r.managedBackendCommand(directory, 'deepseek', 'deepseek', {
    platform: 'darwin', arch: 'x64', env: {},
  }), /npm run start:remote/);
});

test('managed Windows backend starts WSL CUDA without changing Docker', async t => {
  const directory = await temporary(t);
  for (const file of ['compose.yaml', 'pyproject.toml']) await fs.writeFile(path.join(directory, file), 'fixture');
  const calls = [];
  const specification = await r.managedBackendCommand(directory, 'openai', 'qwen', {
    platform: 'win32', env: { OPENAI_API_KEY: 'memory-secret', DASHSCOPE_API_KEY: 'reply-secret' },
    run: async (file, args) => {
      calls.push({ file, args });
      return args[0] === 'wslpath' ? '/mnt/c/VoiceMem-Studio' : '';
    },
  });
  assert.deepEqual(calls, [
    { file: 'wsl.exe', args: ['wslpath', '-a', await fs.realpath(directory)] },
    { file: 'wsl.exe', args: ['--cd', '/mnt/c/VoiceMem-Studio', '--exec', 'test', '-x',
      '/mnt/c/VoiceMem-Studio/.venv-cuda/bin/python'] },
  ]);
  assert.equal(specification.file, 'wsl.exe');
  assert.deepEqual(specification.args.slice(0, 4), [
    '--cd', '/mnt/c/VoiceMem-Studio', '--exec', '/mnt/c/VoiceMem-Studio/.venv-cuda/bin/python',
  ]);
  assert.ok(specification.args.includes('cuda'));
  assert.ok(specification.args.includes('openai'));
  assert.ok(specification.env.WSLENV.split(':').includes('DASHSCOPE_API_KEY'));
  assert.ok(specification.env.WSLENV.split(':').includes('OPENAI_API_KEY'));
  assert.ok(specification.env.WSLENV.split(':').includes('VOICEMEM_MEMORY_API_KEY'));
  assert.ok(specification.env.WSLENV.split(':').includes('VOICEMEM_STUDIO_MODEL'));
  assert.ok(specification.env.WSLENV.split(':').includes('VOICEMEM_STUDIO_BASE_URL'));
  assert.equal(specification.args.includes('memory-secret') || specification.args.includes('reply-secret'), false);
  assert.equal(calls.some(call => call.file === 'docker'), false);
});

test('managed Windows backend reports a missing WSL Python environment clearly', async t => {
  const directory = await temporary(t);
  for (const file of ['compose.yaml', 'pyproject.toml']) await fs.writeFile(path.join(directory, file), 'fixture');
  await assert.rejects(r.managedBackendCommand(directory, 'deepseek', 'deepseek', {
    platform: 'win32', env: {},
    run: async (_file, args) => {
      if (args[0] === 'wslpath') return '/mnt/c/VoiceMem-Studio';
      throw new Error('test failed');
    },
  }), /WSL2 Studio Python 环境.*\.venv-cuda/);
});

test('VoiceMem preparation finishes before Electron startup continues', async t => {
  const directory = await temporary(t);
  const root = await fs.realpath(directory);
  for (const file of ['compose.yaml', 'pyproject.toml']) await fs.writeFile(path.join(directory, file), 'fixture');
  const python = path.join(root, '.venv/bin/python');
  await fs.mkdir(path.dirname(python), { recursive: true }); await fs.writeFile(python, 'fixture');
  let specification;
  await r.prepareManagedMemory(directory, 'deepseek', {
    platform: 'darwin', env: { DEEPSEEK_API_KEY: 'memory-secret' },
    spawnImpl(file, args, options) {
      specification = { file, args, options }; const child = new EventEmitter(); child.killed = false;
      child.kill = () => { child.killed = true; }; setImmediate(() => child.emit('exit', 0, null)); return child;
    },
  });
  assert.equal(specification.file, python);
  assert.deepEqual(specification.args.slice(-2), ['--prepare-stage', 'memory']);
  assert.equal(specification.options.stdio, 'inherit');
});

test('Docker startup preserves compose overrides and uses the published port', async t => {
  const directory = await temporary(t);
  const root = await fs.realpath(directory);
  for (const file of ['compose.yaml', 'pyproject.toml', 'compose.override.yaml']) await fs.writeFile(path.join(directory, file), 'fixture');
  const calls = [];
  const url = await r.startDocker(directory, { platform: 'linux', env: {}, run: async (file, args, options) => {
    calls.push({ file, args, cwd: options.cwd });
    return args[0] === 'context' ? '"unix:///var/run/docker.sock"' : args.includes('port') ? '127.0.0.1:8788' : '';
  } });
  assert.equal(url, 'http://127.0.0.1:8788');
  assert.equal(calls.length, 3);
  assert.deepEqual(calls[1].args.slice(-7), ['up', '-d', '--no-build', '--no-recreate', '--pull', 'never', 'studio']);
  assert.ok(calls[1].args.includes(path.join(root, 'compose.override.yaml')));
  assert.equal(calls.every(call => call.file === 'docker' && call.cwd === root), true);
  assert.equal(calls.some(call => call.args.includes('down') || call.args.includes('build')), false);
});

test('Docker startup refuses remote contexts and unsupported local CUDA platforms', async t => {
  const directory = await temporary(t);
  for (const file of ['compose.yaml', 'pyproject.toml']) await fs.writeFile(path.join(directory, file), 'fixture');
  await assert.rejects(r.startDocker(directory, { platform: 'darwin' }), /Windows/);
  await assert.rejects(r.startDocker(directory, { platform: 'linux', env: { DOCKER_HOST: 'ssh://server' } }), /本机/);
  await assert.rejects(r.startDocker(directory, { platform: 'linux', env: {}, run: async () => '"ssh://server"' }), /context/);
  assert.equal(r.publishedUrl('0.0.0.0:9000'), 'http://127.0.0.1:9000');
  assert.equal(r.publishedUrl('[::]:9001'), 'http://127.0.0.1:9001');
  assert.throws(() => r.publishedUrl('unavailable'));
});

test('Windows Docker startup accepts only a local named pipe', async t => {
  const directory = await temporary(t);
  for (const file of ['compose.yaml', 'pyproject.toml']) await fs.writeFile(path.join(directory, file), 'fixture');
  for (const endpoint of ['npipe:////./pipe/docker_engine', 'npipe:////./pipe/dockerDesktopLinuxEngine']) {
    const calls = [];
    const url = await r.startDocker(directory, { platform: 'win32', env: { DOCKER_HOST: endpoint }, run: async (file, args) => {
      calls.push({ file, args });
      return args[0] === 'context' ? JSON.stringify(endpoint) : args.includes('port') ? '0.0.0.0:8787' : '';
    } });
    assert.equal(url, r.DEFAULTS.serverUrl);
    assert.deepEqual(calls[1].args.slice(-7), ['up', '-d', '--no-build', '--no-recreate', '--pull', 'never', 'studio']);
  }
  for (const endpoint of ['npipe:////remote-server/pipe/docker_engine', 'tcp://localhost:2375', 'ssh://server', 'unix:///var/run/docker.sock']) {
    await assert.rejects(r.startDocker(directory, { platform: 'win32', env: {}, run: async () => JSON.stringify(endpoint) }), /context/);
    await assert.rejects(r.startDocker(directory, { platform: 'win32', env: { DOCKER_HOST: endpoint } }), /本机/);
  }
});

test('readiness checks only GET the existing Web root', async t => {
  const server = http.createServer((req, res) => {
    assert.equal(req.method, 'GET'); assert.equal(req.url, '/');
    res.writeHead(200, { 'content-type': 'text/html' }); res.end('<html><title>VoiceMem</title></html>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  await r.probe(`http://127.0.0.1:${server.address().port}`);
});

test('readiness waiting retries, cancels and times out without starting another service', async () => {
  let count = 0;
  await r.waitForStudio(r.DEFAULTS.serverUrl, { intervalMs: 1, timeoutMs: 100, check: async () => { if (++count < 3) throw new Error('not ready'); } });
  assert.equal(count, 3);
  const cancel = new AbortController();
  const waiting = r.waitForStudio(r.DEFAULTS.serverUrl, { signal: cancel.signal, intervalMs: 100, check: async () => { cancel.abort(); throw new Error('not ready'); } });
  await assert.rejects(waiting, { name: 'AbortError' });
  await assert.rejects(r.waitForStudio(r.DEFAULTS.serverUrl, { timeoutMs: 4, intervalMs: 1, check: async () => { throw new Error('offline'); } }), /尚未就绪/);
});

test('packaging excludes inference weights, credentials, recordings, backend source and tests', async () => {
  const pkg = JSON.parse(await fs.readFile(path.join(__dirname, '..', 'package.json'), 'utf8'));
  assert.ok(pkg.build.files.includes('main.cjs'));
  assert.ok(pkg.build.files.includes('studio-preload.cjs'));
  assert.ok(pkg.build.files.includes('.pet-runtime/**/*'));
  for (const item of pkg.build.files) assert.equal(/\.env|models|record|prompt|tests|evals|voicemem_memoryspace|\.\.\//.test(item), false);
  assert.equal(pkg.devDependencies.electron.startsWith('^'), false);
  assert.equal(pkg.scripts['dist:linux'], undefined);
  assert.equal(pkg.build.linux, undefined);
  assert.ok(pkg.scripts['dist:win'].includes('--win nsis --x64'));
  assert.ok(pkg.scripts['dist:mac'].includes('--mac dmg zip --arm64'));
});
