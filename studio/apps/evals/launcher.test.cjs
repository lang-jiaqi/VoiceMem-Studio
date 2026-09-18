'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const { PassThrough, Readable } = require('node:stream');
const { execFile } = require('node:child_process');
const { promisify } = require('node:util');
const launch = require('../launch.cjs');
const { shouldShowPet } = require('../desktop-visibility.cjs');

test('npm start selects a provider and passes its credential without printing it', async () => {
  let printed = ''; const prepared = [], choices = ['qwen'], secrets = ['shared-secret'];
  const output = { write(value) { printed += value; } };
  const env = await launch.launchEnvironment({
    platform: 'darwin', env: {}, input: Readable.from([]), output,
    savedProviders: null,
    askChoice: async () => choices.shift(), askSecret: async () => secrets.shift(),
    prepareMemory: async value => prepared.push(value),
  });
  assert.equal(env.VOICEMEM_DESKTOP_MEMORY_PROVIDER, 'qwen');
  assert.equal(env.VOICEMEM_DESKTOP_REPLY_PROVIDER, 'qwen');
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'shared-secret');
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, 'shared-secret');
  assert.equal(env.DASHSCOPE_API_KEY, 'shared-secret');
  assert.equal(path.basename(env.VOICEMEM_DESKTOP_PROJECT_ROOT), 'VoiceMem-Studio');
  assert.equal(prepared.length, 1); assert.equal(prepared[0].provider, 'qwen');
  assert.equal(prepared[0].env.VOICEMEM_MEMORY_API_KEY, 'shared-secret');
  assert.equal(choices.length, 0); assert.equal(secrets.length, 0);
  assert.equal(printed.includes('shared-secret'), false);
});

test('provider menu only offers local MLX on macOS', () => {
  assert.equal(launch.selectProvider('4', 'darwin', 'arm64').id, 'local');
  assert.equal(launch.selectProvider('local', 'darwin', 'x64'), undefined);
  assert.equal(launch.selectProvider('local', 'win32'), undefined);
  assert.equal(launch.selectProvider('', 'win32').id, 'deepseek');
});

test('remote source mode skips managed backend state and forwards other Electron arguments', () => {
  assert.deepEqual(launch.launchArguments(['--remote', '--inspect=9229']), {
    remote: true, electronArgs: ['--inspect=9229'],
  });
  const env = launch.remoteEnvironment({
    KEEP: 'yes', ELECTRON_RUN_AS_NODE: '1',
    VOICEMEM_DESKTOP_PROJECT_ROOT: '/fixture',
    VOICEMEM_DESKTOP_MEMORY_PROVIDER: 'deepseek',
    VOICEMEM_DESKTOP_REPLY_PROVIDER: 'deepseek',
    VOICEMEM_DESKTOP_MANAGED_PROVIDER: 'deepseek',
  });
  assert.equal(env.KEEP, 'yes');
  assert.equal(env.VOICEMEM_DESKTOP_CONFIGURE, '1');
  assert.equal(env.VOICEMEM_DESKTOP_PROJECT_ROOT, undefined);
  assert.equal(env.VOICEMEM_DESKTOP_MANAGED_PROVIDER, undefined);
});

test('local Studio replies ask once for the DeepSeek memory key', async () => {
  const choices = ['local'], prompts = [];
  const env = await launch.launchEnvironment({
    platform: 'darwin', env: {}, input: Readable.from([]), output: { write() {} },
    savedProviders: null,
    askChoice: async () => choices.shift(), askSecret: async prompt => { prompts.push(prompt); return 'memory-key'; },
  });
  assert.equal(prompts.length, 1);
  assert.match(prompts[0], /DEEPSEEK_API_KEY/);
  assert.equal(env.VOICEMEM_DESKTOP_MEMORY_PROVIDER, 'deepseek');
  assert.equal(env.VOICEMEM_DESKTOP_REPLY_PROVIDER, 'local');
  assert.equal(env.VOICEMEM_MEMORY_API_KEY, 'memory-key');
  assert.equal(env.VOICEMEM_STUDIO_API_KEY, undefined);
});

test('failed VoiceMem preparation does not ask for another API key', async () => {
  const choices = [];
  await assert.rejects(launch.launchEnvironment({
    platform: 'darwin', env: {}, input: Readable.from([]), output: { write() {} },
    savedProviders: null,
    askChoice: async purpose => { choices.push(purpose); return 'deepseek'; },
    askSecret: async () => 'memory-key',
    prepareMemory: async () => { throw new Error('preparation failed'); },
  }), /preparation failed/);
  assert.deepEqual(choices, ['shared']);
});

test('saved App model services skip the terminal provider prompt', async () => {
  const choices = [];
  const env = await launch.launchEnvironment({
    platform: 'darwin', env: {}, output: { write() {} },
    savedProviders: { memoryProvider: 'qwen', replyProvider: 'openai' },
    askChoice: async purpose => { choices.push(purpose); return 'deepseek'; },
    askSecret: async () => { throw new Error('must not ask'); },
    prepareMemory: async () => { throw new Error('must not prepare'); },
  });
  assert.equal(env.VOICEMEM_DESKTOP_MEMORY_PROVIDER, 'qwen');
  assert.equal(env.VOICEMEM_DESKTOP_REPLY_PROVIDER, 'openai');
  assert.deepEqual(choices, []);
});

test('source launcher discovers saved model providers without reading their secrets', async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-model-settings-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  await fs.writeFile(path.join(directory, 'model-services.json'), JSON.stringify({
    schema: 1,
    memory: { provider: 'deepseek', secret: 'encrypted-memory-value' },
    reply: { provider: 'local', secret: '' },
  }));
  assert.deepEqual(launch.savedModelProviders('darwin', {
    VOICEMEM_DESKTOP_USER_DATA: directory,
  }), { memoryProvider: 'deepseek', replyProvider: 'local' });
});

test('API key input is masked and returns the entered value', async () => {
  const input = new PassThrough();
  input.isTTY = true;
  input.setRawMode = value => { input.raw = value; };
  let printed = '';
  const secret = launch.readSecret('Key: ', { input, output: { write(value) { printed += value; } } });
  input.write('secret-value\n');
  assert.equal(await secret, 'secret-value');
  assert.equal(printed.includes('secret-value'), false);
  assert.match(printed, /\*{12}/);
  assert.equal(input.raw, false);
});

test('connection page uses the same compact visual shell as style selection', async () => {
  const html = await fs.readFile(path.join(__dirname, '../launcher.html'), 'utf8');
  const css = await fs.readFile(path.join(__dirname, '../launcher.css'), 'utf8');
  assert.match(html, /class="launcher-card"/);
  assert.match(html, /class="orb"/);
  assert.match(html, /连接 Studio/);
  assert.match(html, /连接并选择风格/);
  assert.doesNotMatch(html, /class="nav-item"|class="sidebar"/);
  assert.match(css, /--soft:#f7f8fa/);
  assert.match(css, /\.launcher-card\{[^}]*border-radius:24px/);
  const main = await fs.readFile(path.join(__dirname, '../main.cjs'), 'utf8');
  assert.match(main, /width: 680, height: 430/);
  assert.match(main, /label: '连接设置'/);
  assert.match(main, /if \(!managed\) await showLauncher\(configureOnly\)/);
  assert.match(main, /else if \(configureOnly\) publish\('idle'/);
  assert.match(main, /studio-preload\.cjs/);
});

test('Windows exposes Docker startup while macOS keeps the native MLX path', async () => {
  const script = await fs.readFile(path.join(__dirname, '../launcher.js'), 'utf8');
  for (const platform of ['win32', 'darwin', 'linux']) {
    const elements = new Map();
    const byId = id => {
      if (!elements.has(id)) elements.set(id, { addEventListener() {}, checked: false, hidden: false });
      return elements.get(id);
    };
    const state = { platform, version: 'fixture', settings: { serverUrl: 'http://localhost:8787', autoStartDocker: true, projectDir: '' }, status: { kind: 'idle', message: '' } };
    const context = { document: { getElementById: byId }, window: { studioDesktop: { onStatus() {}, state: async () => state } } };
    vm.runInNewContext(script, context);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(byId('docker-options').hidden, platform !== 'win32');
    assert.equal(byId('docker').checked, platform === 'win32');
    assert.equal(byId('docker-options').open, platform === 'win32');
    assert.match(byId('platform-hint').textContent, platform === 'win32' ? /WSL2/ : platform === 'darwin' ? /SSH.*HTTPS/ : /只部署后端/);
  }
});

test('conversation pages omit top mode links and the pet appears only in the background', async () => {
  const [technical, digital, main] = await Promise.all([
    fs.readFile(path.join(__dirname, '../ui/technical.html'), 'utf8'),
    fs.readFile(path.join(__dirname, '../ui/digital.html'), 'utf8'),
    fs.readFile(path.join(__dirname, '../main.cjs'), 'utf8'),
  ]);
  assert.doesNotMatch(technical, /class="app-nav"|\u8fd4\u56de\u9996\u9875|\u6570\u5b57\u4eba \u2197/);
  assert.doesNotMatch(digital, /class="style-nav"|>\u9996\u9875<|\u79d1\u6280\u98ce \u2197/);
  assert.match(main, /followFrontendVisibility\(window\)/);
  assert.match(main, /label: '\u540e\u53f0\u663e\u793a\u684c\u5ba0'/);

  const window = value => ({ isDestroyed: () => false, isFocused: () => value });
  assert.equal(shouldShowPet(true, window(false), undefined), true);
  assert.equal(shouldShowPet(true, window(true), undefined), false);
  assert.equal(shouldShowPet(true, window(false), window(true)), false);
  assert.equal(shouldShowPet(false, window(false), undefined), false);
  assert.equal(shouldShowPet(true, { isDestroyed: () => true }, undefined), false);
});

test('normal Linux entry points reject desktop startup before loading Electron', { skip: process.platform !== 'linux' }, async () => {
  for (const entry of ['../launch.cjs', '../../../pet/launch.cjs']) {
    await assert.rejects(promisify(execFile)(process.execPath, [path.resolve(__dirname, entry)]), error => {
      assert.equal(error.code, 1);
      assert.match(error.stderr, /Windows.*macOS/);
      return true;
    });
  }
});
