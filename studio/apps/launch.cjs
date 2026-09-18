'use strict';
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const readline = require('node:readline/promises');
const { spawn } = require('node:child_process');
const runtime = require('./runtime.cjs');

const PROVIDERS = Object.freeze([
  { id: 'deepseek', label: 'DeepSeek', credential: 'DEEPSEEK_API_KEY' },
  { id: 'qwen', label: 'Qwen / DashScope', credential: 'DASHSCOPE_API_KEY' },
  { id: 'openai', label: 'OpenAI', credential: 'OPENAI_API_KEY' },
  { id: 'local', label: '本地模型 / MLX', credential: '', appleSiliconOnly: true },
]);

const MANAGED_ENVIRONMENT = Object.freeze([
  'VOICEMEM_DESKTOP_PROJECT_ROOT', 'VOICEMEM_DESKTOP_MEMORY_PROVIDER',
  'VOICEMEM_DESKTOP_REPLY_PROVIDER', 'VOICEMEM_DESKTOP_MANAGED_PROVIDER',
]);

function providerChoices(platform = process.platform, arch = process.arch) {
  return PROVIDERS.filter(provider => !provider.appleSiliconOnly
    || (platform === 'darwin' && arch === 'arm64'));
}

function savedModelProviders(platform = process.platform, env = process.env, home = os.homedir()) {
  const configured = String(env.VOICEMEM_DESKTOP_USER_DATA || '').trim();
  const base = configured ? path.resolve(configured) : platform === 'darwin'
    ? path.join(home, 'Library/Application Support')
    : platform === 'win32' ? env.APPDATA : '';
  if (!base) return null;
  try {
    const directory = configured ? base : path.join(base, 'VoiceMem Studio');
    const value = JSON.parse(fs.readFileSync(path.join(directory, 'model-services.json'), 'utf8'));
    const memoryProvider = String(value.memory?.provider || '').toLowerCase();
    const replyProvider = String(value.reply?.provider || '').toLowerCase();
    if (value.schema !== 1 || !runtime.MEMORY_PROVIDERS?.has?.(memoryProvider)) return null;
    if (!runtime.MANAGED_PROVIDERS?.has?.(replyProvider)) return null;
    return { memoryProvider, replyProvider };
  } catch { return null; }
}

function selectProvider(value, platform = process.platform, arch = process.arch) {
  const choices = providerChoices(platform, arch);
  const selected = String(value || '1').trim().toLowerCase();
  return choices.find((provider, index) => selected === provider.id || selected === String(index + 1));
}

async function chooseProvider({ platform, arch, input, output, askChoice } = {}) {
  const choices = providerChoices(platform, arch);
  output.write('\n选择 API（VoiceMem 记忆与 Studio 回复共用）：\n');
  choices.forEach((provider, index) => output.write(`  ${index + 1}. ${provider.label}\n`));
  while (true) {
    let answer;
    if (askChoice) answer = await askChoice('shared', choices);
    else {
      const terminal = readline.createInterface({ input, output });
      try { answer = await terminal.question('请选择 [1]：'); }
      finally { terminal.close(); }
    }
    const provider = selectProvider(answer, platform, arch);
    if (provider) return provider;
    output.write('请输入有效编号或 API 名称。\n');
  }
}

async function readSecret(prompt, { input = process.stdin, output = process.stdout } = {}) {
  if (!input.isTTY || typeof input.setRawMode !== 'function') return '';
  output.write(prompt);
  input.setRawMode(true);
  input.resume();
  return new Promise((resolve, reject) => {
    let secret = '';
    const finish = (error, value) => {
      input.off('data', onData);
      input.setRawMode(false);
      input.pause();
      output.write('\n');
      if (error) reject(error); else resolve(value);
    };
    const onData = chunk => {
      for (const character of chunk.toString('utf8')) {
        if (character === '\u0003') return finish(Object.assign(new Error('已取消启动。'), { code: 'CANCELLED' }));
        if (character === '\r' || character === '\n') return finish(null, secret.trim());
        if (character === '\u007f' || character === '\b') {
          if (secret) { secret = secret.slice(0, -1); output.write('\b \b'); }
        } else if (character >= ' ') { secret += character; output.write('*'); }
      }
    };
    input.on('data', onData);
  });
}

async function launchEnvironment({ platform = process.platform, env = process.env, input = process.stdin,
  arch = process.arch, output = process.stdout, askSecret = readSecret, askChoice,
  prepareMemory = async () => {}, savedProviders = savedModelProviders(platform, env) } = {}) {
  const projectDir = path.resolve(__dirname, '../..');
  const next = { ...env, VOICEMEM_DESKTOP_PROJECT_ROOT: projectDir };
  if (savedProviders) {
    next.VOICEMEM_DESKTOP_MEMORY_PROVIDER = savedProviders.memoryProvider;
    next.VOICEMEM_DESKTOP_REPLY_PROVIDER = savedProviders.replyProvider;
    next.VOICEMEM_DESKTOP_MANAGED_PROVIDER = savedProviders.replyProvider;
    output.write('\n正在使用 App 中保存的模型服务配置。\n');
    return next;
  }
  const provider = await chooseProvider({ platform, arch, input, output, askChoice });
  const memoryProvider = provider.id === 'local' ? PROVIDERS[0] : provider;
  const inherited = String(next.VOICEMEM_MEMORY_API_KEY
    || (provider.id === 'local' ? '' : next.VOICEMEM_STUDIO_API_KEY)
    || next[memoryProvider.credential] || '').trim();
  const secret = await askSecret(
    `请输入 ${memoryProvider.credential}${provider.id === 'local' ? '（供 VoiceMem 记忆处理使用）' : ''}`
      + `${inherited ? '（回车沿用当前环境变量）' : '（回车沿用项目 .env）'}：`,
    { input, output },
  );
  const key = secret || inherited;
  if (secret) next[memoryProvider.credential] = secret;
  if (key) {
    next.VOICEMEM_MEMORY_API_KEY = key;
    if (provider.id !== 'local') next.VOICEMEM_STUDIO_API_KEY = key;
  }
  next.VOICEMEM_DESKTOP_MEMORY_PROVIDER = memoryProvider.id;
  next.VOICEMEM_DESKTOP_REPLY_PROVIDER = provider.id;
  next.VOICEMEM_DESKTOP_MANAGED_PROVIDER = provider.id;
  output.write('\n正在检查 VoiceMem 环境并准备记忆、感知和转写模型…\n');
  await prepareMemory({ projectDir, provider: memoryProvider.id, env: next });
  output.write('VoiceMem 所需模型已准备完成。\n');
  return next;
}

function launchArguments(argv = process.argv.slice(2)) {
  const remote = argv.includes('--remote');
  return { remote, electronArgs: argv.filter(value => value !== '--remote') };
}

function remoteEnvironment(env = process.env) {
  const next = { ...env };
  for (const name of MANAGED_ENVIRONMENT) delete next[name];
  next.VOICEMEM_DESKTOP_CONFIGURE = '1';
  return next;
}

function startElectron(env, args) {
  delete env.ELECTRON_RUN_AS_NODE;
  const child = spawn(require('electron'), ['.', ...args], {
    cwd: __dirname, env, stdio: 'inherit',
  });
  child.on('error', error => { console.error(error.message); process.exitCode = 1; });
  child.on('exit', code => { process.exitCode = code ?? 1; });
}

async function main() {
  if (!['darwin', 'win32'].includes(process.platform)) {
    throw new Error('桌面 App 和桌宠仅面向 Windows / macOS。Linux 请启动 Studio 后端，或从其他电脑连接。');
  }
  const options = launchArguments();
  if (options.remote) {
    startElectron(remoteEnvironment(), options.electronArgs);
    return;
  }
  if (process.platform === 'darwin' && process.arch !== 'arm64') {
    throw new Error('本机 MLX 后端需要 Apple Silicon；Intel Mac 请运行 npm run start:remote 连接已有服务。');
  }
  if (!process.stdin.isTTY) throw new Error('npm start 需要交互终端来选择 API；连接已有服务请运行 npm run start:remote。');
  const env = await launchEnvironment({
    prepareMemory: ({ projectDir, provider, env: childEnv }) =>
      runtime.prepareManagedMemory(projectDir, provider, { platform: process.platform, env: childEnv }),
  });
  startElectron(env, options.electronArgs);
}

if (require.main === module) main().catch(error => {
  console.error(error.message);
  process.exitCode = error.code === 'CANCELLED' ? 130 : 1;
});

module.exports = { PROVIDERS, MANAGED_ENVIRONMENT, providerChoices, selectProvider, chooseProvider,
  savedModelProviders, readSecret, launchEnvironment, launchArguments, remoteEnvironment };
