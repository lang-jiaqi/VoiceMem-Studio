'use strict';
const path = require('node:path');
const readline = require('node:readline/promises');
const { spawn } = require('node:child_process');
const runtime = require('./runtime.cjs');

const PROVIDERS = Object.freeze([
  { id: 'deepseek', label: 'DeepSeek', credential: 'DEEPSEEK_API_KEY' },
  { id: 'qwen', label: 'Qwen / DashScope', credential: 'DASHSCOPE_API_KEY' },
  { id: 'openai', label: 'OpenAI', credential: 'OPENAI_API_KEY' },
  { id: 'local', label: '本地模型 / MLX', credential: '', macOnly: true },
]);

function providerChoices(platform = process.platform) {
  return PROVIDERS.filter(provider => !provider.macOnly || platform === 'darwin');
}

function selectProvider(value, platform = process.platform) {
  const choices = providerChoices(platform);
  const selected = String(value || '1').trim().toLowerCase();
  return choices.find((provider, index) => selected === provider.id || selected === String(index + 1));
}

async function chooseProvider({ platform, input, output, askChoice } = {}) {
  const choices = providerChoices(platform);
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
    const provider = selectProvider(answer, platform);
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
  output = process.stdout, askSecret = readSecret, askChoice, prepareMemory = async () => {} } = {}) {
  const projectDir = path.resolve(__dirname, '../..');
  const next = { ...env, VOICEMEM_DESKTOP_PROJECT_ROOT: projectDir };
  const provider = await chooseProvider({ platform, input, output, askChoice });
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

async function main() {
  if (!['darwin', 'win32'].includes(process.platform)) {
    throw new Error('桌面 App 和桌宠仅面向 Windows / macOS。Linux 请启动 Studio 后端，或从其他电脑连接。');
  }
  if (!process.stdin.isTTY) throw new Error('npm start 需要交互终端来选择 API；请在 Terminal 或 PowerShell 中运行。');
  const env = await launchEnvironment({
    prepareMemory: ({ projectDir, provider, env: childEnv }) =>
      runtime.prepareManagedMemory(projectDir, provider, { platform: process.platform, env: childEnv }),
  });
  delete env.ELECTRON_RUN_AS_NODE;
  const child = spawn(require('electron'), ['.', ...process.argv.slice(2)], {
    cwd: __dirname, env, stdio: 'inherit',
  });
  child.on('error', error => { console.error(error.message); process.exitCode = 1; });
  child.on('exit', code => { process.exitCode = code ?? 1; });
}

if (require.main === module) main().catch(error => {
  console.error(error.message);
  process.exitCode = error.code === 'CANCELLED' ? 130 : 1;
});

module.exports = { PROVIDERS, providerChoices, selectProvider, chooseProvider, readSecret, launchEnvironment };
