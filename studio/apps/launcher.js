'use strict';
const api = window.studioDesktop;
const byId = id => document.getElementById(id);
let projectDir = '';
let busy = false;

function refresh() {
  byId('project-row').hidden = !byId('docker').checked;
  byId('server').disabled = busy || byId('docker').checked;
  byId('connect').disabled = busy || (!byId('docker').checked && !byId('server').value.trim());
  byId('choose-project').disabled = busy;
  byId('cancel').disabled = !busy;
  byId('connect').textContent = byId('docker').checked ? '启动并选择风格' : '连接并选择风格';
}

function status(value) {
  byId('status').className = `status ${value.kind}`;
  byId('message').textContent = value.message;
  if (value.serverUrl && (byId('docker').checked || value.kind === 'connected')) byId('server').value = value.serverUrl;
  busy = value.kind === 'connecting';
  refresh();
}

api.onStatus(status);
api.state().then(state => {
  byId('server').value = state.settings.serverUrl;
  byId('docker').checked = state.settings.autoStartDocker;
  projectDir = state.settings.projectDir;
  byId('project-path').textContent = projectDir || '尚未选择';
  byId('version').textContent = `v${state.version}`;
  byId('docker-options').hidden = state.platform !== 'win32' || Boolean(state.addressRequired);
  byId('docker-options').open = state.platform === 'win32' && byId('docker').checked;
  byId('platform-hint').textContent = state.platform === 'darwin'
    ? '连接本机服务、SSH 转发或 HTTPS 远端。'
    : state.platform === 'win32'
      ? '连接 WSL2 本机服务或 HTTPS 远端。'
      : 'Linux 只部署后端；此桌面页面仅用于开发验证。';
  if (state.platform !== 'win32' || state.addressRequired) {
    byId('docker').disabled = true;
    byId('docker').checked = false;
  }
  status(state.status);
}).catch(error => status({ kind: 'error', message: error.message }));

byId('docker').addEventListener('change', refresh);
byId('server').addEventListener('input', refresh);
byId('choose-project').addEventListener('click', async () => {
  try {
    const selected = await api.chooseProject();
    if (selected) { projectDir = selected; byId('project-path').textContent = selected; }
  } catch (error) { status({ kind: 'error', message: error.message }); }
});
byId('connection').addEventListener('submit', async event => {
  event.preventDefault();
  if (byId('docker').checked && !projectDir) return status({ kind: 'error', message: '请先选择已配置好的项目目录。' });
  if (!byId('docker').checked && !byId('server').value.trim())
    return status({ kind: 'error', message: '请先输入已有 Studio 服务的 IP 或 HTTPS 地址。' });
  try {
    await api.connect({ serverUrl: byId('server').value, autoStartDocker: byId('docker').checked, projectDir });
  } catch (error) { status({ kind: 'error', message: error.message }); }
});
byId('cancel').addEventListener('click', () => { void api.cancel(); });
