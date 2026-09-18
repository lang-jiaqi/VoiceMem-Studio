'use strict';
const { app, BrowserWindow, ipcMain, Menu, dialog, session, systemPreferences, nativeTheme, safeStorage } = require('electron');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const runtime = require('./runtime.cjs');
const { createPet } = require('./pet-window.cjs');

app.setName('VoiceMem Studio');
if (process.env.VOICEMEM_DESKTOP_USER_DATA) app.setPath('userData', path.resolve(process.env.VOICEMEM_DESKTOP_USER_DATA));
const configurationFile = path.join(app.getPath('userData'), 'connection.json');
const modelServicesFile = path.join(app.getPath('userData'), 'model-services.json');
const launcherUrl = pathToFileURL(path.join(__dirname, 'launcher.html')).href;
const icon = path.join(__dirname, 'assets/icon.png');
let launcher, studio, current = { ...runtime.DEFAULTS }, attempt, generation = 0, activeOrigin = '';
let status = { kind: 'idle', message: '连接已运行的服务，或启动本机 Docker。' };
let writes = Promise.resolve();
let quitting = false, managedMode = false;
let pet, petEnabled = true, ownedBackend, managedConfiguration, reconfiguring = false;
const microphoneGrants = new Set();

async function showPet() {
  if (!pet || !petEnabled || !studio || studio.isDestroyed()) return;
  try { await pet.open(activeOrigin); }
  catch (error) {
    dialog.showErrorBox('桌宠加载失败', `Studio 仍可正常使用。请检查 App 的桌宠资源是否完整。\n${error.message}`);
  }
}

function publish(kind, message) {
  status = { kind, message, serverUrl: current.serverUrl };
  if (launcher && !launcher.isDestroyed()) launcher.webContents.send('studio-desktop:status', status);
}

function cancelConnection() {
  generation += 1;
  attempt?.abort();
  attempt = undefined;
}

function backendExitError(result) {
  if (result.error) return new Error(`本机后端启动失败：${result.error.message}`);
  return new Error(`本机后端已退出${result.code === null ? '' : `，状态码 ${result.code}`}${result.signal ? `（${result.signal}）` : ''}。`);
}

function assertLauncher(event) {
  if (!launcher || launcher.isDestroyed() || event.sender !== launcher.webContents
      || event.senderFrame !== launcher.webContents.mainFrame || event.senderFrame.url !== launcherUrl) {
    throw new Error('该操作只允许从本地连接设置页发起。');
  }
}

function assertStudio(event) {
  let pathname = '';
  try { pathname = new URL(event.senderFrame.url).pathname; } catch {}
  if (!studio || studio.isDestroyed() || event.sender !== studio.webContents
      || event.senderFrame !== studio.webContents.mainFrame
      || !runtime.sameOrigin(event.senderFrame.url, activeOrigin)
      || !['/ui/technical.html', '/ui/digital.html'].includes(pathname)) {
    throw new Error('该操作只允许从当前 Studio 设置页发起。');
  }
}

function protectSecret(value) {
  if (!safeStorage.isEncryptionAvailable()) throw new Error('系统安全存储当前不可用，无法保存 API Key。');
  return safeStorage.encryptString(value).toString('base64');
}

function unprotectSecret(value) {
  if (!safeStorage.isEncryptionAvailable()) throw new Error('系统安全存储当前不可用。');
  return safeStorage.decryptString(Buffer.from(String(value || ''), 'base64'));
}

function lockNavigation(window, allowed) {
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', (event, url) => { if (!allowed(url)) event.preventDefault(); });
  window.webContents.on('will-redirect', (event, url) => { if (!allowed(url)) event.preventDefault(); });
  window.webContents.on('will-attach-webview', event => event.preventDefault());
}

async function showLauncher(visible = true) {
  if (launcher && !launcher.isDestroyed()) { launcher.show(); launcher.focus(); return launcher; }
  const window = new BrowserWindow({
    width: 680, height: 430, minWidth: 600, minHeight: 400, title: 'VoiceMem Studio · 连接服务',
    backgroundColor: '#f7f8fa', icon, show: false,
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  launcher = window;
  lockNavigation(window, url => url === launcherUrl);
  window.webContents.session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  window.webContents.session.setPermissionCheckHandler(() => false);
  window.on('closed', () => { if (launcher === window) launcher = undefined; cancelConnection(); });
  await window.loadURL(launcherUrl);
  if (visible && !window.isDestroyed()) window.show();
  return window;
}

function installPermissions(studioSession) {
  function authorized(contents, permission, details, requestingUrl) {
    return studio && !studio.isDestroyed() && contents === studio.webContents
      && runtime.audioPermission(permission, details, activeOrigin, requestingUrl)
      && runtime.sameOrigin(contents.getURL(), activeOrigin);
  }
  studioSession.setPermissionCheckHandler((contents, permission, requestingOrigin, details) =>
    !!authorized(contents, permission, details, requestingOrigin) && microphoneGrants.has(activeOrigin));
  studioSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    if (!authorized(contents, permission, details, details.requestingUrl)) return callback(false);
    const origin = activeOrigin;
    const window = studio;
    if (microphoneGrants.has(origin)) return callback(true);
    void (async () => {
      const response = await dialog.showMessageBox(window, {
        type: 'question', title: '麦克风权限', message: '允许 VoiceMem Studio 使用麦克风？',
        detail: `语音会发送到你配置的服务：${origin}。只允许音频，不授权摄像头或屏幕录制。`,
        buttons: ['允许', '取消'], defaultId: 1, cancelId: 1,
      });
      let granted = response.response === 0;
      if (granted && process.platform === 'darwin') granted = await systemPreferences.askForMediaAccess('microphone');
      granted = granted && origin === activeOrigin && authorized(contents, permission, details, details.requestingUrl);
      if (granted) microphoneGrants.add(origin);
      callback(!!granted);
    })().catch(() => callback(false));
  });
}

async function openStudio(url, ownGeneration) {
  if (ownGeneration !== generation) return;
  const previous = studio;
  const studioSession = session.fromPartition(`persist:studio-${new URL(url).host}`);
  installPermissions(studioSession);
  const window = new BrowserWindow({
    width: 680, height: 430, minWidth: 480, minHeight: 380, title: 'VoiceMem Studio',
    backgroundColor: '#f7f8fa', icon, show: false,
    webPreferences: {
      session: studioSession, preload: path.join(__dirname, 'studio-preload.cjs'),
      contextIsolation: true, nodeIntegration: false, sandbox: true, backgroundThrottling: false,
    },
  });
  let opened = false;
  studio = window;
  activeOrigin = url;
  pet?.close();
  previous?.destroy();
  lockNavigation(window, destination => runtime.sameOrigin(destination, url));
  let choosingMode = true, preparedMode = true;
  const homeOf = destination => ['/', '/index.html', '/ui/', '/ui/index.html'].includes(new URL(destination).pathname);
  function prepareMode(home) {
    if (home === preparedMode) return;
    preparedMode = home;
    if (window.isFullScreen()) window.setFullScreen(false);
    if (window.isMaximized()) window.unmaximize();
    window.setMinimumSize(home ? 480 : 1080, home ? 380 : 680);
    window.setSize(home ? 680 : 1440, home ? 430 : 920);
    window.center();
  }
  window.webContents.on('will-navigate', (_event, destination) => {
    if (runtime.sameOrigin(destination, url)) prepareMode(homeOf(destination));
  });
  window.webContents.on('did-navigate', (_event, destination) => {
    const home = homeOf(destination);
    if (home === choosingMode) return;
    choosingMode = home;
    prepareMode(home);
    if (home) pet?.close(); else void showPet();
  });
  window.webContents.on('page-title-updated', event => event.preventDefault());
  window.webContents.on('render-process-gone', () => {
    if (studio === window && !quitting) {
      publish('error', '界面进程已退出，请重新连接。');
      if (managedMode) {
        dialog.showErrorBox('VoiceMem Studio 界面已退出', '请查看启动终端中的详细日志，然后重新运行 npm start。');
        app.quit();
      } else void showLauncher();
    }
  });
  window.on('closed', () => {
    if (studio !== window) return;
    studio = undefined;
    pet?.close();
    if (opened && launcher && !launcher.isDestroyed()) launcher.destroy();
  });
  try {
    await window.loadURL(`${url}/`);
    if (ownGeneration !== generation || window.isDestroyed()) { if (!window.isDestroyed()) window.destroy(); return; }
    opened = true;
    window.show();
    publish('connected', '已连接。');
    if (launcher && !launcher.isDestroyed()) launcher.hide();
  } catch (error) {
    if (!window.isDestroyed()) window.destroy();
    throw new Error(`无法加载 Studio 页面：${error.message}`);
  }
}

async function connect(value, persist = true) {
  const next = runtime.settings(value);
  cancelConnection();
  const ownGeneration = generation;
  const control = new AbortController();
  attempt = control;
  current = next;
  try {
    if (persist) {
      writes = writes.catch(() => {}).then(() => runtime.saveSettings(configurationFile, next));
      await writes;
    }
    control.signal.throwIfAborted();
    let url = next.serverUrl;
    if (next.autoStartDocker) {
      publish('connecting', '正在启动本机 Docker 中的 studio 服务…');
      url = await runtime.startDocker(next.projectDir, { signal: control.signal });
      control.signal.throwIfAborted();
      current = { ...next, serverUrl: url };
      const discovered = current;
      writes = writes.catch(() => {}).then(() => runtime.saveSettings(configurationFile, discovered));
      await writes;
      control.signal.throwIfAborted();
    }
    publish('connecting', '正在连接 Studio…');
    await runtime.waitForStudio(url, {
      signal: control.signal,
      onWait: seconds => publish('connecting', `等待服务就绪 · ${seconds} 秒。首次模型预热可能需要几分钟。`),
    });
    control.signal.throwIfAborted();
    await openStudio(url, ownGeneration);
  } catch (error) {
    if (ownGeneration !== generation || control.signal.aborted) return;
    publish('error', error.message || '连接失败，请检查服务地址。');
    void showLauncher();
  } finally {
    if (ownGeneration === generation) attempt = undefined;
  }
}

async function connectManaged(config, { quitOnError = true } = {}) {
  cancelConnection();
  const ownGeneration = generation;
  const control = new AbortController();
  attempt = control;
  current = { ...runtime.DEFAULTS };
  try {
    const startupTimeoutMs = runtime.managedStartupTimeout(process.env);
    const previousBackend = ownedBackend;
    ownedBackend = undefined;
    previousBackend?.stop();
    publish('connecting', `正在启动本机 ${process.platform === 'darwin' ? 'MLX' : 'WSL2/CUDA'} 后端…`);
    const backend = await runtime.startManagedBackend(
      config.projectDir, config.memoryProvider, config.replyProvider, {
        signal: control.signal,
        env: runtime.modelServiceEnvironment(process.env, config.services),
      },
    );
    ownedBackend = backend;
    let ready = false;
    const exitWatch = backend.exited.then(result => {
      if (!ready) throw backendExitError(result);
      if (ownedBackend === backend) {
        ownedBackend = undefined;
        if (!quitting && studio && !studio.isDestroyed()) {
          studio.destroy();
          publish('error', backendExitError(result).message);
          dialog.showErrorBox('VoiceMem Studio 后端已退出', `${backendExitError(result).message}\n\n请查看启动终端中的详细日志。`);
          app.quit();
        }
      }
    });
    publish('connecting', `正在准备 Studio ${config.replyProvider} 回复与语音模型，首次预热可能需要几分钟…`);
    await Promise.race([
      runtime.waitForStudio(backend.url, {
        signal: control.signal,
        timeoutMs: startupTimeoutMs,
        onWait: seconds => publish('connecting', `等待本机后端就绪 · ${seconds} 秒。首次模型预热可能需要几分钟。`),
      }),
      exitWatch,
    ]);
    ready = true;
    control.signal.throwIfAborted();
    await openStudio(backend.url, ownGeneration);
    return true;
  } catch (error) {
    if (ownGeneration !== generation || control.signal.aborted) return false;
    ownedBackend?.stop();
    ownedBackend = undefined;
    publish('error', error.message || '本机后端启动失败。');
    dialog.showErrorBox('VoiceMem Studio 启动失败', `${error.message || '本机后端启动失败。'}\n\n请查看启动终端中的详细日志。`);
    if (quitOnError) app.quit();
    return false;
  } finally {
    if (ownGeneration === generation) attempt = undefined;
  }
}

function installMenu() {
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(process.platform === 'darwin' ? [{ label: app.name, submenu: [{ role: 'about' }, { type: 'separator' }, { role: 'hide' }, { role: 'unhide' }, { type: 'separator' }, { role: 'quit' }] }] : []),
    { label: 'Studio', submenu: [
      { label: '连接设置', accelerator: 'CmdOrCtrl+,', click: () => { void showLauncher(); } },
      { label: '重新连接', click: () => { void showLauncher().then(() => connect(current)); } },
      { id: 'show-pet', label: '显示桌宠', type: 'checkbox', checked: petEnabled, click: item => {
        petEnabled = item.checked;
        if (petEnabled) void showPet(); else pet?.close();
      } },
      { label: '找回桌宠位置', click: () => pet?.resetPosition() },
      { label: '桌宠大小', submenu: [
        { label: '缩小', click: () => pet?.resize(-1) },
        { label: '放大', click: () => pet?.resize(1) },
        { label: '恢复原始大小', click: () => pet?.resetSize() },
      ] },
      { type: 'separator' }, { role: 'quit', label: '退出' },
    ] },
    { label: '编辑', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: '视图', submenu: [{ role: 'reload' }, { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'togglefullscreen' }] },
  ]));
}

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { const window = studio || launcher; if (window) { if (window.isMinimized()) window.restore(); window.show(); window.focus(); } });
  app.on('before-quit', () => { quitting = true; cancelConnection(); ownedBackend?.stop(); pet?.close(); });
  app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
  app.on('activate', () => {
    if (studio && !studio.isDestroyed()) studio.show();
    else if (!managedMode) void showLauncher();
  });
  app.whenReady().then(async () => {
    nativeTheme.themeSource = 'dark';
    let configError, managed, persistInitialManaged = false;
    const configureOnly = process.env.VOICEMEM_DESKTOP_CONFIGURE === '1';
    managedMode = Boolean(process.env.VOICEMEM_DESKTOP_PROJECT_ROOT);
    try {
      const launch = runtime.managedLaunch(process.env);
      if (launch) {
        const stored = await runtime.loadModelServices(modelServicesFile, unprotectSecret);
        managed = { ...launch, services: stored || runtime.modelServicesFromLaunch(launch, process.env) };
        persistInitialManaged = !stored;
        managedConfiguration = managed;
      }
      current = managed ? { ...runtime.DEFAULTS } : await runtime.loadSettings(configurationFile);
    } catch (error) { configError = error.message; }
    installMenu();
    try {
      pet = createPet({ onHidden: () => {
        petEnabled = false;
        Menu.getApplicationMenu().getMenuItemById('show-pet').checked = false;
      } });
      await pet.restorePosition();
    } catch (error) {
      petEnabled = false;
      Menu.getApplicationMenu().getMenuItemById('show-pet').checked = false;
      dialog.showErrorBox('桌宠资源缺失', `Studio 仍可正常使用。源码运行请先执行 npm run prepare:pet；安装包请重新安装。\n${error.message}`);
    }
    ipcMain.handle('studio-desktop:state', event => { assertLauncher(event); return { settings: current, status, platform: process.platform, version: app.getVersion() }; });
    ipcMain.handle('studio-desktop:choose-project', async event => {
      assertLauncher(event);
      const result = await dialog.showOpenDialog(launcher, { title: '选择已配置好的 VoiceMem-Studio 项目', properties: ['openDirectory'] });
      if (result.canceled) return null;
      return runtime.validateProject(result.filePaths[0]);
    });
    ipcMain.handle('studio-desktop:connect', async (event, value) => {
      assertLauncher(event);
      const next = runtime.settings(value);
      if (next.autoStartDocker && (!current.autoStartDocker || next.projectDir !== current.projectDir)) {
        const answer = await dialog.showMessageBox(launcher, {
          type: 'question', title: '启用本机 Docker 自动启动', message: '允许 App 按所选项目的 Compose 配置启动服务？',
          detail: '只选择你信任的项目。不会构建镜像、删除数据或停止现有服务；退出 App 后容器继续运行。',
          buttons: ['允许', '取消'], defaultId: 1, cancelId: 1,
        });
        if (answer.response !== 0) return false;
      }
      void connect(next);
      return true;
    });
    ipcMain.handle('studio-desktop:cancel', event => {
      assertLauncher(event);
      cancelConnection();
      if (managed) {
        ownedBackend?.stop();
        ownedBackend = undefined;
        publish('idle', '已取消本机后端启动。');
      } else publish('idle', '已取消等待；已经启动的 Docker 服务不会被停止。');
    });
    ipcMain.handle('studio-desktop:model-services', event => {
      assertStudio(event);
      return managedConfiguration
        ? { managed: true, services: runtime.publicModelServices(managedConfiguration.services), reconfiguring }
        : { managed: false, services: null, reconfiguring: false };
    });
    ipcMain.handle('studio-desktop:update-model-service', (event, role, value) => {
      assertStudio(event);
      if (!managedConfiguration) throw new Error('远程服务需要在服务器端修改模型配置。');
      if (reconfiguring) throw new Error('模型服务正在重启，请稍候。');
      const previous = managedConfiguration;
      const services = runtime.updateModelService(previous.services, role, value);
      const next = { ...previous, memoryProvider: services.memory.provider,
        replyProvider: services.reply.provider, services };
      reconfiguring = true;
      setImmediate(() => { void (async () => {
        managedConfiguration = next;
        const ready = await connectManaged(next, { quitOnError: false });
        if (ready) {
          try {
            await runtime.saveModelServices(modelServicesFile, services, protectSecret);
          } catch (error) {
            dialog.showErrorBox('模型配置未保存', `${error.message}\n\n本次运行已应用配置，但下次启动仍使用原配置。`);
          }
        } else {
          managedConfiguration = previous;
          await connectManaged(previous);
        }
        reconfiguring = false;
      })(); });
      return { restarting: true, services: runtime.publicModelServices(services) };
    });
    if (configError && managedMode) {
      dialog.showErrorBox('VoiceMem Studio 启动配置无效', configError);
      app.quit();
      return;
    }
    if (!managed) await showLauncher(configureOnly);
    if (configError) { publish('error', configError); void showLauncher(); }
    else if (managed) void (async () => {
      const ready = await connectManaged(managed);
      if (ready && persistInitialManaged) {
        try { await runtime.saveModelServices(modelServicesFile, managed.services, protectSecret); }
        catch (error) {
          dialog.showErrorBox('模型配置未保存', `${error.message}\n\n下次启动时需要重新输入 API Key。`);
        }
      }
    })();
    else if (configureOnly) publish('idle', '填写已有 Studio 服务地址后连接。');
    else void connect(current, false);
  }).catch(error => { dialog.showErrorBox('VoiceMem Studio 启动失败', error.message); app.quit(); });
}
