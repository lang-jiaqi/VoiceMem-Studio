const { app, BrowserWindow, ipcMain, screen, Tray, Menu, nativeImage } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { SIZES, RESIZE_CORNERS, clampScale, scaledSize, resizeFromCorner, selectPose, fitBounds } = require('./state.cjs');
let win, tray, mode = 'lie', anchor, dragging, resizing, saveTimer, scale = 1, manuallyCollapsed = false;
const smoke = process.argv.includes('--smoke-test');
// --ws=... 由 VoiceMem 后端拉起时传进来，原样转交给渲染进程里的 voicemem-link.js。
// 不传就是原来那只独立桌宠，不会去连任何东西。
const argOf = name => {
  const prefix = `--${name}=`;
  const hit = process.argv.find(a => a.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : '';
};
const link = argOf('ws');
const model = argOf('model');
const demoGesture = ['tilt-smile', 'nod', 'backchannel', 'surprise'].includes(argOf('gesture')) ? argOf('gesture') : '';
// 跟着 VoiceMem 一起启动时直接以角色形态出现：这种时候小人是被别人叫出来的，
// 再缩成一个小点等人点，等于白启动了。
const expanded = process.argv.includes('--expanded');
const settingsFile = () => path.join(app.getPath('userData'), 'position.json');
function flushSave() { clearTimeout(saveTimer); saveTimer = undefined; try { fs.writeFileSync(settingsFile(), JSON.stringify({ ...anchor, scale })); } catch {} }
function save() { clearTimeout(saveTimer); saveTimer = setTimeout(flushSave, 250); }
function layout() {
  const area = screen.getDisplayNearestPoint(anchor).workArea;
  const bounds = fitBounds(anchor, scaledSize(mode, scale), area);
  win.setIgnoreMouseEvents(false);
  win.setBounds(bounds);
  anchor = { x: bounds.x + bounds.width, y: bounds.y + bounds.height };
}
function setMode(next) {
  if (!Object.hasOwn(SIZES, next) || mode === next) return;
  mode = next;
  if (mode === 'dot') { if (dragging || resizing) save(); dragging = resizing = undefined; }
  layout();
  win.webContents.send('mode', mode);
}
function setScale(next) { scale = clampScale(next); layout(); win.webContents.send('scale', scale); save(); }
function resize(step) { if (step === -1 || step === 1) setScale(Math.round((scale + step * .1) * 100) / 100); }
function collapse() { manuallyCollapsed = true; setMode('dot'); }
function toggle() { if (mode === 'dot') { manuallyCollapsed = false; setMode(selectPose()); } else collapse(); }
if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { if (win) { win.show(); win.focus(); } });
  app.whenReady().then(async () => {
    const area = screen.getPrimaryDisplay().workArea;
    anchor = { x: area.x + area.width - 24, y: area.y + area.height - 24 };
    if (!smoke) { try { const p = JSON.parse(fs.readFileSync(settingsFile())); if (Number.isFinite(p.x) && Number.isFinite(p.y)) anchor = { x: p.x, y: p.y }; scale = clampScale(p.scale); } catch {} }
    win = new BrowserWindow({ ...fitBounds(anchor, scaledSize(mode, scale), screen.getDisplayNearestPoint(anchor).workArea),
      frame: false, transparent: true, alwaysOnTop: true, skipTaskbar: true, resizable: false,
      maximizable: false, fullscreenable: false, show: false, hasShadow: false,
      webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false,
        sandbox: true, backgroundThrottling: false } });
    if(smoke) win.webContents.on('console-message',event=>console.log('RENDER:',event.message));
    win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    win.webContents.on('will-navigate', e => e.preventDefault());
    ipcMain.handle('initial-mode', () => mode);
    ipcMain.handle('initial-scale', () => scale);
    ipcMain.on('toggle', toggle);
    ipcMain.on('activate', (_event, pose) => {
      if (manuallyCollapsed) return;
      if(['sit','lie'].includes(pose)&&mode!==pose)setMode(pose);
      else if(pose===undefined&&mode==='dot')setMode('lie');
    });
    ipcMain.on('collapse', collapse);
    ipcMain.on('resize', (_event, step) => resize(step));
    ipcMain.on('reset-size', () => setScale(1));
    ipcMain.on('pointer', (_e, hit) => { if (!dragging && !resizing && typeof hit === 'boolean') win.setIgnoreMouseEvents(!hit, { forward: true }); });
    ipcMain.on('drag-start', () => { resizing = undefined; dragging = { cursor: screen.getCursorScreenPoint(), bounds: win.getBounds() }; win.setIgnoreMouseEvents(false); });
    ipcMain.on('drag-move', () => {
      if (!dragging) return;
      const p = screen.getCursorScreenPoint(), b = dragging.bounds;
      const target = { x: b.x + b.width + p.x - dragging.cursor.x, y: b.y + b.height + p.y - dragging.cursor.y };
      const bounds = fitBounds(target, [b.width, b.height], screen.getDisplayNearestPoint(p).workArea);
      win.setBounds(bounds); anchor = { x: bounds.x + bounds.width, y: bounds.y + bounds.height };
    });
    ipcMain.on('drag-end', () => { if (dragging) { dragging = undefined; save(); } });
    ipcMain.on('resize-start', (_event, corner = 'se') => {
      if (mode === 'dot' || !Object.hasOwn(RESIZE_CORNERS, corner)) return;
      dragging = undefined;
      resizing = { cursor: screen.getCursorScreenPoint(), bounds: win.getBounds(), scale, corner };
      win.setIgnoreMouseEvents(false);
    });
    ipcMain.on('resize-move', () => {
      if (!resizing) return;
      const point = screen.getCursorScreenPoint(), start = resizing;
      const result = resizeFromCorner(mode, start, point.x - start.cursor.x, point.y - start.cursor.y, screen.getDisplayNearestPoint(point).workArea);
      const bounds = result.bounds;
      scale = result.scale;
      win.setBounds(bounds); anchor = { x: bounds.x + bounds.width, y: bounds.y + bounds.height };
      win.webContents.send('scale', scale);
    });
    ipcMain.on('resize-end', () => { if (resizing) { resizing = undefined; save(); } });
    win.on('close', () => { if (saveTimer || dragging || resizing) flushSave(); });
    const pixels = Buffer.alloc(16 * 16 * 4);
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const i = (y * 16 + x) * 4; pixels[i] = 232; pixels[i+1] = 173; pixels[i+2] = 168; pixels[i+3] = Math.hypot(x-7.5,y-7.5) < 6 ? 255 : 0; }
    tray = new Tray(nativeImage.createFromBitmap(pixels, { width: 16, height: 16 }));
    tray.setToolTip('VoiceMem 白藤');
    tray.setContextMenu(Menu.buildFromTemplate([{ label: '展开 / 收起', click: toggle },
      { label: '缩小', click: () => resize(-1) }, { label: '放大', click: () => resize(1) }, { label: '恢复原始大小', click: () => setScale(1) },
      { label: '找回小点', click: () => { anchor = { x: area.x+area.width-24,y:area.y+area.height-24 }; collapse(); layout(); win.show(); save(); } },
      { type: 'separator' }, { label: '退出', click: () => app.quit() }]));
    tray.on('click', toggle);
    screen.on('display-removed', () => { layout(); save(); });
    const query = new URLSearchParams();
    if (link) query.set('ws', link);
    if (model) query.set('model', model);
    query.set('layout', 'portrait');
    if (process.argv.includes('--debug-avatar')) query.set('debug', '1');
    if (argOf('idle')) query.set('idle', argOf('idle'));   // --idle=4 放大待机幅度
    const search = query.toString();
    await win.loadFile('index.html', search ? { search } : undefined);
    // 无边框窗口没有菜单，调试时允许直接打开开发者工具。
    if (process.argv.includes('--devtools')) win.webContents.openDevTools({ mode: 'detach' });
    if (smoke) {
      try {
        setMode('sit');
        let status;
        for (let i = 0; i < 300; i++) {
          status = await win.webContents.executeJavaScript('avatar.getStatus()');
          if (status.ready || status.modelError) break;
          await new Promise(r => setTimeout(r, 100));
        }
        if (!status?.ready || status.renderer !== 'live2d') throw new Error(JSON.stringify(status));
        await win.webContents.executeJavaScript("avatar.express('blush')");
        await new Promise(r => setTimeout(r, 250));
        status = await win.webContents.executeJavaScript('avatar.getStatus()');
        if (status.expression !== 'blush' || status.expressionError) throw new Error(JSON.stringify(status));
        if (process.argv.includes('--smoke-body-gesture')) {
          await win.webContents.executeJavaScript("avatar.setSpeaking(true); avatar.triggerGesture('sway', { amplitude: 1, cooldown: 0 })");
          await new Promise(r => setTimeout(r, 950));
          status = await win.webContents.executeJavaScript('avatar.getStatus()');
          if (status.gesture !== 'sway' || status.parameters.ParamBodyAngleX < 3) throw new Error(JSON.stringify(status));
        }
        const screenshot = argOf('screenshot');
        if (screenshot) fs.writeFileSync(screenshot, (await win.webContents.capturePage()).toPNG());
        console.log('SMOKE PASS', JSON.stringify(status)); app.exit(0);
      } catch (e) { console.error(e); app.exit(1); }
    } else {
      if (expanded) setMode('lie');
      win.showInactive();
      if (demoGesture) setTimeout(() => win.webContents.executeJavaScript(
        `avatar.wake(); avatar.triggerGesture(${JSON.stringify(demoGesture)}, { amplitude: .8, cooldown: 0 })`
      ).catch(() => {}), 1800);
    }
  });
  app.on('window-all-closed', () => app.quit());
}
