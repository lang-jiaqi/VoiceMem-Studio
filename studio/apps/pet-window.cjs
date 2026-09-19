'use strict';
const { app, BrowserWindow, ipcMain, screen, session } = require('electron');
const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { observerUrl, resourceAllowed, trustedSender } = require('./pet-policy.cjs');

/** Own one optional pet window; the App renderer remains responsible for microphone and playback. */
function createPet({ onHidden = () => {}, onToggleConversation = async () => false,
  onConversationState = async () => false } = {}) {
  const root = path.join(__dirname, '.pet-runtime');
  const { SIZES, RESIZE_CORNERS, clampScale, scaledSize, resizeFromCorner, selectPose, fitBounds } = require(path.join(root, 'state.cjs'));
  const positionFile = path.join(app.getPath('userData'), 'pet-position.json');
  const petSession = session.fromPartition('studio-pet');
  let window, anchor, dragging, resizing, timer, page = '', observer = '', mode = 'lie', scale = 1, manuallyCollapsed = false;
  let writes = Promise.resolve();
  petSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  petSession.setPermissionCheckHandler(() => false);
  petSession.webRequest.onBeforeRequest((details, callback) => callback({ cancel: !resourceAllowed(details.url, root, observer) }));

  function home() {
    const area = screen.getPrimaryDisplay().workArea;
    return { x: area.x + area.width - 24, y: area.y + area.height - 24 };
  }
  function save() {
    clearTimeout(timer);
    timer = undefined;
    if (!anchor) return;
    const value = JSON.stringify({ ...anchor, scale });
    writes = writes.catch(() => {}).then(() => fs.writeFile(positionFile, value, { mode: 0o600 }))
      .catch(error => console.error('[desktop-pet] Cannot save position:', error.message));
  }
  function scheduleSave() { clearTimeout(timer); timer = setTimeout(save, 250); }
  function layout() {
    if (!window || window.isDestroyed()) return;
    window.setIgnoreMouseEvents(false);
    const bounds = fitBounds(anchor, scaledSize(mode, scale), screen.getDisplayNearestPoint(anchor).workArea);
    window.setBounds(bounds);
    anchor = { x: bounds.x + bounds.width, y: bounds.y + bounds.height };
  }
  function setMode(next) {
    if (!window || window.isDestroyed() || !Object.hasOwn(SIZES, next) || mode === next) return;
    mode = next;
    if (mode === 'dot') { if (dragging || resizing) scheduleSave(); dragging = resizing = undefined; }
    layout();
    window.webContents.send('studio-pet:mode', mode);
  }
  function setScale(next) {
    scale = clampScale(next);
    layout();
    window?.webContents.send('studio-pet:scale', scale);
    scheduleSave();
  }
  function resize(step) { if (step === -1 || step === 1) setScale(Math.round((scale + step * .1) * 100) / 100); }
  function collapse() { manuallyCollapsed = true; setMode('dot'); }
  function toggle() {
    if (mode === 'dot') { manuallyCollapsed = false; setMode(selectPose()); }
    else collapse();
  }
  function close() {
    observer = '';
    page = '';
    if (timer || dragging || resizing) save();
    dragging = resizing = undefined;
    const previous = window;
    window = undefined;
    previous?.destroy();
  }
  function listen(name, handler) {
    ipcMain.on(`studio-pet:${name}`, (event, value) => {
      if (trustedSender(event, window, page)) handler(value);
    });
  }
  ipcMain.handle('studio-pet:initial', event => {
    if (!trustedSender(event, window, page)) throw new Error('Pet IPC is restricted to the local pet window.');
    return mode;
  });
  ipcMain.handle('studio-pet:initial-scale', event => {
    if (!trustedSender(event, window, page)) throw new Error('Pet IPC is restricted to the local pet window.');
    return scale;
  });
  ipcMain.handle('studio-pet:toggle-conversation', event => {
    if (!trustedSender(event, window, page)) throw new Error('Pet IPC is restricted to the local pet window.');
    return onToggleConversation();
  });
  ipcMain.handle('studio-pet:conversation-state', event => {
    if (!trustedSender(event, window, page)) throw new Error('Pet IPC is restricted to the local pet window.');
    return onConversationState();
  });
  listen('toggle', toggle);
  listen('collapse', collapse);
  listen('activate', pose => { if (manuallyCollapsed) return; if (['sit', 'lie'].includes(pose)) setMode(pose); else if (pose === undefined && mode === 'dot') setMode('lie'); });
  listen('resize', resize);
  listen('reset-size', () => setScale(1));
  listen('pointer', hit => { if (!dragging && !resizing && typeof hit === 'boolean') window.setIgnoreMouseEvents(!hit, { forward: true }); });
  listen('drag-start', () => { resizing = undefined; dragging = { cursor: screen.getCursorScreenPoint(), bounds: window.getBounds() }; window.setIgnoreMouseEvents(false); });
  listen('drag-move', () => {
    if (!dragging) return;
    const point = screen.getCursorScreenPoint(), bounds = dragging.bounds;
    const target = { x: bounds.x + bounds.width + point.x - dragging.cursor.x, y: bounds.y + bounds.height + point.y - dragging.cursor.y };
    const fitted = fitBounds(target, [bounds.width, bounds.height], screen.getDisplayNearestPoint(point).workArea);
    window.setBounds(fitted);
    anchor = { x: fitted.x + fitted.width, y: fitted.y + fitted.height };
  });
  listen('drag-end', () => { if (dragging) { dragging = undefined; scheduleSave(); } });
  listen('resize-start', (corner = 'se') => {
    if (mode === 'dot' || !Object.hasOwn(RESIZE_CORNERS, corner)) return;
    dragging = undefined;
    resizing = { cursor: screen.getCursorScreenPoint(), bounds: window.getBounds(), scale, corner };
    window.setIgnoreMouseEvents(false);
  });
  listen('resize-move', () => {
    if (!resizing) return;
    const point = screen.getCursorScreenPoint(), start = resizing;
    const result = resizeFromCorner(mode, start, point.x - start.cursor.x, point.y - start.cursor.y, screen.getDisplayNearestPoint(point).workArea);
    const bounds = result.bounds;
    scale = result.scale;
    window.setBounds(bounds);
    anchor = { x: bounds.x + bounds.width, y: bounds.y + bounds.height };
    window.webContents.send('studio-pet:scale', scale);
  });
  listen('resize-end', () => { if (resizing) { resizing = undefined; scheduleSave(); } });
  screen.on('display-removed', () => { layout(); scheduleSave(); });

  async function open(origin) {
    const next = observerUrl(origin);
    if (window && !window.isDestroyed() && observer === next) { window.showInactive(); return; }
    close();
    anchor ||= home();
    observer = next;
    mode = 'lie';
    manuallyCollapsed = false;
    const url = pathToFileURL(path.join(root, 'index.html'));
    url.searchParams.set('ws', observer);
    url.searchParams.set('layout', 'portrait');
    page = url.href;
    const created = new BrowserWindow({
      ...fitBounds(anchor, scaledSize(mode, scale), screen.getDisplayNearestPoint(anchor).workArea),
      title: '白藤 · VoiceMem Studio', frame: false, transparent: true, alwaysOnTop: true,
      skipTaskbar: true, resizable: false, maximizable: false, fullscreenable: false, show: false, hasShadow: false,
      webPreferences: { preload: path.join(__dirname, 'pet-preload.cjs'), session: petSession,
        contextIsolation: true, nodeIntegration: false, sandbox: true, backgroundThrottling: false },
    });
    window = created;
    created.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    for (const name of ['will-navigate', 'will-redirect', 'will-attach-webview']) created.webContents.on(name, event => event.preventDefault());
    created.on('closed', () => { if (window === created) { if (timer || dragging || resizing) save(); dragging = resizing = undefined; window = undefined; observer = ''; onHidden(); } });
    try {
      await created.loadURL(page);
      if (window === created && !created.isDestroyed()) created.showInactive();
    } catch (error) {
      if (created.isDestroyed()) return;
      if (window === created) close();
      throw error;
    }
  }
  async function restorePosition() {
    try {
      const saved = JSON.parse(await fs.readFile(positionFile, 'utf8'));
      if (Number.isFinite(saved.x) && Number.isFinite(saved.y)) anchor = { x: saved.x, y: saved.y };
      scale = clampScale(saved.scale);
    } catch (error) { if (error.code !== 'ENOENT') console.error('[desktop-pet] Ignoring invalid saved position.'); }
  }
  return { open, close, restorePosition, resize,
    setConversationState: active => { if (window && !window.isDestroyed()) window.webContents.send('studio-pet:conversation-state', Boolean(active)); },
    notifyConversationError: message => { if (window && !window.isDestroyed()) window.webContents.send('studio-pet:conversation-error', message); },
    resetSize: () => setScale(1), resetPosition: () => { anchor = home(); layout(); save(); } };
}

module.exports = { createPet };
