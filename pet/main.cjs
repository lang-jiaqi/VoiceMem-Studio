const { app, BrowserWindow, ipcMain, screen, Tray, Menu, nativeImage } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { SIZES, selectPose, fitBounds } = require('./state.cjs');
let win, tray, mode = 'lie', anchor, dragging, saveTimer;
const smoke = process.argv.includes('--smoke-test');
// --ws=... 由 VoiceMem 后端拉起时传进来，原样转交给渲染进程里的 voicemem-link.js。
// 不传就是原来那只独立桌宠，不会去连任何东西。
const argOf = name => {
  const prefix = `--${name}=`;
  const hit = process.argv.find(a => a.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : '';
};
const link = argOf('ws');
// 跟着 VoiceMem 一起启动时直接以角色形态出现：这种时候小人是被别人叫出来的，
// 再缩成一个小点等人点，等于白启动了。
const expanded = process.argv.includes('--expanded');
const settingsFile = () => path.join(app.getPath('userData'), 'position.json');
function save() { clearTimeout(saveTimer); saveTimer = setTimeout(() => { try { fs.writeFileSync(settingsFile(), JSON.stringify(anchor)); } catch {} }, 250); }
function setMode(next) {
  mode = next;
  const area = screen.getDisplayNearestPoint(anchor).workArea;
  win.setIgnoreMouseEvents(false);
  win.setBounds(fitBounds(anchor, SIZES[mode], area));
  win.webContents.send('mode', mode);
}
function toggle() { setMode(mode === 'dot' ? selectPose() : 'dot'); }
if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { if (win) { win.show(); win.focus(); } });
  app.whenReady().then(async () => {
    const area = screen.getPrimaryDisplay().workArea;
    anchor = { x: area.x + area.width - 24, y: area.y + area.height - 24 };
    if (!smoke) { try { const p = JSON.parse(fs.readFileSync(settingsFile())); if (Number.isFinite(p.x) && Number.isFinite(p.y)) anchor = p; } catch {} }
    win = new BrowserWindow({ ...fitBounds(anchor, SIZES[mode], screen.getDisplayNearestPoint(anchor).workArea),
      frame: false, transparent: true, alwaysOnTop: true, skipTaskbar: true, resizable: false,
      maximizable: false, fullscreenable: false, show: false, hasShadow: false,
      webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false, sandbox: true, offscreen:smoke, backgroundThrottling:!smoke } });
    if(smoke) win.webContents.on('console-message',event=>console.log('RENDER:',event.message));
    win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    win.webContents.on('will-navigate', e => e.preventDefault());
    ipcMain.handle('initial-mode', () => mode);
    ipcMain.on('toggle', toggle);
    ipcMain.on('activate', (_event, pose) => {
      if(['sit','lie'].includes(pose)&&mode!==pose)setMode(pose);
      else if(mode==='dot')setMode('lie');
    });
    ipcMain.on('actions', () => Menu.buildFromTemplate([
      { label: '坐姿', type: 'radio', checked: mode === 'sit', click: () => setMode('sit') },
      { label: '躺姿', type: 'radio', checked: mode === 'lie', click: () => setMode('lie') },
      { type: 'separator' },
      ...[['歪头笑', 'tilt'], ['点头', 'Nod'], ['摇头', 'Shake']].map(([label, action]) => ({
        label, enabled: mode === 'sit', click: () => win.webContents.send('action', action)
      }))
    ]).popup({ window: win }));
    ipcMain.on('collapse', () => setMode('dot'));
    ipcMain.on('quit', () => app.quit());
    ipcMain.on('pointer', (_e, hit) => { if (!dragging) win.setIgnoreMouseEvents(!hit, { forward: true }); });
    ipcMain.on('drag-start', () => { dragging = { cursor: screen.getCursorScreenPoint(), bounds: win.getBounds() }; });
    ipcMain.on('drag-move', () => {
      if (!dragging) return;
      const p = screen.getCursorScreenPoint(), b = dragging.bounds;
      const target = { x: b.x + b.width + p.x - dragging.cursor.x, y: b.y + b.height + p.y - dragging.cursor.y };
      const bounds = fitBounds(target, [b.width, b.height], screen.getDisplayNearestPoint(p).workArea);
      win.setBounds(bounds); anchor = { x: bounds.x + bounds.width, y: bounds.y + bounds.height };
    });
    ipcMain.on('drag-end', () => { dragging = undefined; save(); });
    const pixels = Buffer.alloc(16 * 16 * 4);
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const i = (y * 16 + x) * 4; pixels[i] = 232; pixels[i+1] = 173; pixels[i+2] = 168; pixels[i+3] = Math.hypot(x-7.5,y-7.5) < 6 ? 255 : 0; }
    tray = new Tray(nativeImage.createFromBitmap(pixels, { width: 16, height: 16 }));
    tray.setToolTip('雾铃 Noctelle');
    tray.setContextMenu(Menu.buildFromTemplate([{ label: '展开 / 收起', click: toggle }, { label: '找回小点', click: () => { anchor = { x: area.x+area.width-24,y:area.y+area.height-24 }; setMode('dot'); win.show(); save(); } }, { type: 'separator' }, { label: '退出', click: () => app.quit() }]));
    tray.on('click', toggle);
    screen.on('display-removed', () => setMode(mode));
    const query = new URLSearchParams();
    if (link) query.set('ws', link);
    if (argOf('idle')) query.set('idle', argOf('idle'));   // --idle=4 放大待机幅度
    const search = query.toString();
    await win.loadFile('index.html', search ? { search } : undefined);
    // --devtools：调幅度的时候要能敲 petRig.tune()，无边框窗口没有菜单可以开。
    if (process.argv.includes('--devtools')) win.webContents.openDevTools({ mode: 'detach' });
    if (smoke) {
      try {
        const results = [];
        const checksDir = path.join(app.isPackaged ? app.getPath('userData') : __dirname, 'checks');
        fs.mkdirSync(checksDir, { recursive: true });
        for (const pose of ['dot','sit','lie','dot']) {
          setMode(pose);
          await new Promise(r => setTimeout(r, 200));
          const result = await win.webContents.executeJavaScript(`({mode:document.body.dataset.mode, imageReady:document.querySelector('#portrait').complete && document.querySelector('#portrait').naturalWidth > 0, nodeExposed:typeof process !== 'undefined'})`);
          if (result.mode !== pose || (pose !== 'dot' && !result.imageReady) || result.nodeExposed) throw new Error(JSON.stringify(result));
          if(pose!=='dot') {
            let status;
            for(let i=0;i<150;i++) {
              status=await win.webContents.executeJavaScript('window.petRig.status()');
              if((status.ready&&status.pose===pose)||status.error)break;
              await new Promise(r=>setTimeout(r,100));
            }
            if(!status.ready||status.pose!==pose)throw new Error('Cubism load failed: '+JSON.stringify(status));
            const diagnostics=await win.webContents.executeJavaScript(`(()=>{
              const common={ParamAngleX:0,ParamAngleY:0,ParamAngleZ:2,ParamBreath:.5,ParamEyeLOpen:1,ParamEyeROpen:1};
              const closed=petRig.inspectPose({...common,ParamMouthOpenY:0});
              const open=petRig.inspectPose({...common,ParamMouthOpenY:1});
              const equal=(a,b)=>a.length===b.length&&a.every((v,i)=>Math.abs(v-b[i])<1e-7);
              return {ready:true,pose:petRig.status().pose,mouthVertices:open.ArtMeshMouthOpen.length,bodyVertices:open.ArtMeshTopwear.length,mouthChanges:!equal(closed.ArtMeshMouthOpen,open.ArtMeshMouthOpen),headUnaffected:equal(closed.ArtMeshFace,open.ArtMeshFace),bodyUnaffected:equal(closed.ArtMeshTopwear,open.ArtMeshTopwear)};
            })()`);
            if(!diagnostics.mouthVertices||!diagnostics.bodyVertices||!diagnostics.mouthChanges||!diagnostics.headUnaffected||!diagnostics.bodyUnaffected)throw new Error(JSON.stringify(diagnostics));
            results.push(diagnostics);
            for(const [name,params] of Object.entries({neutral:{ParamAngleZ:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamMouthOpenY:0},talking:{ParamAngleZ:2,ParamEyeLOpen:1,ParamEyeROpen:1,ParamMouthOpenY:.8},blink:{ParamAngleZ:-2,ParamEyeLOpen:0,ParamEyeROpen:0,ParamMouthOpenY:0}})){
              await win.webContents.executeJavaScript('petRig.inspectPose('+JSON.stringify(params)+')');
              await new Promise(r=>setTimeout(r,100));
              fs.writeFileSync(path.join(checksDir,pose+'-'+name+'.png'),(await win.webContents.capturePage()).toPNG());
            }
            await win.webContents.executeJavaScript(`petRig.show('${pose}').then(()=>{petRig.talk(3);petRig.tilt()})`);
            await new Promise(r=>setTimeout(r,150));
            const before=await win.webContents.executeJavaScript('petRig.status().parameters');
            await new Promise(r=>setTimeout(r,250));
            const after=await win.webContents.executeJavaScript('petRig.status().parameters');
            const mouthRunning=Math.abs(before.ParamMouthOpenY-after.ParamMouthOpenY)>1e-5;
            const headRunning=pose==='lie'||Math.abs(before.ParamAngleZ-after.ParamAngleZ)>1e-5;
            if(!mouthRunning||!headRunning)throw new Error('Concurrent animation stalled: '+pose);
            const hitCheck=await win.webContents.executeJavaScript(`(()=>{const r=document.querySelector('#live').getBoundingClientRect();return {outside:petRig.hitTest(r.left+1,r.top+1),inside:petRig.hitTest(r.left+r.width*.5,r.top+r.height*.6)}})()`);
            if(hitCheck.outside||!hitCheck.inside)throw new Error('Alpha hit test failed: '+JSON.stringify(hitCheck));
            results.push({pose,concurrentAnimation:true,alphaHitTest:true});
            if(pose==='sit'){
              for(const t of [0,.8,1.6,2.2,3,4]){
                await win.webContents.executeJavaScript(`petRig.inspectPose(TiltedSmile.compose({ParamAngleX:0,ParamAngleY:0,ParamAngleZ:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamMouthForm:0,ParamMouthOpenY:0,ParamBreath:.5},${t}))`);
                await new Promise(r=>setTimeout(r,80));
                fs.writeFileSync(path.join(checksDir,'smile-'+t+'.png'),(await win.webContents.capturePage()).toPNG());
              }
              await win.webContents.executeJavaScript(`petRig.show('sit').then(()=>document.querySelector('#tilt').click())`);
              await new Promise(r=>setTimeout(r,1700));
              const held=await win.webContents.executeJavaScript(`(()=>{const s=petRig.status();return {action:s.action,eyes:s.parameters.ParamEyeLOpen,smile:s.parameters.ParamMouthForm,retrigger:petRig.tilt()}})()`);
              if(held.action!=='tilted-smile'||held.eyes>.05||held.smile<.8||held.retrigger)throw new Error('Smile button / hold failed: '+JSON.stringify(held));
              const coexist=await win.webContents.executeJavaScript(`(()=>{
                const b={ParamAngleZ:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamBreath:.5};
                const closed=petRig.inspectPose(TiltedSmile.compose(b,2,'sit',0));
                const open=petRig.inspectPose(TiltedSmile.compose(b,2,'sit',1));
                const equal=(a,b)=>a.length===b.length&&a.every((v,i)=>Math.abs(v-b[i])<1e-7);
                return !equal(closed.ArtMeshMouthOpen,open.ArtMeshMouthOpen)&&equal(closed.ArtMeshFace,open.ArtMeshFace)&&equal(closed.ArtMeshTopwear,open.ArtMeshTopwear);
              })()`);
              if(!coexist)throw new Error('Speech changed smile head/body');
              results.push({tiltedSmileButton:true,heldSmile:true,retriggerGuard:true,smileSpeechIndependent:true});
            }
          }
          results.push(result);
        }
        await win.webContents.executeJavaScript("document.querySelector('#dot').click()");
        await new Promise(r => setTimeout(r, 200));
        if (!['sit','lie'].includes(mode)) throw new Error('Wake click failed');
        await win.webContents.executeJavaScript("document.querySelector('#collapse').click()");
        await new Promise(r => setTimeout(r, 200));
        if (mode !== 'dot') throw new Error('Collapse click failed');
        results.push({ wakeClick: true, collapseClick: true });
        fs.writeFileSync(path.join(checksDir, 'smoke.json'), JSON.stringify(results,null,2));
        console.log('SMOKE PASS', JSON.stringify(results)); app.exit(0);
      } catch (e) { console.error(e); app.exit(1); }
    } else {
      if (expanded) setMode('lie');
      win.showInactive();
    }
  });
  app.on('window-all-closed', () => app.quit());
}
