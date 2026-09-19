'use strict';
// Virtual-display test of the actual app with a fake backend and fake Docker CLI.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const { createHash } = require('node:crypto');
const { setTimeout: delay } = require('node:timers/promises');

async function until(operation, timeout = 30000) {
  const started = Date.now();
  let error;
  while (Date.now() - started < timeout) {
    try { const result = await operation(); if (result) return result; } catch (caught) { error = caught; }
    await delay(100);
  }
  throw error || new Error('Smoke test timed out');
}

async function devtools(url) {
  const socket = new WebSocket(url);
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }); });
  let next = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    const call = pending.get(message.id);
    if (call) { pending.delete(message.id); message.error ? call.reject(new Error(message.error.message)) : call.resolve(message.result); }
  });
  return {
    send: (method, params = {}) => new Promise((resolve, reject) => { const id = ++next; pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })); }),
    close: () => socket.close(),
  };
}

async function main() {
  if (process.platform !== 'linux') throw new Error('This virtual-display smoke check runs on Linux.');
  if (process.getuid?.() === 0 && process.env.VOICEMEM_SMOKE_ALLOW_ROOT !== '1') throw new Error('Use a normal desktop user, or explicitly allow the isolated root-only smoke test.');
  const apps = path.resolve(__dirname, '..');
  if (!process.env.VOICEMEM_DESKTOP_BINARY) await require('./prepare-pet.cjs').preparePet();
  const directory = await fs.mkdtemp(path.join(process.env.VOICEMEM_TEST_TMP || os.tmpdir(), 'studio-desktop-smoke-'));
  let appEntry = apps;
  const asar = process.argv.includes('--asar');
  if (asar) {
    if (process.env.VOICEMEM_DESKTOP_BINARY) throw new Error('Choose an ASAR payload check or an installed-binary check, not both.');
    const payload = path.join(directory, 'payload');
    await fs.mkdir(payload);
    const pkg = JSON.parse(await fs.readFile(path.join(apps, 'package.json'), 'utf8'));
    for (const file of pkg.build.files) {
      const resource = file === '.pet-runtime/**/*' ? '.pet-runtime' : file;
      assert.equal(resource.includes('*'), false, 'Update the payload test for this file selector.');
      await fs.mkdir(path.dirname(path.join(payload, resource)), { recursive: true });
      await fs.cp(path.join(apps, resource), path.join(payload, resource), { recursive: true });
    }
    appEntry = path.join(directory, 'app.asar');
    const archive = require('@electron/asar');
    await archive.createPackage(payload, appEntry);
    const files = archive.listPackage(appEntry);
    assert.ok(files.includes('/.pet-runtime/avatar-controller.js'));
    assert.ok(files.includes('/.pet-runtime/assets/scene/call-background.png'));
    assert.equal(files.some(file => /previous-|node_modules|vendor\/|models\/|checks\//.test(file)), false);
  }
  const profile = path.join(directory, 'profile'), project = path.join(directory, 'project'), bin = path.join(directory, 'bin');
  for (const dir of [profile, project, bin]) await fs.mkdir(dir);
  for (const name of ['compose.yaml', 'pyproject.toml']) await fs.writeFile(path.join(project, name), 'synthetic fixture');
  await fs.copyFile(path.join(apps, 'evals/fixtures/docker.cjs'), path.join(bin, 'docker'));
  await fs.chmod(path.join(bin, 'docker'), 0o755);
  const html = await fs.readFile(path.join(apps, '../web/voicemem.html'));
  const readyAt = Date.now() + 5000;
  const handler = (req, res) => {
    if (req.url === '/') {
      res.writeHead(Date.now() < readyAt ? 503 : 200, { 'content-type': 'text/html' }); return res.end(html);
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify(req.url === '/api/spaces'
      ? { active: 'desktop-smoke', spaces: [{ id: 'desktop-smoke', name: '桌面测试', count: 0, language: 'zh' }] }
      : { left: [], right: [], lang: 'zh' }));
  };
  const backend = http.createServer(handler), replacement = http.createServer(handler);
  const observers = new Map([[backend, new Set()], [replacement, new Set()]]);
  for (const server of [backend, replacement]) server.on('upgrade', (req, socket) => {
    if (req.url !== '/ws-pet' || !req.headers['sec-websocket-key']) return socket.destroy();
    const accept = createHash('sha1').update(req.headers['sec-websocket-key'] + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
    socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
    observers.get(server).add(socket);
    socket.on('data', data => { if ((data[0] & 0x0f) === 8) socket.end(Buffer.from([0x88, 0x00])); });
    socket.on('end', () => socket.end());
    socket.on('close', () => observers.get(server).delete(socket));
    socket.on('error', () => socket.destroy());
  });
  function broadcast(server, message) {
    const payload = Buffer.from(JSON.stringify(message));
    assert.ok(payload.length < 126);
    for (const socket of observers.get(server)) socket.write(Buffer.concat([Buffer.from([0x81, payload.length]), payload]));
  }
  await new Promise(resolve => backend.listen(0, '127.0.0.1', resolve));
  await new Promise(resolve => replacement.listen(0, '127.0.0.1', resolve));
  const port = backend.address().port, origin = `http://127.0.0.1:${port}`;
  await fs.writeFile(path.join(profile, 'connection.json'), JSON.stringify({ serverUrl: 'http://127.0.0.1:1', autoStartDocker: true, projectDir: project }));
  const reserve = http.createServer();
  await new Promise(resolve => reserve.listen(0, '127.0.0.1', resolve));
  const debugPort = reserve.address().port;
  await new Promise(resolve => reserve.close(resolve));
  const display = spawn(process.env.VOICEMEM_XVFB || 'Xvfb', ['-displayfd', '1', '-screen', '0', '1440x1000x24', '-nolisten', 'tcp'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let displayOutput = '', displayErrors = '', application, appOutput = '', client, petClient;
  display.stdout.on('data', data => { displayOutput += data; });
  display.stderr.on('data', data => { displayErrors += data; });
  const displayFailure = new Promise((_, reject) => { display.once('error', reject); display.once('exit', code => reject(new Error(`Xvfb exited ${code}: ${displayErrors}`))); });
  try {
    const number = await Promise.race([until(() => /^\d+/.exec(displayOutput)?.[0], 5000), displayFailure]);
    const env = { ...process.env, DISPLAY: `:${number}`, PATH: `${bin}${path.delimiter}${process.env.PATH}`, VOICEMEM_DESKTOP_USER_DATA: profile,
      VOICEMEM_SMOKE_DOCKER_LOG: path.join(directory, 'docker.jsonl'), VOICEMEM_SMOKE_PORT: String(port) };
    delete env.ELECTRON_RUN_AS_NODE;
    delete env.DOCKER_HOST;
    delete env.DOCKER_CONTEXT;
    const executable = process.env.VOICEMEM_DESKTOP_BINARY || require('electron');
    const args = [...(process.env.VOICEMEM_DESKTOP_BINARY ? [] : [appEntry]), '--disable-gpu', '--remote-debugging-address=127.0.0.1', `--remote-debugging-port=${debugPort}`];
    if (process.getuid?.() === 0) args.push('--no-sandbox');
    application = spawn(executable, args, { env, detached: true, stdio: ['ignore', 'pipe', 'pipe'] });
    application.stdout.on('data', data => { appOutput += data; });
    application.stderr.on('data', data => { appOutput += data; });
    application.on('error', error => { appOutput += error.stack; });
    const pages = async () => (await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json();
    const connectionPage = await until(async () => (await pages()).find(page => page.url.endsWith('/launcher.html')));
    client = await devtools(connectionPage.webSocketDebuggerUrl);
    const initial = await client.send('Page.captureScreenshot');
    await fs.writeFile(path.join(directory, 'connection.png'), Buffer.from(initial.data, 'base64'));
    client.close();
    const target = await until(async () => (await pages()).find(page => page.url === `${origin}/`));
    client = await devtools(target.webSocketDebuggerUrl);
    await delay(500);
    const result = await client.send('Runtime.evaluate', { expression: 'JSON.stringify({title:document.title,background:getComputedStyle(document.body).backgroundColor,hasApp:!!document.getElementById("app"),bridge:typeof window.studioDesktop,node:typeof require})', returnByValue: true });
    const state = JSON.parse(result.result.value);
    assert.equal(state.title, 'VoiceMem');
    assert.equal(state.background, 'rgb(33, 33, 33)');
    assert.equal(state.hasApp, true);
    assert.equal(state.bridge, 'undefined');
    assert.equal(state.node, 'undefined');
    const petTarget = await until(async () => (await pages()).find(page => page.url.includes('/.pet-runtime/index.html')));
    petClient = await devtools(petTarget.webSocketDebuggerUrl);
    async function petState() {
      const result = await petClient.send('Runtime.evaluate', { expression: 'JSON.stringify({ ...window.avatar.getStatus(), node:typeof require, settings:typeof window.studioDesktop, mode:document.body.dataset.mode, callButton:!!document.getElementById("talkButton"), numberedActions:!!document.getElementById("avatar-debug"), pixi:typeof window.PIXI })', returnByValue: true });
      return JSON.parse(result.result.value);
    }
    await until(async () => { const state = await petState(); if (state.error) throw new Error(state.error); return state.ready && state.pose === 'lie'; });
    assert.equal((await petState()).node, 'undefined');
    assert.equal((await petState()).settings, 'undefined');
    assert.equal((await petState()).callButton, true);
    assert.equal((await petState()).numberedActions, false);
    assert.equal((await petState()).pixi, 'object');
    await until(() => observers.get(backend).size === 1);
    broadcast(backend, { type: 'conversation_started' });
    await until(async () => (await petState()).pose === 'sit');
    broadcast(backend, { type: 'backchannel' });
    await until(async () => (await petState()).gesture === 'backchannel');
    broadcast(backend, { type: 'playback_checkpoint', output_id: 'smoke-output', state: 'playing' });
    broadcast(backend, { type: 'avatar_audio_level', output_id: 'smoke-output', rms: .7, rendered_samples: 2400 });
    await until(async () => (await petState()).parameters.ParamMouthOpenY > 0.05, 3000);
    broadcast(backend, { type: 'playback_checkpoint', output_id: 'smoke-output', state: 'paused' });
    await until(async () => !(await petState()).speaking);
    broadcast(backend, { type: 'playback_checkpoint', output_id: 'smoke-output', state: 'playing' });
    broadcast(backend, { type: 'avatar_audio_level', output_id: 'smoke-output', rms: .7, rendered_samples: 4800 });
    await until(async () => (await petState()).parameters.ParamMouthOpenY > 0.05, 3000);
    broadcast(backend, { type: 'answer_interrupt' });
    await until(async () => !(await petState()).speaking);
    const petScreenshot = await petClient.send('Page.captureScreenshot');
    await fs.writeFile(path.join(directory, 'pet.png'), Buffer.from(petScreenshot.data, 'base64'));
    await petClient.send('Runtime.evaluate', { expression: 'window.pet.collapse()' });
    await until(async () => (await petState()).mode === 'dot');
    await petClient.send('Runtime.evaluate', { expression: 'window.pet.toggle()' });
    await until(async () => (await petState()).mode !== 'dot');
    const screenshot = await client.send('Page.captureScreenshot');
    await fs.writeFile(path.join(directory, 'studio.png'), Buffer.from(screenshot.data, 'base64'));
    const calls = (await fs.readFile(path.join(directory, 'docker.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
    assert.equal(calls.filter(args => args.includes('up')).length, 1);
    assert.equal(calls.some(args => args.includes('build') || args.includes('down')), false);
    assert.ok(calls.some(args => args.includes('--no-build') && args.includes('--no-recreate') && args.includes('never')));
    client.close();
    petClient.close();
    const replacementOrigin = `http://127.0.0.1:${replacement.address().port}`;
    client = await devtools(connectionPage.webSocketDebuggerUrl);
    await client.send('Runtime.evaluate', { expression: `window.studioDesktop.connect(${JSON.stringify({ serverUrl: replacementOrigin, autoStartDocker: false, projectDir: project })})`, awaitPromise: true });
    await until(() => observers.get(backend).size === 0 && observers.get(replacement).size === 1);
    const targets = await pages();
    assert.equal(targets.filter(page => page.url.includes('/.pet-runtime/index.html')).length, 1);
    petClient = await devtools(targets.find(page => page.url.includes('/.pet-runtime/index.html')).webSocketDebuggerUrl);
    await until(async () => (await petState()).ready);
    broadcast(replacement, { type: 'playback_checkpoint', output_id: 'disconnect-output', state: 'playing' });
    broadcast(replacement, { type: 'avatar_audio_level', output_id: 'disconnect-output', rms: .7, rendered_samples: 2400 });
    await until(async () => (await petState()).parameters.ParamMouthOpenY > 0.05, 3000);
    for (const socket of observers.get(replacement)) socket.destroy();
    await until(async () => !(await petState()).speaking);
    const hidePet = !process.argv.includes('--keep-pet-on-exit');
    if (hidePet) {
      await petClient.send('Runtime.evaluate', { expression: 'setTimeout(() => window.close(), 100)' });
      await until(async () => !(await pages()).some(page => page.url.includes('/.pet-runtime/index.html')));
    }
    assert.ok((await pages()).some(page => page.url === `${replacementOrigin}/`));
    client.close();
    client = await devtools((await pages()).find(page => page.url === `${replacementOrigin}/`).webSocketDebuggerUrl);
    await client.send('Runtime.evaluate', { expression: 'setTimeout(() => window.close(), 100)' });
    try { await until(() => application.exitCode !== null || application.signalCode !== null, 5000); }
    catch (error) { console.error('Remaining windows:', (await pages()).map(page => ({ title: page.title, url: page.url }))); throw error; }
    assert.equal(application.exitCode, 0, `App exited by signal ${application.signalCode}`);
    await until(() => observers.get(replacement).size === 0);
    console.log(JSON.stringify({ passed: true, packaged: !!process.env.VOICEMEM_DESKTOP_BINARY, asar, dockerStartedOnce: true, reusedWebUi: true,
      rendererIsolated: true, live2dPetLoaded: true, backchannelGesture: true, petPlaybackAndDisconnect: true, petServiceSwitch: true,
      petHiddenSeparately: hidePet, petClosedWithStudio: !hidePet, appExit: true, screenshots: directory }, null, 2));
  } catch (error) {
    console.error(appOutput.slice(-6000));
    throw error;
  } finally {
    client?.close();
    petClient?.close();
    if (application?.pid && application.exitCode === null) {
      try { process.kill(-application.pid, 'SIGTERM'); } catch {}
      await Promise.race([once(application, 'exit'), delay(2000)]);
      if (application.exitCode === null) { try { process.kill(-application.pid, 'SIGKILL'); } catch {} }
    }
    display.removeAllListeners('exit');
    display.kill('SIGTERM');
    for (const server of [backend, replacement]) {
      for (const socket of observers.get(server)) socket.destroy();
      server.closeAllConnections();
      server.close();
    }
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
