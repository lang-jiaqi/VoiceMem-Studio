'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const vm = require('node:vm');
const { CUBISM_CORE, observerUrl, resourceAllowed, trustedSender } = require('../pet-policy.cjs');
const { preparePet, petSource, sourceFiles } = require('../scripts/prepare-pet.cjs');

test('pet observes only the selected service and bundled local assets', () => {
  const root = path.resolve('/fixture/app/.pet-runtime');
  const asset = file => pathToFileURL(path.join(root, file)).href;
  const ws = observerUrl('http://127.0.0.1:8787');
  assert.equal(ws, 'ws://127.0.0.1:8787/ws-pet');
  assert.equal(observerUrl('https://studio.example.com'), 'wss://studio.example.com/ws-pet');
  assert.equal(observerUrl('http://[::1]:8787'), 'ws://[::1]:8787/ws-pet');
  assert.throws(() => observerUrl('http://untrusted.example.com'));
  for (const url of [asset('index.html'), asset('assets/live2d/rattan/rattan.moc3'), CUBISM_CORE, ws]) assert.equal(resourceAllowed(url, root, ws), true);
  for (const url of [asset('../launcher.html'), asset('../.pet-runtime-other/secret'), 'file:///etc/passwd', `${ws}?other=1`,
    'ws://localhost:8787/ws-pet', 'ws://127.0.0.1:8787/ws', 'https://studio.example.com/script.js']) {
    assert.equal(resourceAllowed(url, root, ws), false, url);
  }
  assert.equal(resourceAllowed(ws, root, ''), false);
});

test('pet window controls reject another renderer, frame or document', () => {
  const page = 'file:///fixture/pet/index.html?ws=fixture';
  const frame = { url: page }, contents = { mainFrame: frame };
  const window = { webContents: contents, isDestroyed: () => false };
  const event = { sender: contents, senderFrame: frame };
  assert.equal(trustedSender(event, window, page), true);
  assert.equal(trustedSender({ ...event, sender: {} }, window, page), false);
  assert.equal(trustedSender({ ...event, senderFrame: { url: page } }, window, page), false);
  assert.equal(trustedSender(event, window, `${page}-other`), false);
  assert.equal(trustedSender(event, { ...window, isDestroyed: () => true }, page), false);
});

test('pet build prefers Studio-owned sources and falls back to the legacy repository pet', async t => {
  const project = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-pet-source-'));
  t.after(() => fs.rm(project, { recursive: true, force: true }));
  const apps = path.join(project, 'studio/apps');
  await fs.mkdir(apps, { recursive: true });
  assert.equal(await petSource(apps), path.join(project, 'pet'));
  await fs.mkdir(path.join(project, 'studio/pet'));
  assert.equal(await petSource(apps), path.join(project, 'studio/pet'));
  const external = path.join(project, 'reviewed-pet');
  await fs.mkdir(external);
  assert.equal(await petSource(apps, { STUDIO_PET_DIR: external }), external);
});

test('pet bundle includes the Live2D runtime, rattan model and provenance notice', async t => {
  const destination = await fs.mkdtemp(path.join(process.env.VOICEMEM_TEST_TMP || os.tmpdir(), 'studio-pet-test-'));
  t.after(() => fs.rm(destination, { recursive: true, force: true }));
  const inventory = await preparePet({ destination });
  const source = path.resolve(__dirname, '../../pet');
  assert.equal(await petSource(path.resolve(__dirname, '..')), source);
  for (const file of sourceFiles) {
    const original = path.join(file.startsWith('node_modules/') ? path.resolve(__dirname, '..') : source, file);
    assert.deepEqual(await fs.readFile(path.join(destination, file)), await fs.readFile(original), file);
  }
  const html = await fs.readFile(path.join(destination, 'index.html'), 'utf8');
  for (const [, file] of html.matchAll(/(?:src|href)="([^"]+)"/g)) await fs.access(path.join(destination, file));
  assert.ok(inventory.includes('node_modules/pixi.js/dist/browser/pixi.min.js'));
  assert.ok(inventory.includes('node_modules/pixi-live2d-display/dist/cubism4.min.js'));
  assert.ok(html.includes("script-src 'self' 'unsafe-eval' https://cubism.live2d.com;"));
  for (const name of ['live2d-renderer.js', 'style.css']) {
    const content = await fs.readFile(path.join(destination, name), 'utf8');
    for (const [file] of content.matchAll(/assets\/[\w/.-]+\.png/g)) {
      assert.ok(inventory.includes(file), file);
      await fs.access(path.join(destination, file));
    }
  }
  assert.equal(inventory.some(file => /checks|package-lock|\.env/.test(file)), false);
  assert.equal(inventory.filter(file => file.endsWith('.png')).length, 6);
  assert.ok(inventory.includes('assets/scene/call-background.png'));
  assert.equal(inventory.includes('debug-panel.js'), false);
  assert.equal(inventory.includes('scene.js'), false);
  assert.doesNotMatch(html, /动作 0|debug-panel\.js|curtains\.png|scene\.js/);
  assert.match(html, /id="talkButton"/);
  assert.doesNotMatch(html, /随时陪你聊聊|call-identity|callState/);
  assert.ok(inventory.includes('THIRD_PARTY_NOTICES.md'));
  assert.ok(inventory.includes('assets/live2d/rattan/README-SOURCE.md'));
  const model = JSON.parse(await fs.readFile(
    path.join(destination, 'assets/live2d/rattan/rattan.model3.json'), 'utf8'));
  assert.equal(model.FileReferences.Textures.length, 5);
  assert.ok(model.FileReferences.Textures.every(file => file.includes('rattan.2048/')));
  assert.equal(model.FileReferences.Expressions.length, 22);
  assert.deepEqual(model.Groups.find(group => group.Name === 'LipSync').Ids,
    ['ParamMouthOpenY']);
  await fs.writeFile(path.join(destination, 'unexpected-private-file'), 'synthetic fixture');
  await assert.rejects(preparePet({ destination }), /Unexpected files/);
  assert.equal(await fs.readFile(path.join(destination, 'unexpected-private-file'), 'utf8'), 'synthetic fixture');
});

test('old Canvas pet resources are backed up once and excluded from the Live2D payload', async t => {
  const directory = await fs.mkdtemp(path.join(process.env.VOICEMEM_TEST_TMP || os.tmpdir(), 'studio-pet-migration-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const destination = path.join(directory, 'bundle');
  await fs.mkdir(path.join(destination, 'assets/avatar'), { recursive: true });
  await fs.writeFile(path.join(destination, 'avatar-rig.js'), 'old Canvas fixture');
  await fs.writeFile(path.join(destination, 'assets/avatar/calm.png'), 'old image fixture');
  await preparePet({ destination });
  const backups = (await fs.readdir(directory)).filter(file => file.startsWith('bundle.previous-'));
  assert.equal(backups.length, 1);
  assert.equal(await fs.readFile(path.join(directory, backups[0], 'resources/avatar-rig.js'), 'utf8'), 'old Canvas fixture');
  await assert.rejects(fs.access(path.join(destination, 'avatar-rig.js')), { code: 'ENOENT' });
  await assert.rejects(fs.access(path.join(destination, 'assets/avatar')), { code: 'ENOENT' });
  await preparePet({ destination });
  assert.deepEqual((await fs.readdir(directory)).filter(file => file.startsWith('bundle.previous-')), backups);
});

test('desktop preload exposes the same reduced window API as the new pet', async () => {
  async function exposed(file) {
    let api;
    vm.runInNewContext(await fs.readFile(file, 'utf8'), { require: name => {
      assert.equal(name, 'electron');
      return { contextBridge: { exposeInMainWorld: (key, value) => { assert.equal(key, 'pet'); api = value; } }, ipcRenderer: {} };
    } });
    return Object.keys(api).sort();
  }
  assert.deepEqual(await exposed(path.join(__dirname, '../pet-preload.cjs')), await exposed(path.resolve(__dirname, '../../pet/preload.cjs')));
});

test('backchannels trigger named gestures without a numbered action panel', async () => {
  const gestures = [];
  let socket;
  class WebSocket {
    static OPEN = 1;
    constructor() { socket = this; this.readyState = 1; }
    close() {}
  }
  const context = {
    URLSearchParams, WebSocket, performance: { now: () => 1000 },
    setTimeout: () => 1, clearTimeout() {}, location: { search: '?ws=ws://test' },
    window: { avatar: {
      triggerGesture: gesture => gestures.push(gesture), express() {}, wake() {}, sleep() {},
      setState() {}, setSpeaking() {}, feedAudioLevel() {}, setEmotion() {},
    } },
  };
  const link = await fs.readFile(path.resolve(__dirname, '../../pet/voicemem-link.js'), 'utf8');
  vm.runInNewContext(link, context);
  socket.onopen();
  for (let i = 1; i <= 3; i++) socket.onmessage({ data: JSON.stringify({
    type: 'backchannel', session_id: 'session', event_id: `bc-${i}`,
  }) });
  assert.deepEqual(gestures, ['backchannel', 'backchannel', 'backchannel']);
});

test('pointer gaze overrides idle wandering and speaking schedules a spaced body gesture', () => {
  const { AvatarBehaviorController, avatarFrameRate } = require('../../pet/avatar-behavior-controller.js');
  const { AvatarParameterController } = require('../../pet/avatar-parameter-controller.js');
  assert.equal(avatarFrameRate('sleeping'), 30);
  assert.equal(avatarFrameRate('idle'), 30);
  assert.equal(avatarFrameRate('speaking'), 60);
  assert.equal(avatarFrameRate('sleeping', true), 60);
  const parameters = new AvatarParameterController();
  const behavior = new AvatarBehaviorController(parameters, { random: () => 0 });
  behavior.setState('speaking');
  behavior.setPointerGaze(.8, -.6);
  behavior.update(.1);
  assert.equal(parameters.layers.get('pointer-eyes').values.ParamEyeBallX, .68);
  assert.equal(parameters.layers.get('pointer-eyes').values.ParamEyeBallY, -.39);
  behavior.update(.1);
  assert.equal(behavior.gesture?.name, 'nod');
  for (let i = 0; i < 5; i++) behavior.update(.1);
  assert.ok(parameters.layers.get('gesture').values.ParamBodyAngleY < -3);
  assert.ok(behavior.armAccents.Param47 > .15);
  assert.ok(behavior.nextSpeakingGesture > behavior.time + 3);
  for (let i = 0; i < 9; i++) behavior.update(.1);
  assert.equal(behavior.armAccents, null);
  behavior.setPointerGaze(null, null);
  assert.equal(parameters.layers.get('pointer-eyes').remove, true);
  behavior.setState('idle');
  assert.equal(behavior.nextSpeakingGesture, Infinity);
});

test('arm accents add to model physics only during a gesture', () => {
  const { Live2DRenderer } = require('../../pet/live2d-renderer.js');
  const renderer = new Live2DRenderer({ addEventListener() {} });
  const applied = [];
  renderer.model = { internalModel: { coreModel: {
    addParameterValueById: (id, value) => applied.push([id, value]),
  } } };
  renderer.applyArmAccents();
  assert.deepEqual(applied, []);
  renderer.setArmAccents({ Param47: .24, Param50: -.18 });
  renderer.applyArmAccents();
  assert.deepEqual(applied, [['Param47', .24], ['Param50', -.18]]);
  renderer.nativeMotion = true;
  renderer.applyArmAccents();
  assert.equal(applied.length, 2);
  renderer.nativeMotion = false;
  renderer.setArmAccents(null);
  renderer.applyArmAccents();
  assert.equal(applied.length, 2);
});

test('a backchannel does not defer the first speaking gesture for several seconds', () => {
  const { AvatarBehaviorController } = require('../../pet/avatar-behavior-controller.js');
  const { AvatarParameterController } = require('../../pet/avatar-parameter-controller.js');
  const behavior = new AvatarBehaviorController(new AvatarParameterController(), { random: () => 0 });
  behavior.setState('listening');
  assert.equal(behavior.triggerGesture('backchannel'), true);
  behavior.setState('speaking');
  behavior.update(.2);
  assert.equal(behavior.gesture.name, 'backchannel');
  assert.ok(behavior.nextSpeakingGesture < 1);
  for (let i = 0; i < 16; i++) behavior.update(.1);
  assert.equal(behavior.gesture?.name, 'nod');
});

test('native body motions keep live audio control of mouth opening', () => {
  const { Live2DRenderer } = require('../../pet/live2d-renderer.js');
  const renderer = new Live2DRenderer({ addEventListener() {} });
  const applied = [];
  renderer.model = { internalModel: { coreModel: {
    setParameterValueById: (id, value) => applied.push([id, value]),
  } } };
  renderer.parameters = { ParamAngleX: 18, ParamMouthOpenY: .72 };
  renderer.nativeMotion = true;
  renderer.applyParameters();
  assert.deepEqual(applied, [['ParamMouthOpenY', .72]]);
});

test('rattan expressions are dispatched through the Live2D expression manager', async () => {
  const { Live2DRenderer } = require('../../pet/live2d-renderer.js');
  const renderer = new Live2DRenderer({ addEventListener() {} });
  let selected = '';
  renderer.model = {
    expression: async name => { selected = name; return true; },
    internalModel: { expressionManager: { resetExpression() {} } },
  };
  assert.equal(renderer.playExpression('blush', 0), true);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(selected, 'blush');
});
