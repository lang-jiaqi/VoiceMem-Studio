'use strict';
// Execute the actual page lifecycle with deferred browser APIs, never a microphone.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../../studio/web/voicemem.html'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function harness(options = {}) {
  const sockets = [], tracks = [], sources = [], processors = [], permissions = [];
  const modules = [], resumes = [], memories = [], messages = [], playback = [], notices = [];
  const events = {};
  const button = () => ({ disabled: false, classList: { toggle() {} }, addEventListener(event, fn) { this[event] = fn; } });
  const buttons = { talkBtn: button(), pauseBtn: button() };
  class GraphNode {
    constructor(kind) { this.kind = kind; this.connections = new Set(); }
    connect(target) {
      if (this.kind === 'reference' && options.failReference && target.kind === 'worklet') throw new Error('synthetic reference failure');
      if (this.kind === 'script' && options.failFallback) throw new Error('synthetic fallback failure');
      this.connections.add(target);
    }
    disconnect(target) { target ? this.connections.delete(target) : this.connections.clear(); }
  }
  class Worklet extends GraphNode {
    constructor() {
      super('worklet');
      if (options.failWorklet) throw new Error('synthetic worklet failure');
      this.port = { onmessage: null, closed: false, close() { this.closed = true; } };
      processors.push(this);
    }
  }
  const reference = new GraphNode('reference');
  const audio = {
    state: options.deferResume ? 'suspended' : 'running',
    sampleRate: options.sampleRate || 24000, destination: {},
    audioWorklet: { addModule() {
      const request = deferred(); modules.push(request);
      if (!options.deferModule) request.resolve();
      return request.promise;
    } },
    resume() {
      const request = deferred(); resumes.push(request);
      return request.promise.then(() => { audio.state = 'running'; });
    },
    createMediaStreamSource(stream) {
      const source = new GraphNode('source'); source.stream = stream; sources.push(source); return source;
    },
    createScriptProcessor() {
      const processor = new GraphNode('script'); processors.push(processor); return processor;
    },
  };
  class Socket {
    static OPEN = 1;
    constructor() {
      if (options.failSocket) throw new Error('synthetic socket failure');
      this.readyState = 0; this.sent = []; sockets.push(this);
    }
    send(value) { assert.equal(this.readyState, Socket.OPEN); this.sent.push(value); }
    close() { this.readyState = 3; }
  }
  const context = vm.createContext({
    console: { info() {}, warn() {}, error() {} },
    window: { addEventListener: (event, fn) => { events[event] = fn; } },
    navigator: { mediaDevices: { getUserMedia() {
      const request = deferred(); permissions.push(request); return request.promise;
    } } },
    WebSocket: Socket, AudioWorkletNode: Worklet,
    ensurePlayback: () => audio, playBus: reference, stopPlayback() {},
    setComfortNoise() {}, pcmComfortDb: -50, pcmNode: null,
    SAMPLE_RATE: 24000, WS_URL: 'ws://fixture/ws', WS_HOST: 'fixture',
    HALF_DUPLEX: false, assistantAudible: () => true, replaying: false, lvOn: false,
    $: id => buttons[id], i18n: key => key, toast: value => notices.push(value),
    ensureSpaceObj() {}, createSession() {}, activeSpace: 'fixture',
    loadMemories() {
      const request = deferred(); memories.push(request);
      if (!options.deferMemory) request.resolve();
      return request.promise;
    },
    setVoice() {}, handle: value => messages.push(value), playPcm: value => playback.push(value),
    aiSpeaking: false, Float32Array, Int16Array, ArrayBuffer,
  });
  const declaration = html.match(/^let ws=null.*$/m);
  const guard = html.match(/^function currentLiveRun\(.*$/m);
  assert(declaration && guard);
  const micStart = html.indexOf('let micWorkletLoad='), micEnd = html.indexOf('/* --- 后端的记忆', micStart);
  const controlsStart = html.indexOf("const talkBtn=$('talkBtn')"), controlsEnd = html.indexOf('/* ── 语言切换', controlsStart);
  assert(micStart >= 0 && micEnd > micStart && controlsStart >= 0 && controlsEnd > controlsStart);
  vm.runInContext(`${declaration[0]}\n${guard[0]}\n${html.slice(micStart, micEnd)}\n${html.slice(controlsStart, controlsEnd)}`, context);

  function grant(index = permissions.length - 1) {
    const track = { stopped: false, stop() { this.stopped = true; }, getSettings: () => ({ echoCancellation: true }) };
    tracks.push(track);
    permissions[index].resolve({ getAudioTracks: () => [track], getTracks: () => [track] });
    return track;
  }
  function open(socket = sockets.at(-1)) { socket.readyState = Socket.OPEN; return socket.onopen(); }
  function emit(processor, count = 1) {
    for (let i = 0; i < count; i++) {
      const samples = new Float32Array(Math.round(audio.sampleRate * .02)).fill(.1);
      if (processor.kind === 'worklet') processor.port.onmessage?.({ data: { type: 'mic', samples } });
      else processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => samples } });
    }
  }
  function clean() {
    assert.equal(tracks.filter(track => !track.stopped).length, 0);
    assert.equal(reference.connections.size, 0);
    for (const node of [...sources, ...processors]) assert.equal(node.connections.size, 0);
    for (const node of processors) {
      if (node.port) { assert.equal(node.port.onmessage, null); assert.equal(node.port.closed, true); }
      assert.equal(node.onaudioprocess, null);
    }
  }
  return { sockets, tracks, processors, permissions, modules, resumes, memories, messages, playback, notices,
    events, buttons, context, audio, grant, open, emit, clean,
    click: () => buttons.talkBtn.click(), end: () => vm.runInContext('endLive()', context),
    run: () => vm.runInContext('liveRun', context),
  };
}

async function ready(h) {
  await h.click(); const opening = h.open(); await tick(); h.grant(); await opening;
  return h.processors.at(-1);
}

test('a second click cancels connecting instead of opening another socket', async () => {
  const h = harness(); await h.click();
  const staleOpen = h.sockets[0].onopen;
  assert.equal(h.buttons.talkBtn.textContent, 'cancelStart');
  assert.equal(h.buttons.pauseBtn.disabled, true);
  await h.click(); await staleOpen();
  assert.equal(h.sockets.length, 1);
  assert.equal(h.sockets[0].readyState, 3);
  assert.equal(h.permissions.length, 0);
  assert.equal(h.run(), null);
});

for (const oldFirst of [true, false]) test(`late permission from cancelled startup is stopped (old first: ${oldFirst})`, async () => {
  const h = harness(); await h.click(); const oldOpen = h.open(); await tick();
  await h.click(); await h.click(); const newOpen = h.open(); await tick();
  let oldTrack, newTrack;
  if (oldFirst) { oldTrack = h.grant(0); await oldOpen; newTrack = h.grant(1); await newOpen; }
  else { newTrack = h.grant(1); await newOpen; oldTrack = h.grant(0); await oldOpen; }
  assert.equal(oldTrack.stopped, true); assert.equal(newTrack.stopped, false);
  assert.equal(h.processors.length, 1);
  h.emit(h.processors[0], 50);
  assert.equal(h.sockets[0].sent.length, 0);
  assert.equal(h.sockets[1].sent.length, 50);
  assert(h.sockets[1].sent.every(pcm => pcm.byteLength === 960));
  h.end(); h.clean();
});

test('duplicate open and mic-start callbacks share one permission request', async () => {
  const h = harness(); await h.click(); const opening = h.open(); await tick();
  await h.sockets[0].onopen();
  const duplicate = vm.runInContext('startMic(liveRun)', h.context);
  assert.equal(h.permissions.length, 1);
  h.grant(); await Promise.all([opening, duplicate]);
  assert.equal(h.processors.length, 1);
  h.end(); h.clean();
});

test('cancelled memory setup cannot start a microphone in the new session', async () => {
  const h = harness({ deferMemory: true }); await h.click(); const old = h.open();
  await h.click(); await h.click(); const fresh = h.open();
  h.memories[0].resolve(); await old;
  assert.equal(h.permissions.length, 0);
  h.memories[1].resolve(); await tick(); h.grant(); await fresh;
  assert.equal(h.permissions.length, 1);
  h.end(); h.clean();
});

test('cancelled memory response does not modify the page', async () => {
  const request = deferred(); let active = true;
  const context = vm.createContext({ activeSpace: 'fixture', ensureSpaceObj() {},
    fetch: () => request.promise });
  const start = html.indexOf('async function loadMemories('), end = html.indexOf('\n}\n', start) + 2;
  vm.runInContext(html.slice(start, end), context);
  const loading = context.loadMemories(() => active);
  active = false;
  request.resolve({ ok: true, json: async () => ({ left: [], right: [] }) });
  // Any mutation past the ownership check would access deliberately absent UI globals.
  await loading;
});

test('cancellation while resuming audio prevents stale node creation', async () => {
  const h = harness({ deferResume: true }); await h.click(); const old = h.open(); await tick();
  h.grant(); await tick(); assert.equal(h.resumes.length, 1);
  h.end(); await h.click(); const fresh = h.open(); await tick(); h.grant(); await tick();
  h.resumes[1].resolve(); await fresh;
  h.resumes[0].resolve(); await old;
  assert.equal(h.processors.length, 1);
  h.end(); h.clean();
});

for (const fail of [false, true]) test(`pending worklet loading is shared without reviving old capture (failure: ${fail})`, async () => {
  const h = harness({ deferModule: true }); await h.click(); const old = h.open(); await tick(); h.grant(); await tick();
  h.end(); await h.click(); const fresh = h.open(); await tick(); h.grant(); await tick();
  assert.equal(h.modules.length, 1);
  fail ? h.modules[0].reject(new Error('synthetic module failure')) : h.modules[0].resolve();
  await Promise.all([old, fresh]);
  assert.equal(h.processors.length, 1);
  assert.equal(h.processors[0].kind, fail ? 'script' : 'worklet');
  h.emit(h.processors[0]); assert.equal(h.sockets[1].sent.length, 1);
  h.end(); h.clean();
});

test('partial worklet setup is detached before ScriptProcessor fallback', async () => {
  const h = harness({ failReference: true }); await ready(h);
  assert.equal(h.processors.length, 2);
  assert.equal(h.processors[0].port.closed, true);
  h.emit(h.processors[0]); h.emit(h.processors[1]);
  assert.equal(h.sockets[0].sent.length, 1);
  h.end(); h.clean();
});

test('failed fallback stops tracks and closes the owning socket', async () => {
  const h = harness({ failWorklet: true, failFallback: true }); await ready(h);
  assert.equal(h.run(), null);
  assert.equal(h.sockets[0].readyState, 3);
  h.clean();
});

test('permission denial resets startup and allows a clean retry', async () => {
  const h = harness(); await h.click(); const opening = h.open(); await tick();
  h.permissions[0].reject(new Error('synthetic permission denied')); await opening;
  assert.equal(h.run(), null);
  assert.equal(h.sockets[0].readyState, 3);
  assert.equal(h.buttons.talkBtn.textContent, 'talkStart');
  await ready(h); h.end(); h.clean();
});

test('stale permission failure cannot terminate a newer connection', async () => {
  const h = harness(); await h.click(); const old = h.open(); await tick(); h.end();
  await ready(h); const current = h.run(), notices = h.notices.length;
  h.permissions[0].reject(new Error('late permission failure')); await old;
  assert.equal(h.run(), current); assert.equal(h.notices.length, notices);
  h.end(); h.clean();
});

test('old socket and capture callbacks cannot send to or stop a new connection', async () => {
  const h = harness(); const oldNode = await ready(h), oldSocket = h.sockets[0];
  const callbacks = { message: oldSocket.onmessage, close: oldSocket.onclose, error: oldSocket.onerror,
    mic: oldNode.port.onmessage };
  h.end(); const fresh = await ready(h), current = h.run(), notices = h.notices.length;
  callbacks.mic({ data: { type: 'mic', samples: new Float32Array(480) } });
  callbacks.message({ data: JSON.stringify({ type: 'old' }) });
  callbacks.message({ data: new ArrayBuffer(2) });
  callbacks.close(); callbacks.error();
  assert.equal(h.run(), current); assert.equal(h.notices.length, notices);
  assert.equal(h.messages.length, 0); assert.equal(h.playback.length, 0);
  assert.equal(h.sockets[1].sent.length, 0);
  h.emit(fresh); assert.equal(h.sockets[1].sent.length, 1);
  h.sockets[1].onmessage({ data: JSON.stringify({ type: 'current' }) });
  assert.equal(h.messages[0].type, 'current');
  h.end(); h.clean();
});

for (const trigger of ['disconnect', 'pagehide']) test(`${trigger} invalidates a pending permission request`, async () => {
  const h = harness(); await h.click(); const opening = h.open(); await tick();
  if (trigger === 'disconnect') { h.sockets[0].readyState = 3; h.sockets[0].onclose(); }
  else h.events.pagehide();
  const track = h.grant(); await opening;
  assert.equal(track.stopped, true); assert.equal(h.processors.length, 0); assert.equal(h.run(), null);
});

test('repeated start and stop leave no active tracks or audio edges', async () => {
  const h = harness();
  for (let i = 0; i < 5; i++) {
    const node = await ready(h); h.emit(node);
    assert.equal(h.sockets[i].sent.length, 1);
    assert.equal(h.tracks.filter(track => !track.stopped).length, 1);
    h.end(); h.end(); h.clean();
  }
  assert.equal(h.modules.length, 1);
});

test('48k input is still resampled and pause/replay/half-duplex guards are preserved', async () => {
  const h = harness({ sampleRate: 48000 }); const node = await ready(h);
  h.emit(node); assert.equal(h.sockets[0].sent[0].byteLength, 960);
  h.buttons.pauseBtn.click(); h.emit(node); assert.equal(h.sockets[0].sent.length, 1);
  h.buttons.pauseBtn.click(); h.context.replaying = true; h.emit(node);
  h.context.replaying = false; h.context.HALF_DUPLEX = true; h.emit(node);
  assert.equal(h.sockets[0].sent.length, 1);
  h.context.HALF_DUPLEX = false; h.emit(node); assert.equal(h.sockets[0].sent.length, 2);
  h.end(); h.clean();
});

test('WebSocket creation failure does not leave startup locked', async () => {
  const h = harness({ failSocket: true }); await h.click();
  assert.equal(h.run(), null); assert.equal(h.buttons.talkBtn.textContent, 'talkStart');
  assert.equal(h.permissions.length, 0);
});
