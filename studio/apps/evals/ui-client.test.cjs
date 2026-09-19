'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../ui/studio-client.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
function fixture({ desktop = false } = {}) {
  const sockets = [], nodes = [], sources = [], events = [], notices = [], states = [], contexts = [], listeners = {};
  let grant;
  class Socket {
    static OPEN = 1;
    constructor() { this.readyState = 1; this.sent = []; sockets.push(this); }
    send(message) { this.sent.push(typeof message === 'string' ? JSON.parse(message) : message); }
    close() { this.readyState = 3; }
    receive(message) { this.onmessage?.({ data: message instanceof ArrayBuffer ? message : typeof message === 'object' ? JSON.stringify(message) : message }); }
  }
  class Node {
    constructor(_context, name) {
      this.name = name; this.messages = [];
      this.port = { postMessage: m => this.messages.push(m), close() {} }; nodes.push(this);
    }
    connect() {} disconnect() {}
  }
  class Context {
    constructor(options = {}) { this.sampleRate = options.sampleRate || 24000; this.destination = {}; this.buffers = []; this.audioWorklet = {addModule: async () => {}}; contexts.push(this); }
    async resume() {} async close() {this.closed = true;}
    createGain() {return {connect() {}, disconnect() {}, gain:{value:1}};}
    createMediaStreamSource() {return {connect() {}, disconnect() {}};}
    createBuffer(_channels, length, sampleRate = this.sampleRate) {
      const data = new Float32Array(length), buffer = {length, sampleRate, getChannelData: () => data};
      this.buffers.push(buffer); return buffer;
    }
    createBufferSource() {
      const source = {connect() {}, disconnect() {}, start() {this.started = true;}, stop() {this.stopped = true;}};
      sources.push(source); return source;
    }
  }
  const context = {URL, ArrayBuffer, Float32Array, Int16Array, setTimeout, clearTimeout, console,
    AudioContext:Context, AudioWorkletNode:Node, WebSocket:Socket,
    location:{href:'http://localhost:8787/ui/technical.html',protocol:'http:'},
    document:{hidden:false, addEventListener(name, callback) {listeners[name] = callback;}},
    navigator:{mediaDevices:{getUserMedia:() => new Promise(resolve => {grant = resolve;})}},
    VMUI:{notify:message => notices.push(message)}, atob:text => Buffer.from(text, 'base64').toString('binary'),
  };
  context.window = {addEventListener(name, callback) {listeners[name] = callback;}};
  if (desktop) context.window.studioModelServices = {};
  vm.runInNewContext(source, context);
  const api = context.window.VMStudio;
  const client = api.create({onEvent:e => events.push(e), onState:state => states.push(state)});
  async function connected() {
    const sending = client.send('fixture'); await tick(); sockets[0].receive({type:'session_ready',mode:'llm_tts'}); await sending;
  }
  return {api, client, sockets, nodes, sources, events, notices, states, contexts, listeners,
    document:context.document, connected, grant:stream => grant(stream)};
}
test('desktop voice remains active when minimized while a browser session still ends', async () => {
  const desktop = fixture({ desktop: true });
  const starting = desktop.client.start(); await tick();
  desktop.sockets[0].receive({ type: 'session_ready' }); await tick();
  desktop.document.hidden = true; desktop.listeners.visibilitychange();
  assert.equal(desktop.states.at(-1), true);
  desktop.grant({ getTracks: () => [{ stop() {} }] }); await starting;
  assert.equal(desktop.sockets[0].readyState, 1);
  desktop.client.cancel();

  const browser = fixture(); await browser.connected();
  browser.document.hidden = true; browser.listeners.visibilitychange();
  assert.equal(browser.sockets[0].readyState, 3);
});
test('text waits for session readiness and uses the existing user_text contract', async () => {
  const f = fixture(); const sent = f.client.send('hello'); await tick();
  assert.equal(f.sockets[0].sent.length, 0);
  f.sockets[0].receive({type:'session_ready',mode:'llm_tts'}); await sent;
  assert.deepEqual(f.sockets[0].sent, [{type:'user_text',text:'hello'}]); f.client.cancel();
});
test('cancelled connection rejects pending text and ignores old socket events', async () => {
  const f = fixture(); const sent = f.client.send('hello'); await tick();
  f.client.cancel(); await assert.rejects(sent, /结束/);
  const count = f.events.length; f.sockets[0].receive({type:'answer_start',output_id:'stale'});
  assert.equal(f.events.length, count); assert.equal(f.contexts[0].closed, true);
});
test('cancelling a microphone permission request disposes its late stream', async () => {
  const f = fixture(); const starting = f.client.start(); await tick();
  f.sockets[0].receive({type:'session_ready'}); await tick(); f.client.cancel();
  let stopped = 0; f.grant({getTracks:() => [{stop() {stopped++;}}]}); await starting;
  assert.equal(stopped, 1); assert.equal(f.nodes.some(node => node.name.includes('mic-capture')), false);
});
test('pause/resume, drain, checkpoints and stale outputs retain output identity', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one',sample_rate:24000});
  socket.receive({type:'answer_pause',output_id:'one'}); socket.receive({type:'answer_resume',output_id:'one'});
  socket.receive({type:'answer_start',output_id:'two',sample_rate:24000});
  const count = player.messages.length; socket.receive({type:'answer_done',output_id:'one'});
  assert.equal(player.messages.length, count);
  socket.receive({type:'answer_done',output_id:'two'}); assert.equal(player.messages.at(-1).type, 'drain');
  player.port.onmessage({data:{type:'drained',outputId:'two',renderedSamples:24000,sampleRate:24000}});
  assert.equal(socket.sent.at(-1).rendered_samples, 24000); assert.equal(socket.sent.at(-1).state, 'drained');
  assert.equal(f.events.at(-1).type, 'playback_done'); f.client.cancel();
});
test('played audio levels drive the App-owned Live2D pet without becoming checkpoints', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one',sample_rate:24000});
  player.port.onmessage({data:{type:'level',outputId:'one',rms:.25,peak:.7,renderedSamples:1200,sampleRate:24000}});
  assert.deepEqual(socket.sent.at(-1), {
    type:'avatar_audio_level',output_id:'one',rms:.25,peak:.7,rendered_samples:1200,sample_rate:24000,
  });
  assert.equal(socket.sent.at(-1).state, undefined); f.client.cancel();
});
test('interrupt clears queued audio and preserves the server heard prefix', async () => {
  const f = fixture(); await f.connected();
  f.sockets[0].receive({type:'answer_start',output_id:'one'});
  f.sockets[0].receive({type:'answer_interrupt',output_id:'one',heard_text:'heard'});
  assert.equal(f.nodes[0].messages.at(-1).reason, 'interrupted');
  assert.equal(f.events.at(-1).heard_text, 'heard'); f.client.cancel();
});
test('reply replay uses the original PCM without sending new playback checkpoints', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one',sample_rate:24000});
  socket.receive(new Int16Array([0, 16384, -16384, 32767]).buffer);
  socket.receive({type:'answer_done',output_id:'one'});
  player.port.onmessage({data:{type:'drained',outputId:'one',renderedSamples:4,sampleRate:24000}});
  const sent = socket.sent.length, replayStates = [];
  assert.equal(await f.client.replay('one', state => replayStates.push(state)), true);
  const buffer = f.contexts[0].buffers.at(-1), data = buffer.getChannelData(0);
  assert.equal(buffer.length, 4); assert.equal(buffer.sampleRate, 24000);
  assert.deepEqual([...data].map(value => Number(value.toFixed(5))), [0, .5, -.5, .99997]);
  assert.equal(socket.sent.length, sent); assert.deepEqual(replayStates, [true]);
  f.sources.at(-1).onended(); assert.deepEqual(replayStates, [true, false]); f.client.cancel();
});
test('interrupted reply replay is trimmed to samples rendered by the worklet', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one',sample_rate:24000});
  socket.receive(new Int16Array([100, 200, 300, 400]).buffer);
  socket.receive({type:'answer_interrupt',output_id:'one',heard_text:'heard'});
  player.port.onmessage({data:{type:'interrupted',outputId:'one',renderedSamples:2,sampleRate:24000}});
  f.client.cancel();
  assert.equal(await f.client.replay('one'), true);
  assert.equal(f.contexts[1].buffers.at(-1).length, 2);
  f.client.stopReplay();
});
test('final transcript replaces optimistic text once and continuation keeps the original bubble', () => {
  const f = fixture(), messages = [{role:'user',text:'hello',pending:true}];
  f.api.applyUser(messages,{text:'hello',input_turn_id:'one'},'user');
  f.api.applyUser(messages,{text:'hello',input_turn_id:'one'},'user');
  assert.equal(messages.length, 1);
  f.api.applyUser(messages,{text:'hello again',input_turn_id:'two',replace_input_turn_id:'one'},'user');
  assert.equal(messages.length, 1); assert.equal(messages[0].text,'hello again');
});
test('disconnect releases audio resources and a new attempt gets a new socket', async () => {
  const f = fixture(); await f.connected(); f.sockets[0].onclose();
  assert.equal(f.contexts[0].closed, true); assert.equal(f.events.at(-1).type,'disconnected');
  const pending = f.client.send('new'); await tick(); assert.equal(f.sockets.length, 2);
  f.sockets[1].receive({type:'session_ready'}); await pending; f.client.cancel();
});
test('finalized transcripts reject stale partials and pre-merge final events', () => {
  const f = fixture(), messages = [];
  f.api.applyUser(messages,{text:'first',input_turn_id:'one'},'user');
  assert.equal(f.api.acceptsPartial(messages,{input_turn_id:'one'}), false);
  assert.equal(f.api.acceptsPartial(messages,{input_turn_id:'two'}), true);
  f.api.applyUser(messages,{text:'merged',input_turn_id:'two',replace_input_turn_id:'one'},'user');
  f.api.applyUser(messages,{text:'stale',input_turn_id:'one'},'user');
  assert.equal(messages.length, 1); assert.equal(messages[0].text, 'merged');
});
test('the new home page passes the unchanged desktop readiness probe', async () => {
  const html = fs.readFileSync(require('node:path').join(__dirname, '../ui/index.html'), 'utf8');
  await require('../runtime.cjs').probe('http://localhost:8787', {
    fetchImpl: async (_url, options) => {
      assert.equal(options.redirect, 'error');
      return new Response(html, {status:200});
    },
  });
});
