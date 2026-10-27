'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../../studio/apps/ui/studio-client.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
function fixture({ desktop = false, language = 'zh', prompts = {} } = {}) {
  const sockets = [], nodes = [], sources = [], events = [], notices = [], states = [], contexts = [], listeners = {}, documentEvents = [];
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
  class CustomEvent { constructor(type, options = {}) { this.type = type; this.detail = options.detail; } }
  const context = {URL, ArrayBuffer, Float32Array, Int16Array, setTimeout, clearTimeout, console, CustomEvent,
    AudioContext:Context, AudioWorkletNode:Node, WebSocket:Socket,
    location:{href:'http://localhost:8787/ui/technical.html',protocol:'http:'},
    document:{hidden:false, addEventListener(name, callback) {listeners[name] = callback;}, dispatchEvent(event) {documentEvents.push(event);listeners[event.type]?.(event);}},
    navigator:{mediaDevices:{getUserMedia:() => new Promise(resolve => {grant = resolve;})}},
    VMUI:{notify:message => notices.push(message)}, atob:text => Buffer.from(text, 'base64').toString('binary'),
  };
  context.window = {addEventListener(name, callback) {listeners[name] = callback;}};
  context.window.VMSettings = {language};
  context.window.VMHarnessPrompts = {saved:() => prompts};
  if (desktop) context.window.studioModelServices = {};
  vm.runInNewContext(source, context);
  const api = context.window.VMStudio;
  const client = api.create({onEvent:e => events.push(e), onState:state => states.push(state)});
  async function connected() {
    const sending = client.send('fixture'); await tick(); sockets[0].receive({type:'session_ready',mode:'llm_tts'}); await sending;
  }
  return {api, client, sockets, nodes, sources, events, notices, states, contexts, listeners, documentEvents,
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
  assert.deepEqual(f.sockets[0].sent, [{type:'conversation_language',language:'zh'}, {type:'user_text',text:'hello'}]); f.client.cancel();
});
test('chained playback enables bounded start grace while realtime keeps immediate admission', async () => {
  for (const mode of ['llm_tts', 'realtime']) {
    const f = fixture(), pending = f.client.send('fixture'); await tick();
    f.sockets[0].receive({type:'session_ready', mode}); await pending;
    const config = f.nodes[0].messages.find(message => message.type === 'config');
    assert.equal(config.prebuffer, mode === 'realtime' ? .08 : .16);
    assert.equal(config.startGrace, mode === 'realtime' ? 0 : .04);
    f.client.cancel();
  }
});
test('settings pause the microphone without discarding the session', async () => {
  const f = fixture(); const starting = f.client.start(); await tick();
  f.sockets[0].receive({type:'session_ready'}); await tick();
  let stopped = 0; f.grant({getTracks:() => [{stop() {stopped++;}}]}); await starting;
  f.listeners['settings-open']();
  assert.equal(stopped, 1); assert.equal(f.sockets[0].readyState, 1);
  f.client.cancel();
});
test('Self Harness state is published to settings and explicit choices use the typed protocol', async () => {
  const f = fixture(); await f.connected();
  const state = {type:'self_harness_state',schema:{},snapshot:{profile:{}},error:''};
  f.sockets[0].receive(state);
  assert.equal(f.documentEvents.at(-1).type, 'self-harness-state');
  assert.equal(JSON.stringify(f.documentEvents.at(-1).detail), JSON.stringify(state));
  await f.client.getHarness();
  await f.client.setHarness({turn_taking:{backchannel:'less'}});
  await f.client.setHarnessPrompt('persona', '多听一点');
  await f.client.setBackchannelCurve([1.8,.8,1.8,1.6]);
  assert.deepEqual(f.sockets[0].sent.slice(-4), [
    {type:'self_harness_get'},
    {type:'self_harness_update',update:{turn_taking:{backchannel:'less'}}},
    {type:'harness_prompt_update',name:'persona',value:'多听一点'},
    {type:'backchannel_curve_update',values:[1.8,.8,1.8,1.6]},
  ]);
  f.client.cancel();
});
test('language and saved prompts are restored before the first input', async () => {
  const f = fixture({language:'en', prompts:{main_system:'Synthetic system'}});
  await f.connected();
  assert.deepEqual(f.sockets[0].sent.slice(0, 3), [
    {type:'conversation_language',language:'en'},
    {type:'harness_prompts_update',prompts:{main_system:'Synthetic system'}},
    {type:'user_text',text:'fixture'},
  ]);
  f.client.cancel();
});
test('the selected base language reaches the server before microphone opening', async () => {
  for (const language of ['zh', 'en']) {
    const f = fixture({language});
    const starting = f.client.start(); await tick();
    f.sockets[0].receive({type:'session_ready'}); await tick();
    f.grant({getTracks:() => [{stop(){}}]}); await starting;
    const messages = f.sockets[0].sent;
    const configured = messages.findIndex(message => message.type === 'conversation_language');
    const opening = messages.findIndex(message => message.type === 'conversation_start');
    assert.equal(messages[configured].language, language);
    assert.ok(configured >= 0 && opening > configured);
    f.client.cancel();
  }
});
test('prompt saves await matching acknowledgements and reject on error or disconnect', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0];
  const first = f.client.setHarnessPrompts({main_system:'Synthetic system'}, 'en'); await tick();
  const request = socket.sent.at(-1);
  assert.equal(request.type, 'harness_prompts_update');
  assert.equal(request.prompt_language, 'en');
  let accepted = false; first.then(() => {accepted = true;});
  socket.receive({type:'self_harness_state',request_id:'other',error:''}); await tick();
  assert.equal(accepted, false);
  socket.receive({type:'self_harness_state',request_id:request.request_id,error:''}); await first;
  const invalid = f.client.setHarnessPrompts({main_system:''}); await tick();
  socket.receive({type:'self_harness_state',request_id:socket.sent.at(-1).request_id,error:'Invalid prompt'});
  await assert.rejects(invalid, /Invalid prompt/);
  const disconnected = f.client.setHarnessPrompts({}); await tick();
  f.client.cancel(); await assert.rejects(disconnected, /结束/);
});
test('confirmed preparation shows memory before answer start and rejects older enrichment', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0];
  socket.receive({type:'answer_start',output_id:'old'});
  socket.receive({type:'reply_preparing',output_id:'new'});
  socket.receive({type:'memory_hits',output_id:'new',stage:'retrieved'});
  assert.equal(f.events.at(-1).type, 'memory_hits');
  const count = f.events.length;
  socket.receive({type:'memory_hits',output_id:'old'});
  socket.receive({type:'tag_update',output_id:'old'});
  assert.equal(f.events.length, count);
  socket.receive({type:'answer_start',output_id:'new'});
  socket.receive({type:'memory_hits',output_id:'new'});
  assert.equal(f.events.at(-1).output_id, 'new');
  f.client.cancel();
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
test('live orb phases follow preparation, actual playback, interruption, and cancellation', async () => {
  const phases=[],f=fixture();
  const client=f.api.create({onPhase:value=>phases.push(value)});
  const connecting=client.send('Synthetic input');await tick();
  const socket=f.sockets[0];socket.receive({type:'session_ready',mode:'llm_tts'});await connecting;
  assert.equal(phases.at(-1),'short-thinking');
  socket.receive({type:'reply_preparing',output_id:'one',reply_mode:'memory_cot'});
  assert.equal(phases.at(-1),'long-thinking');
  socket.receive({type:'answer_start',output_id:'one'});
  assert.ok(phases.at(-1).includes('thinking'));
  const player=f.nodes[0];
  player.port.onmessage({data:{type:'started',outputId:'stale'}});
  assert.ok(phases.at(-1).includes('thinking'));
  player.port.onmessage({data:{type:'started',outputId:'one'}});
  assert.equal(phases.at(-1),'speaking');
  socket.receive({type:'answer_pause',output_id:'one'});assert.equal(phases.at(-1),'listening');
  socket.receive({type:'answer_resume',output_id:'one'});
  player.port.onmessage({data:{type:'resumed',outputId:'one'}});assert.equal(phases.at(-1),'speaking');
  socket.receive({type:'answer_interrupt',output_id:'one'});assert.equal(phases.at(-1),'idle');
  client.cancel();assert.equal(phases.at(-1),'idle');
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
test('speech errors clear their own output and stale errors leave a new output playing', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one'});
  socket.receive({type:'answer_start',output_id:'two'});
  const count = player.messages.length;
  socket.receive({type:'error',output_id:'one',code:'speech_failed',message:'old failure'});
  assert.equal(player.messages.length, count);
  assert.equal(f.notices.length, 0);
  socket.receive({type:'error',output_id:'two',code:'speech_failed',message:'speech failed'});
  assert.equal(player.messages.at(-1).type, 'clear');
  assert.equal(f.notices.at(-1), 'speech failed');
  assert.equal(f.events.at(-1).code, 'speech_failed'); f.client.cancel();
});
test('interrupt clears queued audio and preserves the server heard prefix', async () => {
  const f = fixture(); await f.connected();
  f.sockets[0].receive({type:'answer_start',output_id:'one'});
  f.sockets[0].receive({type:'answer_interrupt',output_id:'one',heard_text:'heard'});
  assert.equal(f.nodes[0].messages.at(-1).reason, 'interrupted');
  assert.equal(f.events.at(-1).heard_text, 'heard'); f.client.cancel();
});
test('reply text and completion wait for main playback, including short replies', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one'});
  socket.receive({type:'answer_delta',output_id:'one',text:'first'});
  player.port.onmessage({data:{type:'resumed',outputId:'one'}});
  socket.receive(new Int16Array([100,200]).buffer);
  socket.receive({type:'answer_delta',output_id:'one',text:'second'});
  socket.receive({type:'answer_done',output_id:'one'});
  const replies = () => f.events.filter(e => ['answer_delta','answer_done'].includes(e.type));
  assert.equal(replies().length,0);
  assert.equal(player.messages.at(-1).type,'drain');
  player.port.onmessage({data:{type:'started',outputId:'one'}});
  assert.deepEqual(replies().map(e => e.text || e.type),['first','second','answer_done']);
  player.port.onmessage({data:{type:'drained',outputId:'one',renderedSamples:2}});
  assert.equal(f.events.at(-1).type,'playback_done'); f.client.cancel();
});
test('candidate pauses hold text and completion while PCM continues buffering', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one'});
  player.port.onmessage({data:{type:'started',outputId:'one'}});
  socket.receive({type:'answer_delta',output_id:'one',text:'visible'});
  socket.receive({type:'answer_pause',output_id:'one'});
  socket.receive({type:'answer_delta',output_id:'one',text:'held'});
  socket.receive(new Int16Array([100,200]).buffer);
  assert.equal(player.messages.at(-1).type,'audio');
  socket.receive({type:'answer_done',output_id:'one'});
  assert.deepEqual(f.events.filter(e => e.type==='answer_delta').map(e => e.text),['visible']);
  assert.equal(f.events.some(e => e.type==='answer_done'),false);
  socket.receive({type:'answer_resume',output_id:'one'});
  assert.deepEqual(f.events.filter(e => ['answer_delta','answer_done'].includes(e.type)).map(e => e.text || e.type),['visible','held','answer_done']);
  f.client.cancel();
});
test('resume before the first audio never releases text and a racing start respects pause', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one'});
  socket.receive({type:'answer_delta',output_id:'one',text:'held'});
  socket.receive({type:'answer_pause',output_id:'one'});
  socket.receive({type:'answer_resume',output_id:'one'});
  assert.equal(f.events.some(e => e.type==='answer_delta'),false);
  socket.receive({type:'answer_pause',output_id:'one'});
  player.port.onmessage({data:{type:'started',outputId:'one'}});
  assert.equal(f.events.some(e => e.type==='answer_delta'),false);
  socket.receive({type:'answer_resume',output_id:'one'});
  assert.deepEqual(f.events.filter(e => e.type==='answer_delta').map(e => e.text),['held']);
  f.client.cancel();
});
test('interruption and replacement discard private text and ignore late events', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one'});
  socket.receive({type:'answer_delta',output_id:'one',text:'discarded'});
  socket.receive({type:'answer_done',output_id:'one'});
  socket.receive({type:'answer_interrupt',output_id:'one',heard_text:''});
  player.port.onmessage({data:{type:'started',outputId:'one'}});
  socket.receive({type:'answer_delta',output_id:'one',text:'late'});
  socket.receive({type:'answer_start',output_id:'two'});
  socket.receive({type:'answer_delta',output_id:'two',text:'replaced'});
  socket.receive({type:'answer_start',output_id:'three'});
  socket.receive({type:'answer_delta',output_id:'three',text:'current'});
  socket.receive({type:'answer_pause',output_id:'two'});
  player.port.onmessage({data:{type:'started',outputId:'two'}});
  player.port.onmessage({data:{type:'started',outputId:'three'}});
  assert.deepEqual(f.events.filter(e => e.type==='answer_delta').map(e => e.text),['current']);
  socket.receive({type:'answer_pause',output_id:'three'});
  socket.receive({type:'answer_delta',output_id:'three',text:'also discarded'});
  socket.receive({type:'answer_resume',output_id:'two'});
  socket.receive({type:'error',output_id:'three',message:'synthetic failure'});
  player.port.onmessage({data:{type:'resumed',outputId:'three'}});
  assert.deepEqual(f.events.filter(e => e.type==='answer_delta').map(e => e.text),['current']);
  f.client.cancel();
});
test('a successful text-only drain keeps completion ordered, including during pause', async () => {
  for (const paused of [false,true]) {
    const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
    socket.receive({type:'answer_start',output_id:'one'});
    if (paused) socket.receive({type:'answer_pause',output_id:'one'});
    socket.receive({type:'answer_delta',output_id:'one',text:'text only'});
    socket.receive({type:'answer_done',output_id:'one'});
    player.port.onmessage({data:{type:'drained',outputId:'one',renderedSamples:0}});
    if (paused) {
      assert.equal(f.events.some(e => ['answer_delta','answer_done','playback_done'].includes(e.type)),false);
      socket.receive({type:'answer_resume',output_id:'one'});
    }
    assert.deepEqual(f.events.filter(e => ['answer_delta','answer_done','playback_done'].includes(e.type)).map(e => e.text || e.type),['text only','answer_done','playback_done']);
    f.client.cancel();
  }
});
test('backchannel completion cannot release reply text before main playback', async () => {
  const f = fixture(); await f.connected(); const socket = f.sockets[0], player = f.nodes[0];
  socket.receive({type:'backchannel',pcm:Buffer.from([0,0,0,0]).toString('base64'),filler_id:'filler'});
  socket.receive({type:'answer_start',output_id:'one'});
  socket.receive({type:'answer_delta',output_id:'one',text:'main reply'});
  f.sources[0].onended();
  player.port.onmessage({data:{type:'resumed',outputId:'one'}});
  assert.equal(f.events.some(e => e.type==='answer_delta'),false);
  player.port.onmessage({data:{type:'started',outputId:'one'}});
  assert.deepEqual(f.events.filter(e => e.type==='answer_delta').map(e => e.text),['main reply']);
  f.client.cancel();
});
test('disconnect drops held reply text before reconnecting', async () => {
  const f = fixture(); await f.connected();
  f.sockets[0].receive({type:'answer_start',output_id:'one'});
  f.sockets[0].receive({type:'answer_delta',output_id:'one',text:'discarded'});
  f.client.cancel();
  const pending=f.client.send('new'); await tick(); f.sockets[1].receive({type:'session_ready'}); await pending;
  f.sockets[0].receive({type:'answer_resume',output_id:'one'});
  f.sockets[1].receive({type:'answer_start',output_id:'two'});
  f.sockets[1].receive({type:'answer_delta',output_id:'two',text:'new reply'});
  f.nodes.at(-1).port.onmessage({data:{type:'started',outputId:'two'}});
  assert.deepEqual(f.events.filter(e => e.type==='answer_delta').map(e => e.text),['new reply']);
  f.client.cancel();
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
test('repeated continuations retain aliases and reject stale replacement events', () => {
  const f = fixture(), messages = [];
  f.api.applyUser(messages,{text:'first',input_turn_id:'one'},'user');
  f.api.applyUser(messages,{text:'first second',input_turn_id:'two',replace_input_turn_id:'one'},'user');
  f.api.applyUser(messages,{text:'first second third',input_turn_id:'three',replace_input_turn_id:'one'},'user');
  f.api.applyUser(messages,{text:'stale second',input_turn_id:'two',replace_input_turn_id:'one'},'user');
  assert.equal(messages.length, 1);
  assert.equal(messages[0].text, 'first second third');
  for (const id of ['one','two','three']) assert.equal(f.api.acceptsPartial(messages,{input_turn_id:id}),false);
  f.api.applyUser(messages,{text:'a new question',input_turn_id:'four'},'user');
  assert.equal(messages.length, 2);
});
test('the new home page passes the unchanged desktop readiness probe', async () => {
  const html = fs.readFileSync(require('node:path').join(__dirname, '../../studio/apps/ui/index.html'), 'utf8');
  await require('../../studio/apps/runtime.cjs').probe('http://localhost:8787', {
    fetchImpl: async (_url, options) => {
      assert.equal(options.redirect, 'error');
      return new Response(html, {status:200});
    },
  });
});

test('a completed short caption appears in full at first playback, with completion held until drain', async () => {
  const f=fixture();await f.connected();const socket=f.sockets[0], player=f.nodes[0];
  const text='好呀😀，我们就从这件事开始。你慢慢说，我在听。';
  socket.receive({type:'answer_start',output_id:'short',caption_sync:'playback'});
  for (const char of text) socket.receive({type:'answer_delta',output_id:'short',text:char});
  socket.receive({type:'answer_text_done',output_id:'short'});
  assert.equal(f.events.some(event=>event.type==='answer_delta'),false);
  player.port.onmessage({data:{type:'resumed',outputId:'short'}});
  assert.equal(f.events.some(event=>event.type==='answer_delta'),false);
  player.port.onmessage({data:{type:'started',outputId:'short'}});
  assert.deepEqual(f.events.filter(event=>event.type==='answer_delta').map(event=>event.text),[text]);
  assert.equal(f.events.some(event=>event.type==='answer_done'),false);
  assert.equal(player.messages.some(message=>message.type==='drain'),false);
  socket.receive({type:'answer_done',output_id:'short'});
  player.port.onmessage({data:{type:'drained',outputId:'short'}});
  assert.deepEqual(f.events.slice(-2).map(event=>event.type),['answer_done','playback_done']);
  f.client.cancel();
});

test('long captions preview whole phrases and use heard code-point offsets to advance', async () => {
  const f=fixture();await f.connected();const socket=f.sockets[0], player=f.nodes[0];
  const deltaText=()=>f.events.filter(event=>event.type==='answer_delta').map(event=>event.text).join('');
  const first='先把今天最想完成的一件事写下来😀。';
  const second='然后把它拆成几个容易开始的小步骤，';
  const tail='每一步只安排一个明确的动作，不用一开始就把整天塞得很满。做完第一步以后再看看实际花了多少时间，最后根据精力调整后面的安排。';
  const text=first+second+tail, firstEnd=Array.from(first).length;
  assert.ok(Array.from(text).length>80);
  socket.receive({type:'answer_start',output_id:'sync',caption_sync:'playback'});
  socket.receive({type:'answer_delta',output_id:'sync',text});
  socket.receive({type:'answer_done',output_id:'sync'});
  socket.receive({type:'answer_progress',output_id:'sync',text_end:3});
  assert.equal(deltaText(),'');
  player.port.onmessage({data:{type:'started',outputId:'sync'}});
  assert.equal(deltaText(),first);
  socket.receive({type:'answer_progress',output_id:'old',text_end:100});
  socket.receive({type:'answer_progress',output_id:'sync',text_end:NaN});
  socket.receive({type:'answer_progress',output_id:'sync',text_end:-1});
  socket.receive({type:'answer_progress',output_id:'sync',text_end:firstEnd-1});
  player.port.onmessage({data:{type:'underflow',outputId:'sync'}});
  assert.equal(deltaText(),first);
  socket.receive({type:'answer_pause',output_id:'sync'});
  socket.receive({type:'answer_progress',output_id:'sync',text_end:firstEnd});
  assert.equal(deltaText(),first);
  assert.equal(f.events.filter(event=>event.type==='answer_done').length,0);
  socket.receive({type:'answer_resume',output_id:'sync'});
  assert.equal(deltaText(),first+second);
  socket.receive({type:'answer_progress',output_id:'sync',text_end:4});
  assert.equal(deltaText(),first+second);
  player.port.onmessage({data:{type:'drained',outputId:'sync'}});
  assert.equal(deltaText(),text);
  assert.deepEqual(f.events.slice(-3).map(event=>event.type),['answer_delta','answer_done','playback_done']);
  f.client.cancel();
});

test('streaming captions wait for a phrase rather than revealing individual tokens', async () => {
  const f=fixture();await f.connected();const socket=f.sockets[0], player=f.nodes[0];
  const deltas=()=>f.events.filter(event=>event.type==='answer_delta').map(event=>event.text);
  socket.receive({type:'answer_start',output_id:'stream',caption_sync:'playback'});
  player.port.onmessage({data:{type:'started',outputId:'stream'}});
  const first='我们先从一个简单的步骤开始。', second='接下来再根据实际情况调整。';
  for (const char of first.slice(0,-1)) {
    socket.receive({type:'answer_delta',output_id:'stream',text:char});
    assert.equal(deltas().length,0);
  }
  socket.receive({type:'answer_delta',output_id:'stream',text:'。'});
  assert.deepEqual(deltas(),[first]);
  for (const char of second) socket.receive({type:'answer_delta',output_id:'stream',text:char});
  assert.deepEqual(deltas(),[first]);
  socket.receive({type:'answer_progress',output_id:'stream',text_end:Array.from(first).length});
  assert.deepEqual(deltas(),[first,second]);
  socket.receive({type:'answer_delta',output_id:'stream',text:'最后这句没有标点'});
  assert.deepEqual(deltas(),[first,second]);
  socket.receive({type:'answer_text_done',output_id:'stream'});
  assert.deepEqual(deltas(),[first,second,'最后这句没有标点']);
  f.client.cancel();
});

test('short text completion during a candidate pause waits for resume', async () => {
  const f=fixture();await f.connected();const socket=f.sockets[0], player=f.nodes[0];
  const first='我们先从这一件事开始。', tail='之后再一起看看下一步怎么走。';
  socket.receive({type:'answer_start',output_id:'paused',caption_sync:'playback'});
  socket.receive({type:'answer_delta',output_id:'paused',text:first});
  socket.receive({type:'answer_pause',output_id:'paused'});
  player.port.onmessage({data:{type:'started',outputId:'paused'}});
  socket.receive({type:'answer_delta',output_id:'paused',text:tail});
  socket.receive({type:'answer_text_done',output_id:'paused'});
  assert.equal(f.events.some(event=>event.type==='answer_delta'),false);
  socket.receive({type:'answer_resume',output_id:'paused'});
  assert.deepEqual(f.events.filter(event=>event.type==='answer_delta').map(event=>event.text),[first+tail]);
  assert.equal(f.events.some(event=>event.type==='answer_done'),false);
  socket.receive({type:'answer_interrupt',output_id:'paused',heard_text:'我们先'});
  socket.receive({type:'answer_text_done',output_id:'paused'});
  socket.receive({type:'answer_delta',output_id:'paused',text:'late text'});
  assert.equal(f.events.filter(event=>event.type==='answer_delta').length,1);
  assert.equal(f.events.find(event=>event.type==='answer_interrupt').heard_text,'我们先');
  f.client.cancel();
});

test('unpunctuated captions use bounded blocks and keep Unicode and English words intact', async () => {
  for (const text of ['😀'.repeat(70), 'A long explanation without punctuation should still arrive in readable groups of complete words']) {
    const f=fixture();await f.connected();const socket=f.sockets[0], player=f.nodes[0];
    socket.receive({type:'answer_start',output_id:'block',caption_sync:'playback'});
    player.port.onmessage({data:{type:'started',outputId:'block'}});
    socket.receive({type:'answer_delta',output_id:'block',text});
    const prefix=f.events.find(event=>event.type==='answer_delta').text;
    assert.ok(Array.from(prefix).length<=64);
    if (text.startsWith('😀')) assert.equal(prefix,'😀'.repeat(64));
    else assert.ok(prefix.endsWith(' '));
    assert.ok(text.startsWith(prefix));
    socket.receive({type:'answer_done',output_id:'block'});
    player.port.onmessage({data:{type:'drained',outputId:'block'}});
    assert.equal(f.events.filter(event=>event.type==='answer_delta').map(event=>event.text).join(''),text);
    f.client.cancel();
  }
});

test('caption boundaries do not split streamed decimals or grouped numbers', async () => {
  const f=fixture();await f.connected();const socket=f.sockets[0], player=f.nodes[0];
  socket.receive({type:'answer_start',output_id:'numbers',caption_sync:'playback'});
  player.port.onmessage({data:{type:'started',outputId:'numbers'}});
  for (const text of ['The value is 3.','14 and the count is 1,','024']) {
    socket.receive({type:'answer_delta',output_id:'numbers',text});
    assert.equal(f.events.some(event=>event.type==='answer_delta'),false);
  }
  socket.receive({type:'answer_delta',output_id:'numbers',text:'. '});
  assert.deepEqual(f.events.filter(event=>event.type==='answer_delta').map(event=>event.text),
    ['The value is 3.14 and the count is 1,024. ']);
  f.client.cancel();
});

test('synced caption interruption, output replacement and disconnect discard unseen text', async () => {
  for (const terminal of ['answer_interrupt','error','disconnect','replacement']) {
    const f=fixture();await f.connected();const socket=f.sockets[0], player=f.nodes[0];
    socket.receive({type:'answer_start',output_id:'one',caption_sync:'playback'});
    socket.receive({type:'answer_delta',output_id:'one',text:'unheard text'});
    if (terminal==='disconnect') f.client.cancel();
    else if (terminal==='replacement') socket.receive({type:'answer_start',output_id:'two',caption_sync:'playback'});
    else socket.receive({type:terminal,output_id:'one',heard_text:'heard'});
    socket.receive({type:'answer_progress',output_id:'one',text_end:100});
    socket.receive({type:'answer_text_done',output_id:'one'});
    player.port.onmessage?.({data:{type:'started',outputId:'one'}});
    assert.equal(f.events.filter(event=>event.type==='answer_delta').length,0);
    if (terminal==='answer_interrupt') assert.equal(f.events.find(event=>event.type===terminal).heard_text,'heard');
    f.client.cancel();
  }
});

test('synced text-only drain and paused completion preserve event order', async () => {
  const f=fixture();await f.connected();const socket=f.sockets[0], player=f.nodes[0];
  socket.receive({type:'answer_start',output_id:'one',caption_sync:'playback'});
  socket.receive({type:'answer_delta',output_id:'one',text:'text only'});
  socket.receive({type:'answer_pause',output_id:'one'});
  socket.receive({type:'answer_done',output_id:'one'});
  player.port.onmessage({data:{type:'drained',outputId:'one'}});
  assert.equal(f.events.filter(event=>event.type==='answer_delta').length,0);
  socket.receive({type:'answer_resume',output_id:'one'});
  assert.deepEqual(f.events.filter(event=>['answer_delta','answer_done','playback_done'].includes(event.type)).map(event=>event.type),
    ['answer_delta','answer_done','playback_done']);
  f.client.cancel();
});
