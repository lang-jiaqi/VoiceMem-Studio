// Exercise the real browser filler completion protocol without an audio device.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../studio/web/voicemem.html'), 'utf8');
const start = html.indexOf('function finishBackchannel(src){');
const end = html.indexOf('\n}', start) + 2;
const playEnd = html.indexOf('\n}', html.indexOf('function playBackchannel(m){')) + 2;
assert(start >= 0 && end > start && playEnd > end);

let source;
const sent = [];
const context = {
  atob: value => Buffer.from(value, 'base64').toString('binary'),
  bcGain: 0.8,
  activeBackchannels: new Set(),
  backchannelUntil: 0,
  pcmBackchannelPaused: false,
  pcmBargePaused: false,
  pcmNode: null,
  replyBus: null,
  playBus: {},
  playCtx: {
    createBuffer: (_channels, samples) => ({
      getChannelData: () => new Float32Array(samples),
    }),
    createBufferSource: () => (source = {
      playbackRate: {value: 1},
      connect: () => {},
      start: () => {},
      stop: () => {},
    }),
    createGain: () => ({gain: {value: 1}, connect: () => {}}),
  },
  ws: {readyState: 1, send: value => sent.push(JSON.parse(value))},
  WebSocket: {OPEN: 1},
};

vm.runInNewContext(html.slice(start, playEnd), context);
context.playBackchannel({
  pcm: Buffer.from([0, 0, 0, 0]).toString('base64'),
  sample_rate: 24000,
  interruptible: true,
  filler_id: 'filler-test',
});
assert(context.activeBackchannels.has(source));
assert.deepEqual(sent, []);

source.onended();
assert(!context.activeBackchannels.has(source));
assert.deepEqual(sent, [{type: 'filler_done', filler_id: 'filler-test'}]);
console.log('filler playback completion acknowledgement PASS');
