// Browser-free regression for independent backchannel playback.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../studio/web/voicemem.html'), 'utf8');
const start = html.indexOf('let playCtx=null');
const end = html.indexOf('/* --- 麦克风', start);
assert(start >= 0 && end > start);

const sources = [];
const audioContext = {
  currentTime: 10,
  sampleRate: 24000,
  destination: {},
  state: 'running',
  createGain() {
    return {gain: {value: 1}, connect() {}};
  },
  createBuffer(_channels, length, sampleRate) {
    const channel = new Float32Array(length);
    return {
      duration: length / sampleRate,
      getChannelData: () => channel,
      copyToChannel(samples) { channel.set(samples); },
    };
  },
  createBufferSource() {
    const source = {
      playbackRate: {value: 1},
      stopCalls: 0,
      connect() {},
      start(at = audioContext.currentTime) { this.startedAt = at; },
      stop() { this.stopCalls += 1; },
    };
    sources.push(source);
    return source;
  },
  resume() {},
};

const context = {
  SAMPLE_RATE: 24000,
  URLSearchParams,
  location: {search: ''},
  window: {AudioContext: function AudioContext() { return audioContext; }},
  console,
  Math,
  Float32Array,
  Int16Array,
  WebSocket: {OPEN: 1},
  atob: value => Buffer.from(value, 'base64').toString('binary'),
  setTimeout,
  clearTimeout,
  setSpokenCaptionPlaying() {},
  releaseMic() {},
};
vm.createContext(context);
vm.runInContext(html.slice(start, end), context);

context.ensurePlayback();
const pcm = Buffer.alloc(2400 * 2).toString('base64');
context.playBackchannel({pcm, sample_rate: 24000});
const backchannel = sources[0];
const expectedEnd = backchannel.startedAt + 0.1 / backchannel.playbackRate.value;

context.pausePlaybackForBarge();
assert.equal(backchannel.stopCalls, 0, 'candidate pause must preserve the acknowledgement');
context.resumePlaybackAfterBarge();
assert.equal(backchannel.stopCalls, 0, 'candidate resume must preserve the acknowledgement');

context.stopPlayback('interrupted');
assert.equal(backchannel.stopCalls, 0, 'interruption must not stop a backchannel');

context.holdReplyForBackchannel();
context.legacyPlayPcm(new Float32Array(240));
const reply = sources[1];
assert(reply.startedAt >= expectedEnd, 'main reply must start after the backchannel');

backchannel.onended();
assert.equal(backchannel.stopCalls, 0, 'a backchannel must finish naturally');

const answerStart = html.slice(
  html.indexOf("case 'answer_start':"), html.indexOf("case 'answer_delta':"));
assert.match(answerStart, /holdReplyForBackchannel\(\)/);
console.log('backchannel completion and reply handoff PASS');
