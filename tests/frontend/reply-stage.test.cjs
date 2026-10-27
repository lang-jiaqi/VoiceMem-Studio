'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function fixture() {
  const node = {}, timers = new Map(), listeners = {};
  let sequence = 0;
  const window = {addEventListener:(name, fn) => {listeners[name] = fn;}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/reply-stage.js'), 'utf8'), {
    window,document:{querySelectorAll:() => [node],addEventListener:(name, fn) => {listeners[name] = fn;}},
    setTimeout:fn => {timers.set(++sequence, fn); return sequence;}, clearTimeout:id => timers.delete(id),
  });
  return {node, timers, listeners, event:message => window.VMReplyStage.onEvent(message),
    flush() {for (const fn of timers.values()) fn(); timers.clear();}};
}
test('retrieved memory precedes generation and speech without delaying playback', () => {
  const f = fixture();
  f.event({type:'user_transcript'}); assert.equal(f.node.textContent, '检索记忆…');
  f.event({type:'reply_preparing',output_id:'one',provider:'deepseek',reply_mode:'memory'});
  f.event({type:'memory_hits',output_id:'one',stage:'retrieved',left_brain:[{}]});
  assert.equal(f.node.textContent, '已找到相关记忆');
  f.flush(); assert.equal(f.node.textContent, 'DeepSeek 正在生成…');
  f.event({type:'playback_started',output_id:'one'}); assert.equal(f.node.textContent, '正在说话…');
  f.event({type:'playback_done',output_id:'one'}); assert.equal(f.node.hidden, true);
});
test('stale events, cancellation, and space changes cannot resurrect an older stage', () => {
  const f = fixture();
  f.event({type:'reply_preparing',output_id:'one',provider:'deepseek',reply_mode:'memory_cot'});
  f.event({type:'memory_hits',output_id:'one',stage:'retrieved'}); f.flush();
  assert.equal(f.node.textContent, 'DeepSeek 正在思考…');
  f.event({type:'reply_preparing',output_id:'two'});
  f.event({type:'memory_hits',output_id:'two',stage:'retrieved'});
  f.event({type:'answer_interrupt',output_id:'one'}); assert.equal(f.node.hidden, false);
  f.event({type:'playback_started',output_id:'two'}); f.flush();
  assert.equal(f.node.textContent, '正在说话…');
  f.listeners['memory-space-change'](); f.flush(); assert.equal(f.node.hidden, true);
  f.event({type:'user_transcript'}); f.event({type:'disconnected'}); assert.equal(f.node.hidden, true);
});
