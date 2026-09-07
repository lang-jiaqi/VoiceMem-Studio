// Exercise the real demo message handler without a browser/audio device.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../web/voicemem.html'), 'utf8');
const start = html.indexOf('function handle(m){');
const end = html.indexOf('\n}\n', start) + 2;
assert(start >= 0 && end > start);
const session = {ui:{input:'', speaker:'你', tags:{emo:'平静'}}};
const voices = [], turns = [];
const context = {
  window:{}, console, currentSession:()=>session,
  setVoice:v=>voices.push(v), paint:()=>{},
  addTurn:(_, role, text)=>turns.push({role,text}),
};
vm.runInNewContext(html.slice(start, end), context);
context.handle({type:'partial_transcript', text:'ok.', non_interrupting:true});
context.handle({type:'user_backchannel', text:'ok.'});
assert.equal(session.ui.input, 'ok.');
assert.deepEqual(voices, []);
assert.deepEqual(turns, [{role:'user', text:'ok.'}]);
assert.equal(session.ui.tags.emo, '平静');
context.handle({type:'partial_transcript', text:''});
assert.equal(session.ui.input, '');
assert.deepEqual(voices, []);
context.handle({type:'partial_transcript', text:'停一下'});
assert.deepEqual(voices, ['user']);
console.log('ack display, echo clearing, and normal interrupt UI PASS');
