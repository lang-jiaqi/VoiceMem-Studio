// Exercise the real demo message handler without a browser/audio device.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../web/voicemem.html'), 'utf8');
const helperStart = html.indexOf('function addTurn(');
const helperEnd = html.indexOf('function paint(', helperStart);
const start = html.indexOf('function handle(m){');
const end = html.indexOf('\n}\n', start) + 2;
assert(helperStart >= 0 && helperEnd > helperStart && start >= 0 && end > start);
const session = {id:'test', named:false, turns:[], ui:{input:'', speaker:'你', tags:{emo:'平静'}}};
const voices = [], turns = [];
const context = {
  window:{}, console, Date, currentSession:()=>session,
  setVoice:v=>voices.push(v), paint:()=>{}, renderSessions:()=>{}, renderChat:()=>{},
  liveChatDraft:null, lvOn:false,
};
vm.runInNewContext(html.slice(helperStart, helperEnd), context);
vm.runInNewContext(html.slice(start, end), context);
context.handle({type:'partial_transcript', text:'ok.', non_interrupting:true});
context.handle({type:'user_backchannel', text:'ok.'});
assert.equal(session.ui.input, 'ok.');
assert.deepEqual(voices, []);
assert.deepEqual(session.turns.map(({role,text})=>({role,text})), [{role:'user', text:'ok.'}]);
assert.equal(session.ui.tags.emo, '平静');
context.handle({type:'partial_transcript', text:''});
assert.equal(session.ui.input, '');
assert.deepEqual(voices, []);
context.handle({type:'partial_transcript', text:'停一下'});
assert.deepEqual(voices, ['user']);
session.turns.length=0;
context.handle({type:'user_transcript', text:'我想问一下'});
context.handle({type:'user_transcript', text:'你还能更快吗'});
assert.deepEqual(session.turns.map(({role,text})=>({role,text})), [
  {role:'user', text:'我想问一下你还能更快吗'},
]);
console.log('ack display, continuation merge, echo clearing, and normal interrupt UI PASS');
