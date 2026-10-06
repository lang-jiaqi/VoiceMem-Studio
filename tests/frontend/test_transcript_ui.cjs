// Exercise the real demo message handler without a browser/audio device.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../../studio/web/voicemem.html'), 'utf8');
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

function reset(){
  session.turns=[]; session.ui.input=''; session.ui.tags={emo:'平静'};
  delete session.inputTranscripts;
  context.liveChatDraft=null;
  voices.length=0;
}
function partial(id,text){ context.handle({type:'partial_transcript',input_turn_id:id,text}); }
function final(id,text,replace=''){
  context.handle({type:'user_transcript',input_turn_id:id,text,replace_input_turn_id:replace});
}

reset();
const full='请帮我比较两种方案，重点看准确率、响应速度和维护成本。';
partial('one','请帮我比较'); final('one',full);
partial('one','请帮我'); partial('one',''); final('one','旧的短结果');
assert.equal(session.ui.input,full);
assert.deepEqual(session.turns.map(t=>t.text),[full]);
partial('two','第二个问题'); partial('one','迟到的旧句子');
assert.equal(session.ui.input,'第二个问题');
final('two','第二个完整问题。');
assert.deepEqual(session.turns.map(t=>t.text),[full,'第二个完整问题。']);
console.log('identified finals survive stale partials, clears and duplicate finals PASS');

reset();
partial('one','第一个问题'); partial('two','第二个问题');
session.ui.tags={emo:'current'};
final('one','第一个完整问题。');
assert.equal(session.ui.input,'第二个问题');
assert.equal(session.ui.tags.emo,'current');
assert.equal(session.turns[0].text,'第一个完整问题。');
partial('one','再次迟到');
assert.equal(session.ui.input,'第二个问题');
console.log('older final updates its record without overwriting current input PASS');

reset();
let now=10000;
context.Date={now:()=>now};
final('prefix','我想问一下');
context.liveChatDraft={sessionId:session.id,said:{textContent:'existing draft'}};
now+=10000;
partial('continuation','模型应该怎么选');
final('continuation','我想问一下模型应该怎么选？','prefix');
final('third','我想问一下模型应该怎么选？还要考虑资源。','continuation');
partial('prefix','旧前缀'); final('continuation','旧的合并文本','prefix');
assert.deepEqual(session.turns.map(t=>t.text),['我想问一下模型应该怎么选？还要考虑资源。']);
assert.equal(session.ui.input,session.turns[0].text);
console.log('server-authorized continuation replaces one bubble without a time heuristic PASS');

reset();
final('one',full); final('two',full);
assert.equal(session.turns.length,2,'distinct input IDs must not merge repeated questions');
context.replyPlaybackGeneration=0;
context.spokenCaption={shown:'已播内容'};
context.setSpokenCaptionPlaying=()=>{};
context.paintSpokenCaption=()=>{};
context.finishLiveChatDraft=()=>{};
context.stopPlayback=()=>{};
context.handle({type:'answer_interrupt',heard_text:'已播内容',output_id:'assistant-output'});
assert.deepEqual(session.turns.map(t=>t.text),[full,full]);
assert.equal(session.ui.input,full);
console.log('reply interruption cannot erase confirmed user bubbles PASS');

reset();
context.handle({type:'partial_transcript',input_turn_id:'ack',text:'嗯',non_interrupting:true});
context.handle({type:'user_backchannel',input_turn_id:'ack',text:'嗯'});
context.handle({type:'user_backchannel',input_turn_id:'ack',text:'嗯'});
partial('ack','迟到文字');
assert.deepEqual(voices,[]);
assert.deepEqual(session.turns.map(t=>t.text),['嗯']);
assert.equal(session.ui.tags.emo,'平静');
console.log('identified acknowledgements remain display-only and non-interrupting PASS');

reset();
const longInput=full.repeat(20);
final('long-input',longInput);
partial('long-input',full);
assert.equal(session.ui.input,longInput);
assert.equal(session.turns[0].text,longInput);
console.log('long confirmed input is retained without a display character limit PASS');
