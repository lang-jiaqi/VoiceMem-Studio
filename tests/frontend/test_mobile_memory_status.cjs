const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');
const source = fs.readFileSync('studio/apps/ui/pet-mobile.js', 'utf8');
const showStored = source.match(/  async function showStored\(event\) \{[\s\S]*?\n  \}\n(?=  function renderChat)/)[0];
const eventBody = source.match(/    onEvent\(event\) \{([\s\S]*?)\n    \},/)[1];
const tick = () => new Promise(resolve => setImmediate(resolve));

function environment(fetch) {
  const lists = new Map(), calls = [];
  const context = vm.createContext({
    list: (id, values, empty) => lists.set(id, { values: [...values], empty }),
    memoryKey: (kind, item) => `${kind}:${item.id}:${item.text}`,
    $: () => ({hidden: false, open: false}), renderChat: () => {},
    VMStudio: {applyUser: () => {}},
    fetch: (...args) => { calls.push(args); return fetch(...args); },
  });
  vm.runInContext(`let memoryPoll=0,memoryTurnId='',known=new Set(),baselineReady=true;
    const memorySnapshotReady=Promise.resolve(),messages=[];let replyMessage=null;
    ${showStored}
    function handle(event){${eventBody}}`, context);
  return { context, lists, calls, handle: event => context.handle(event) };
}

test('a long response does not poll or report empty before durable completion', async () => {
  const e = environment(async () => ({ok: true, json: async () => ({left:[],right:[]})}));
  e.handle({type:'user_transcript',input_turn_id:'current',text:'synthetic input'});
  await tick();
  assert.equal(e.calls.length, 0);
  assert.equal(e.lists.get('storedMemories').empty, '对话结束后写入记忆…');
  e.handle({type:'memory_store_status',input_turn_id:'previous',status:'empty'});
  assert.equal(e.lists.get('storedMemories').empty, '对话结束后写入记忆…');
  e.handle({type:'memory_store_status',input_turn_id:'current',status:'empty'});
  assert.equal(e.calls.length, 0);
  assert.equal(e.lists.get('storedMemories').empty, '本轮没有新入库记忆。');
});

test('completion fetches once and shows only factual IDs written by that turn', async () => {
  const e = environment(async () => ({ok:true,json:async () => ({left:[
    {id:'previous',text:'earlier turn'}, {id:'new',text:'current turn'}],right:[]})}));
  e.handle({type:'user_transcript',input_turn_id:'current'});
  e.handle({type:'memory_store_status',input_turn_id:'current',status:'stored',memory_ids:['new']});
  await tick();
  assert.equal(e.calls.length, 1);
  assert.deepEqual(e.lists.get('storedMemories').values, ['事实 · current turn']);
});

test('a snapshot completing after the next input cannot overwrite its status', async () => {
  let release;
  const response = new Promise(resolve => { release=resolve; });
  const e = environment(() => response);
  e.handle({type:'user_transcript',input_turn_id:'first'});
  e.handle({type:'memory_store_status',input_turn_id:'first',status:'stored',memory_ids:['first']});
  await tick();
  e.handle({type:'user_transcript',input_turn_id:'second'});
  release({ok:true,json:async () => ({left:[{id:'first',text:'old'}],right:[]})});
  await tick();
  assert.equal(e.lists.get('storedMemories').empty, '对话结束后写入记忆…');
});

test('snapshot failure does not falsely claim that durable memory was empty', async () => {
  const e = environment(async () => { throw new Error('synthetic unavailable'); });
  e.handle({type:'user_transcript',input_turn_id:'current'});
  e.handle({type:'memory_store_status',input_turn_id:'current',status:'stored'});
  await tick();
  assert.equal(e.lists.get('storedMemories').empty, '记忆已入库，暂时无法加载详情。');
});
