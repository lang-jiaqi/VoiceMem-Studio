'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('conversations select one of the backend Memory Spaces and retain its identity', async () => {
  const calls = [], events = [];
  let active = 'studio-zh', dialog;
  const spaces = [{id:'studio-zh',name:'工作',count:3}, {id:'project',name:'project',count:2}];
  const element = () => ({children:[], attrs:{}, append(...items){this.children.push(...items);}, setAttribute(name,value){this.attrs[name]=value;},
    replaceChildren(...items){this.children=items;}, addEventListener(){}, close(){}, remove(){}, showModal(){},
    focus(){}});
  const context = {
    window: {dispatchEvent(event){events.push(event.detail.space);}},
    document: {createElement: element, body: {append(value){dialog=value;}}},
    CustomEvent: class {constructor(_name, options){this.detail=options.detail;}},
    fetch: async (url, options = {}) => {
      calls.push([url, options.method || 'GET']);
      if (url === '/api/spaces') return {ok:true,json:async()=>({spaces,active})};
      if (url === '/api/spaces/project/use') {active='project';return {ok:true,json:async()=>({active})};}
      throw new Error(`Unexpected request: ${url}`);
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/conversation-spaces.js'), 'utf8'), context);
  const api = context.window.VMConversationSpaces;
  assert.equal(await api.ready, 'studio-zh');
  const choice = api.choose('studio-zh');
  await new Promise(resolve => setImmediate(resolve));
  const list = dialog.children[2];
  assert.equal(list.children[0].children[0].textContent,'工作');
  assert.equal(list.children[0].children[0].attrs['data-no-i18n'],'');
  assert.equal(list.children[0].children[2].textContent,'3 条记忆');
  list.children[1].onclick();
  assert.equal(list.children.length, spaces.length);
  const footer = dialog.children.at(-1);
  assert.equal(footer.className, 'space-picker__footer');
  assert.equal(footer.children[1].textContent, '创建对话');
  footer.children[1].onclick();
  assert.equal(await choice, 'project');
  assert.equal(await api.use('project'), 'project');
  assert.equal(api.name('project'), 'project');
  assert.deepEqual(events, ['studio-zh', 'project']);
  assert.deepEqual(calls.at(-1), ['/api/spaces/project/use', 'POST']);
});

test('an exact memory root keeps the startup Space and hides creation controls', async () => {
  const calls = []; let dialog;
  const element = () => ({children:[], append(...items){this.children.push(...items);}, setAttribute(){},
    replaceChildren(...items){this.children=items;}, addEventListener(){}, close(){}, remove(){}, showModal(){}});
  const context = {
    window: {dispatchEvent(){}},
    document: {createElement: element, body: {append(value){dialog=value;}}},
    CustomEvent: class {constructor(_name, options){this.detail=options.detail;}},
    fetch: async (url, options = {}) => {
      calls.push([url, options.method || 'GET']);
      assert.equal(url, '/api/spaces');
      return {ok:true,json:async()=>({active:'mine', spaces:[{id:'mine',name:'mine',count:1,fixed:true}]})};
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/conversation-spaces.js'), 'utf8'), context);
  const api = context.window.VMConversationSpaces;
  await api.ready;
  const choosing = api.choose('old-history-space');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(dialog.children[3].hidden, true);
  assert.match(dialog.children[1].textContent, /单个记忆库/);
  assert.equal(dialog.children[2].children[0].className, 'selected');
  dialog.children.at(-1).children[1].onclick();
  assert.equal(await choosing, 'mine');
  assert.equal(await api.use('mine'), 'mine');
  await assert.rejects(api.use('old-history-space'), /不存在/);
  assert.equal(calls.some(([, method]) => method === 'POST'), false);
});

function mutableSpaces(language, switchRequest) {
  const calls = [], events = [];
  let active = 'one', dialog;
  const spaces = [{id:'one',name:'One',count:0}, {id:'two',name:'Two',count:0}];
  const element = () => ({children:[],append(...items){this.children.push(...items);},setAttribute(){},
    replaceChildren(...items){this.children=items;},addEventListener(){},close(){},remove(){},showModal(){},focus(){}});
  const context = {window:{VMSettings:{language},dispatchEvent:event=>events.push(event.detail.space)},
    document:{createElement:element,body:{append:value=>{dialog=value;}}},
    CustomEvent:class {constructor(_name, options){this.detail=options.detail;}},
    fetch:async (url, options={})=>{
      calls.push({url,...options});
      if (url === '/api/spaces' && !options.method) return {ok:true,json:async()=>({spaces,active})};
      if (url === '/api/spaces') {
        const body = JSON.parse(options.body);
        const created = {id:body.name,name:body.name,count:0,language:body.language};
        spaces.push(created);
        return {ok:true,json:async()=>created};
      }
      if (switchRequest) return switchRequest(url);
      active = decodeURIComponent(url.split('/')[3]);
      return {ok:true,json:async()=>({active})};
    }};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/conversation-spaces.js'), 'utf8'), context);
  return {api:context.window.VMConversationSpaces,calls,events,get dialog(){return dialog;}};
}

test('new Spaces inherit the chosen browser language rather than backend startup defaults', async () => {
  for (const [language, expected] of [['en','en'],['zh-CN','zh']]) {
    const f = mutableSpaces(language);
    await f.api.ready;
    const choosing = f.api.choose('one');
    await new Promise(resolve => setImmediate(resolve));
    const row = f.dialog.children[3];
    row.children[0].value = 'synthetic-new';
    await row.children[1].onclick();
    const created = f.calls.find(call=>call.method==='POST');
    assert.deepEqual(JSON.parse(created.body), {name:'synthetic-new',language:expected});
    f.dialog.children.at(-1).children[1].onclick();
    assert.equal(await choosing, 'synthetic-new');
  }
});

test('rapid switches are ordered and a rejected switch does not poison subsequent selection', async () => {
  const pending = [];
  const f = mutableSpaces('en',url=>new Promise(resolve=>pending.push({url,resolve})));
  await f.api.ready;
  const first = f.api.use('two'), second = f.api.use('one');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(pending.length,1);
  assert.match(pending[0].url,/two\/use$/);
  const failed = assert.rejects(first,/503/);
  pending[0].resolve({ok:false,status:503,text:async()=>'synthetic failure'});
  await failed;
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(pending.length,2);
  assert.match(pending[1].url,/one\/use$/);
  pending[1].resolve({ok:true,json:async()=>({active:'one'})});
  assert.equal(await second,'one');
  assert.equal(f.api.active,'one');
  assert.deepEqual(f.events,['one']);
});
