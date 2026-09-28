'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('conversations select one of the backend Memory Spaces and retain its identity', async () => {
  const calls = [], events = [];
  let active = 'studio-zh', dialog;
  const spaces = [{id:'studio-zh',name:'studio-zh',count:3}, {id:'project',name:'project',count:2}];
  const element = () => ({children:[], append(...items){this.children.push(...items);}, setAttribute(){},
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
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../ui/conversation-spaces.js'), 'utf8'), context);
  const api = context.window.VMConversationSpaces;
  assert.equal(await api.ready, 'studio-zh');
  const choice = api.choose('studio-zh');
  await new Promise(resolve => setImmediate(resolve));
  const list = dialog.children[2];
  list.children[1].onclick();
  assert.equal(list.children.length, spaces.length);
  const footer = dialog.children.at(-1);
  assert.equal(footer.className, 'space-picker__footer');
  assert.equal(footer.children[1].textContent, '确认并创建对话');
  footer.children[1].onclick();
  assert.equal(await choice, 'project');
  assert.equal(await api.use('project'), 'project');
  assert.equal(api.name('project'), 'project');
  assert.deepEqual(events, ['studio-zh', 'project']);
  assert.deepEqual(calls.at(-1), ['/api/spaces/project/use', 'POST']);
});
