'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function fixture() {
  const storage = new Map(), listeners = {}, dialogs = [], calls = [], languages = [];
  const select = {}, status = {}, save = {}, reset = {};
  const decode = value => value.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
  const editors = {
    sections:[],
    set innerHTML(html) {
      this.html = html;
      this.sections = [...html.matchAll(/data-key="([^"]+)"[^]*?<textarea[^>]*>([^]*?)<\/textarea>/g)].map(([, key, value]) => {
        const input = {value:decode(value),style:{},scrollTop:0}, numbers = {}, start = {}, end = {};
        const measure = {rows:[],set innerHTML(html) {
          this.rows = [...html.matchAll(/<span>([^]*?)<\/span>/g)].map(() => ({offsetHeight:20}));
        }};
        const elements = {'textarea':input,'.prompt-line-numbers':numbers,'.prompt-start-number':start,
          '.prompt-end-number':end,'.prompt-measure':measure};
        return {dataset:{key},input,measure,numbers,start,end,
          querySelector:selector => elements[selector],querySelectorAll:() => measure.rows};
      });
    },
    querySelectorAll(selector) {return selector === 'textarea' ? this.sections.map(section => section.input) : this.sections;},
  };
  const elements = {'select':select,'.prompt-status':status,'.prompt-editors':editors,'.prompt-save':save,'.prompt-reset':reset};
  const root = {querySelector:selector => elements[selector]};
  const document = {body:{append(){}},createElement() {
    const handlers = {};
    const dialog = {returnValue:'',addEventListener:(name, fn) => {handlers[name] = fn;},
      showModal(){},remove(){},querySelector:() => ({focus(){}}),
      async choose(action) {this.returnValue = action; await handlers.close();}};
    dialogs.push(dialog); return dialog;
  }};
  const window = {addEventListener:(name, fn) => {listeners[name] = fn;},
    ResizeObserver:class {constructor(callback) {listeners.resize = callback;} observe(){}}};
  const client = {async setHarnessPrompts(value, language) {calls.push(value); languages.push(language); if (client.error) throw new Error(client.error);}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/harness-prompts.js'), 'utf8'), {
    window, document, localStorage:{getItem:key => storage.get(key),setItem:(key, value) => storage.set(key, value)},
  });
  const api = window.VMHarnessPrompts.mount(root, {client:() => client,refresh:async () => {}});
  const receive = (overrides = {}, language = 'zh') => api.receive({prompt_catalog:{
    main_system:{group:'main',variable:'SYSTEM_PROMPT',default:language === 'en' ? 'Synthetic English default' : 'Synthetic default',max_length:20000},
    speaking_style:{group:'main',variable:'SPEAKING_PROMPT',default:language === 'en' ? 'Synthetic English style' : 'Synthetic style',max_length:20000},
  },snapshot:{prompt_overrides:overrides,prompt_language:language}});
  receive();
  const edit = text => {const input = editors.sections[0].input; input.value = text; input.oninput();};
  return {api,window,storage,client,calls,languages,status,edit,editors,dialogs,listeners,receive,save,reset};
}

test('drafts survive unrelated state updates and only accepted saves persist', async () => {
  const f = fixture(); f.edit('Synthetic edited prompt');
  f.receive(); assert.equal(f.editors.sections[0].input.value, 'Synthetic edited prompt');
  assert.equal(f.api.dirty(), true);
  f.client.error = 'Rejected'; f.save.onclick(); await new Promise(setImmediate);
  assert.equal(f.storage.size, 0); assert.equal(f.status.textContent, 'Rejected');
  f.client.error = ''; f.save.onclick(); await new Promise(setImmediate);
  assert.equal(f.api.dirty(), false);
  assert.equal(JSON.stringify(f.window.VMHarnessPrompts.saved()), JSON.stringify({main_system:'Synthetic edited prompt'}));
  f.receive({main_system:'Synthetic edited prompt'});
  assert.equal(f.editors.sections[0].input.value, 'Synthetic edited prompt');
});
test('closing supports keep editing, discard, and save without losing rejected drafts', async () => {
  const f = fixture(); f.edit('Synthetic draft');
  const cancelled = f.api.confirmLeave(); await f.dialogs.at(-1).choose('cancel');
  assert.equal(await cancelled, false); assert.equal(f.api.dirty(), true);
  f.client.error = 'Rejected';
  const rejected = f.api.confirmLeave(); await f.dialogs.at(-1).choose('save');
  assert.equal(await rejected, false); assert.equal(f.api.dirty(), true);
  const discarded = f.api.confirmLeave(); await f.dialogs.at(-1).choose('discard');
  assert.equal(await discarded, true); assert.equal(f.editors.sections[0].input.value, 'Synthetic default');
  f.client.error = ''; f.edit('Synthetic accepted');
  const accepted = f.api.confirmLeave(); await f.dialogs.at(-1).choose('save');
  assert.equal(await accepted, true); assert.equal(f.api.dirty(), false);
});
test('source strings are escaped and refresh warns only for unsaved edits', () => {
  const f = fixture();
  let prevented = false; f.listeners.beforeunload({preventDefault(){prevented = true;}});
  assert.equal(prevented, false);
  f.receive({main_system:'</textarea><script>Synthetic</script>'});
  assert.ok(!f.editors.html.includes('<script>'));
  assert.equal(f.editors.sections[0].input.value, '</textarea><script>Synthetic</script>');
  f.edit('Unsaved'); f.listeners.beforeunload({preventDefault(){prevented = true;}});
  assert.equal(prevented, true);
});
test('language changes replace untouched defaults while preserving individual dirty drafts', async () => {
  const f = fixture(); f.edit('自定义角色'); f.receive({}, 'en');
  assert.equal(f.editors.sections[0].input.value, '自定义角色');
  assert.equal(f.editors.sections[1].input.value, 'Synthetic English style');
  f.save.onclick(); await new Promise(setImmediate);
  assert.equal(f.languages.at(-1), 'en');
  assert.equal(JSON.stringify(f.calls.at(-1)), JSON.stringify({main_system:'自定义角色'}));
  f.receive({main_system:'自定义角色'}, 'zh');
  assert.equal(f.editors.sections[0].input.value, '自定义角色');
  assert.equal(f.editors.sections[1].input.value, 'Synthetic style');
  assert.equal(f.api.dirty(), false);
  f.receive({main_system:'自定义角色'}, 'en'); f.reset.onclick();
  assert.equal(f.editors.sections[0].input.value, 'Synthetic English default');
  f.save.onclick(); await new Promise(setImmediate);
  assert.equal(JSON.stringify(f.calls.at(-1)), '{}');
});
test('rejected stale-language saves retain drafts and do not persist old defaults', async () => {
  const f = fixture(); f.edit('Synthetic custom identity');
  let reject;
  f.client.setHarnessPrompts = () => new Promise((_, fail) => {reject = fail;});
  f.save.onclick();
  f.receive({}, 'en'); reject(new Error('Prompt language changed'));
  await new Promise(setImmediate);
  assert.equal(f.storage.size, 0);
  assert.equal(f.api.dirty(), true);
  assert.equal(f.editors.sections[0].input.value, 'Synthetic custom identity');
  assert.equal(f.editors.sections[1].input.value, 'Synthetic English style');
});
test('wrapped display keeps source newlines and continuous logical line numbers', async () => {
  const f = fixture(), text = 'Synthetic long line\n\n    {history}\n';
  f.edit(text);
  const [first, second] = f.editors.sections;
  first.measure.rows[0].offsetHeight = 80;
  f.listeners.resize();
  assert.equal(first.start.textContent, 1);
  assert.match(first.numbers.innerHTML, /height:80px">2<\/span>/);
  assert.equal(first.end.textContent, 6);
  assert.equal(second.start.textContent, 7);
  f.save.onclick(); await new Promise(setImmediate);
  assert.equal(f.calls.at(-1).main_system, text);
});
