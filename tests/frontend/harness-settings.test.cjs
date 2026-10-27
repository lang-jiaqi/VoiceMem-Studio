'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/settings.js'), 'utf8');

function state() {
  const choice = (value, values = [value]) => ({default:value, values});
  return {
    schema:{
      persona:{interaction_style:choice('warm')},
      speaking_style:{speech_rate:choice('normal'), tone:choice('warm'), reply_length:choice('normal')},
      reply_modes:{reasoning_depth:choice('auto')},
      turn_taking:{work_filler:choice('auto', ['silent','auto','reassuring']), backchannel:choice('auto')},
    },
    snapshot:{profile:{speaking_style:{speech_rate:'normal'}}, prompts:{persona:'<em title="fixture">A & B\'s</em>'}},
    prompt_schema:{persona:{max_length:2000}},
    backchannel_curve_schema:{min:0, max:3, step:.1, profiles:{auto:[1,1,1,1]}, phases:[
      {label:'第一阶段'}, {label:'第二阶段'}, {label:'第三阶段'}, {label:'第四阶段'},
    ]},
  };
}

function fixture(style) {
  const listeners = new Map();
  const status = {textContent:''};
  const body = {innerHTML:'', querySelector:() => null, querySelectorAll:() => []};
  const panel = {querySelector:selector => selector === '.harness-status' ? status : body};
  const document = {
    body:{nodeType:1, dataset:{style}, matches:() => false, hasAttribute:() => false, childNodes:[]},
    documentElement:{style:{setProperty(){}}}, getElementById:() => null,
    addEventListener(name, callback) { listeners.set(name, callback); },
    dispatchEvent(event) { listeners.get(event.type)?.(event); },
  };
  const window = {addEventListener(){}};
  const context = {window, document, localStorage:{getItem:() => null},
    CustomEvent:class { constructor(type, options = {}) { this.type = type; this.detail = options.detail; } },
  };
  const instrumented = source.replace(/\}\)\(\);\s*$/, 'window.mountHarnessForTest = panel => (harnessApi = setupHarnessPanel(panel));\nwindow.translateForTest = translate;\n})();');
  assert.notEqual(instrumented, source);
  vm.runInNewContext(instrumented, context);
  const api = window.mountHarnessForTest(panel);
  return {api, status, body, settings:window.VMSettings, translate:window.translateForTest,
    bind:client => window.VMSettings.bindHarness(client),
    receive:detail => document.dispatchEvent({type:'self-harness-state', detail})};
}

test('English conversation controls translate tone labels without changing protocol values', () => {
  const f = fixture('digital');
  f.settings.setLanguage('en');
  for (const [source, translated] of [
    ['温和','Warm'],['共情','Empathetic'],['轻快','Upbeat'],['认真','Serious'],
    ['鼓励','Encouraging'],['俏皮','Playful'],['抱歉','Apologetic'],['平静','Calm'],
  ]) {
    assert.equal(f.settings.t(source), translated);
  }
  const message = state();
  message.schema.speaking_style.tone = {default:'温和', values:['温和','共情'], labels:{'温和':'温和','共情':'共情'}};
  f.receive(message);
  assert.match(f.body.innerHTML, /value="温和" selected/);
  f.settings.setLanguage('zh-CN');
  assert.equal(f.settings.t('Warm'), '温和');
});

test('locale changes translate marked placeholders and preserve actual dialogue text', () => {
  const f = fixture('technical');
  const text = (value, placeholder) => ({nodeType:3,nodeValue:value,parentElement:{
    closest:selector => selector === '[data-ui-placeholder]' ? placeholder : true,
  }});
  const placeholder = text('等待你的下一句话…', true);
  const dialogue = text('等待你的下一句话…', false);
  f.settings.setLanguage('en');
  f.translate(placeholder); f.translate(dialogue);
  assert.equal(placeholder.nodeValue, 'Waiting for you…');
  assert.equal(dialogue.nodeValue, '等待你的下一句话…');
  f.settings.setLanguage('zh-CN');
  f.translate(placeholder); f.translate(dialogue);
  assert.equal(placeholder.nodeValue, '等待你的下一句话…');
  assert.equal(dialogue.nodeValue, '等待你的下一句话…');
});

test('English New chat chrome includes the Space picker and live memory status', () => {
  const f=fixture('digital');
  f.settings.setLanguage('en');
  for (const [source,translated] of [
    ['新对话','New chat'],['选择记忆空间','Choose a Memory Space'],
    ['同一空间的对话共享长期记忆。','Conversations in the same Space share memory.'],
    ['新空间名称','New Space name'],['创建空间','Create Space'],['取消','Cancel'],
    ['创建对话','Create conversation'],['详情','Details'],['领域','Domain'],
    ['0 条记忆','0 memories'],['1 条记忆','1 memory'],['23 条记忆','23 memories'],
    ['本轮召回 1 条相关记忆','1 memory retrieved'],['工作 · 本轮召回','Work · Retrieved'],
    ['本轮暂无新入库记忆','No new memories this turn'],
  ]) assert.equal(f.settings.t(source),translated);
});

for (const style of ['technical','digital']) {
  test(`${style} conversation preferences render all controls after state synchronization`, () => {
    const f = fixture(style);
    f.api.render();
    assert.equal(f.body.innerHTML, '');
    assert.equal(f.status.textContent, '连接后即可调整当前对话。');
    f.receive(state());
    assert.equal(f.status.textContent, '已同步');
    for (const label of ['陪伴方式','说话速度','语气','回答长度','思考深度','等待反馈','附和频率']) {
      assert.ok(f.body.innerHTML.includes(label), label);
    }
    assert.match(f.body.innerHTML, /value="normal" selected/);
    assert.match(f.body.innerHTML, /class="backchannel-curve"/);
    assert.ok(!f.body.innerHTML.includes('<textarea'));
    assert.ok(!f.body.innerHTML.includes('<em title="fixture">'));
  });

  test(`${style} conversation preferences reuse a synchronized state and refresh on activation`, async () => {
    const f = fixture(style);
    let requested = 0;
    f.bind({async getHarness() { requested++; }});
    f.api.render();
    assert.equal(f.status.textContent, '正在连接 Studio…');
    f.receive(state());
    f.api.activate();
    assert.equal(requested, 1);
    assert.equal(f.status.textContent, '已同步');
    const next = state();
    next.snapshot.prompts.persona = 'Updated fixture';
    f.receive(next);
    assert.ok(!f.body.innerHTML.includes('Updated fixture'));
    assert.ok(!f.body.innerHTML.includes('A &amp; B'));
  });
}
