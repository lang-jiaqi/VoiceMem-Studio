'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('classic transcripts, tags and recalled memories keep untrusted values as text', () => {
  const source = fs.readFileSync(path.join(__dirname, '../../studio/web/index.html'), 'utf8');
  const hooks = source.match(/window\.vm = (\{[\s\S]*?\n  \});/)[1];
  function node(tag = '') {
    return {tag, children:[], textContent:'', className:'',
      set innerHTML(_value) { throw new Error('HTML insertion is forbidden'); },
      append(...items) {this.children.push(...items);},
      replaceChildren(...items) {this.children = items;}};
  }
  const hosts = Object.fromEntries(['liveInput','tags','recallL','recallR'].map(id => [id,node()]));
  let activation;
  const context = {setVoiceState(){}, $:id => hosts[id],
    window:{brainActivate:value => {activation=value;}},
    document:{createElement:node, createTextNode:text => ({textContent:text})}};
  vm.runInNewContext(`window.vm = ${hooks}`, context);
  const unsafe = '<img src=x onerror="synthetic()">';
  context.window.vm.setInput(unsafe);
  assert.equal(hosts.liveInput.children[0].textContent, unsafe);
  assert.equal(hosts.liveInput.children[1].className, 'caret');
  context.window.vm.setTags([{k:'sch',b:unsafe,v:unsafe}]);
  assert.equal(hosts.tags.children[0].children[0].textContent, unsafe);
  assert.equal(hosts.tags.children[0].children[1].textContent, unsafe);
  assert.equal(activation.schemas[0], unsafe);
  for (const side of ['l','r']) {
    context.window.vm.setRecall(side, [{text:unsafe,score:'0.9'}]);
    const row = hosts[side === 'l' ? 'recallL' : 'recallR'].children[0];
    assert.equal(row.children[0].textContent, unsafe);
    assert.equal(row.children[1].textContent, '0.9');
    context.window.vm.setRecall(side, []);
    assert.equal(hosts[side === 'l' ? 'recallL' : 'recallR'].children.length, 0);
  }
});
