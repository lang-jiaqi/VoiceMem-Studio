'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {Node} = require('../helpers/dom.cjs');

const source = fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/digital.js'), 'utf8');

test('stored panel follows every committed snapshot and ignores portrait-only rewrites', () => {
  const events = new Map(), rows = [];
  const host = {replaceChildren(){rows.length=0;},append(item){rows.push(item.textContent);}};
  const context = {
    window:{addEventListener:(name,callback)=>events.set(name,callback)},
    $:()=>host,tr:text=>text,el:(tag,className,text)=>new Node(tag,text),
  };
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf('let knownMemories'), source.indexOf('const reduced =')), context);
  const initial = {left:[],right:[{id:'trait',raw:'Prefers brief replies',text:'Initial label',notes:[]}]};
  const emit = data=>events.get('memory-snapshot')({detail:data});
  emit(initial);
  vm.runInContext('watchStoredMemories()', context);
  emit({...initial,right:[{...initial.right[0],text:'A prepared label'}]});
  assert.deepEqual(rows, ['正在等待入库…']);
  emit({...initial,left:[{id:'first',text:'First late fact',slot:'work'}]});
  emit({...initial,left:[{id:'second',text:'Second late fact',slot:'work'}]});
  assert.deepEqual(rows, ['事实 · First late fact','事实 · Second late fact']);
  emit({...initial,left:[{id:'second',text:'Updated late fact',slot:'work'}]});
  assert.deepEqual(rows, ['事实 · First late fact','事实 · Updated late fact']);
  events.get('memory-space-change')();
  emit({left:[{id:'other-space',text:'Unrelated fact'}],right:[]});
  assert.deepEqual(rows, ['本轮暂无新入库记忆']);
});
