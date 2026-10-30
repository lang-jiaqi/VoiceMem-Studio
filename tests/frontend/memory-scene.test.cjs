'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const sceneSource = fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/memory-scene.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

async function portraitScene(style, initialData, live = false) {
  let data = initialData, nextId = 0, fetchCount = 0;
  const timers = new Map(), frames = new Map(), windowEvents = new Map(), documentEvents = new Map();
  const requested = [];
  const elements = new Map(), rounded = [], warnings = [];
  const context = new Proxy({
    measureText: value => ({width: value.length * 8}),
    createRadialGradient: () => ({addColorStop(){}}),
    clearRect: () => { rounded.length = 0; },
    roundRect: (...args) => { rounded.push(args); },
  }, {get: (target, key) => target[key] ?? (() => {})});
  const bounds = {left:0,top:0,width:800,height:600};
  const canvas = {style:{},getContext:()=>context,getBoundingClientRect:()=>bounds};
  const stage = {getBoundingClientRect:()=>bounds,addEventListener(){},append(){}};
  const element = () => {
    const result = {style:{},hidden:false,textContent:'',open:false,children:[],events:{},classList:{contains:()=>true,add(){},remove(){}},
      setAttribute(){},append(...children){this.children.push(...children);},replaceChildren(){this.children=[];},
      addEventListener(name,callback){this.events[name]=callback;},showModal(){this.open=true;},close(){this.open=false;},getContext:()=>context};
    Object.defineProperty(result, 'className', {set: name => { elements.set(name, result); }});
    return result;
  };
  const defaultFetch = async () => {
    const snapshot = structuredClone(data);
    return {ok:true,json:async()=>snapshot};
  };
  let fetcher = defaultFetch;
  const sockets = [];
  class Socket {
    constructor(){sockets.push(this);}
    close(){this.onclose?.();}
    receive(value){this.onmessage?.({data:JSON.stringify(value)});}
  }
  const environment = {
    window:{addEventListener:(name, callback)=>windowEvents.set(name, callback),dispatchEvent(){}},
    document:{hidden:false,
      body:{dataset:{style},classList:{contains: name => name === 'memory-on'},append(){}},
      createElement:element,getElementById:id=>id==='memoryIn'?element():canvas,
      querySelector:()=>stage,addEventListener:(name, callback)=>documentEvents.set(name, callback)},
    Image:class {complete=false;naturalWidth=0;},
    ResizeObserver:class {observe(){}},MutationObserver:class {observe(){}},
    matchMedia:()=>({matches:true}),
    requestAnimationFrame:callback=>{const id=++nextId;frames.set(id,callback);return id;},
    cancelAnimationFrame:id=>frames.delete(id),
    setTimeout:(callback,ms)=>{const id=++nextId;timers.set(id,{callback,ms});return id;},
    clearTimeout:id=>timers.delete(id),
    fetch:(...args)=>{fetchCount++;requested.push(args[0]);return fetcher(...args);},
    innerWidth:800,innerHeight:600,devicePixelRatio:1,
    console:{warn:(...args)=>warnings.push(args)},
    CustomEvent:class {constructor(type,options){this.type=type;this.detail=options?.detail;}},
    ...(live ? {WebSocket:Socket,location:{protocol:'http:',host:'fixture.local'}} : {}),
  };
  vm.runInNewContext(sceneSource, environment);
  await flush();
  const draw = () => {
    const [id, callback] = frames.entries().next().value;
    frames.delete(id);callback(0);
  };
  return {
    scene:environment.window.VMMemoryScene, timers, warnings, sockets, elements, requested,
    setLanguage(value){environment.window.VMSettings={language:value,t:text=>text,graphLabel:text=>text};documentEvents.get('display-settings-change')();},
    setData:value=>{data=value;},setFetch:value=>{fetcher=value || defaultFetch;},
    get fetchCount(){return fetchCount;},
    setHidden:value=>{environment.document.hidden=value;documentEvents.get('visibilitychange')();},
    event:(name,detail)=>windowEvents.get(name)({detail}),
    async tick(ms) {
      const timer = [...timers].find(([, value])=>value.ms===ms);
      assert.ok(timer, `Expected a ${ms}ms timer`);
      timers.delete(timer[0]);await timer[1].callback();await flush();
    },
    portraitTitle() {
      windowEvents.get('resize')();draw();
      const [x,y,width,height] = rounded[0];
      windowEvents.get('pointermove')({clientX:x+width/2,clientY:y+height/2,target:{closest:()=>null}});
      draw();
      return elements.get('memory-node-card__title').textContent;
    },
  };
}

const portrait = (text='Prefers short replies', pending=true) => ({left:[],right:[{
  raw:'Prefers short replies',text,cluster:'personality',label_pending:pending,notes:[{text:'Keep it brief.'}],
}]});

test('both styles refresh prepared portraits regardless of page-open order', async () => {
  for (const order of [['digital','technical'],['technical','digital']]) {
    const first = await portraitScene(order[0], portrait());
    assert.equal(first.portraitTitle(), 'Prefers short replies');
    await first.tick(1000);
    assert.equal(first.timers.size, 1, 'Unchanged pending labels must keep refreshing');

    const ready = portrait('Fixture Owner likes brief answers', false);
    const second = await portraitScene(order[1], ready);
    first.setData(ready);
    await first.tick(1000);
    assert.equal(first.portraitTitle(), second.portraitTitle());
    assert.equal(first.portraitTitle(), 'Fixture Owner likes brief answers');
    assert.equal(first.timers.size, 0);
    assert.equal(second.timers.size, 0);
  }
});

test('language switches refetch localized labels and reject an earlier language response', async () => {
  for(const style of ['digital','technical']){
    const f=await portraitScene(style,portrait('原来的画像',false));
    let resolve;
    f.setFetch(()=>new Promise(done=>{resolve=done;}));
    const old=f.scene.refresh();
    f.setLanguage('en');
    resolve({ok:true,json:async()=>portrait('迟到的中文画像',false)});
    await old;
    assert.equal(f.portraitTitle(),'原来的画像');
    f.setFetch(null);f.setData(portrait('English portrait',false));
    await f.tick(0);
    assert.equal(f.requested.at(-1),'/api/memories?lang=en');
    assert.equal(f.portraitTitle(),'English portrait');
    assert.equal(f.elements.get('memory-node-card__kind').textContent,'Right brain · Portraits');
  }
});

test('an open full-list dialog accepts its new-language response without waiting for the graph', async () => {
  const f=await portraitScene('technical',portrait('旧画像',false));
  f.elements.get('memory-all-button').events.click();await flush();
  f.setData(portrait('English portrait',false));f.setLanguage('en');await flush();
  assert.equal(f.requested.at(-1),'/api/memories?all=true&lang=en');
  const rows=f.elements.get('memory-all-body').children;
  assert.ok(rows.some(row=>row.children?.some(child=>child.textContent==='English portrait')));
});

test('both styles keep observing late writes, edits and deletions after portraits are ready', async () => {
  for (const style of ['technical','digital']) {
    const ready = {...portrait('Fixture Owner likes brief answers', false), space:'fixture',revision:0};
    const ui = await portraitScene(style, ready, true);
    ui.scene.onStudioEvent({type:'answer_done'});
    assert.equal(ui.timers.size, 0, 'Writes must not depend on a bounded post-reply poll');
    ui.setData({...ready,revision:1,left:[{id:'fact',text:'A late synthetic memory',slot:'work'}]});
    ui.sockets[0].receive({type:'memory_changed',space:'fixture',revision:1});
    await ui.tick(0);
    assert.equal(ui.scene.stats().left, 1);
    ui.setData({...ready,revision:2,right:[]});
    ui.sockets[0].receive({type:'memory_changed',space:'fixture',revision:2});
    await ui.tick(0);
    assert.equal(ui.scene.stats().left, 0);
    assert.equal(ui.scene.stats().right, 0);
    assert.equal(ui.timers.size, 0);
  }
});

test('reconnect catches missed writes and accepts a restarted backend revision counter', async () => {
  const ui = await portraitScene('technical', {left:[],right:[],space:'fixture',epoch:'old',revision:12}, true);
  ui.sockets[0].receive({type:'memory_changed',space:'fixture',epoch:'old',revision:12});
  await ui.tick(0);
  ui.sockets[0].close();
  await ui.tick(1000);
  ui.setData({left:[{id:'new',text:'After restart',slot:'work'}],right:[],space:'fixture',epoch:'new',revision:0});
  ui.sockets[1].receive({type:'memory_changed',space:'fixture',epoch:'new',revision:0});
  await ui.tick(0);
  assert.equal(ui.scene.stats().left, 1);
  ui.sockets[0].receive({type:'memory_changed',space:'fixture',epoch:'old',revision:13});
  ui.sockets[1].receive({type:'memory_changed',space:'another-space',epoch:'new',revision:4});
  assert.equal(ui.timers.size, 0);
});

test('a snapshot behind the notified commit retries until the requested revision is visible', async () => {
  const initial = {left:[],right:[],space:'fixture',epoch:'epoch',revision:0};
  const ui = await portraitScene('digital', initial, true);
  ui.sockets[0].receive({type:'memory_changed',space:'fixture',epoch:'epoch',revision:2});
  await ui.tick(0);
  ui.setData({...initial,revision:2,left:[{id:'fact',text:'Final committed state',slot:'work'}]});
  await ui.tick(100);
  assert.equal(ui.scene.stats().left, 1);
  assert.equal(ui.timers.size, 0);
});

test('full-list dialog loads uncapped records and closes on a Space switch', async () => {
  const ui = await portraitScene('technical', {space:'fixture',revision:1,left:[],right:[],total:{left:60,right:0}});
  const full = {space:'fixture',revision:1,left:Array.from({length:60}, (_,i)=>({id:`fact-${i}`,text:`Fact ${i}`})),right:[]};
  const urls = [];
  ui.setFetch(async url=>{urls.push(url);return {ok:true,json:async()=>full};});
  ui.elements.get('memory-all-button').events.click();
  await flush();
  assert.deepEqual(urls, ['/api/memories?all=true&lang=zh']);
  assert.equal(ui.elements.get('memory-all-body').children.filter(row=>row.children.length).length, 60);
  assert.equal(ui.elements.get('memory-all-dialog').open, true);
  ui.setFetch(null);
  ui.setData({space:'another',left:[],right:[]});
  ui.event('memory-space-change', {space:'another'});
  assert.equal(ui.elements.get('memory-all-dialog').open, false);
  await ui.tick(0);
});

test('a late snapshot cannot overwrite prepared labels or restart polling', async () => {
  const ui = await portraitScene('digital', portrait('An earlier label', false));
  let finish;
  ui.setFetch(()=>new Promise(resolve=>{finish=resolve;}));
  const old = ui.scene.refresh();
  ui.setFetch(null);
  ui.setData(portrait('The latest label', false));
  await ui.scene.refresh();
  finish({ok:true,json:async()=>portrait()});
  assert.equal(await old, false);
  assert.equal(ui.portraitTitle(), 'The latest label');
  assert.equal(ui.timers.size, 0);
});

test('switching Spaces discards previous label timers and in-flight snapshots', async () => {
  const ui = await portraitScene('technical', portrait());
  let finish;
  ui.setFetch(()=>new Promise(resolve=>{finish=resolve;}));
  const old = ui.scene.refresh();
  ui.setFetch(null);
  ui.setData({left:[],right:[]});
  ui.event('memory-space-change');
  await ui.tick(0);
  finish({ok:true,json:async()=>portrait('An old Space label', false)});
  await old;
  assert.equal(ui.scene.stats().right, 0);
  assert.equal(ui.timers.size, 0);
});

test('pending labels retry transient fetch errors and stop when hidden or departed', async () => {
  const ui = await portraitScene('digital', portrait());
  ui.setFetch(async()=>{throw new Error('Synthetic network failure');});
  await ui.tick(1000);
  assert.equal(ui.warnings.length, 1);
  assert.equal(ui.timers.size, 1);
  ui.setHidden(true);
  assert.equal(ui.timers.size, 0);
  ui.setFetch(null);
  ui.setData(portrait('A prepared label', false));
  ui.setHidden(false);
  await ui.tick(0);
  assert.equal(ui.portraitTitle(), 'A prepared label');

  ui.setData(portrait());
  await ui.scene.refresh();
  let fail;
  ui.setFetch(()=>new Promise((resolve,reject)=>{fail=reject;}));
  await ui.tick(1000);
  ui.event('pagehide');
  fail(new Error('Synthetic departed request'));
  await flush();
  assert.equal(ui.timers.size, 0);
  ui.setFetch(null);
  ui.setData(portrait('A restored label', false));
  ui.event('pageshow');
  await ui.tick(0);
  assert.equal(ui.portraitTitle(), 'A restored label');
});

test('memory graph nodes follow the current backend snapshot', async () => {
  let data = {left:[{text:'喜欢拿铁',slot:'daily_life'}],right:[]};
  let backgroundDraws = 0;
  const canvas = context => ({style:{},getContext:()=>context,
    getBoundingClientRect:()=>({left:0,top:0,width:800,height:600})});
  const foreground = canvas({setTransform(){},measureText:value=>({width:value.length*8})});
  const background = canvas({setTransform(){},fillRect(){backgroundDraws++;}});
  const stageBounds = {left:0,top:0,width:800,height:600};
  const stage = {getBoundingClientRect:()=>stageBounds,addEventListener(){},append(){}};
  const body = {dataset:{style:'technical'},classList:{contains:()=>false},addEventListener(){},append(){}};
  const element = () => ({className:'',style:{},hidden:false,textContent:'',
    classList:{add(){},remove(){}},setAttribute(){},append(){},addEventListener(){}});
  const environment = {
    window:{addEventListener(){},dispatchEvent(){}},
    document:{body,createElement:element,getElementById:id=>id==='scene-canvas'?foreground:background,
      querySelector:selector=>selector==='#viewSpace .space'?stage:{addEventListener(){}} ,
      addEventListener(){}},
    Image:class {complete=false;naturalWidth=0;},
    ResizeObserver:class {observe(){}}, MutationObserver:class {observe(){}},
    matchMedia:()=>({matches:false}), requestAnimationFrame:()=>1, cancelAnimationFrame(){},
    setTimeout, clearTimeout,
    CustomEvent:class {},
    fetch:async()=>({ok:true,json:async()=>data}),
    innerWidth:800,innerHeight:600,devicePixelRatio:2,
    console,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/memory-scene.js'), 'utf8'), environment);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(environment.window.VMMemoryScene.stats().left, 1);
  assert.equal(environment.window.VMMemoryScene.stats().right, 0);
  assert.equal(foreground.width, 1600);
  assert.equal(background.width, 1600);
  data = {left:[{...data.left[0],hit:true}],right:[]};
  assert.equal(await environment.window.VMMemoryScene.refresh(), false);
  data = {left:[...data.left,{text:'周五开会',slot:'work'}],
    right:[{text:'咖啡偏好',cluster:'preference',notes:[{text:'喜欢拿铁'}]}]};
  assert.equal(await environment.window.VMMemoryScene.refresh(), true);
  assert.equal(environment.window.VMMemoryScene.stats().left, 2);
  assert.equal(environment.window.VMMemoryScene.stats().right, 1);
  assert.equal(await environment.window.VMMemoryScene.refresh(), false);
  assert.equal(backgroundDraws, 1);
  const slots = ['work','health','relationships','finance','knowledge','goals','daily_life'];
  data = {
    left: slots.flatMap(slot => Array.from({length:12}, (_, index) => ({text:`${slot}-${index}`,slot}))),
    right: ['emotion','preference','personality'].flatMap(cluster =>
      Array.from({length:8}, (_, index) => ({text:`${cluster}-${index}`,cluster}))),
  };
  await environment.window.VMMemoryScene.refresh();
  assert.equal(environment.window.VMMemoryScene.stats().left, 84);
  assert.equal(environment.window.VMMemoryScene.stats().right, 24);
  assert.ok(environment.window.VMMemoryScene.stats().minGap >= 8);
  stageBounds.width = 500; stageBounds.height = 420;
  data = {left:data.left.map(item => ({...item,text:`small-${item.text}`})),
    right:data.right.map(item => ({...item,text:`small-${item.text}`}))};
  await environment.window.VMMemoryScene.refresh();
  assert.ok(environment.window.VMMemoryScene.stats().minGap >= 8);
});
