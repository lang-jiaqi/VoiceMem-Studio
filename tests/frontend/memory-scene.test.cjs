'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('memory graph nodes follow the current backend snapshot', async () => {
  let data = {left:[{text:'喜欢拿铁',slot:'daily_life'}],right:[]};
  let backgroundDraws = 0;
  const canvas = context => ({style:{},getContext:()=>context,
    getBoundingClientRect:()=>({left:0,top:0,width:800,height:600})});
  const foreground = canvas({setTransform(){},measureText:value=>({width:value.length*8})});
  const background = canvas({setTransform(){},fillRect(){backgroundDraws++;}});
  const stageBounds = {left:0,top:0,width:800,height:600};
  const stage = {getBoundingClientRect:()=>stageBounds,addEventListener(){}};
  const body = {dataset:{style:'technical'},classList:{contains:()=>false},addEventListener(){},append(){}};
  const element = () => ({className:'',style:{},hidden:false,textContent:'',
    classList:{add(){},remove(){}},setAttribute(){},append(){}});
  const environment = {
    window:{addEventListener(){},dispatchEvent(){}},
    document:{body,createElement:element,getElementById:id=>id==='scene-canvas'?foreground:background,
      querySelector:selector=>selector==='#viewSpace .space'?stage:{addEventListener(){}} ,
      addEventListener(){}},
    Image:class {complete=false;naturalWidth=0;},
    ResizeObserver:class {observe(){}}, MutationObserver:class {observe(){}},
    matchMedia:()=>({matches:false}), requestAnimationFrame:()=>1, cancelAnimationFrame(){},
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
