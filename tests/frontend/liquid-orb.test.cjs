'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../studio/apps/ui/liquid-orb.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));
function deferred() {
  let resolve;
  const promise = new Promise(done => {resolve = done;});
  return {promise, resolve};
}
function fixture({available = true, adapterFailures = 0, permanentFailure = false,
                  shaderError = false, pendingDevice, reduced = false} = {}) {
  const frames = new Map(), timers = new Map(), devices = [], events = new Map();
  const classes = new Set();
  let sequence = 0, adapterCalls = 0, deviceCalls = 0, unconfigured = 0, drawError = false;
  const on = (name, callback) => {
    if (!events.has(name)) events.set(name, []);
    events.get(name).push(callback);
  };
  const context = {configure(){}, unconfigure(){unconfigured++;},
    getCurrentTexture(){if(drawError) throw new Error('texture unavailable');return {createView(){return {};}};}};
  const canvas = {hidden:false,dataset:{},clientWidth:280,clientHeight:280,
    parentElement:{classList:{add:name=>classes.add(name),remove:name=>classes.delete(name)}},
    getContext:()=>context};
  const support = {hidden:true}, label = {};
  function device() {
    const loss = deferred(), listeners = {};
    const uniform = {destroyed:false,destroy(){this.destroyed=true;}};
    const result = {lost:loss.promise,limits:{maxTextureDimension2D:4096},destroyed:false,
      submissions:0,uniform,lose:()=>loss.resolve({message:'device lost'}),
      error:()=>listeners.uncapturederror({preventDefault(){},error:new Error('GPU error')}),
      destroy(){this.destroyed=true;loss.resolve({message:'destroyed'});},
      addEventListener:(name,callback)=>{listeners[name]=callback;},
      createShaderModule:()=>({getCompilationInfo:async()=>({messages:shaderError?[{type:'error',message:'invalid shader'}]:[]})}),
      createRenderPipeline:()=>({getBindGroupLayout(){return {};}}),
      createBuffer:()=>uniform,createBindGroup:()=>({}),
      createCommandEncoder:()=>({beginRenderPass:()=>({setPipeline(){},setBindGroup(){},draw(){},end(){}}),finish(){return {};}}),
      queue:{writeBuffer(){},submit(){result.submissions++;}}};
    devices.push(result);
    return result;
  }
  const gpu = {getPreferredCanvasFormat:()=> 'bgra8unorm',requestAdapter:async()=>{
    adapterCalls++;
    if(permanentFailure || adapterCalls<=adapterFailures)return null;
    return {requestDevice:async()=>{
      deviceCalls++;
      if(pendingDevice && deviceCalls===1)return pendingDevice.promise;
      return device();
    }};
  }};
  const document = {hidden:false,getElementById:id=>id==='liquidOrb'?canvas:id==='orbPhase'?label:support,
    querySelectorAll:()=>[],addEventListener:on};
  const window = {addEventListener:on};
  let intersection;
  const environment = {window,document,navigator:available?{gpu}:{},
    console:{warn(){}},performance:{now:()=>100},devicePixelRatio:2,
    matchMedia:()=>({matches:reduced,addEventListener(){}}),
    GPUBufferUsage:{UNIFORM:1,COPY_DST:2},
    requestAnimationFrame:callback=>{const id=++sequence;frames.set(id,callback);return id;},
    cancelAnimationFrame:id=>frames.delete(id),
    setTimeout:(callback,delay)=>{const id=++sequence;timers.set(id,{callback,delay});return id;},
    clearTimeout:id=>timers.delete(id),
    IntersectionObserver:class {constructor(callback){intersection=callback;}observe(){}},
    ResizeObserver:class {observe(){}},
  };
  window.IntersectionObserver=environment.IntersectionObserver;
  window.ResizeObserver=environment.ResizeObserver;
  vm.runInNewContext(source, environment);
  return {canvas,support,classes,document,window,frames,timers,devices,device,
    get adapterCalls(){return adapterCalls;},get unconfigured(){return unconfigured;},
    set drawError(value){drawError=value;},
    event:name=>{for(const callback of events.get(name)||[])callback();},
    intersect:value=>intersection([{isIntersecting:value}]),
    async retry(){const [id,{callback}]=timers.entries().next().value;timers.delete(id);callback();await flush();},
    async frame(now=1000){const current=[...frames.values()];frames.clear();current.forEach(callback=>callback(now));await flush();}};
}

test('transient adapter failure recovers the live voice state and removes the preview', async () => {
  const f=fixture({adapterFailures:1});await flush();
  assert.equal(f.canvas.hidden,true);assert.equal(f.timers.size,1);
  f.window.liquidOrb.speaking();
  await f.retry();
  assert.equal(f.canvas.hidden,false);assert.equal(f.support.hidden,true);
  assert.equal(f.classes.has('orb-fallback'),false);
  assert.equal(f.window.liquidOrb.getState(),'speaking');
  await f.frame();assert.equal(f.devices[0].submissions,1);
  assert.equal(f.frames.size,1);assert.equal(f.timers.size,0);
});

test('device loss releases resources and stale errors cannot stop the replacement', async () => {
  const f=fixture();await flush();const previous=f.devices[0];
  f.window.liquidOrb.longThinking();previous.lose();await flush();
  assert.equal(previous.destroyed,true);assert.equal(previous.uniform.destroyed,true);
  assert.equal(f.frames.size,0);await f.retry();
  previous.error();await flush();
  assert.equal(f.canvas.hidden,false);assert.equal(f.timers.size,0);
  assert.equal(f.window.liquidOrb.getState(),'long-thinking');
  await f.frame();assert.equal(f.devices[1].submissions,1);
});

test('a rendering exception can recover without changing motion settings', async () => {
  const f=fixture();await flush();f.drawError=true;await f.frame();
  assert.equal(f.canvas.hidden,true);assert.equal(f.devices[0].destroyed,true);
  f.drawError=false;await f.retry();await f.frame();
  assert.equal(f.canvas.hidden,false);assert.equal(f.devices[1].submissions,1);
});

test('persistent failure stops after three retries instead of polling forever', async () => {
  const f=fixture({permanentFailure:true});await flush();
  for(let i=0;i<3;i++)await f.retry();
  assert.equal(f.adapterCalls,4);assert.equal(f.timers.size,0);
  f.window.liquidOrb.speaking();f.intersect(true);await flush();
  assert.equal(f.adapterCalls,4);assert.equal(f.timers.size,0);
  assert.equal(f.canvas.hidden,true);
});

test('unsupported WebGPU and invalid shader keep the preview without retrying', async () => {
  for(const options of [{available:false},{shaderError:true}]){
    const f=fixture(options);await flush();
    assert.equal(f.canvas.hidden,true);assert.equal(f.timers.size,0);
    f.event('pageshow');await flush();assert.equal(f.timers.size,0);
    if(options.shaderError)assert.equal(f.devices[0].destroyed,true);
  }
});

test('hidden and offscreen views pause recovery until they become active', async () => {
  const f=fixture({adapterFailures:2});await flush();
  f.document.hidden=true;f.event('visibilitychange');assert.equal(f.timers.size,0);
  f.document.hidden=false;f.event('visibilitychange');assert.equal(f.timers.size,1);
  f.intersect(false);assert.equal(f.timers.size,0);
  f.intersect(true);await f.retry();await f.retry();
  f.event('settings-open');assert.equal(f.frames.size,0);
  f.event('settings-close');assert.equal(f.frames.size,1);
});

test('page departure releases the GPU and restoration rebuilds the same voice state', async () => {
  const f=fixture();await flush();f.window.liquidOrb.listening();
  f.event('pagehide');await flush();
  assert.equal(f.devices[0].destroyed,true);assert.equal(f.frames.size,0);
  assert.equal(f.timers.size,0);assert.equal(f.unconfigured,1);
  f.event('pageshow');await flush();
  assert.equal(f.devices.length,2);assert.equal(f.canvas.hidden,false);
  assert.equal(f.window.liquidOrb.getState(),'listening');
});

test('a late device from an abandoned page is destroyed instead of replacing the new one', async () => {
  const pending=deferred(),f=fixture({pendingDevice:pending});await flush();
  f.event('pagehide');f.event('pageshow');await flush();
  const replacement=f.devices[0],late=f.device();pending.resolve(late);await flush();
  assert.equal(late.destroyed,true);assert.equal(replacement.destroyed,false);
  await f.frame();assert.equal(replacement.submissions,1);assert.equal(f.frames.size,1);
});

test('recovery still respects reduced motion and does not start a continuous loop', async () => {
  const f=fixture({reduced:true,adapterFailures:1});await flush();await f.retry();
  await f.frame();assert.equal(f.devices[0].submissions,1);assert.equal(f.frames.size,0);
});
