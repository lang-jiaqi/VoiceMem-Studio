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
                  shaderError = false, pendingDevice, pendingGPU, reduced = false} = {}) {
  const frames = new Map(), timers = new Map(), devices = [], events = new Map();
  const classes = new Set();
  let sequence = 0, adapterCalls = 0, deviceCalls = 0, unconfigured = 0, drawError = false, clock = 100;
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
      submissions:0,uniform,uniforms:[],lose:()=>loss.resolve({message:'device lost'}),
      error:()=>listeners.uncapturederror({preventDefault(){},error:new Error('GPU error')}),
      destroy(){this.destroyed=true;loss.resolve({message:'destroyed'});},
      addEventListener:(name,callback)=>{listeners[name]=callback;},
      createShaderModule:()=>({getCompilationInfo:async()=>({messages:shaderError?[{type:'error',message:'invalid shader'}]:[]})}),
      createRenderPipeline:()=>({getBindGroupLayout(){return {};}}),
      createBuffer:()=>uniform,createBindGroup:()=>({}),
      createCommandEncoder:()=>({beginRenderPass:()=>({setPipeline(){},setBindGroup(){},draw(){},end(){}}),finish(){return {};}}),
      queue:{writeBuffer(_buffer, _offset, values){result.uniforms.push(Array.from(values));},submit(){result.submissions++;},
        onSubmittedWorkDone:()=>pendingGPU?.promise || Promise.resolve()}};
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
    console:{warn(){}},performance:{now:()=>clock},devicePixelRatio:2,
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
  return {canvas,support,label,classes,document,window,frames,timers,devices,device,
    get adapterCalls(){return adapterCalls;},get unconfigured(){return unconfigured;},
    set drawError(value){drawError=value;},
    event:name=>{for(const callback of events.get(name)||[])callback();},
    intersect:value=>intersection([{isIntersecting:value}]),
    async retry(){const [id,{callback}]=timers.entries().next().value;timers.delete(id);callback();await flush();},
    async frame(now=1000){clock=now;const current=[...frames.values()];frames.clear();current.forEach(callback=>callback(now));await flush();},
    async advance(ms){const end=clock+ms;while(clock<end)await this.frame(Math.min(end,clock+16));}};
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

test('listening keeps the interior flowing even when microphone levels are silent or stale', async () => {
  const f=fixture();await flush();await f.frame(100);
  const uniforms=f.devices[0].uniforms,idle=uniforms.at(-1);
  f.window.liquidOrb.listening();f.window.liquidOrb.setAudioLevel(0);await f.advance(1000);
  const listening=uniforms.at(-1);await f.advance(1000);const stale=uniforms.at(-1);
  assert.ok(listening[3]>idle[3]*4);
  assert.ok(listening[6]>idle[6]*2);
  assert.ok(stale[2]-listening[2]>3);
  assert.equal(stale[10],0);
});

test('speaking moves above and below center while idle settles without a rhythm', async () => {
  const f=fixture();await flush();await f.frame(100);f.window.liquidOrb.speaking();await f.advance(2000);
  const positions=f.devices[0].uniforms.map(values=>values[10]);
  assert.ok(Math.min(...positions)<-.05);assert.ok(Math.max(...positions)>.05);
  f.window.liquidOrb.setState('idle');await f.advance(1500);
  assert.ok(Math.abs(f.devices[0].uniforms.at(-1)[10])<.001);
});

test('thinking lowers once, stays lowered across both phases, then springs back', async () => {
  const f=fixture();await flush();await f.frame(100);f.window.liquidOrb.shortThinking();
  await f.advance(250);assert.ok(f.devices[0].uniforms.at(-1)[10]<-.16);
  await f.advance(800);const held=f.devices[0].uniforms.at(-1)[10];
  assert.ok(Math.abs(held+.2)<.001);
  const switched=f.devices[0].uniforms.length;
  f.window.liquidOrb.longThinking();await f.advance(1000);
  const recent=f.devices[0].uniforms.slice(switched).map(values=>values[10]);
  assert.ok(Math.max(...recent)-Math.min(...recent)<.001);
  f.window.liquidOrb.setState('idle');const begin=f.devices[0].uniforms.length;await f.advance(1200);
  const returning=f.devices[0].uniforms.slice(begin).map(values=>values[10]);
  assert.ok(Math.max(...returning)>.015);
  assert.ok(Math.abs(returning.at(-1))<.001);
  assert.ok(f.devices[0].uniforms.every(values=>values[4]+Math.abs(values[10])<1));
});

test('reduced motion keeps all live states centered and does not spin continuously', async () => {
  const f=fixture({reduced:true});await flush();await f.frame(100);
  for(const state of ['listening','speaking','short-thinking','long-thinking','idle']){
    f.window.liquidOrb.setState(state);await f.advance(1000);
    assert.equal(f.devices[0].uniforms.at(-1)[10],0);
    assert.equal(f.devices[0].uniforms.at(-1)[2],0);
    assert.equal(f.frames.size,0);
  }
});

test('rendering has a bounded frame and pixel budget on a high refresh Retina display', async () => {
  const f=fixture();await flush();f.window.liquidOrb.listening();
  for(let time=100;time<=1100;time+=1000/120)await f.frame(time);
  assert.ok(f.devices[0].submissions<=31);
  assert.equal(f.canvas.width,350);assert.equal(f.canvas.height,350);
  f.window.liquidOrb.setState('idle');await f.advance(2000);
  const before=f.devices[0].submissions;await f.advance(1000);
  assert.ok(f.devices[0].submissions-before<=13);
  f.canvas.clientWidth=f.canvas.clientHeight=1200;await f.advance(100);
  assert.equal(f.canvas.width,384);assert.equal(f.canvas.height,384);
});

test('a busy GPU never accumulates orb submissions and resumes the latest state', async () => {
  const pending=deferred(),f=fixture({pendingGPU:pending});await flush();await f.frame(100);
  f.window.liquidOrb.listening();await f.advance(500);
  assert.equal(f.devices[0].submissions,1);assert.equal(f.frames.size,0);
  pending.resolve();await flush();await f.frame(700);
  assert.equal(f.devices[0].submissions,2);
  assert.equal(f.window.liquidOrb.getState(),'listening');
});

test('live phase labels follow the interface language', async () => {
  const f=fixture();await flush();f.window.VMSettings={language:'en'};
  f.window.liquidOrb.listening();assert.equal(f.label.textContent,'Listening');
  f.window.VMSettings.language='zh-CN';f.event('display-settings-change');
  assert.equal(f.label.textContent,'倾听');
});

test('completion of abandoned GPU work cannot restart a departed page', async () => {
  const pending=deferred(),f=fixture({pendingGPU:pending});await flush();await f.frame(100);
  f.event('pagehide');pending.resolve();await flush();
  assert.equal(f.frames.size,0);assert.equal(f.devices[0].destroyed,true);
  f.event('pageshow');await flush();await f.frame(200);
  assert.equal(f.devices[1].submissions,1);
});
