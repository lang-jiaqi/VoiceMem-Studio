const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
let Capture;
const messages = [];
const context = {
  sampleRate:24000, Float32Array, Math,
  AudioWorkletProcessor:class {constructor(){this.port={postMessage:m=>messages.push(m)};}},
  registerProcessor:(_, cls)=>{Capture=cls;},
};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../studio/web/mic-capture-worklet.js'),'utf8'),context);
let seed = 817;
const noise = () => {seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed/2**32-.5)*.4;};
const rms = a => Math.sqrt(a.reduce((s,x)=>s+x*x,0)/a.length);
const ref = Float32Array.from({length:24000},noise);
const user = Float32Array.from({length:24000},noise);
for(const scenario of ['echo','user','double-talk','silence']) {
  const p = new Capture();
  let before=0,after=0;
  for(let at=0;at<ref.length;at+=480){
    const r=ref.slice(at,at+480);
    const m=Float32Array.from(r,(_,i)=>{
      const echo=(ref[at+i-2400]||0)*.3;
      if(scenario==='echo')return echo;
      if(scenario==='double-talk')return echo+user[at+i]*.7;
      if(scenario==='user')return user[at+i];
      return 0;
    });
    before+=rms(m); after+=rms(p.clean(m,r));
  }
  if(scenario==='echo') assert(after/before<.05,'delayed echo must be attenuated');
  else if(scenario!=='silence') assert(after/before>.99,'near-end speech must survive');
  else assert.equal(after,0);
  console.log(scenario,'PASS',before?`ratio=${(after/before).toFixed(3)}`:'');
}
const p=new Capture();
for(let i=0;i<15;i++){
  const out=new Float32Array(128).fill(1);
  p.process([[new Float32Array(128).fill(.1)],[new Float32Array(128)]],[[out]]);
  assert(out.every(x=>x===0),'microphone must never play through speakers');
}
const sent=messages.filter(m=>m.type==='mic');
assert.equal(sent.length,4);
assert(sent.every(m=>m.samples.length===480),'capture batches must be 20ms');
console.log('20ms capture and silent output PASS');
const html=fs.readFileSync(path.join(__dirname,'../studio/web/voicemem.html'),'utf8');
for(const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
console.log('demo JavaScript syntax PASS');
