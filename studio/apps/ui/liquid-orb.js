/* Liquid Orb: the supplied Opal preset only; no unused shader/editor banks. */
(()=>{
'use strict';
const canvas=document.getElementById('liquidOrb');
if(!canvas)return;
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const buttons=[...document.querySelectorAll('[data-orb-state]')];
const label=document.getElementById('orbPhase');
const names={idle:'待机',listening:'倾听',speaking:'说话','short-thinking':'短思考','long-thinking':'长思考'};
const aliases={responding:'speaking',thinking:'short-thinking',shortThinking:'short-thinking',longThinking:'long-thinking'};
// Compact layout: size/time/speed, radius/zoom/warp/shade, exposure/glow/padding.
const idle=[1,1,0,.45,.78,.282,1.456,.1,.7616,0,0,0];
const thinking=[1,1,0,1.5,.78,.64,4.75,.35,1.08,.26,0,0];
const seeds={idle,listening:[...idle],speaking:[...thinking],'short-thinking':thinking,'long-thinking':[...thinking]};
seeds.listening[8]=1.08;seeds.speaking[3]=15;
// Both thinking interfaces keep the supplied thinking material and motion.
let state='idle',from=new Float32Array(idle),target=new Float32Array(idle),values=new Float32Array(idle);
let transitionAt=0,duration=0,phase=0,previous=0,scale=1,audio=0,audioAt=-Infinity,listeningAt=0;
let device,context,pipeline,buffer,group,format,raf=0,visible=true,failed=false;
let generation=0,initializing=false,pageHidden=false,retryTimer=0,retries=0,retryable=true;
const maxRetries=3;
let demo=false,demoRAF=0,demoStarted=0;
let floatPhase=0,floatAmplitude=0,floatSpeed=2.8,settingsOpen=false;
function sample(now){
 const raw=duration?Math.min(1,Math.max(0,(now-transitionAt)/duration)):1;
 const k=state.includes('thinking')?1-(1-raw)**3:raw*raw*(3-2*raw);
 for(let i=3;i<values.length;i++)values[i]=from[i]+(target[i]-from[i])*k;
 return values;
}
function setState(next){
 next=aliases[next]||next;
 if(!(next in seeds))throw new TypeError('Unknown orb state: '+next);
 if(next!==state){const now=performance.now();sample(now);from=new Float32Array(values);target=new Float32Array(seeds[next]);transitionAt=now;duration=next.includes('thinking')?220:650;state=next;if(next==='listening'){listeningAt=now;audioAt=-Infinity;}}
 canvas.dataset.state=state;
 buttons.forEach(b=>b.setAttribute('aria-pressed',String(demo&&b.dataset.orbState===state)));
 label.textContent=(demo?'演示 · ':'')+names[state];
 sync();
}
function setAudioLevel(level){if(!Number.isFinite(level))throw new TypeError('Audio level must be finite');audio=Math.min(1,Math.max(0,level));audioAt=performance.now();}
function stopDemo(){demo=false;cancelAnimationFrame(demoRAF);demoRAF=0;}
function demoState(next){
 stopDemo();demo=true;demoStarted=performance.now();setState(next);
 if(next==='listening'){
  const feed=now=>{if(!demo||state!=='listening')return;const t=(now-demoStarted)/1000;setAudioLevel((.1+.7*Math.abs(Math.sin(t*9.7)))*(.5+.5*Math.sin(t*1.55)**2));demoRAF=requestAnimationFrame(feed);};
  if(!reduced.matches)demoRAF=requestAnimationFrame(feed);
 }
}
buttons.forEach(b=>b.addEventListener('click',()=>{if(demo&&state===b.dataset.orbState){stopDemo();setState('idle');}else demoState(b.dataset.orbState);}));
function liveState(next){stopDemo();setState(next);}
window.liquidOrb=Object.freeze({getState:()=>state,setState:liveState,setAudioLevel,listening:()=>liveState('listening'),speaking:()=>liveState('speaking'),shortThinking:()=>liveState('short-thinking'),longThinking:()=>liveState('long-thinking'),stopVoiceDemo:()=>{stopDemo();setState('idle');}});
const shader=`
struct U {size:vec2<f32>,time:f32,speed:f32,radius:f32,zoom:f32,warp:f32,shade:f32,exposure:f32,glow:f32,pad:vec2<f32>};
@group(0) @binding(0) var<uniform> u:U;
const A=vec3<f32>(.968627451,.960784314,.952941176);
const B=vec3<f32>(.97254902,.188235294,.776470588);
const C=vec3<f32>(.568627451,.956862745,1.);
const D=vec3<f32>(.156862745,.494117647,.964705882);
fn ramp(v:f32,a:vec3<f32>,b:vec3<f32>,c:vec3<f32>,d:vec3<f32>)->vec3<f32>{var col=mix(a,b,smoothstep(0.,.45,v));col=mix(col,c,smoothstep(.38,.72,v));return mix(col,d,smoothstep(.68,1.,v));}
fn fluid(p:vec2<f32>,t:f32)->vec3<f32>{
 let q=p*(.8+u.zoom*.64);let complexity=.76+u.warp*.085;
 var d=-t*.42;var a=0.;
 for(var i=0;i<8;i++){let fi=f32(i);a=a+cos(fi-d-a*q.x*complexity);d=d+sin(q.y*fi*complexity+a);}
 d=d+t*.42;
 let c1=cos(q*vec2<f32>(d,a))*.6+vec2<f32>(.4);let c2=cos(a+d)*.5+.5;
 let interference=.5+.5*cos(vec3<f32>(c1.x,c1.y,c2)*cos(vec3<f32>(d,a,2.5))*.5+vec3<f32>(.5));
 let tone=fract(interference.r*.37+interference.g*.51+interference.b*.73+c1.x*.22-c1.y*.15);
 var color=ramp(tone,B,C,D,A);color=mix(color,A,.16+.1*interference.b);color=color/(vec3<f32>(1.)+color*.16);
 color=mix(color,vec3<f32>(1.),u.shade*.22*smoothstep(.15,1.15,dot(p,vec2<f32>(-.32,.78))));
 color=color*(1.-u.shade*.34*smoothstep(-.1,1.2,dot(p,vec2<f32>(.45,-.62))));
 color=color*(1.-u.shade*.22*smoothstep(.72,1.08,length(p)));
 return clamp(color,vec3<f32>(0.),vec3<f32>(1.));
}
fn over(dst:vec3<f32>,src:vec3<f32>,a:f32)->vec3<f32>{return mix(dst,src,clamp(a,0.,1.));}
fn lobe(n:vec2<f32>,dir:vec2<f32>,cut:f32,power:f32)->f32{return pow(clamp((dot(n,dir)-cut)/(1.-cut),0.,1.),power);}
fn halo(r:f32,rad:f32)->vec3<f32>{return vec3<f32>(.564705882,.549019608,1.)*(u.glow*exp(-max(r-rad,0.)*11.)*smoothstep(rad-.01,rad+.01,r));}
struct V {@builtin(position) pos:vec4<f32>,@location(0) uv:vec2<f32>};
@vertex fn vs(@builtin(vertex_index) i:u32)->V{
 var p=array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
 var v:V;v.pos=vec4<f32>(p[i],0.,1.);v.uv=(p[i]+vec2<f32>(1.))*.5;return v;
}
@fragment fn fs(v:V)->@location(0) vec4<f32>{
 let fc=v.uv*u.size;let minSize=max(min(u.size.x,u.size.y),1.);let uv=(2.*fc-u.size)/minSize-vec2<f32>(0.,u.pad.x);
 let rad=max(u.radius,.05);let r=length(uv);let t=u.time;
 var col=vec3<f32>(0.);var alpha=0.;
 if(r>rad*1.015){col=clamp(halo(r,rad),vec3<f32>(0.),vec3<f32>(1.));alpha=max(col.r,max(col.g,col.b));}
 else{
 let p=uv/rad;let pd=length(p);let normal=uv/max(r,.0001);
 let clearFa=1.-smoothstep(.995,1.04,pd);let depth=max(1.-pd,0.);
 let refT=clamp(depth/(.015+.95*.15),0.,1.);
 let profile=pow(1.-sqrt(max(1.-(1.-refT)*(1.-refT),0.)),.68);
 let refracted=p-normal*(1.6*.51*profile);let split=.14*.32*.51*profile;
 let red=fluid(refracted-normal*split,t);let green=fluid(refracted,t);let blue=fluid(refracted+normal*split,t);
 let fcol=vec3<f32>(red.r,green.g,blue.b);let lum=dot(fcol,vec3<f32>(.213,.715,.072));
 col=clamp(vec3<f32>(lum)+(fcol-vec3<f32>(lum))*1.22,vec3<f32>(0.),vec3<f32>(1.))*.99*clearFa;
 let rim=pow((1.-smoothstep(0.,.026+.055*.17,depth))*clearFa,1.8);
 col=over(col,vec3<f32>(1.),rim*.51*.45);
 let dispersion=rim*.32*(.8+.8*.17);
 col=over(col,vec3<f32>(.803921569,.898039216,1.),dispersion*lobe(normal,normalize(vec2<f32>(.84,.54)),-.32,1.8));
 col=over(col,vec3<f32>(.850980392,.784313725,1.),dispersion*lobe(normal,normalize(vec2<f32>(-.62,-.78)),-.28,2.));
 col=col*(1.-rim*(.015+.15*.17)*(.15+.85*max(dot(normal,vec2<f32>(.45,-.89)),0.)));
 col=over(col,vec3<f32>(.917647059,.956862745,1.),rim*lobe(normal,normalize(vec2<f32>(-.68,.73)),.2,2.8)*.48*1.4);
 col=over(col,vec3<f32>(.862745098,.917647059,1.),rim*lobe(normal,normalize(vec2<f32>(.74,-.67)),.4,3.6)*.48);
 let ballA=1.-smoothstep(.985,1.015,pd);col=clamp(col*u.exposure,vec3<f32>(0.),vec3<f32>(1.))*ballA;
 col=clamp(col+halo(r,rad),vec3<f32>(0.),vec3<f32>(1.));alpha=clamp(max(ballA,max(col.r,max(col.g,col.b))),0.,1.);
 }
 let q=(2.*fc-u.size)/u.size;let fit=1.-smoothstep(min((rad+1.)*.5,1.-2./minSize),1.,max(abs(q.x),abs(q.y)));
 return vec4<f32>(col*fit,alpha*fit);
}`;
function active(){return !pageHidden&&!document.hidden&&visible&&!settingsOpen;}
function cancelRetry(){clearTimeout(retryTimer);retryTimer=0;}
function release(){
 // Invalidate callbacks before destroying the device: destroy also resolves device.lost.
 generation++;initializing=false;cancelAnimationFrame(raf);raf=0;previous=0;
 context?.unconfigure();buffer?.destroy();device?.destroy();
 device=context=pipeline=buffer=group=undefined;
}
function scheduleRecovery(){
 if(!active()||!retryable||retryTimer||initializing||retries>=maxRetries)return;
 retryTimer=setTimeout(()=>{retryTimer=0;if(active()){retries++;void init();}},250*2**retries);
}
function fallback(error){
 release();failed=true;canvas.hidden=true;canvas.parentElement.classList.add('orb-fallback');
 document.getElementById('orbSupport').hidden=false;
 console.warn('Liquid Orb fallback:',error.message);scheduleRecovery();
}
function draw(now){
 raf=0;if(!device||failed||initializing||!active())return;
 try{
 const dpr=Math.min(devicePixelRatio||1,2);const limit=device.limits.maxTextureDimension2D;
 const w=Math.min(limit,Math.max(1,Math.round(canvas.clientWidth*dpr))),h=Math.min(limit,Math.max(1,Math.round(canvas.clientHeight*dpr)));
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 const dt=previous?Math.min(.1,(now-previous)/1000):0;previous=now;sample(now);
 let desired=1;
 if(state==='listening'&&!reduced.matches){const elapsed=(now-listeningAt)/1000;const level=audioAt>=listeningAt?(now-audioAt<180?audio:0):.28+.22*Math.sin(elapsed*7.2)*Math.sin(elapsed*4.1);desired+=.12*Math.pow(Math.max(0,(level-.025)/.975),.65);}
 scale+=(desired-scale)*(1-Math.exp(-dt/(desired>scale?.065:.18)));
 if(!reduced.matches)phase+=dt*Math.max(values[3],0);
 const thinking=state.includes('thinking');
 const amp=!reduced.matches&&thinking?(state==='long-thinking'?.085:.035):0;
 const ease=1-Math.exp(-dt/.35);
 floatAmplitude+=(amp-floatAmplitude)*ease;
 floatSpeed+=((state==='long-thinking'?1.5:2.8)-floatSpeed)*ease;
 if(!reduced.matches)floatPhase+=dt*floatSpeed;
 const frameValues=new Float32Array(values);frameValues[0]=w;frameValues[1]=h;frameValues[2]=phase;frameValues[4]*=scale;frameValues[10]=reduced.matches?0:Math.sin(floatPhase)*floatAmplitude;
 device.queue.writeBuffer(buffer,0,frameValues);
 const encoder=device.createCommandEncoder();const pass=encoder.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:'clear',storeOp:'store'}]});
 pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();device.queue.submit([encoder.finish()]);
 if(!reduced.matches||now-transitionAt<duration)raf=requestAnimationFrame(draw);
 }catch(error){fallback(error);}
}
function sync(){
 if(!active()){cancelAnimationFrame(raf);raf=0;previous=0;cancelRetry();return;}
 if(initializing)return;
 if(failed){scheduleRecovery();return;}
 if(!device){void init();return;}
 if(!raf)raf=requestAnimationFrame(draw);
}
async function init(){
 if(initializing||!active())return;
 const current=++generation;initializing=true;
 try{
 if(!navigator.gpu){retryable=false;throw new Error('WebGPU unavailable');}
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('WebGPU adapter unavailable');
 if(current!==generation)return;
 const acquired=await adapter.requestDevice();
 if(current!==generation){acquired.destroy();return;}
 device=acquired;
 device.lost.then(info=>{if(current===generation)fallback(new Error(info.message||'Device lost'));});
 device.addEventListener('uncapturederror',e=>{e.preventDefault();if(current===generation)fallback(e.error);});
 context=canvas.getContext('webgpu');if(!context)throw new Error('WebGPU context unavailable');
 format=navigator.gpu.getPreferredCanvasFormat();context.configure({device,format,alphaMode:'premultiplied'});
 const module=device.createShaderModule({code:shader});const info=await module.getCompilationInfo();
 if(current!==generation)return;
 const errors=info.messages.filter(m=>m.type==='error');
 if(errors.length){retryable=false;throw new Error(errors.map(m=>m.message).join('\n'));}
 pipeline=device.createRenderPipeline({layout:'auto',vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:'fs',targets:[{format,blend:{color:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'}}}]},primitive:{topology:'triangle-list'}});
 buffer=device.createBuffer({size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}}]});
 initializing=false;failed=false;canvas.hidden=false;
 canvas.parentElement.classList.remove('orb-fallback');document.getElementById('orbSupport').hidden=true;sync();
 }catch(error){if(current===generation)fallback(error);}
}
if('IntersectionObserver'in window)new IntersectionObserver(es=>{visible=es[0].isIntersecting;sync();}).observe(canvas.parentElement);
if('ResizeObserver'in window)new ResizeObserver(sync).observe(canvas);
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopDemo();else retries=0;sync();});
window.addEventListener('pagehide',()=>{pageHidden=true;stopDemo();cancelRetry();release();});
document.addEventListener('settings-open',()=>{settingsOpen=true;stopDemo();sync();});
document.addEventListener('settings-close',()=>{settingsOpen=false;sync();});
window.addEventListener('pageshow',()=>{pageHidden=false;retries=0;sync();});
reduced.addEventListener('change',()=>{stopDemo();sync();});
setState('idle');
})();
