(()=>{

'use strict';



const $ = id => document.getElementById(id);
const tr=text=>window.VMSettings?.t(text)||text;
const el = (t,c,x) => { const n=document.createElement(t); if(c)n.className=c; if(x!=null)n.textContent=x; return n; };

const conversations = [{id:'c1',title:'新对话',pinned:false,messages:[]}];
let current = conversations[0], replyMessage = null;
const state = {get messages(){return current.messages;},busy:false,listening:false,memory:false};
let toastTimer, typeTimer, markTimer;
const reduced = VMUI.reduced;
const backgroundVideo = $('bgVideo');
const backgroundMotion = matchMedia('(prefers-reduced-motion: reduce)');

function syncBackgroundPlayback(){
  if(document.hidden || state.memory || backgroundMotion.matches){
    backgroundVideo.pause();
    backgroundVideo.classList.remove('playing');
    return;
  }
  const attempt = backgroundVideo.play();
  attempt?.catch(() => backgroundVideo.classList.remove('playing'));
}
backgroundVideo.addEventListener('playing', () => {
  if(document.hidden || state.memory || backgroundMotion.matches){backgroundVideo.pause();return;}
  backgroundVideo.classList.add('playing');
});
backgroundVideo.addEventListener('pause', () => backgroundVideo.classList.remove('playing'));
backgroundVideo.addEventListener('error', () => backgroundVideo.classList.remove('playing'));
backgroundMotion.addEventListener('change', syncBackgroundPlayback);

function renderRail(){
  $('pinList').replaceChildren();$('chatList').replaceChildren();
  conversations.forEach(c=>{
    const row=el('div','session-row');
    const b=el('button','item'+(c.pinned?' pin':''));
    b.append(el('i','dot'),el('span',null,c.title));b.title=c.title;
    b.setAttribute('aria-current',String(c===current));
    b.onclick=()=>selectConversation(c);
    const pin=el('button','pin-chat',c.pinned?'◆':'◇');
    pin.setAttribute('aria-label',c.pinned?'取消置顶':'置顶对话');pin.setAttribute('aria-pressed',String(c.pinned));
    pin.onclick=()=>{c.pinned=!c.pinned;renderRail();};
    row.append(b,pin);$(c.pinned?'pinList':'chatList').append(row);
  });
}
function selectConversation(c){
  voiceInput.cancel();clearTimeout(typeTimer);clearTimeout(markTimer);
  document.body.classList.remove('talking');current=c;state.busy=!!c.busy;
  const me=[...c.messages].reverse().find(m=>m.role==='me');
  const her=[...c.messages].reverse().find(m=>m.role==='her');
  $('said').textContent=me?.text||tr('说点什么，她在听。');$('said').classList.add('on');
  replyMessage=null;renderMarks($('marks'),me?.marks||[],false);$('voice').textContent=her?.text||tr('我在。今天想从哪里说起？');
  $('say').value='';renderLog();renderRail();syncSend();setRail(innerWidth>1024 && !document.body.classList.contains('rail-collapsed'));
}
function toast(text){
  $('toast').textContent = text; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(()=>$('toast').hidden = true, 3000);
}

/* ── 情绪 / 说话人 / 实体 / 簇 ── */
const KINDS = { emo:['mk-emo','情绪'], spk:['mk-spk','说话人'], ent:['mk-ent','实体'], clu:['mk-clu','簇'] };
function renderMarks(host, marks, animate){
  host.replaceChildren();
  marks.forEach(([kind,value],i) => {
    const [cls,label] = KINDS[kind];
    const chip = el('span','mk '+cls);
    const tail = el('span',null,label); tail.style.opacity = '.62';
    chip.append(el('i'), el('b',null,value), tail);
    if(animate) chip.style.animationDelay = (0.12 + i*0.11)+'s';
    else { chip.style.animation='none'; chip.style.opacity='1'; chip.style.transform='none'; }
    host.append(chip);
  });
}

/* ── 聊天记录 ── */
function renderLog(){
  const host = $('logScroll');
  host.replaceChildren();
  if(!state.messages.length){ host.append(el('p','log-empty','还没有记录。说第一句话，它会出现在这里。')); return; }
  state.messages.forEach((m, idx) => {
    const turn = el('article','turn '+(m.role==='me'?'me':'her'));
    turn.style.animationDelay = (idx*0.05)+'s';
    turn.append(el('div','from', m.role==='me'?'你':'Echo'), el('p',null,m.text));
    if(m.marks){ const mk = el('div','marks'); renderMarks(mk, m.marks, false); turn.append(mk); }
    if(m.role==='her'){
      const actions=el('div','turn-actions');const copy=el('button',null,'复制');copy.onclick=()=>VMUI.copy(m.text);
      const read=el('button',null,m.replaying?'停止重播':'重播原声');
      read.setAttribute('aria-pressed',String(!!m.replaying));
      read.onclick=()=>{
        if(state.busy){VMUI.notify('请等当前回复完成。');return;}
        void voiceInput.replay(m.outputId,on=>{m.replaying=on;read.textContent=on?'停止重播':'重播原声';read.setAttribute('aria-pressed',String(on));document.body.classList.toggle('talking',on);});
      };
      actions.append(copy,read);turn.append(actions);
    }
    host.append(turn);
  });
  requestAnimationFrame(()=>host.scrollTop = host.scrollHeight);
}

/* ── 她说话 ── */
function speak(text){
  clearTimeout(typeTimer);
  if(reduced){$('voice').textContent=text;return;}
  const box = $('voice'); box.textContent = '';
  const cursor = el('span','cursor'); box.append(cursor);
  let i = 0;
  document.body.classList.add('talking');
  const step = () => {
    if(i >= text.length){ cursor.remove(); document.body.classList.remove('talking'); return; }
    cursor.before(document.createTextNode(text[i++]));
    typeTimer = setTimeout(step, text[i-1]==='\n' ? 180 : 38);
  };
  step();
}

function handleStudio(message) {
  if (message.type === 'partial_transcript') {
    if (!VMStudio.acceptsPartial(current.messages, message)) return;
    $('said').textContent = message.text; $('said').classList.add('on');
  } else if (message.type === 'user_transcript' || message.type === 'user_backchannel') {
    VMStudio.applyUser(current.messages, message, 'me');
    $('said').textContent = message.text; $('said').classList.add('on');
    if (current.title === '新对话') current.title = message.text.slice(0, 26);
    if (message.type === 'user_transcript') $('marks').replaceChildren();
    renderRail(); renderLog();
  } else if (message.type === 'answer_start') {
    clearTimeout(typeTimer); current.busy = state.busy = true;
    replyMessage = {role:'her',text:'',outputId:message.output_id}; current.messages.push(replyMessage);
    $('voice').textContent = ''; renderLog(); syncSend();
  } else if (message.type === 'answer_delta' && replyMessage) {
    replyMessage.text += message.text || ''; $('voice').textContent = replyMessage.text; renderLog();
  } else if (message.type === 'answer_interrupt') {
    if (replyMessage) {replyMessage.text = message.heard_text || ''; $('voice').textContent = replyMessage.text;}
    replyMessage = null; current.busy = state.busy = false; renderLog(); syncSend();
  } else if (['playback_done','disconnected','error'].includes(message.type)) {
    if (message.type !== 'playback_done' && replyMessage) {
      const index = current.messages.indexOf(replyMessage);
      if (index >= 0) current.messages.splice(index, 1);
      $('voice').textContent = ''; renderLog();
    }
    replyMessage = null; current.busy = state.busy = false; syncSend();
  } else if (message.type === 'memory_hits' || message.type === 'tag_update') {
    const marks = [];
    if (message.emotion) marks.push(['emo', message.emotion]);
    if (message.speaker_id) marks.push(['spk', message.speaker_id]);
    for (const entity of message.entities || []) marks.push(['ent', entity]);
    for (const slot of message.slots || []) marks.push(['clu', slot]);
    if (message.type === 'tag_update') {
      const last = [...current.messages].reverse().find(item => item.role === 'me');
      if (last?.marks) marks.push(...last.marks.filter(([kind]) => kind !== 'emo'));
    }
    const last = [...current.messages].reverse().find(item => item.role === 'me');
    if (last) last.marks = marks;
    renderMarks($('marks'), marks, false); renderLog();
    if (message.type === 'memory_hits') {
      const memories = [...(message.left_brain || []).map(item => item.text), ...(message.right_brain_hits || []).filter(item => !item.internal).map(item => item.content)];
      document.querySelector('.memory-head p').textContent = `本轮召回 ${memories.length} 条相关记忆`;
    }
  }
}
async function send(text) {
  text = String(text).trim().slice(0,2000);
  if (!text || state.busy) return;
  const conversation = current; conversation.busy = state.busy = true; syncSend();
  const item = {role:'me',text,pending:true}; conversation.messages.push(item);
  if (conversation.title === '新对话') conversation.title = text.slice(0, 26);
  $('said').textContent = text; $('said').classList.add('on'); $('say').value = '';
  renderRail(); renderLog();
  try { await voiceInput.send(text); }
  catch (error) {
    const index = conversation.messages.indexOf(item); if (index >= 0) conversation.messages.splice(index,1);
    conversation.busy = false;
    if (current === conversation) { state.busy = false; $('say').value = text; renderLog(); syncSend(); }
    VMUI.notify(error.message);
  }
}
function syncSend(){ $('sendBtn').disabled = state.busy || !$('say').value.trim(); }

/* ── 墨染：逐帧生长，一点点连成片 ── */
const cvs = $('inkCanvas'), ctx = cvs.getContext('2d');
const STOPS = [[0,'#161322'],[.07,'#09080f'],[.155,'#000000'],[1,'#000000']];
let blots = [], W = 0, H = 0, ink = 0, inkTarget = 0, raf = 0, lastT = 0;
const DUR = 1500;

function mulberry(a){ return function(){ a|=0; a = a+0x6D2B79F5|0; let t = Math.imul(a^a>>>15, 1|a); t = t+Math.imul(t^t>>>7, 61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function seedBlots(){
  const rnd = mulberry(20260915);
  blots = [];
  const push = (nx,ny,scale,bias) => {
    const dx = (nx - .5), dy = (ny - .05) * .82;       // 从顶部那颗按钮落墨，向四周洇开
    const dist = Math.min(1, Math.sqrt(dx*dx + dy*dy) / .92);
    blots.push({ nx, ny, s:scale, d:Math.max(0, Math.min(.66, dist*0.72 + bias + rnd()*0.09)),
      g:0.30 + rnd()*0.16, p:[rnd()*6.28, rnd()*6.28, rnd()*6.28] });
  };
  for(let r=0;r<5;r++) for(let c=0;c<7;c++)
    push((c-0.5)/5.6 + (rnd()-.5)*0.07, (r-0.4)/3.9 + (rnd()-.5)*0.07, 0.20 + rnd()*0.075, 0);
  for(let i=0;i<16;i++) push(rnd()*1.05-.02, rnd()*1.05-.02, 0.018+rnd()*0.03, -0.1);  // 先溅出去的小点
}
function resizeInk(){
  const r = cvs.getBoundingClientRect(), dpr = Math.min(devicePixelRatio||1, 2);
  W = r.width; H = r.height;
  cvs.width = Math.max(1, Math.round(W*dpr)); cvs.height = Math.max(1, Math.round(H*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0);
  paintInk(ink);
}
function paintInk(t){
  ctx.clearRect(0,0,W,H);
  if(t <= 0) return;
  ctx.beginPath();
  const unit = Math.hypot(W,H);
  for(const b of blots){
    const p = (t - b.d) / b.g;
    if(p <= 0) continue;
    const k = p >= 1 ? 1 : 1 - Math.pow(1-p, 2.4);        // 先快后慢，像墨在纸上洇开
    const R = b.s * unit * k;
    if(R < .4) continue;
    const cx = b.nx*W, cy = b.ny*H, N = 44;
    for(let i=0;i<=N;i++){
      const a = i/N*Math.PI*2;
      const wob = 1 + 0.17*Math.sin(3*a+b.p[0]) + 0.10*Math.sin(5*a+b.p[1]) + 0.055*Math.sin(9*a+b.p[2]);
      const x = cx + Math.cos(a)*R*wob, y = cy + Math.sin(a)*R*wob*0.94;
      i ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
    }
    ctx.closePath();
  }
  if(t > 0.985) ctx.rect(0,0,W,H);                        // 收尾补满四角
  const grad = ctx.createLinearGradient(0,0,W,0);
  STOPS.forEach(([o,c]) => grad.addColorStop(o,c));
  ctx.fillStyle = grad;
  ctx.fill();
}
function tick(now){
  const dt = Math.min(64, now - (lastT || now)); lastT = now;
  const up = inkTarget > ink;
  const step = dt / (up ? DUR : DUR*0.62);
  ink = up ? Math.min(inkTarget, ink+step) : Math.max(inkTarget, ink-step);
  paintInk(ink);
  $('memoryIn').classList.toggle('show', ink > 0.9);
  if(ink !== inkTarget && !document.hidden) raf = requestAnimationFrame(tick);
  else { raf = 0; lastT = 0; }
}
function setMemory(on){
  state.memory = on;
  syncBackgroundPlayback();
  document.querySelector('.log').inert=on;
  $('memoryIn').inert=!on;
  document.body.classList.toggle('memory-on', on);
  $('tabMem').setAttribute('aria-pressed', String(on));
  $('tabLog').setAttribute('aria-pressed', String(!on));
  inkTarget = on ? 1 : 0;
  if(reduced){ ink = inkTarget; paintInk(ink); $('memoryIn').classList.toggle('show', on); return; }
  lastT = 0;
  if(!raf) raf = requestAnimationFrame(tick);
}

/* ── 左栏收起 / 展开 ── */
function setRail(open){
  if(innerWidth <= 1024){ document.body.classList.toggle('rail-open',open);document.querySelector('.rail').inert=!open;$('peekLogo').setAttribute('aria-expanded',String(open));return; }
  document.querySelector('.rail').inert=!open;
  document.body.classList.toggle('rail-collapsed', !open);
  $('railLogo').setAttribute('aria-expanded', String(open));
  setTimeout(resizeInk, 460);
}

/* ── 事件 ── */
$('ask').addEventListener('submit', e => { e.preventDefault(); send($('say').value); });
$('say').addEventListener('input', syncSend);
const newChat = () => {
  const c={id:crypto.randomUUID(),title:'新对话',pinned:false,messages:[]};conversations.unshift(c);selectConversation(c);

  renderLog();
  $('said').classList.remove('on'); $('said').textContent = tr('说点什么，她在听。');
  $('marks').replaceChildren(); speak('我在。今天想从哪里说起？');
  requestAnimationFrame(()=>$('said').classList.add('on'));
  document.body.classList.remove('rail-open'); $('say').focus();
};
$('newChat').onclick = newChat;
$('peekNew').onclick = newChat;
$('railLogo').onclick = () => setRail(false);
$('peekLogo').onclick = () => setRail(true);
function setDrawer(on){
  document.body.classList.toggle('show-right',on);$('drawerBtn').setAttribute('aria-expanded',String(on));
  $('drawerBtn').setAttribute('aria-label',on?'返回对话':'查看聊天记录');
  $('drawerBtn').innerHTML=on?'<svg viewBox="0 0 24 24"><path d="m6 6 12 12M6 18 18 6"/></svg>':'<svg><use href="#i-chat"/></svg>';
  if(!on)setMemory(false);
}
$('drawerBtn').onclick=()=>setDrawer(!document.body.classList.contains('show-right'));
$('tabLog').onclick = () => setMemory(false);
$('tabMem').onclick = () => setMemory(true);
addEventListener('keydown', e => { if(e.key === 'Escape' && state.memory) setMemory(false); });
addEventListener('resize', resizeInk);
const voiceInput=VMStudio.create({
  onEvent: handleStudio,
  onPhase(value){document.body.classList.toggle('talking', value === 'speaking');},
  onState(on){state.listening=on;document.body.classList.toggle('listening',on);$('micBtn').setAttribute('aria-label',on?'停止说话':'开始说话');$('micBtn').setAttribute('aria-pressed',String(on));$('startTalk').textContent=on?'结束对话':'开始对话';$('startTalk').setAttribute('aria-pressed',String(on));window.studioModelServices?.reportConversationState(on);},
  onInterim(text){$('said').classList.add('on');$('said').textContent=text;$('say').value=text;syncSend();},
  onFinal:send
});
window.VMSettings?.bindHarness(voiceInput);
$('micBtn').onclick=()=>voiceInput.toggle();
$('startTalk').onclick=()=>{$('say').focus();voiceInput.toggle();};
addEventListener('click', e => {
  if(document.body.classList.contains('rail-open') && !e.target.closest('.rail') && !e.target.closest('#peekLogo'))
    setRail(false);
}, true);

renderRail(); renderLog(); syncSend(); seedBlots(); resizeInk(); syncBackgroundPlayback();
if(innerWidth <= 1024){document.body.classList.add('rail-collapsed');setRail(false);}

window.addEventListener('memory-domain-select',e=>{
 const names={work:'工作',health:'健康',relationships:'人物',finance:'财务',knowledge:'研究',goals:'项目',daily_life:'日常',emotion:'情绪',preference:'偏好',personality:'人格'};
 const domain=tr(names[e.detail.domain]||'全部记忆');
 document.querySelector('.memory-head h2').textContent=domain;

});
document.querySelector('.switch').addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();setMemory(!state.memory);$(state.memory?'tabMem':'tabLog').focus();}});
addEventListener('keydown',e=>{if(e.key==='Escape'){setDrawer(false);setRail(innerWidth>1024 && !document.body.classList.contains('rail-collapsed'));voiceInput.cancel();}});
document.addEventListener('visibilitychange',()=>{syncBackgroundPlayback();cancelAnimationFrame(raf);raf=0;lastT=0;if(!document.hidden && ink!==inkTarget)raf=requestAnimationFrame(tick);});
addEventListener('pageshow',e=>{syncBackgroundPlayback();if(e.persisted){selectConversation(current);resizeInk();if(ink!==inkTarget&&!raf)raf=requestAnimationFrame(tick);}});
document.addEventListener('settings-open',()=>voiceInput.pause());
addEventListener('pagehide',()=>{backgroundVideo.pause();cancelAnimationFrame(raf);clearTimeout(typeTimer);clearTimeout(markTimer);clearTimeout(toastTimer);});

})();
