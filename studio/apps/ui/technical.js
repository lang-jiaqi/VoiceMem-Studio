(()=>{
'use strict';
/* ============================================================
   VoiceMem — demo state
   ============================================================ */
const DOMAINS = {
  work:          { zh: '工作',   side: 'L', color: '#7aa2ff' },
  knowledge:     { zh: '知识',   side: 'L', color: '#7aa2ff' },
  health:        { zh: '健康',   side: 'L', color: '#7aa2ff' },
  daily_life:    { zh: '日常',   side: 'L', color: '#7aa2ff' },
  relationships: { zh: '关系',   side: 'L', color: '#7aa2ff' },
  goals:         { zh: '目标',   side: 'L', color: '#7aa2ff' },
  finance:       { zh: '财务',   side: 'L', color: '#7aa2ff' },
  emotion:       { zh: '情绪',   side: 'R', color: '#ff5fa2', shape: 'square' },
  preference:    { zh: '偏好',   side: 'R', color: '#4ade80', shape: 'square' },
  personality:   { zh: '性格',   side: 'R', color: '#ff9d4d', shape: 'triangle' }
};

const MEMORIES = {facts: [], traits: []};
const CONVERSATIONS = [{id:'c1',title:'New conversation',messages:[]}];
let activeConv = 'c1', activeDomain = null, replyMessage = null;
let perception = {};

/* ---------- rendering ---------- */
const $ = id => document.getElementById(id);
const orbState=state=>window.liquidOrb?.setState(state);
const tr=text=>window.VMSettings?.t(text)||text;
let thoughtTimer;
let flowVersion=0;

function renderStack(host, items) {
  host.replaceChildren();
  for (const item of items) {
    const node = document.createElement('div'); node.className = 'mem enter';
    node.textContent = item.text;
    const score = document.createElement('span'); score.className = 'score';
    score.textContent = `${DOMAINS[item.domain]?.zh || item.domain || ''} · ${Number(item.score || 0).toFixed(2)}`;
    node.append(score); host.append(node);
  }
}
function renderTopK() {
  const filter = items => activeDomain ? items.filter(item => item.domain === activeDomain) : items;
  renderStack($('stackFacts'), filter(MEMORIES.facts));
  renderStack($('stackTraits'), filter(MEMORIES.traits));
  $('topkScope').textContent = activeDomain ? `${DOMAINS[activeDomain]?.zh || activeDomain} · 本轮召回` : '本轮召回';
  renderPerception();
}

function esc(s) { return s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

function messageNode(m) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (m.role === 'ai' ? 'assistant' : 'user');
  const avatar = m.role === 'ai'
    ? `<div class="avatar ai"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 12v2M8 8v8M12 5v14M16 8v8M20 11v3"/></svg></div>`
    : `<div class="avatar">Y</div>`;
  const head = m.role === 'ai'
    ? `<span class="msg-name">Voicemem</span><span class="tag">Memory AI</span>`
    : `<span class="msg-name">You</span>`;
  const grounded = m.grounded ? `
    <div class="grounded">
      <div class="grounded-label">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19V5m0 14h16"/></svg>
        引用的记忆
      </div>
      <button class="chip">${esc(m.grounded)}</button>
    </div>` : '';
  const foot = m.role === 'ai' ? `
    <div class="msg-foot">
      <button title="复制"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg></button>
      <button title="重播原声"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M11 5 6 9H3v6h3l5 4z"/><path d="M16 9a4 4 0 0 1 0 6"/></svg></button>
      
    </div>` : '';
  wrap.innerHTML = `${avatar}<div>
      <div class="msg-head">${head}<span class="time">${m.time}</span></div>
      <div class="msg-body">${esc(m.text)}</div>
      ${grounded}${foot}
    </div>`;
  const buttons=wrap.querySelectorAll('.msg-foot button');
  if(buttons[0]) {buttons[0].setAttribute('aria-label','复制回复');buttons[0].onclick=()=>VMUI.copy(m.text);}
  if(buttons[1]) {
    const replayState=on=>{m.replaying=on;buttons[1].setAttribute('aria-pressed',String(on));buttons[1].title=on?'停止重播':'重播原声';buttons[1].setAttribute('aria-label',on?'停止重播':'重播原声回复');};
    replayState(!!m.replaying);
    buttons[1].onclick=()=>{
      if(CONVERSATIONS.find(c=>c.id===activeConv)?.busy){VMUI.notify('请等当前回复完成。');return;}
      const version=flowVersion;
      void voice.replay(m.outputId,on=>{replayState(on);if(version===flowVersion)orbState(on?'speaking':'idle');});
    };
  }
  const chip=wrap.querySelector('.chip');
  if(chip) chip.onclick=()=>{
    const memory=[...MEMORIES.facts,...MEMORIES.traits].find(x=>m.grounded.startsWith(x.id));
    if(memory) {activeDomain=memory.domain;renderTopK(memory.text);VMUI.notify(memory.text);switchTab('space');}
  };
  return wrap;
}

function renderThread() {
  const conv = CONVERSATIONS.find(c => c.id === activeConv);
  const t = $('thread');
  t.innerHTML = '';
  const card = document.createElement('div');
  card.className = 'thread-card';
  if (!conv.messages.length) {
    const empty = document.createElement('div');
    empty.className = 'thread-empty';
    empty.textContent = '说点什么，VoiceMem 会先去记忆里找相关的内容，再回答。';
    card.appendChild(empty);
  }
  conv.messages.forEach(m => card.appendChild(messageNode(m)));
  t.appendChild(card);
  t.scrollTop = t.scrollHeight;
}

function renderPerception() {
  $('emotionTag').textContent = perception.emotion || '待识别';
  $('speakerTag').textContent = perception.speaker_id || '—';
  $('entityTag').textContent = (perception.entities || []).join(' · ') || '未匹配';
  $('clusterTag').textContent = (perception.slots || []).join(' · ') || '未匹配';
}

function syncConversation() {
  flowVersion++;clearTimeout(thoughtTimer);orbState('idle');
  voice.cancel();
  replyMessage = null; perception = {}; MEMORIES.facts = []; MEMORIES.traits = [];
  clearInterval(typing);
  activeDomain = null;
  const conv = CONVERSATIONS.find(c => c.id === activeConv);
  const lastUser = [...conv.messages].reverse().find(m => m.role === 'user');
  const lastAI = [...conv.messages].reverse().find(m => m.role === 'ai');
  $('composer').value = '';
  $('liveEcho').textContent = lastUser?.text || tr('等待你的下一句话…');
  $('liveEcho').classList.toggle('empty', !lastUser);
  $('liveEchoPrev').textContent = '';
  $('aiEcho').textContent = lastAI?.text || tr('回答会同步显示在这里。');
  $('aiEcho').classList.toggle('empty', !lastAI);
  renderTopK(lastUser?.text || '');
  renderConvList();
  renderThread();
}

function renderConvList() {
  $('convList').innerHTML = '';
  $('pinnedList').innerHTML = '';
  CONVERSATIONS.forEach(c => {
    const li = document.createElement('li');
    li.className = 'conv' + (c.id === activeConv ? ' active' : '');
    const select = document.createElement('button');
    select.className = 'conv-select';
    select.title = c.title;
    select.setAttribute('aria-current', c.id === activeConv ? 'true' : 'false');
    select.innerHTML = `<span class="conv-title">${esc(c.title)}</span><span class="dot"></span>`;
    select.onclick = () => { voice.cancel(); activeConv = c.id; syncConversation(); };
    const pin = document.createElement('button');
    pin.className = 'pin-conv';
    pin.title = c.pinned ? '取消置顶' : '置顶对话';
    pin.setAttribute('aria-label', pin.title + '：' + c.title);
    pin.setAttribute('aria-pressed', String(!!c.pinned));
    pin.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="m8 3 8 0-1 7 4 4H5l4-4-1-7ZM12 14v7"/></svg>';
    pin.onclick = () => { c.pinned = !c.pinned; renderConvList(); };
    li.append(select, pin);
    $(c.pinned ? 'pinnedList' : 'convList').appendChild(li);
  });
  ['pinnedList', 'convList'].forEach(id => {
    if (!$(id).children.length) {
      const empty = document.createElement('li');
      empty.className = 'conv-empty';
      empty.textContent = id === 'pinnedList' ? '暂无置顶对话' : '暂无聊天记录';
      $(id).appendChild(empty);
    }
  });
}

/* ---------- send flow ---------- */
let typing = null;
function typeInto(el, text, done) {
  clearInterval(typing);
  if(VMUI.reduced) {el.textContent=text;el.classList.remove('empty');done?.();return;}
  el.classList.remove('empty');
  el.innerHTML = '';
  const caret = document.createElement('span');
  caret.className = 'caret';
  el.appendChild(caret);
  let i = 0;
  typing = setInterval(() => {
    i += 2;
    el.textContent = text.slice(0, i);
    el.appendChild(caret);
    if (i >= text.length) { clearInterval(typing); caret.remove(); done && done(); }
  }, 16);
}

function now() {
  return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function handleStudio(message) {
  const conv = CONVERSATIONS.find(c => c.id === activeConv);
  if (message.type === 'partial_transcript') {
    if (!VMStudio.acceptsPartial(conv.messages, message)) return;
    $('liveEcho').textContent = message.text; $('liveEcho').classList.remove('empty');
  } else if (message.type === 'user_transcript' || message.type === 'user_backchannel') {
    VMStudio.applyUser(conv.messages, message, 'user');
    $('liveEcho').textContent = message.text; $('liveEcho').classList.remove('empty');
    if (conv.title.startsWith('New conversation')) conv.title = message.text.slice(0, 34);
    if (message.type === 'user_transcript') perception = {};
    renderPerception(); renderThread(); renderConvList();
  } else if (message.type === 'answer_start') {
    conv.busy = true;
    replyMessage = {role:'ai',text:'',time:now(),outputId:message.output_id};
    conv.messages.push(replyMessage); $('aiEcho').textContent = ''; renderThread();
  } else if (message.type === 'answer_delta' && replyMessage) {
    replyMessage.text += message.text || ''; $('aiEcho').textContent = replyMessage.text;
    $('aiEcho').classList.remove('empty'); renderThread();
  } else if (message.type === 'answer_interrupt') {
    if (replyMessage) { replyMessage.text = message.heard_text || ''; $('aiEcho').textContent = replyMessage.text; }
    replyMessage = null; conv.busy = false; renderThread();
  } else if (['playback_done','disconnected','error'].includes(message.type)) {
    if (message.type !== 'playback_done' && replyMessage) {
      const index = conv.messages.indexOf(replyMessage);
      if (index >= 0) conv.messages.splice(index, 1);
      $('aiEcho').textContent = ''; renderThread();
    }
    conv.busy = false; replyMessage = null;
  } else if (message.type === 'memory_hits') {
    perception = message;
    MEMORIES.facts = (message.left_brain || []).map(item => ({...item, domain:item.slot || item.domain || 'daily_life'}));
    MEMORIES.traits = (message.right_brain_hits || []).filter(item => !item.internal).map(item => ({...item, text:item.content, domain:item.source || 'personality', score:item.priority}));
    renderTopK();
  } else if (message.type === 'tag_update') {
    perception = {...perception, ...message}; renderPerception();
  }
}
function send(text) {
  text = text.trim().slice(0, 2000);
  const conv = CONVERSATIONS.find(c => c.id === activeConv);
  if (!text) return false;
  if (conv.busy) { VMUI.notify('请等当前回复完成。'); return false; }
  conv.busy = true;
  const item = {role:'user',text,time:now(),pending:true}; conv.messages.push(item);
  if (conv.title.startsWith('New conversation')) conv.title = text.slice(0, 34);
  $('liveEcho').textContent = text; $('liveEcho').classList.remove('empty');
  renderThread(); renderConvList();
  voice.send(text).catch(error => {
    const index = conv.messages.indexOf(item); if (index >= 0) conv.messages.splice(index, 1);
    conv.busy = false;
    if (conv.id === activeConv) { $('composer').value = text; renderThread(); }
    VMUI.notify(error.message);
  });
  return true;
}

/* ---------- wiring ---------- */
$('send').onclick = () => { if(send($('composer').value)) $('composer').value = ''; };
$('composer').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.isComposing) { if(send($('composer').value)) $('composer').value = ''; }
});
$('composer').addEventListener('input', e => {
  const v = e.target.value;
  $('liveEcho').classList.remove('interim');
  $('liveEcho').textContent = v || tr('等待你的下一句话…');
  $('liveEcho').classList.toggle('empty', !v);
  renderPerception(v);
});
$('newConv').onclick = () => {
  const id = 'c' + (CONVERSATIONS.length + 1);
  CONVERSATIONS.unshift({ id, title: 'New conversation', messages: [] });
  voice.cancel();
  activeConv = id;
  syncConversation();
  $('composer').focus();
};
$('tabChat').onclick = () => switchTab('chat');
$('tabSpace').onclick = () => switchTab('space');
function switchTab(which) {
  const chat = which === 'chat';
  document.body.classList.toggle('show-brain',!chat);
  document.body.classList.toggle('chat-background-only',chat);
  $('tabChat').tabIndex=chat?0:-1;$('tabSpace').tabIndex=chat?-1:0;
  $('tabChat').setAttribute('aria-selected', chat);
  $('tabSpace').setAttribute('aria-selected', !chat);
  $('viewChat').classList.toggle('on', chat);
  $('viewSpace').classList.toggle('on', !chat);
}

/* ---------- sidebar collapse ---------- */
document.querySelector('.switch').addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const next=e.key==='Home'?'chat':e.key==='End'?'space':$('tabChat').getAttribute('aria-selected')==='true'?'space':'chat';switchTab(next);$(next==='chat'?'tabChat':'tabSpace').focus();}});
const gridEl = document.querySelector('.grid');
function setSidebar(hidden) {
  gridEl.classList.toggle('side-hidden', hidden);
  $('toggleSide').setAttribute('aria-expanded', String(!hidden));
  $('toggleSide').setAttribute('aria-label', hidden ? '展开对话栏' : '收起对话栏');
  $('toggleSide').title = hidden ? '展开对话栏' : '收起对话栏';
  $('conversationSidebar').inert = hidden;
}
$('toggleSide').onclick = () => setSidebar(!gridEl.classList.contains('side-hidden'));
document.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
    e.preventDefault();
    setSidebar(!gridEl.classList.contains('side-hidden'));
  }
});

/* Voice lifecycle is local to this page. */
const voice = VMStudio.create({
  onEvent: handleStudio,
  onPhase: orbState,
  onState(on) {
    if(on){clearInterval(typing);clearTimeout(thoughtTimer);orbState('listening');}
    else if(window.liquidOrb?.getState()==='listening')orbState('idle');
    document.querySelector('.input-wrap').classList.toggle('listening',on);
    $('mic').classList.toggle('on',on); $('liveBadge').classList.toggle('on',on);
    $('mic').setAttribute('aria-label',on?'停止说话':'语音输入');
    $('mic').setAttribute('aria-pressed',String(on));
    $('startTalk').textContent=on?'结束对话':'开始对话';
    $('startTalk').setAttribute('aria-pressed',String(on));
    window.studioModelServices?.reportConversationState(on);
  },
  onInterim(text) { $('liveEcho').textContent=text;$('liveEcho').classList.remove('empty');$('composer').value=text;renderPerception(text); },
  onFinal(text) {if(send(text))$('composer').value='';}
});
window.VMSettings?.bindHarness(voice);
$('mic').onclick=()=>voice.toggle();
$('startTalk').onclick=()=>{ $('composer').focus();voice.toggle(); };
$('settingsBtn').onclick=()=>window.VMSettings?.open();

syncConversation();
switchTab('space');
if(matchMedia('(max-width:700px)').matches) setSidebar(true);

document.addEventListener('settings-open',()=>{voice.pause();window.liquidOrb?.stopVoiceDemo();});
window.addEventListener('pagehide',()=>{clearInterval(typing);clearTimeout(thoughtTimer);orbState('idle');});
window.addEventListener('pageshow',e=>{if(e.persisted)syncConversation();});

window.addEventListener('memory-domain-select', e => {
  activeDomain = e.detail.domain;
  renderTopK(document.getElementById('composer').value || document.getElementById('liveEcho').textContent);
});

})();
