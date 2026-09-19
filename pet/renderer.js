const dot = document.querySelector('#dot'), character = document.querySelector('#character'), live = document.querySelector('#live'), stage = document.querySelector('#stage'), bubble = document.querySelector('#bubble');
// Browser preview uses the same interaction UI without desktop privileges.
const api = window.pet || { initial: async () => 'dot', onMode: () => {}, toggle: () => render(document.body.dataset.mode === 'dot' ? 'sit' : 'dot'), collapse: () => render('dot'), pointer: () => {}, dragStart: () => {}, dragMove: () => {}, dragEnd: () => {}, initialScale: async () => 1, onScale: () => {}, resize: () => {}, resetSize: () => {}, resizeStart: () => {}, resizeMove: () => {}, resizeEnd: () => {}, conversationState: async () => false, toggleConversation: async () => { throw new Error('请在 Studio App 中开启对话。'); } };
const talkButton = document.querySelector('#talkButton');
let renderEpoch=0, callStateRevision=0;
function callState(active) {
  callStateRevision++;
  talkButton.setAttribute('aria-pressed', String(active));
  document.querySelector('#talkLabel').textContent = active ? '结束对话' : '开始对话';
  document.querySelector('#callHint').textContent = active ? '通话进行中 · 再次点击结束' : '点击下方按钮，开始语音对话';
}
let bubbleTimer;
function notice(message) {
  clearTimeout(bubbleTimer); bubble.textContent = message; bubble.hidden = false;
  bubbleTimer = setTimeout(() => { bubble.hidden = true; }, 3600);
}
talkButton.addEventListener('click', async () => {
  talkButton.disabled = true;
  try { callState(await api.toggleConversation()); bubble.hidden = true; }
  catch (error) { notice(error.message || '无法启动对话，请先打开 Studio App。'); }
  finally { talkButton.disabled = false; }
});
document.querySelector('#collapseCall').addEventListener('click', () => api.collapse());
api.onConversationState?.(callState);
api.onConversationError?.(notice);
const initialCallStateRevision = callStateRevision;
api.conversationState().then(active => {
  if (callStateRevision === initialCallStateRevision) callState(active);
}).catch(() => {});
async function render(mode) {
  const epoch=++renderEpoch;
  document.body.dataset.mode = mode; dot.hidden = mode !== 'dot'; character.hidden = mode === 'dot';
  if(mode==='dot')window.avatar.hide();bubble.hidden=true;
  stage.hidden=mode==='dot';
  if(mode!=='dot') {
    try {await window.avatar.show(mode);}
    catch {if(epoch===renderEpoch)notice('模型暂时没有加载成功，请重新启动。');}
  }
}
api.onMode(render); api.initial().then(render);
let gesture, suppressClick = false;
for (const el of [dot,character]) {
  el.addEventListener('pointerdown', e => {
    if (e.button !== 0 || e.target.closest?.('button')) return;
    const corner = e.target.closest('[data-resize-corner]')?.dataset.resizeCorner;
    el.setPointerCapture(e.pointerId);
    gesture = { x:e.screenX,y:e.screenY,moved:false,resize:Boolean(corner) };
    if(corner)api.resizeStart(corner);else api.dragStart();
  });
  el.addEventListener('pointermove', e => { if (!gesture) return; if (Math.hypot(e.screenX-gesture.x,e.screenY-gesture.y)>4) gesture.moved=true; if (gesture.moved) { if(gesture.resize)api.resizeMove();else api.dragMove(); } });
  const end = () => { if (!gesture) return; suppressClick=gesture.moved||gesture.resize; if(gesture.resize)api.resizeEnd();else api.dragEnd(); gesture=undefined; };
  el.addEventListener('pointerup',end); el.addEventListener('pointercancel',end); el.addEventListener('lostpointercapture',end);
}
dot.addEventListener('click', () => { if (!suppressClick) api.toggle(); suppressClick=false; });
character.addEventListener('dblclick', e => { if (!suppressClick && !e.target.closest?.('button')) api.collapse(); });
document.addEventListener('keydown',e => { if(e.key==='Escape') api.collapse(); });
document.addEventListener('pointermove',e => {
  const inside = character.contains(e.target) && !character.hidden;
  api.pointer(Boolean(gesture || e.target===dot || inside));
  if (inside) {
    const bounds = stage.getBoundingClientRect();
    window.avatar.setPointerGaze((e.clientX - bounds.left) / bounds.width * 2 - 1,
      1 - (e.clientY - bounds.top) / bounds.height * 2);
  } else window.avatar.setPointerGaze(null, null);
});
document.addEventListener('pointerleave',() => { window.avatar.setPointerGaze(null, null); if (!gesture) api.pointer(false); });
