/* Device-local display preferences, shared across the two independent pages. */
(()=>{
'use strict';
const KEY='voicemem.display',COMPONENT_KEY='voicemem.components.layout.v1',UI_BASE=1.2,CONTENT_BASE=.75;
let prefs={lang:'zh-CN',uiLevel:1,contentLevel:1,schema:2};
try{
 const saved=JSON.parse(localStorage.getItem(KEY)||'null');
 if(saved){
  if(saved.schema===2) prefs={...prefs,...saved};
  else {
   const oldUI=Number(saved.uiScale??saved.scale??1);
   const oldContent=Number(saved.contentScale??saved.scale??1);
   prefs={lang:saved.lang||'zh-CN',uiLevel:oldUI/UI_BASE,contentLevel:oldContent/CONTENT_BASE,schema:2};
  }
 }
}catch{}
function normalize(){const clamp=value=>Math.min(2,Math.max(.5,Number(value)||1));prefs={lang:prefs.lang==='en'?'en':'zh-CN',uiLevel:clamp(prefs.uiLevel),contentLevel:clamp(prefs.contentLevel),schema:2};}normalize();
const pairs=[
['等待你的下一句话…','Waiting for your next words…'],['回答会同步显示在这里。','Responses will appear here.'],['说点什么，她在听。','Say something. She is listening.'],['我在。今天想从哪里说起？','I’m here. Where would you like to start?'],['还没有记录。说第一句话，它会出现在这里。','No history yet. Your first message will appear here.'],['说点什么，VoiceMem 会先去记忆里找相关的内容，再回答。','Say something. VoiceMem will retrieve relevant memories before responding.'],['你','You'],['人物','People'],['研究','Research'],['项目','Projects'],['人格','Personality'],['全部记忆','All memories'],['娱乐','Entertainment'],['设置','Settings'],['显示设置','Display settings'],['返回对话','Back to conversation'],['选择模式','Choose your mode'],['科技风','Technical'],['数字人','Digital human'],['专注、清晰，尽在掌握','Focus, clarity, and control'],['有回应，也有陪伴','A voice and a companion'],['系统语言','System language'],['字体大小','Font size'],['UI 字号','UI font size'],['内容字号','Content font size'],['按钮、导航与标签','Buttons, navigation, and labels'],['ASR、记忆、回复与对话记录','ASR, memories, responses, and conversations'],['恢复默认','Reset to default'],['当前模式','Current mode'],['预览','Preview'],['记忆','Memory'],['识别到的语音显示在这里','Your recognized speech appears here'],['相关记忆显示在这里','Relevant memories appear here'],['回复与对话记录显示在这里','Responses and conversation history appear here'],
['首页','Home'],['返回首页','Back to home'],['科技风 ↗','Technical ↗'],['数字人 ↗','Digital human ↗'],['新对话','New conversation'],['置顶','Pinned'],['聊天记录','Chat history'],['对话记录','Conversation'],['记忆空间','Memory Space'],['对话','Chat'],['模型回复','Response'],['本轮感知','Perception'],['本轮感知 · ASR','Perception · ASR'],['开始对话','Start conversation'],['结束对话','End conversation'],['开始说话','Start speaking'],['停止说话','Stop speaking'],['正在聆听…','Listening…'],['语音输入','Voice input'],['发送','Send'],['倾听','Listening'],['说话','Speaking'],['短思考','Short thinking'],['长思考','Long thinking'],['待机','Idle'],['演示 · 倾听','Demo · Listening'],['演示 · 说话','Demo · Speaking'],['演示 · 短思考','Demo · Short thinking'],['演示 · 长思考','Demo · Long thinking'],['当前浏览器使用静态球体预览','This browser uses a static orb preview'],
['输入这一轮想说的话…','Type what you want to say…'],['本轮文字输入','Message input'],['说点什么…','Say something…'],['对 Echo 说','Talk to Echo'],['待识别','Not identified'],['未匹配','No match'],['情感','Emotion'],['情绪','Emotion'],['实体','Entities'],['簇','Clusters'],['说话人','Speaker'],['事实与经历','Facts and experiences'],['偏好','Preference'],['性格','Personality'],['情绪与人格','Emotion and personality'],['记忆节点','Memory nodes'],['点击节点查看记忆域','Select a node to explore memories'],['点击任一记忆域，Top-K 会切换到该域','Select a memory domain to filter Top-K'],['全部记忆域 · 按相似度排序','All memory domains · Ranked by similarity'],['引用的记忆','Referenced memory'],['已与你的记忆同步','Synced with your memory'],['暂无置顶对话','No pinned conversations'],['暂无聊天记录','No conversations yet'],['已复制','Copied'],['复制','Copy'],['复制回复','Copy response'],['重播原声','Replay original voice'],['重播原声回复','Replay original response voice'],['停止重播','Stop replay'],['今天','Today'],['与 Echo 的对话','Conversation with Echo'],['深湖','Deep Lake'],['记忆空间 · 深湖','Memory Space · Deep Lake'],['你说过的话在这里沉下去，彼此连成一片。','Your words settle here and connect into memories.'],
['收起对话栏','Collapse conversations'],['展开对话栏','Expand conversations'],['展开对话列表','Expand conversations'],['收起对话列表','Collapse conversations'],['查看聊天记录','Show chat history'],['关闭聊天记录','Close chat history'],['取消置顶','Unpin'],['置顶对话','Pin conversation'],['页面导航','Page navigation'],['对话列表','Conversations'],['右侧显示','Right panel'],['左右脑记忆','Left and right brain memories'],['对话设置','Conversation settings'],['液态语音球','Liquid voice orb'],['液态语音状态动画','Liquid voice state animation'],['球体动效演示','Orb animation demos'],['脑部记忆图谱：245 个记忆节点','Brain memory graph: 245 memory nodes'],['245 个记忆节点','245 memory nodes'],
['交互原型','Interactive prototype'],['想以哪种方式开始？','How would you like to begin?'],['同一个助手，两种相处方式。','One assistant, two ways to connect.'],['选择一个空间，开始对话','Choose a space and start a conversation'],['选择风格','Choose a style'],['银发数字人助手','Silver-haired digital assistant'],['工作','Work'],['知识','Knowledge'],['健康','Health'],['日常','Daily life'],['关系','Relationships'],['目标','Goals'],['财务','Finance'],['紧张','Anxious'],['开心','Happy'],['低落','Low'],['疲惫','Tired'],['轻快','Upbeat'],['迟疑','Hesitant'],['平静','Calm'],
['请等当前回复完成。','Please wait for the current response.'],['这条回复的原始语音暂不可用。','The original voice for this response is unavailable.'],['原始语音播放失败。','Original voice playback failed.'],['复制未成功，请选中文字后复制。','Copy failed. Select the text and copy it manually.'],['此浏览器不支持语音识别，请在输入框中开始对话。','Speech recognition is unavailable. Please type your message.'],['麦克风权限未开启，可继续文字对话。','Microphone access is off. You can continue by typing.'],['未识别到语音，请重试或输入文字。','No speech recognized. Try again or type your message.'],['语音未确认，文字已保留在输入框。','Speech was not confirmed. The text remains in the input.'],['麦克风暂不可用，请使用文字输入。','Microphone unavailable. Please type your message.'],['根据本轮文字关键词展示','Based on keywords in this message'],['尚未接入说话人识别','Speaker identification is not connected']
];
const en=new Map(pairs),zh=new Map(pairs.map(([a,b])=>[b,a]));
[['风格','Style'],['语言与字体','Language & Type'],['组件','Components'],['组件画板','Component canvas']].forEach(([a,b])=>{en.set(a,b);zh.set(b,a);});
[
 ['拖动整理，点击组件查看配置','Drag to arrange. Select a component to view its settings'],
 ['恢复布局','Reset layout'],['语音输入','Voice input'],['麦克风与流式识别','Microphone and streaming recognition'],
 ['记忆系统','Memory system'],['记忆检索与整理','Memory retrieval and organization'],
 ['回复模型','Response model'],['流式模型回复','Streaming model response'],
 ['语音合成','Speech synthesis'],['Breeze 本地语音','Local Breeze voice'],
 ['桌宠','Desktop pet'],['跟随回复状态','Follows response state'],['组件配置','Component settings'],
 ['关闭组件配置','Close component settings'],['运行状态','Status'],['已就绪','Ready'],
 ['服务商','Provider'],['API Key','API Key'],['已安全配置','Configured securely'],['未配置','Not configured'],
 ['不需要','Not required'],['配置方式','Configuration'],['启动终端','Startup terminal'],
 ['后端','Backend'],['模式','Mode'],['流式输入','Streaming input'],['模型','Model'],
 ['管理方式','Managed in'],['App 菜单','App menu'],['固定组件','Fixed component'],
 ['所有核心组件都会保留，只能调整位置。','All core components stay available; only their positions can change.'],
 ['重新配置 API','Reconfigure API'],['完成','Done'],
 ['退出 App 后重新运行 npm start，在终端选择服务商并输入一次 Key。VoiceMem 与 Studio 会共用这次配置。','Quit the app and run npm start again. Choose a provider and enter one key; VoiceMem and Studio will share it.'],
 ['拖动组件','Drag component'],['本机','Local'],['本地模型','Local model'],['启动配置','Startup config'],
].forEach(([a,b])=>{en.set(a,b);zh.set(b,a);});
// Existing English chrome gets a Chinese equivalent too.
zh.set('Technical','科技风');zh.set('Chat','对话记录');zh.set('info','事实记忆');zh.set('emo&persona','情绪与人格');zh.set('Speaker ID','说话人 ID');
function t(value){
 const dict=prefs.lang==='en'?en:zh;if(dict.has(value))return dict.get(value);
 if(prefs.lang==='en'){
  const domain=value.match(/^(.+?) 记忆域 · 按相似度排序$/);if(domain)return t(domain[1])+' memory domain · Ranked by similarity';
  const score=value.match(/^(\w+ · )(.+?)( · 相似度 )(.*)$/);if(score)return score[1]+t(score[2])+' · Similarity '+score[4];
  if(value.startsWith('置顶对话：'))return 'Pin conversation: '+value.slice(5);
  if(value.startsWith('取消置顶：'))return 'Unpin conversation: '+value.slice(5);
 }
 return value;
}
en.set('事实记忆','info');en.set('说话人 ID','Speaker ID');
const textCache=new WeakMap(),attrCache=new WeakMap();
const skip='script,style,textarea,input,[data-no-i18n],.msg-body,.turn p,.conv-title,.conv-select,.item,#liveEcho,#liveEchoPrev,#aiEcho,#said,#voice';
function translate(root=document.body){
 const walk=node=>{
  if(node.nodeType===3){const parent=node.parentElement;if(!parent||parent.closest(skip))return;
   const current=node.nodeValue;let record=textCache.get(node);if(!record||current!==record.output)record={source:current};
   const trimmed=record.source.trim();const out=t(trimmed);record.output=record.source.replace(trimmed,out);textCache.set(node,record);if(current!==record.output)node.nodeValue=record.output;
  }else if(node.nodeType===1){if(node.matches('script,style,[data-no-i18n]'))return;
   for(const key of ['title','aria-label','placeholder','alt'])if(node.hasAttribute(key)){
    let records=attrCache.get(node);if(!records){records={};attrCache.set(node,records);}const current=node.getAttribute(key);let rec=records[key];if(!rec||rec.output!==current)rec={source:current};rec.output=t(rec.source);records[key]=rec;if(current!==rec.output)node.setAttribute(key,rec.output);
   }
   for(const child of node.childNodes)walk(child);
  }
 };walk(root);
}
function apply(save=true){
 normalize();document.documentElement.lang=prefs.lang;document.documentElement.style.setProperty('--font-scale',prefs.uiLevel*UI_BASE);document.documentElement.style.setProperty('--content-scale',prefs.contentLevel*CONTENT_BASE);
 if(save)try{localStorage.setItem(KEY,JSON.stringify(prefs));}catch{}
 document.dispatchEvent(new CustomEvent('display-settings-change',{detail:{...prefs}}));
 translate();
 for(const id of ['liveEcho','aiEcho','said','voice']){const node=document.getElementById(id);if(node&&pairs.slice(0,4).some(pair=>pair.includes(node.textContent)))node.textContent=t(node.textContent);}
 document.title='VoiceMem · '+t(document.body.dataset.style==='technical'?'科技风':document.body.dataset.style==='digital'?'数字人':'首页');
 if(language)language.value=prefs.lang;if(uiScale){uiScale.value=String(Math.round(prefs.uiLevel*100));uiOutput.textContent=Math.round(prefs.uiLevel*100)+'%';contentScale.value=String(Math.round(prefs.contentLevel*100));contentOutput.textContent=Math.round(prefs.contentLevel*100)+'%';}
}
let language,uiScale,contentScale,uiOutput,contentOutput,dialog,opener;
function controls(){
 const section=document.createElement('section');section.className='display-options';
 section.innerHTML=`<div class="setting-row"><label for="systemLanguage">系统语言</label><select id="systemLanguage"><option value="zh-CN">简体中文</option><option value="en">English</option></select></div>
 <div class="setting-row"><label for="uiFont">UI 字号<small>按钮、导航与标签</small></label><output id="uiFontPercent" for="uiFont">100%</output></div>
 <input id="uiFont" class="font-range" type="range" min="50" max="200" step="5" value="100" aria-label="UI 字号">
 <div class="setting-row"><label for="contentFont">内容字号<small>ASR、记忆、回复与对话记录</small></label><output id="contentFontPercent" for="contentFont">100%</output></div>
 <input id="contentFont" class="font-range" type="range" min="50" max="200" step="5" value="100" aria-label="内容字号">
 <div class="font-preview"><small>预览</small><p class="preview-asr">识别到的语音显示在这里</p><p class="preview-memory">相关记忆显示在这里</p><p class="preview-response">回复与对话记录显示在这里</p></div>
 <button type="button" id="resetDisplay">恢复默认</button>`;
 language=section.querySelector('#systemLanguage');uiScale=section.querySelector('#uiFont');uiOutput=section.querySelector('#uiFontPercent');contentScale=section.querySelector('#contentFont');contentOutput=section.querySelector('#contentFontPercent');
 language.onchange=()=>{prefs.lang=language.value;apply();};uiScale.oninput=()=>{prefs.uiLevel=Number(uiScale.value)/100;apply();};contentScale.oninput=()=>{prefs.contentLevel=Number(contentScale.value)/100;apply();};
 section.querySelector('#resetDisplay').onclick=()=>{prefs={lang:'zh-CN',uiLevel:1,contentLevel:1,schema:2};apply();};return section;
}
const COMPONENTS=[
 {id:'input',title:'语音输入',description:'麦克风与流式识别'},
 {id:'memory',title:'记忆系统',description:'记忆检索与整理'},
 {id:'reply',title:'回复模型',description:'流式模型回复'},
 {id:'speech',title:'语音合成',description:'Breeze 本地语音'},
 {id:'pet',title:'桌宠',description:'跟随回复状态'},
];
const DEFAULT_COMPONENT_LAYOUT={
 input:{x:.03,y:.12},memory:{x:.27,y:.57},reply:{x:.51,y:.12},speech:{x:.75,y:.57},pet:{x:.99,y:.12},
};
const COMPONENT_LINKS=[['input','memory'],['memory','reply'],['reply','speech'],['speech','pet']];
function providerName(value){const labels={deepseek:'DeepSeek',qwen:'Qwen',openai:'OpenAI',local:'本地模型',breeze_mlx:'Breeze MLX',breeze_cuda:'Breeze CUDA'};return labels[value]||value||'启动配置';}
function componentLayout(){
 let saved={};try{saved=JSON.parse(localStorage.getItem(COMPONENT_KEY)||'{}')||{};}catch{}
 return Object.fromEntries(COMPONENTS.map(item=>{const value=saved[item.id]||DEFAULT_COMPONENT_LAYOUT[item.id];return[item.id,{x:Math.min(1,Math.max(0,Number(value.x)||0)),y:Math.min(1,Math.max(0,Number(value.y)||0))}];}));
}
function setupComponentBoard(board){
 let layout=componentLayout(),runtime={},selected='',loaded=false;
 board.innerHTML=`<div class="component-board-head"><div><strong>组件画板</strong><span>拖动整理，点击组件查看配置</span></div><button type="button" class="component-reset">恢复布局</button></div>
 <svg class="component-links" aria-hidden="true"><defs><marker id="componentArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 8 4 0 8Z"></path></marker></defs>${COMPONENT_LINKS.map(([a,b])=>`<path data-from="${a}" data-to="${b}" marker-end="url(#componentArrow)"></path>`).join('')}</svg>
 <div class="component-nodes">${COMPONENTS.map(item=>`<article class="component-node component-${item.id}" data-component="${item.id}"><button type="button" class="component-drag" aria-label="拖动组件" title="拖动组件"><span></span><span></span><span></span><span></span><span></span><span></span></button><button type="button" class="component-open"><span class="component-copy"><strong>${item.title}</strong><small>${item.description}</small></span><span class="component-state"><i></i><b>本机</b></span></button></article>`).join('')}</div>
 <aside class="component-editor" aria-live="polite" hidden><header><div><small>组件配置</small><h3></h3></div><button type="button" class="component-editor-close" aria-label="关闭组件配置">×</button></header><div class="component-editor-body"></div><footer><button type="button" class="component-reconfigure" hidden>重新配置 API</button><button type="button" class="component-done">完成</button></footer></aside>`;
 const nodes=new Map([...board.querySelectorAll('.component-node')].map(node=>[node.dataset.component,node]));
 const editor=board.querySelector('.component-editor');
 function saveLayout(){try{localStorage.setItem(COMPONENT_KEY,JSON.stringify(layout));}catch{}}
 function updateLinks(){
  const bounds=board.getBoundingClientRect();
  for(const path of board.querySelectorAll('.component-links path')){
   const from=nodes.get(path.dataset.from)?.getBoundingClientRect(),to=nodes.get(path.dataset.to)?.getBoundingClientRect();
   if(!from||!to||!bounds.width)continue;
   const x1=from.right-bounds.left,y1=from.top+from.height/2-bounds.top,x2=to.left-bounds.left,y2=to.top+to.height/2-bounds.top;
   const bend=Math.max(24,Math.abs(x2-x1)*.45);path.setAttribute('d',`M${x1} ${y1} C${x1+bend} ${y1},${x2-bend} ${y2},${x2} ${y2}`);
  }
 }
 function positionNodes(){
  for(const [id,node] of nodes){const x=Math.max(0,board.clientWidth-node.offsetWidth),y=Math.max(0,board.clientHeight-node.offsetHeight);node.style.left=`${layout[id].x*x}px`;node.style.top=`${layout[id].y*y}px`;}
  updateLinks();
 }
 function componentBadge(id){
  if(id==='input')return runtime.backend?.toUpperCase()||'本机';
  if(id==='memory')return providerName(runtime.memory?.provider);
  if(id==='reply')return providerName(runtime.reply?.provider);
  if(id==='speech')return providerName(runtime.speech?.provider||'breeze_mlx');
  return 'App';
 }
 function updateBadges(){for(const [id,node] of nodes)node.querySelector('.component-state b').textContent=componentBadge(id);}
 function field(label,value,secret=false){return `<div class="component-field"><span>${label}</span><strong class="${secret?'component-secret':''}">${value}</strong></div>`;}
 function renderEditor(id){
  selected=id;for(const [key,node] of nodes)node.classList.toggle('selected',key===id);
  const item=COMPONENTS.find(value=>value.id===id);if(!item)return;
  editor.querySelector('h3').textContent=item.title;
  let rows=field('运行状态','已就绪');
  if(id==='input')rows+=field('后端',runtime.backend?.toUpperCase()||'本机')+field('模式','流式输入');
  if(id==='memory')rows+=field('服务商',providerName(runtime.memory?.provider))+field('API Key',runtime.memory?.configured?'••••••••••••':'未配置',true)+field('模式',runtime.space||'Memory Space');
  if(id==='reply')rows+=field('服务商',providerName(runtime.reply?.provider))+field('API Key',runtime.reply?.provider==='local'?'不需要':runtime.reply?.configured?'••••••••••••':'未配置',true)+field('模式',runtime.mode||'llm_tts');
  if(id==='speech')rows+=field('模型',providerName(runtime.speech?.provider||'breeze_mlx'))+field('后端',runtime.backend?.toUpperCase()||'本机');
  if(id==='pet')rows+=field('管理方式','App 菜单')+field('模式','跟随回复状态');
  const api=['memory','reply'].includes(id);
  editor.querySelector('.component-editor-body').innerHTML=`${rows}<div class="component-lock"><strong>固定组件</strong><p>所有核心组件都会保留，只能调整位置。</p></div>${api?'<p class="component-reconfigure-note" hidden>退出 App 后重新运行 npm start，在终端选择服务商并输入一次 Key。VoiceMem 与 Studio 会共用这次配置。</p>':''}`;
  const configure=editor.querySelector('.component-reconfigure');configure.hidden=!api;configure.onclick=()=>{const note=editor.querySelector('.component-reconfigure-note');if(note)note.hidden=false;configure.hidden=true;};
  editor.hidden=false;board.classList.add('editing');editor.querySelector('.component-editor-close').focus();
 }
 function closeEditor(){selected='';editor.hidden=true;board.classList.remove('editing');for(const node of nodes.values())node.classList.remove('selected');}
 for(const [id,node] of nodes){
  node.querySelector('.component-open').onclick=()=>renderEditor(id);
  const handle=node.querySelector('.component-drag');let drag;
  handle.onpointerdown=event=>{if(event.button!==0)return;event.preventDefault();const rect=node.getBoundingClientRect();drag={dx:event.clientX-rect.left,dy:event.clientY-rect.top};handle.setPointerCapture(event.pointerId);node.classList.add('dragging');};
  handle.onpointermove=event=>{if(!drag)return;const rect=board.getBoundingClientRect(),maxX=Math.max(1,board.clientWidth-node.offsetWidth),maxY=Math.max(1,board.clientHeight-node.offsetHeight);const left=Math.min(maxX,Math.max(0,event.clientX-rect.left-drag.dx)),top=Math.min(maxY,Math.max(0,event.clientY-rect.top-drag.dy));layout[id]={x:left/maxX,y:top/maxY};node.style.left=`${left}px`;node.style.top=`${top}px`;updateLinks();};
  const end=event=>{if(!drag)return;drag=undefined;node.classList.remove('dragging');if(handle.hasPointerCapture(event.pointerId))handle.releasePointerCapture(event.pointerId);saveLayout();};
  handle.onpointerup=end;handle.onpointercancel=end;
  handle.onkeydown=event=>{const steps={ArrowLeft:[-.025,0],ArrowRight:[.025,0],ArrowUp:[0,-.04],ArrowDown:[0,.04]};if(!steps[event.key])return;event.preventDefault();const [x,y]=steps[event.key];layout[id]={x:Math.min(1,Math.max(0,layout[id].x+x)),y:Math.min(1,Math.max(0,layout[id].y+y))};positionNodes();saveLayout();};
 }
 board.querySelector('.component-reset').onclick=()=>{layout=JSON.parse(JSON.stringify(DEFAULT_COMPONENT_LAYOUT));saveLayout();positionNodes();};
 editor.querySelector('.component-editor-close').onclick=closeEditor;editor.querySelector('.component-done').onclick=closeEditor;
 async function refresh(){
  if(loaded)return;loaded=true;
  try{const response=await fetch('/api/components',{cache:'no-store'});if(response.ok)runtime=await response.json();}catch{}
  updateBadges();if(selected)renderEditor(selected);
 }
 const activate=()=>{requestAnimationFrame(()=>{positionNodes();void refresh();});};
 if(window.ResizeObserver)new ResizeObserver(positionNodes).observe(board);else window.addEventListener('resize',positionNodes);
 return{activate};
}
function openSettings(){
 opener=document.activeElement;
 if(!dialog){dialog=document.createElement('dialog');dialog.className='settings-page';dialog.setAttribute('aria-labelledby','settingsTitle');
 dialog.innerHTML=`<div class="settings-inner"><header class="settings-header"><h1 id="settingsTitle">设置</h1><button type="button" id="closeSettings">返回对话</button></header>
 <nav class="settings-tabs" role="tablist" aria-label="设置">
  <button role="tab" aria-selected="true" aria-controls="settingsStyle">1 · 风格</button>
  <button role="tab" aria-selected="false" aria-controls="settingsDisplay">2 · 语言与字体</button>
  <button role="tab" aria-selected="false" aria-controls="settingsHarness">3 · Harness</button>
  <button role="tab" aria-selected="false" aria-controls="settingsComponents">4 · 组件</button>
 </nav>
 <section class="settings-panel on" id="settingsStyle" role="tabpanel"><h2>选择模式</h2><div class="settings-modes">
  <a href="technical.html" class="mode-card mode-technical"><div class="mode-orb"></div><strong>科技风</strong><p>专注、清晰，尽在掌握</p></a>
  <a href="digital.html" class="mode-card mode-digital"><img src="assets/avatar.jpg" alt="银发数字人助手"><strong>数字人</strong><p>有回应，也有陪伴</p></a></div></section>
 <section class="settings-panel" id="settingsDisplay" role="tabpanel"></section>
 <section class="settings-panel settings-empty" id="settingsHarness" role="tabpanel" aria-label="Harness"></section>
 <section class="settings-panel" id="settingsComponents" role="tabpanel"><div class="component-board" aria-label="组件画板"></div></section></div>`;
 dialog.querySelector('#settingsDisplay').append(controls());document.body.append(dialog);
 const componentApi=setupComponentBoard(dialog.querySelector('.component-board'));
 const tabs=[...dialog.querySelectorAll('.settings-tabs button')],panels=[...dialog.querySelectorAll('.settings-panel')];
 function showPanel(index){tabs.forEach((tab,i)=>{const on=i===index;tab.setAttribute('aria-selected',String(on));tab.tabIndex=on?0:-1;panels[i].classList.toggle('on',on);});if(index===3)componentApi.activate();}
 tabs.forEach((tab,i)=>{tab.tabIndex=i?-1:0;tab.onclick=()=>showPanel(i);tab.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;showPanel(next);tabs[next].focus();};});
 dialog.querySelector('#closeSettings').onclick=()=>dialog.close();dialog.addEventListener('close',()=>{document.dispatchEvent(new Event('settings-close'));opener?.focus();});
 const current=dialog.querySelector('.mode-'+document.body.dataset.style);current?.setAttribute('aria-current','page');current?.addEventListener('click',e=>{e.preventDefault();dialog.close();});
 }
 apply(false);document.dispatchEvent(new Event('settings-open'));dialog.showModal();dialog.querySelector('#closeSettings').focus();
}
window.VMSettings={graphLabel(value){const labels={work:'工作',health:'健康',person:'人物',research:'研究',projects:'项目',entertainment:'娱乐',you:'你','经济':'财务'};return prefs.lang==='en'?(value==='经济'?'finance':value):labels[value]||value;},get language(){return prefs.lang;},get uiScale(){return prefs.uiLevel*UI_BASE;},get contentScale(){return prefs.contentLevel*CONTENT_BASE;},t,open:openSettings};
const btn=document.getElementById('settingsBtn');if(btn)btn.onclick=openSettings;
apply(false);
let queued=false;
if(window.MutationObserver)new MutationObserver(()=>{if(queued)return;queued=true;queueMicrotask(()=>{queued=false;translate();});}).observe(document.body,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['title','aria-label','placeholder','alt']});
window.addEventListener('storage',e=>{if(e.key===KEY){try{prefs=JSON.parse(e.newValue)||{lang:'zh-CN',uiLevel:1,contentLevel:1,schema:2};apply(false);}catch{}}});
})();
