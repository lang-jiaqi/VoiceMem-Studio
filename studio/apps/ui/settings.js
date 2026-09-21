/* Device-local display preferences, shared across the two independent pages. */
(()=>{
'use strict';
const KEY='voicemem.display',COMPONENT_KEY='voicemem.components.layout.v1',UI_BASE=1.2,CONTENT_BASE=.75;
const desktopModels=window.studioModelServices;
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
 ['保存并重启服务','Save and restart service'],['模型名称','Model'],['服务地址','Service URL'],
 ['留空则保留当前 Key','Leave blank to keep the current key'],['正在保存并重启服务…','Saving and restarting the service…'],
 ['配置已提交，正在等待服务重新就绪。','Configuration submitted. Waiting for the service to become ready.'],
 ['远程服务的模型配置需要在服务器端修改。','Configure models on the remote server.'],
 ['只允许托管的本机后端从 App 重启。','Only an app-managed local backend can be restarted here.'],
 ['拖动组件','Drag component'],['本机','Local'],['本地模型','Local model'],['启动配置','Startup config'],
 ['外观','Appearance'],['文字','Text'],['对话偏好','Conversation'],['聊天','Chat'],['声音','Voice'],['回答','Answers'],['接话','Turn taking'],['垫话','Backchannels'],
 ['Persona Prompt','Persona Prompt'],['当前','Current'],['保存','Save'],['例如：多听我说，别急着给建议。','For example: listen first and do not rush to give advice.'],
 ['选择你喜欢的对话画面。','Choose the conversation view you prefer.'],['调整语言和字号。','Adjust language and text size.'],['查看运行状态或调整布局。','View runtime status or adjust the layout.'],
 ['聊天节奏','Chat rhythm'],['说话速度','Speaking speed'],['语气','Tone'],['回答长短','Answer length'],['思考时间','Thinking time'],
 ['附和频率','Acknowledgement frequency'],['等待反馈','Waiting feedback'],['当前对话','Current conversation'],
 ['前 6 秒合计最多附和 2 次。','At most 2 acknowledgements in the first 6 seconds.'],
 ['这些设置只在当前对话中生效。你在聊天里说“慢一点”时，这里也会同步。','These settings apply to this conversation. If you say “speak slower,” the control here updates too.'],
 ['正在连接 Studio…','Connecting to Studio…'],['连接后即可调整当前对话。','Connect to adjust the current conversation.'],
 ['已同步','Synced'],['正在保存…','Saving…'],['等你在对话里再确认一次','Waiting for your confirmation in chat'],
 ['默认陪伴方式','Default'],['多倾听、少建议','Listen more, advise less'],['主动推进和给下一步','Actively suggest next steps'],['专业克制','Professional and restrained'],
 ['很慢','Very slow'],['稍慢','Slower'],['正常','Normal'],['稍快','Faster'],['很快','Very fast'],
 ['自动','Auto'],['简短','Short'],['详细','Detailed'],['自动判断','Automatic'],['优先快速','Prefer speed'],['优先深思','Prefer deeper thought'],
 ['默认附和频率','Default frequency'],['不插话','Do not interject'],['少附和','Fewer acknowledgements'],['多附和','More acknowledgements'],['默认等待反馈','Default feedback'],['安静等待','Wait silently'],['更常说明正在思考','More thinking updates'],
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
let harnessClient=null,harnessMessage=null,harnessApi=null;
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
const HARNESS_GROUPS=[
 {title:'声音',fields:[['speaking_style','speech_rate','说话速度','slider'],['speaking_style','tone','语气','choice']]},
 {title:'回答',fields:[['speaking_style','reply_length','回答长短','slider'],['reply_modes','reasoning_depth','思考时间','slider']]},
 {id:'backchannel',title:'垫话',fields:[['turn_taking','backchannel','附和频率','curve'],['turn_taking','work_filler','等待反馈','slider']]},
];
const HARNESS_ORDER={
 'turn_taking.backchannel':['off','less','auto','more'],
 'turn_taking.work_filler':['silent','auto','reassuring'],
};
function setupHarnessPanel(panel){
 let requested=false;
 const status=panel.querySelector('.harness-status');
 const body=panel.querySelector('.harness-controls');
 function label(spec,value){return spec?.labels?.[value]||value||'';}
 function curveGeometry(values,spec){
  const xs=[58,220,382,544],top=22,bottom=142,height=bottom-top;
  const points=values.map((value,index)=>({x:xs[index],y:bottom-(value-spec.min)/(spec.max-spec.min)*height}));
  let path=`M${points[0].x} ${points[0].y}`;
  for(let i=1;i<points.length;i++){const previous=points[i-1],point=points[i],mid=(previous.x+point.x)/2;path+=` C${mid} ${previous.y},${mid} ${point.y},${point.x} ${point.y}`;}
  return{points,path,area:`${path} L${points.at(-1).x} ${bottom} L${points[0].x} ${bottom} Z`,top,bottom};
 }
 function render(){
  const schema=harnessMessage?.schema,snapshot=harnessMessage?.snapshot;
  if(!schema||!snapshot?.profile){
   status.textContent=harnessClient?'正在连接 Studio…':'连接后即可调整当前对话。';body.innerHTML='';return;
  }
  status.textContent=harnessMessage.error||'已同步';
  const persona=schema.persona?.interaction_style,promptSpec=harnessMessage.prompt_schema?.persona,curveSpec=harnessMessage.backchannel_curve_schema;
  const personaValue=snapshot.profile.persona?.interaction_style??persona?.default;
  const personaPrompt=snapshot.prompts?.persona||'';
  const promptCard=promptSpec?`<section class="harness-group"><h3>Persona</h3><article class="harness-prompt"><header><span>Persona Prompt</span><small><span>当前</span>：<b>${escapeHtml(label(persona,personaValue))}</b></small></header><textarea maxlength="${promptSpec.max_length}" placeholder="例如：多听我说，别急着给建议。">${escapeHtml(personaPrompt)}</textarea><footer><span class="harness-count">${personaPrompt.length}/${promptSpec.max_length}</span><button type="button" class="harness-prompt-reset">恢复默认</button><button type="button" class="harness-prompt-save">保存</button></footer></article></section>`:'';
  body.innerHTML=promptCard+HARNESS_GROUPS.map(group=>`<section class="harness-group${group.id?` harness-group-${group.id}`:''}"><h3>${group.title}</h3><div class="harness-grid">${group.fields.map(([domain,name,title,kind])=>{
   const spec=schema[domain]?.[name];if(!spec)return'';
   const path=`${domain}.${name}`,values=HARNESS_ORDER[path]||spec.values,current=snapshot.profile[domain]?.[name]??spec.default,index=Math.max(0,values.indexOf(current));
   const pending=snapshot.pending?.[path];
   if(kind==='curve'&&curveSpec){
    const quotas=Array.isArray(snapshot.backchannel_curve)?snapshot.backchannel_curve:curveSpec.profiles[current]||curveSpec.profiles.auto,geometry=curveGeometry(quotas,curveSpec);
    const baseline=geometry.bottom-(1.5-curveSpec.min)/(curveSpec.max-curveSpec.min)*(geometry.bottom-geometry.top);
    return `<article class="backchannel-curve" data-values="${escapeHtml(JSON.stringify(quotas))}"><header><strong>${title}</strong><button type="button" class="harness-reset curve-reset" title="恢复默认" aria-label="${title} 恢复默认">↶</button></header><div class="curve-chart"><svg viewBox="0 0 600 190" role="img" aria-label="各阶段附和次数"><line class="curve-baseline" x1="28" y1="${baseline}" x2="574" y2="${baseline}"/>${geometry.points.map(point=>`<line class="curve-guide" x1="${point.x}" y1="14" x2="${point.x}" y2="150"/>`).join('')}<path class="curve-area" d="${geometry.area}"/><path class="curve-line" d="${geometry.path}"/>${geometry.points.map((point,i)=>`<circle class="curve-point point-${i}" cx="${point.x}" cy="${point.y}" r="10"/>`).join('')}</svg>${quotas.map((quota,i)=>`<input class="curve-input" style="--curve-x:${geometry.points[i].x/6}%" type="range" min="${curveSpec.min}" max="${curveSpec.max}" step="${curveSpec.step}" value="${quota}" aria-label="${curveSpec.phases[i].label} 附和 ${quota.toFixed(1)} 次"><span class="curve-label" style="--curve-x:${geometry.points[i].x/6}%"><b>${quota.toFixed(1)}</b><small>${curveSpec.phases[i].label}</small></span>`).join('')}</div><p>前 6 秒合计最多附和 2 次。</p></article>`;
   }
   if(kind==='choice')return `<article class="harness-control harness-choice" data-domain="${domain}" data-name="${name}"><header><span>${title}</span><button type="button" class="harness-reset" title="恢复默认" aria-label="${title} 恢复默认" data-default="${escapeHtml(spec.default)}">↶</button></header><strong class="harness-value">${escapeHtml(label(spec,current))}</strong><select aria-label="${title}">${values.map(item=>`<option value="${escapeHtml(item)}" ${item===current?'selected':''}>${escapeHtml(label(spec,item))}</option>`).join('')}</select><small class="harness-pending">${pending?'等你在对话里再确认一次':''}</small></article>`;
   const dots=values.map((_,i)=>`<i class="${i<=index?'on':''}"></i>`).join('');
   return `<article class="harness-control harness-slider" data-domain="${domain}" data-name="${name}" data-values="${escapeHtml(JSON.stringify(values))}">
    <header><span>${title}</span><button type="button" class="harness-reset" title="恢复默认" aria-label="${title} 恢复默认" data-default="${escapeHtml(spec.default)}">↶</button></header>
    <strong class="harness-value">${escapeHtml(label(spec,current))}</strong>
    <div class="harness-range-wrap"><div class="harness-track"><span style="width:${values.length>1?index/(values.length-1)*100:0}%"></span><div class="harness-dots">${dots}</div></div><input type="range" min="0" max="${values.length-1}" step="1" value="${index}" aria-label="${title}" aria-valuetext="${escapeHtml(label(spec,current))}"></div>
    <small class="harness-pending">${pending?'等你在对话里再确认一次':''}</small>
   </article>`;
  }).join('')}</div></section>`).join('');
  const prompt=body.querySelector('.harness-prompt');
  if(prompt){const textarea=prompt.querySelector('textarea'),count=prompt.querySelector('.harness-count');textarea.oninput=()=>{count.textContent=`${textarea.value.length}/${textarea.maxLength}`;};prompt.querySelector('.harness-prompt-save').onclick=async()=>{status.textContent='正在保存…';try{await harnessClient?.setHarnessPrompt('persona',textarea.value);}catch(error){status.textContent=error.message;}};prompt.querySelector('.harness-prompt-reset').onclick=()=>{textarea.value='';textarea.oninput();void harnessClient?.setHarnessPrompt('persona','');};}
  for(const card of body.querySelectorAll('.harness-slider')){
   const input=card.querySelector('input'),value=card.querySelector('.harness-value'),fill=card.querySelector('.harness-track span'),dots=[...card.querySelectorAll('.harness-dots i')];
   const values=JSON.parse(card.dataset.values),spec=schema[card.dataset.domain][card.dataset.name];
   const paint=()=>{const index=Number(input.value),selected=values[index];value.textContent=t(label(spec,selected));input.setAttribute('aria-valuetext',label(spec,selected));fill.style.width=`${values.length>1?index/(values.length-1)*100:0}%`;dots.forEach((dot,i)=>dot.classList.toggle('on',i<=index));};
   const save=async selected=>{if(!harnessClient)return;status.textContent='正在保存…';try{await harnessClient.setHarness({[card.dataset.domain]:{[card.dataset.name]:selected}});}catch(error){status.textContent=error.message;}};
   input.oninput=paint;input.onchange=()=>void save(values[Number(input.value)]);
   card.querySelector('.harness-reset').onclick=()=>{input.value=String(Math.max(0,values.indexOf(spec.default)));paint();void save(spec.default);};
  }
  for(const card of body.querySelectorAll('.harness-choice')){
   const select=card.querySelector('select'),value=card.querySelector('.harness-value'),spec=schema[card.dataset.domain][card.dataset.name];
   const save=async selected=>{if(!harnessClient)return;status.textContent='正在保存…';try{await harnessClient.setHarness({[card.dataset.domain]:{[card.dataset.name]:selected}});}catch(error){status.textContent=error.message;}};
   select.onchange=()=>{value.textContent=t(label(spec,select.value));void save(select.value);};
   card.querySelector('.harness-reset').onclick=()=>{select.value=spec.default;value.textContent=t(label(spec,spec.default));void save(spec.default);};
  }
  const curve=body.querySelector('.backchannel-curve');
  if(curve){const inputs=[...curve.querySelectorAll('.curve-input')],labels=[...curve.querySelectorAll('.curve-label b')],svg=curve.querySelector('svg');
   const paint=()=>{const quotas=inputs.map(input=>Number(input.value)),geometry=curveGeometry(quotas,curveSpec);svg.querySelector('.curve-line').setAttribute('d',geometry.path);svg.querySelector('.curve-area').setAttribute('d',geometry.area);geometry.points.forEach((point,i)=>{svg.querySelector(`.point-${i}`).setAttribute('cy',point.y);labels[i].textContent=quotas[i].toFixed(1);inputs[i].setAttribute('aria-label',`${curveSpec.phases[i].label} 附和 ${quotas[i].toFixed(1)} 次`);});return quotas;};
   const save=async quotas=>{if(!harnessClient)return;status.textContent='正在保存…';try{await harnessClient.setBackchannelCurve(quotas);}catch(error){status.textContent=error.message;}};
   inputs.forEach(input=>{input.oninput=paint;input.onchange=()=>void save(paint());});
   curve.querySelector('.curve-reset').onclick=()=>{status.textContent='正在保存…';void harnessClient?.setBackchannelCurve(null);};
  }
  translate(panel);
 }
 function activate(){
  render();if(!harnessClient)return;
  requested=true;void harnessClient.getHarness().catch(error=>{status.textContent=error.message;});
 }
 return{activate,render,get requested(){return requested;}};
}
function bindHarness(client){harnessClient=client;if(harnessApi?.requested)harnessApi.activate();else harnessApi?.render();}
document.addEventListener('self-harness-state',event=>{harnessMessage=event.detail;harnessApi?.render();});
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
const MODEL_DEFAULTS={deepseek:{model:'deepseek-v4-flash',baseUrl:'https://api.deepseek.com'},qwen:{model:'qwen3.6-flash',baseUrl:'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'},openai:{model:'gpt-4o',baseUrl:'https://api.openai.com/v1'},local:{model:'mlx-community/Qwen3.5-4B-4bit',baseUrl:''}};
function providerName(value){const labels={deepseek:'DeepSeek',qwen:'Qwen',openai:'OpenAI',local:'本地模型',breeze_mlx:'Breeze MLX',breeze_cuda:'Breeze CUDA'};return labels[value]||value||'启动配置';}
function escapeHtml(value){return String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));}
function componentLayout(){
 let saved={};try{saved=JSON.parse(localStorage.getItem(COMPONENT_KEY)||'{}')||{};}catch{}
 return Object.fromEntries(COMPONENTS.map(item=>{const value=saved[item.id]||DEFAULT_COMPONENT_LAYOUT[item.id];return[item.id,{x:Math.min(1,Math.max(0,Number(value.x)||0)),y:Math.min(1,Math.max(0,Number(value.y)||0))}];}));
}
function setupComponentBoard(board){
 let layout=componentLayout(),runtime={},manager={managed:false,services:null},selected='',loaded=false;
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
 function field(label,value,secret=false){return `<div class="component-field"><span>${label}</span><strong class="${secret?'component-secret':''}">${escapeHtml(value)}</strong></div>`;}
 function serviceForm(id){
  if(!manager.managed)return `<div class="component-reconfigure-note"><strong>远程服务的模型配置需要在服务器端修改。</strong><br>只允许托管的本机后端从 App 重启。</div>`;
  const service=manager.services?.[id]||runtime[id]||{},local=id==='reply'&&service.provider==='local';
  const providers=[['deepseek','DeepSeek'],['qwen','Qwen / DashScope'],['openai','OpenAI-compatible'],...(id==='reply'?[['local','本地模型 / MLX']]:[])];
  return `<form class="component-service-form" data-role="${id}">
   <label>服务商<select name="provider">${providers.map(([value,label])=>`<option value="${value}"${service.provider===value?' selected':''}>${label}</option>`).join('')}</select></label>
   <label>模型名称<input name="model" value="${escapeHtml(service.model||'')}" maxlength="300" required${local?' disabled':''}></label>
   <label>服务地址<input name="baseUrl" type="url" value="${escapeHtml(service.baseUrl||'')}" spellcheck="false"${local?' disabled':' required'}></label>
   <label>API Key<input name="apiKey" type="password" value="" autocomplete="new-password" placeholder="留空则保留当前 Key"${local?' disabled':''}></label>
   <p class="component-service-status" role="status" aria-live="polite"></p>
   <button class="component-service-save" type="submit">保存并重启服务</button>
  </form>`;
 }
 function bindServiceForm(id){
  const form=editor.querySelector('.component-service-form');if(!form)return;
  const provider=form.elements.provider,model=form.elements.model,base=form.elements.baseUrl,key=form.elements.apiKey,status=form.querySelector('.component-service-status'),save=form.querySelector('.component-service-save');
  provider.onchange=()=>{const value=MODEL_DEFAULTS[provider.value],local=provider.value==='local';model.value=value.model;base.value=value.baseUrl;for(const input of [model,base,key])input.disabled=local;base.required=!local;model.required=!local;};
  form.onsubmit=async event=>{event.preventDefault();save.disabled=true;status.textContent='正在保存并重启服务…';
   try{await desktopModels.update(id,{provider:provider.value,model:model.value,baseUrl:base.value,apiKey:key.value});status.textContent='配置已提交，正在等待服务重新就绪。';}
   catch(error){status.textContent=error.message;save.disabled=false;}
  };
 }
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
  editor.querySelector('.component-editor-body').innerHTML=`${rows}${api?serviceForm(id):''}<div class="component-lock"><strong>固定组件</strong><p>所有核心组件都会保留，只能调整位置。</p></div>`;
  const configure=editor.querySelector('.component-reconfigure');configure.hidden=true;
  if(api)bindServiceForm(id);
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
  if(desktopModels)try{manager=await desktopModels.state();}catch{}
  updateBadges();if(selected)renderEditor(selected);
 }
 const activate=()=>{requestAnimationFrame(()=>{positionNodes();void refresh();});};
 if(window.ResizeObserver)new ResizeObserver(positionNodes).observe(board);else window.addEventListener('resize',positionNodes);
 return{activate};
}
function openSettings(){
 opener=document.activeElement;
 if(!dialog){dialog=document.createElement('dialog');dialog.className='settings-page';dialog.setAttribute('aria-labelledby','settingsTitle');
 dialog.innerHTML=`<div class="settings-shell"><aside class="settings-sidebar"><div class="settings-brand"><span class="settings-logo">V</span><strong id="settingsTitle">设置</strong></div>
 <nav class="settings-tabs" role="tablist" aria-label="设置">
  <button role="tab" aria-selected="true" aria-controls="settingsStyle"><svg viewBox="0 0 24 24"><path d="M12 3a9 9 0 1 0 9 9c0-1.1-.9-2-2-2h-1.6a2 2 0 0 1-1.8-2.8l.6-1.4A2 2 0 0 0 14.4 3H12Z"/><circle cx="7.5" cy="11.5" r="1"/><circle cx="10" cy="7.5" r="1"/><circle cx="8.5" cy="16" r="1"/></svg><span>外观</span></button>
  <button role="tab" aria-selected="false" aria-controls="settingsDisplay"><svg viewBox="0 0 24 24"><path d="M4 19 10.5 5h3L20 19M7 14h10"/></svg><span>文字</span></button>
  <button role="tab" aria-selected="false" aria-controls="settingsHarness"><svg viewBox="0 0 24 24"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/></svg><span>对话偏好</span></button>
  <button role="tab" aria-selected="false" aria-controls="settingsComponents"><svg viewBox="0 0 24 24"><rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/></svg><span>组件</span></button>
 </nav></aside>
 <main class="settings-main"><header class="settings-header"><span class="settings-current">外观</span><button type="button" id="closeSettings" aria-label="返回对话" title="返回对话">×</button></header><div class="settings-scroll">
 <section class="settings-panel on" id="settingsStyle" role="tabpanel"><div class="settings-panel-copy"><h2>外观</h2><p>选择你喜欢的对话画面。</p></div><div class="settings-modes">
  <a href="technical.html?v=20260921-4" class="mode-card mode-technical"><div class="mode-orb"></div><strong>科技风</strong><p>专注、清晰，尽在掌握</p></a>
  <a href="digital.html?v=20260921-4" class="mode-card mode-digital"><img src="assets/avatar.jpg" alt="银发数字人助手"><strong>数字人</strong><p>有回应，也有陪伴</p></a></div></section>
 <section class="settings-panel" id="settingsDisplay" role="tabpanel"><div class="settings-panel-copy"><h2>文字</h2><p>调整语言和字号。</p></div><div class="settings-panel-body"></div></section>
 <section class="settings-panel" id="settingsHarness" role="tabpanel"><div class="settings-panel-copy"><h2>对话偏好</h2><p>这些设置只在当前对话中生效。你在聊天里说“慢一点”时，这里也会同步。</p><span class="harness-status" aria-live="polite">正在连接 Studio…</span></div><div class="harness-controls"></div></section>
 <section class="settings-panel" id="settingsComponents" role="tabpanel"><div class="settings-panel-copy"><h2>组件</h2><p>查看运行状态或调整布局。</p></div><div class="component-board" aria-label="组件画板"></div></section>
 </div></main></div>`;
 dialog.querySelector('#settingsDisplay .settings-panel-body').append(controls());document.body.append(dialog);
 const componentApi=setupComponentBoard(dialog.querySelector('.component-board'));
 harnessApi=setupHarnessPanel(dialog.querySelector('#settingsHarness'));
 const tabs=[...dialog.querySelectorAll('.settings-tabs button')],panels=[...dialog.querySelectorAll('.settings-panel')];
 function showPanel(index){tabs.forEach((tab,i)=>{const on=i===index;tab.setAttribute('aria-selected',String(on));tab.tabIndex=on?0:-1;panels[i].classList.toggle('on',on);});dialog.querySelector('.settings-current').textContent=tabs[index].querySelector('span').textContent;if(index===2)harnessApi.activate();if(index===3)componentApi.activate();}
 tabs.forEach((tab,i)=>{tab.tabIndex=i?-1:0;tab.onclick=()=>showPanel(i);tab.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;showPanel(next);tabs[next].focus();};});
 dialog.querySelector('#closeSettings').onclick=()=>dialog.close();dialog.addEventListener('close',()=>{document.dispatchEvent(new Event('settings-close'));opener?.focus();});
 const current=dialog.querySelector('.mode-'+document.body.dataset.style);current?.setAttribute('aria-current','page');current?.addEventListener('click',e=>{e.preventDefault();dialog.close();});
 }
 apply(false);document.dispatchEvent(new Event('settings-open'));dialog.showModal();dialog.querySelector('#closeSettings').focus();
}
window.VMSettings={graphLabel(value){const labels={work:'工作',health:'健康',person:'人物',research:'研究',projects:'项目',entertainment:'娱乐',you:'你','经济':'财务'};return prefs.lang==='en'?(value==='经济'?'finance':value):labels[value]||value;},get language(){return prefs.lang;},get uiScale(){return prefs.uiLevel*UI_BASE;},get contentScale(){return prefs.contentLevel*CONTENT_BASE;},t,bindHarness,open:openSettings};
const btn=document.getElementById('settingsBtn');if(btn)btn.onclick=openSettings;
apply(false);
let queued=false;
if(window.MutationObserver)new MutationObserver(()=>{if(queued)return;queued=true;queueMicrotask(()=>{queued=false;translate();});}).observe(document.body,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['title','aria-label','placeholder','alt']});
window.addEventListener('storage',e=>{if(e.key===KEY){try{prefs=JSON.parse(e.newValue)||{lang:'zh-CN',uiLevel:1,contentLevel:1,schema:2};apply(false);}catch{}}});
})();
