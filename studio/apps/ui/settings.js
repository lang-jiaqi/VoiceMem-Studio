/* Device-local display preferences, shared across the two independent pages. */
(()=>{
'use strict';
const KEY='voicemem.display',UI_BASE=1.2,CONTENT_BASE=.75;
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
['关闭','Close'],['读取中…','Loading…'],['事实','Facts'],['画像','Portraits'],['读取失败，请关闭后重试。','Could not load memories. Close and try again.'],
['脑部记忆图谱：记忆域导航示意图','Brain memory graph: memory domains'],
['记忆域导航示意图','Memory domains'],
['启动时指定了单个记忆库，当前对话固定使用此空间。','A single memory directory was specified at startup. This conversation uses its fixed Space.'],
['等待你的下一句话…','Waiting for you…'],['回答会同步显示在这里。','Responses will appear here.'],['说点什么，她在听。','Say something. I’m listening.'],['我在。今天想从哪里说起？','I’m here. Where would you like to start?'],['还没有记录。说第一句话，它会出现在这里。','No history yet. Your first message will appear here.'],['说点什么，VoiceMem 会先去记忆里找相关的内容，再回答。','Say something. VoiceMem will retrieve relevant memories before responding.'],['你','You'],['人物','People'],['研究','Research'],['项目','Projects'],['人格','Personality'],['全部记忆','All memories'],['娱乐','Entertainment'],['设置','Settings'],['显示设置','Display settings'],['返回对话','Back to conversation'],['选择模式','Choose your mode'],['科技风','Technical'],['数字人','Digital human'],['专注、清晰，尽在掌握','Focus, clarity, and control'],['有回应，也有陪伴','A voice and a companion'],['界面语言','Interface language'],['字体大小','Font size'],['界面字号','UI font size'],['内容字号','Content font size'],['按钮、导航与标签','Buttons, navigation, and labels'],['ASR、记忆、回复与对话记录','ASR, memories, responses, and conversations'],['恢复默认','Reset to default'],['当前模式','Current mode'],['预览','Preview'],['记忆','Memory'],['识别到的语音显示在这里','Your recognized speech appears here'],['相关记忆显示在这里','Relevant memories appear here'],['回复与对话记录显示在这里','Responses and conversation history appear here'],
['首页','Home'],['返回首页','Back to home'],['科技风 ↗','Technical ↗'],['数字人 ↗','Digital human ↗'],['新对话','New chat'],['置顶','Pinned'],['聊天记录','Chat history'],['对话记录','Chat'],['记忆空间','Memory Space'],['对话','Chat'],['模型回复','Response'],['本轮感知','Live input'],['本轮感知 · ASR','Live input · ASR'],['开始对话','Start chat'],['结束对话','End chat'],['开始说话','Start speaking'],['停止说话','Stop speaking'],['正在聆听…','Listening…'],['语音输入','Voice input'],['发送','Send'],['倾听','Listening'],['说话','Speaking'],['短思考','Quick thinking'],['长思考','Deep thinking'],['待机','Idle'],['演示 · 倾听','Demo · Listening'],['演示 · 说话','Demo · Speaking'],['演示 · 短思考','Demo · Quick thinking'],['演示 · 长思考','Demo · Deep thinking'],['当前浏览器使用静态球体预览','This browser uses a static orb preview'],
['输入这一轮想说的话…','Message…'],['本轮文字输入','Message input'],['说点什么…','Say something…'],['对 Echo 说','Talk to Echo'],['待识别','Not identified'],['未匹配','No match'],['情感','Emotion'],['情绪','Emotion'],['实体','Entities'],['簇','Clusters'],['说话人','Speaker'],['事实与经历','Facts and experiences'],['偏好','Preference'],['性格','Personality'],['情绪与人格','Emotion and personality'],['记忆节点','Memory nodes'],['点击节点查看记忆域','Select a node to explore memories'],['点击任一记忆域，Top-K 会切换到该域','Select a memory domain to filter Top-K'],['全部记忆域 · 按相似度排序','All memories · Relevance'],['引用的记忆','Referenced memory'],['已与你的记忆同步','Synced with your memory'],['暂无置顶对话','No pinned conversations'],['暂无聊天记录','No conversations yet'],['已复制','Copied'],['复制','Copy'],['复制回复','Copy response'],['重播原声','Replay original voice'],['重播原声回复','Replay original response voice'],['停止重播','Stop replay'],['今天','Today'],['与 Echo 的对话','Conversation with Echo'],['深湖','Deep Lake'],['记忆空间 · 深湖','Memory Space · Deep Lake'],['你说过的话在这里沉下去，彼此连成一片。','Your words settle here and connect into memories.'],
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
 ['流式识别与最终转写','Streaming and final transcription'],['回复与附和共用音色','One voice for replies and acknowledgements'],
 ['轮次控制','Turn control'],['语音检测、结束判断与思考路由','Speech detection, turn endings and reasoning'],
 ['流式识别','Streaming recognition'],['最终转写','Final transcription'],['语音检测','Speech detection'],['结束判断','End-of-turn detection'],['思考路由','Reasoning router'],
 ['待应用','Pending'],['项待应用','pending changes'],['撤销更改','Discard changes'],['应用并重启','Apply and restart'],['正在重启…','Restarting…'],
 ['接入方式','Provider'],['使用推荐链路','Using the recommended pipeline'],['恢复推荐配置','Restore recommended settings'],['高级','Advanced'],
 ['自定义适配器','Custom adapter'],['Python 适配器','Python adapter'],['适配器参数','Adapter options'],['音色 ID','Voice ID'],['识别语言','Recognition language'],
 ['本地模型路径','Local model path'],['模型目录或 Hugging Face ID','Model directory or Hugging Face ID'],['已保存，留空保留','Saved; leave blank to keep'],['请输入 API Key','Enter API key'],
 ['可选','Optional'],['16 kHz 音频 · 累计转写 · 独立会话','16 kHz audio · Cumulative text · Independent sessions'],['16 kHz 音频 → 完整文本','16 kHz audio → Complete transcript'],
 ['16 kHz 音频 → 是否说话 · 独立会话','16 kHz audio → Speech detection · Independent sessions'],['16 kHz 音频 → 结束概率','16 kHz audio → End-of-turn probability'],
 ['思考深度 · 垫话生成 · 可取消','Reasoning depth · Spoken filler · Cancellable'],['流式 PCM16 · 单声道 24 kHz · 可取消','Streaming PCM16 · Mono 24 kHz · Cancellable'],
 ['结构化记忆提取 · OpenAI Chat 协议','Structured memory extraction · OpenAI Chat protocol'],['流式文本 · 对话历史 · 可取消','Streaming text · Conversation history · Cancellable'],
 ['首次启用此音色会合成 2 段附和音频。','This voice generates 2 acknowledgement clips on first use.'],
 ['此服务由外部管理，请在服务器端修改配置。','This service is managed externally. Configure it on the server.'],
 ['此服务版本不支持组件配置。','This server version does not support component settings.'],['读取失败，请重试。','Could not load. Please retry.'],['重试','Retry'],
 ['X-ASR · 推荐','X-ASR · Recommended'],['SenseVoice · 推荐','SenseVoice · Recommended'],['Silero VAD · 推荐','Silero VAD · Recommended'],['Smart Turn · 推荐','Smart Turn · Recommended'],['Qwen3 0.6B · 推荐','Qwen3 0.6B · Recommended'],['Breeze · 推荐','Breeze · Recommended'],['DeepSeek · 推荐','DeepSeek · Recommended'],
 ['外观','Appearance'],['文字','Text'],['对话偏好','Conversation'],['聊天','Chat'],['声音','Voice'],['回答','Answers'],['接话','Turn taking'],['垫话','Backchannels'],
 ['返回组件','Back to components'],['角色设定','Persona prompt'],['角色','Persona'],['选择记忆空间','Choose a Memory Space'],['创建对话','Create conversation'],['当前','Current'],['保存','Save'],['例如：多听我说，别急着给建议。','For example: listen first and do not rush to give advice.'],
 ['选择你喜欢的对话画面。','Choose the conversation view you prefer.'],['调整语言和字号。','Adjust language and text size.'],['查看运行状态或调整布局。','View runtime status or adjust the layout.'],
 ['聊天节奏','Chat rhythm'],['说话速度','Speaking speed'],['语气','Tone'],['回答长度','Answer length'],['思考深度','Thinking depth'],
 ['附和频率','Acknowledgements'],['等待反馈','Thinking updates'],['当前对话','Current conversation'],
 ['温和','Warm'],['共情','Empathetic'],['认真','Serious'],['鼓励','Encouraging'],['俏皮','Playful'],['抱歉','Apologetic'],
 ['VoiceMem · 本轮记忆','VoiceMem · This turn'],['本轮召回','Retrieved'],['新入库','Saved'],['等待对话','No messages yet'],
 ['详情','Details'],['领域','Domain'],['同一空间的对话共享长期记忆。','Conversations in the same Space share memory.'],
 ['新空间名称','New Space name'],['创建空间','Create Space'],['取消','Cancel'],
 ['正在等待入库…','Waiting for memory updates…'],['本轮暂无新入库记忆','No new memories this turn'],
 ['本轮入库未完成','Memory update incomplete'],['正在更新记忆…','Updating memories…'],['检索中…','Retrieving…'],['本轮没有召回相关记忆','No relevant memories this turn'],
 ['前 6 秒合计最多附和 2 次。','At most 2 acknowledgements in the first 6 seconds.'],
 ['这些设置只在当前对话中生效。你在聊天里说“慢一点”时，这里也会同步。','These settings apply to this conversation. If you say “speak slower,” the control here updates too.'],
 ['正在连接 Studio…','Connecting to Studio…'],['连接后即可调整当前对话。','Connect to adjust the current conversation.'],
 ['主模型对话','Main model'],['未保存','Unsaved'],['已保存','Saved'],['已保存，下轮生效','Saved · Next turn'],['保存 Prompt 修改？','Save prompt changes?'],['继续编辑','Keep editing'],['不保存','Discard'],['连接后即可编辑 Prompt。','Connect to edit prompts.'],['基础语言','Default language'],['陪伴方式','Interaction style'],['检索记忆…','Retrieving memories…'],['已找到相关记忆','Relevant memories found'],['本轮无相关记忆','No relevant memories'],['DeepSeek 正在思考…','DeepSeek is thinking…'],['DeepSeek 正在生成…','DeepSeek is generating…'],['正在生成回复…','Generating a reply…'],['正在说话…','Speaking…'],
 ['已同步','Synced'],['正在保存…','Saving…'],['等你在对话里再确认一次','Waiting for your confirmation in chat'],
 ['默认陪伴方式','Default'],['多倾听、少建议','Listen first'],['主动推进和给下一步','Suggest next steps'],['专业克制','Professional'],
 ['很慢','Very slow'],['稍慢','Slower'],['正常','Normal'],['稍快','Faster'],['很快','Very fast'],
 ['自动','Auto'],['简短','Short'],['详细','Detailed'],['自动判断','Automatic'],['优先快速','Quick'],['优先深思','Deep'],
 ['默认附和频率','Default frequency'],['不插话','Do not interject'],['少附和','Fewer acknowledgements'],['多附和','More acknowledgements'],['默认等待反馈','Default updates'],['安静等待','Silent'],['更常说明正在思考','More updates'],
 ].forEach(([a,b])=>{en.set(a,b);zh.set(b,a);});
// Existing English chrome gets a Chinese equivalent too.
zh.set('Technical','科技风');zh.set('Chat','对话记录');zh.set('info','事实记忆');zh.set('emo&persona','情绪与人格');zh.set('Speaker ID','说话人 ID');
en.set('info','Facts');en.set('emo&persona','Traits');
function t(value){
 const dict=prefs.lang==='en'?en:zh;if(dict.has(value))return dict.get(value);
 if(prefs.lang==='en'){
  const count=value.match(/^(\d+) 条记忆$/);if(count)return `${count[1]} ${count[1]==='1'?'memory':'memories'}`;
  const retrieved=value.match(/^本轮召回 (\d+) 条相关记忆$/);if(retrieved)return `${retrieved[1]} ${retrieved[1]==='1'?'memory':'memories'} retrieved`;
  const recall=value.match(/^(.+?) · 本轮召回$/);if(recall)return `${t(recall[1])} · Retrieved`;
  const domain=value.match(/^(.+?) 记忆域 · 按相似度排序$/);if(domain)return t(domain[1])+' memory domain · Ranked by similarity';
  const score=value.match(/^(\w+ · )(.+?)( · 相似度 )(.*)$/);if(score)return score[1]+t(score[2])+' · Similarity '+score[4];
  if(value.startsWith('置顶对话：'))return 'Pin conversation: '+value.slice(5);
  if(value.startsWith('取消置顶：'))return 'Unpin conversation: '+value.slice(5);
 }
 return value;
}
en.set('事实记忆','info');en.set('说话人 ID','Speaker ID');
const textCache=new WeakMap(),attrCache=new WeakMap();
const skip='script,style,textarea,input,[data-no-i18n],.msg-body,.turn p,.turn .turn-body,.conv-title,.conv-select,.item,#liveEcho,#liveEchoPrev,#aiEcho,#said,#voice';
function translate(root=document.body){
 const walk=node=>{
  if(node.nodeType===3){const parent=node.parentElement;if(!parent||(parent.closest(skip)&&!parent.closest('[data-ui-placeholder]')))return;
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
 document.title='VoiceMem Studio · '+t(document.body.dataset.style==='technical'?'科技风':document.body.dataset.style==='digital'?'数字人':'首页');
 if(language)language.value=prefs.lang;if(uiScale){uiScale.value=String(Math.round(prefs.uiLevel*100));uiOutput.textContent=Math.round(prefs.uiLevel*100)+'%';contentScale.value=String(Math.round(prefs.contentLevel*100));contentOutput.textContent=Math.round(prefs.contentLevel*100)+'%';}
}
let language,uiScale,contentScale,uiOutput,contentOutput,dialog,opener;
let harnessClient=null,harnessMessage=null,harnessApi=null,promptApi=null;
function controls(){
 const section=document.createElement('section');section.className='display-options';
 section.innerHTML=`<div class="setting-row"><label for="systemLanguage">界面语言</label><select id="systemLanguage"><option value="zh-CN">简体中文</option><option value="en">English</option></select></div>
 <div class="setting-row"><label for="uiFont">界面字号<small>按钮、导航与标签</small></label><output id="uiFontPercent" for="uiFont">100%</output></div>
 <input id="uiFont" class="font-range" type="range" min="50" max="200" step="5" value="100" aria-label="界面字号">
 <div class="setting-row"><label for="contentFont">内容字号<small>ASR、记忆、回复与对话记录</small></label><output id="contentFontPercent" for="contentFont">100%</output></div>
 <input id="contentFont" class="font-range" type="range" min="50" max="200" step="5" value="100" aria-label="内容字号">
 <div class="font-preview"><small>预览</small><p class="preview-asr">识别到的语音显示在这里</p><p class="preview-memory">相关记忆显示在这里</p><p class="preview-response">回复与对话记录显示在这里</p></div>
 <button type="button" id="resetDisplay">恢复默认</button>`;
 language=section.querySelector('#systemLanguage');uiScale=section.querySelector('#uiFont');uiOutput=section.querySelector('#uiFontPercent');contentScale=section.querySelector('#contentFont');contentOutput=section.querySelector('#contentFontPercent');
 language.onchange=()=>{prefs.lang=language.value;apply();};uiScale.oninput=()=>{prefs.uiLevel=Number(uiScale.value)/100;apply();};contentScale.oninput=()=>{prefs.contentLevel=Number(contentScale.value)/100;apply();};
 section.querySelector('#resetDisplay').onclick=()=>{prefs={lang:'zh-CN',uiLevel:1,contentLevel:1,schema:2};apply();};return section;
}
const HARNESS_GROUPS=[
 {title:'角色',fields:[['persona','interaction_style','陪伴方式','choice']]},
 {title:'声音',fields:[['speaking_style','speech_rate','说话速度','choice'],['speaking_style','tone','语气','choice']]},
 {title:'回答',fields:[['speaking_style','reply_length','回答长度','segments'],['reply_modes','reasoning_depth','思考深度','segments']]},
 {id:'backchannel',title:'垫话',fields:[['turn_taking','work_filler','等待反馈','choice'],['turn_taking','backchannel','附和频率','curve']]},
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
  const curveSpec=harnessMessage.backchannel_curve_schema;
  body.innerHTML=HARNESS_GROUPS.map(group=>`<section class="harness-group${group.id?` harness-group-${group.id}`:''}"><h3>${group.title}</h3><div class="harness-grid">${group.fields.map(([domain,name,title,kind])=>{
   const spec=schema[domain]?.[name];if(!spec)return'';
   const path=`${domain}.${name}`,values=HARNESS_ORDER[path]||spec.values,current=snapshot.profile[domain]?.[name]??spec.default;
   const pending=snapshot.pending?.[path];
   if(kind==='curve'&&curveSpec){
    const quotas=Array.isArray(snapshot.backchannel_curve)?snapshot.backchannel_curve:curveSpec.profiles[current]||curveSpec.profiles.auto,geometry=curveGeometry(quotas,curveSpec);
    const baseline=geometry.bottom-(1.5-curveSpec.min)/(curveSpec.max-curveSpec.min)*(geometry.bottom-geometry.top);
    return `<article class="backchannel-curve" data-values="${escapeHtml(JSON.stringify(quotas))}"><header><strong>${title}</strong><button type="button" class="harness-reset curve-reset" title="恢复默认" aria-label="${title} 恢复默认">↶</button></header><div class="curve-chart"><svg viewBox="0 0 600 190" role="img" aria-label="各阶段附和次数"><line class="curve-baseline" x1="28" y1="${baseline}" x2="574" y2="${baseline}"/>${geometry.points.map(point=>`<line class="curve-guide" x1="${point.x}" y1="14" x2="${point.x}" y2="150"/>`).join('')}<path class="curve-area" d="${geometry.area}"/><path class="curve-line" d="${geometry.path}"/>${geometry.points.map((point,i)=>`<circle class="curve-point point-${i}" cx="${point.x}" cy="${point.y}" r="10"/>`).join('')}</svg><div class="curve-handles">${quotas.map((quota,i)=>`<input class="curve-input" style="--curve-x:${geometry.points[i].x/6}%" type="range" min="${curveSpec.min}" max="${curveSpec.max}" step="${curveSpec.step}" value="${quota}" aria-label="${curveSpec.phases[i].label} 附和 ${quota.toFixed(1)} 次">`).join('')}</div>${quotas.map((quota,i)=>`<span class="curve-label" style="--curve-x:${geometry.points[i].x/6}%"><b>${quota.toFixed(1)}</b><small>${curveSpec.phases[i].label}</small></span>`).join('')}</div><p>前 6 秒合计最多附和 2 次。</p></article>`;
   }
   const reset=`<button type="button" class="harness-reset" title="恢复默认" aria-label="${title} 恢复默认" data-default="${escapeHtml(spec.default)}">↶</button>`;
   const pendingText=`<small class="harness-pending">${pending?'等你在对话里再确认一次':''}</small>`;
   if(kind==='segments')return `<article class="harness-control harness-segments" data-domain="${domain}" data-name="${name}"><span class="harness-label">${title}</span><div class="harness-row-value"><div class="harness-options" role="group" aria-label="${title}">${values.map(item=>`<button type="button" class="harness-option" data-value="${escapeHtml(item)}" aria-pressed="${item===current}">${escapeHtml(label(spec,item))}</button>`).join('')}</div>${reset}</div>${pendingText}</article>`;
   return `<article class="harness-control harness-choice" data-domain="${domain}" data-name="${name}"><span class="harness-label">${title}</span><div class="harness-row-value"><select aria-label="${title}">${values.map(item=>`<option value="${escapeHtml(item)}" ${item===current?'selected':''}>${escapeHtml(label(spec,item))}</option>`).join('')}</select>${reset}</div>${pendingText}</article>`;
  }).join('')}</div></section>`).join('');
  for(const card of body.querySelectorAll('.harness-segments,.harness-choice')){
   const spec=schema[card.dataset.domain][card.dataset.name],select=card.querySelector('select');
   const options=[...card.querySelectorAll('.harness-option')];
   const paint=selected=>{if(select)select.value=selected;for(const option of options)option.setAttribute('aria-pressed',String(option.dataset.value===selected));};
   const save=async selected=>{if(!harnessClient)return;status.textContent='正在保存…';try{await harnessClient.setHarness({[card.dataset.domain]:{[card.dataset.name]:selected}});}catch(error){status.textContent=error.message;}};
   if(select)select.onchange=()=>void save(select.value);
   for(const option of options)option.onclick=()=>{paint(option.dataset.value);void save(option.dataset.value);};
   card.querySelector('.harness-reset').onclick=()=>{paint(spec.default);void save(spec.default);};
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
document.addEventListener('self-harness-state',event=>{harnessMessage=event.detail;harnessApi?.render();promptApi?.receive(harnessMessage);});
const COMPONENT_ICONS={
 turn:'<path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/>',
 input:'<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11v1a7 7 0 0 0 14 0v-1M12 19v2"/>',
 memory:'<path d="m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5"/>',
 reply:'<path d="M20 15a3 3 0 0 1-3 3H8l-4 3V6a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3Z"/><path d="M8 9h8M8 13h5"/>',
 speech:'<path d="M4 10v4M8 6v12M12 3v18M16 6v12M20 10v4"/>',
 pet:'<ellipse cx="7" cy="7" rx="2" ry="3"/><ellipse cx="17" cy="7" rx="2" ry="3"/><path d="M5 17c0-3 4-6 7-6s7 3 7 6c0 4-4 3-7 2s-7 2-7-2Z"/>',
};
const MODEL_DEFAULTS={deepseek:{model:'deepseek-v4-flash',baseUrl:'https://api.deepseek.com'},qwen:{model:'qwen3.6-flash',baseUrl:'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'},openai:{model:'gpt-4o',baseUrl:'https://api.openai.com/v1'},local:{model:'mlx-community/Qwen3.5-4B-4bit',baseUrl:''}};
function providerName(value){const labels={deepseek:'DeepSeek',qwen:'Qwen',openai:'OpenAI',local:'本地模型',breeze_mlx:'Breeze MLX',breeze_cuda:'Breeze CUDA'};return labels[value]||value||'启动配置';}
function escapeHtml(value){return String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));}
function setupComponents(container){
 return window.VMComponents.mount(container,{desktop:desktopModels,defaults:MODEL_DEFAULTS,providerName,icons:COMPONENT_ICONS,t,translate});
}
function openSettings(){
 opener=document.activeElement;
 if(!dialog){dialog=document.createElement('dialog');dialog.className='settings-page';dialog.setAttribute('aria-labelledby','settingsTitle');
 dialog.innerHTML=`<div class="settings-shell"><aside class="settings-sidebar"><div class="settings-brand"><strong id="settingsTitle">设置</strong></div>
 <nav class="settings-tabs" role="tablist" aria-label="设置">
  <button role="tab" aria-selected="true" aria-controls="settingsStyle"><svg viewBox="0 0 24 24"><path d="M12 3a9 9 0 1 0 9 9c0-1.1-.9-2-2-2h-1.6a2 2 0 0 1-1.8-2.8l.6-1.4A2 2 0 0 0 14.4 3H12Z"/><circle cx="7.5" cy="11.5" r="1"/><circle cx="10" cy="7.5" r="1"/><circle cx="8.5" cy="16" r="1"/></svg><span>外观</span></button>
  <button role="tab" aria-selected="false" aria-controls="settingsDisplay"><svg viewBox="0 0 24 24"><path d="M4 19 10.5 5h3L20 19M7 14h10"/></svg><span>文字</span></button>
  <button role="tab" aria-selected="false" aria-controls="settingsHarness"><svg viewBox="0 0 24 24"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/></svg><span>对话偏好</span></button>
  <button role="tab" aria-selected="false" aria-controls="settingsPrompts"><svg viewBox="0 0 24 24"><path d="m8 7-5 5 5 5m8-10 5 5-5 5M14 4l-4 16"/></svg><span>Harness</span></button>
  <button role="tab" aria-selected="false" aria-controls="settingsComponents"><svg viewBox="0 0 24 24"><rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/></svg><span>组件</span></button>
 </nav></aside>
 <main class="settings-main"><header class="settings-header"><span class="settings-current">外观</span><button type="button" id="closeSettings" aria-label="返回对话" title="返回对话">×</button></header><div class="settings-scroll">
 <section class="settings-panel on" id="settingsStyle" role="tabpanel"><div class="settings-modes">
  <a href="technical.html" class="style-card mode-card mode-technical"><div class="style-preview style-preview-technical" aria-hidden="true"><span class="style-orb"></span></div><span class="style-card-name">科技风</span></a>
  <a href="digital.html" class="style-card mode-card mode-digital"><div class="style-preview" aria-hidden="true"><img src="assets/digital-background.png" alt=""></div><span class="style-card-name">数字人</span></a></div></section>
 <section class="settings-panel" id="settingsDisplay" role="tabpanel"><div class="settings-panel-body"></div></section>
 <section class="settings-panel" id="settingsHarness" role="tabpanel"><div class="settings-panel-copy"><p>当前对话</p><span class="harness-status" aria-live="polite">正在连接 Studio…</span></div><div class="harness-controls"></div></section>
 <section class="settings-panel" id="settingsPrompts" role="tabpanel"><div class="harness-prompt-editor"></div></section>
 <section class="settings-panel" id="settingsComponents" role="tabpanel"><div class="component-settings"></div></section>
 </div></main></div>`;
 dialog.querySelector('#settingsDisplay .settings-panel-body').append(controls());document.body.append(dialog);
 const componentApi=setupComponents(dialog.querySelector('.component-settings'));
 harnessApi=setupHarnessPanel(dialog.querySelector('#settingsHarness'));
 promptApi=window.VMHarnessPrompts?.mount(dialog.querySelector('.harness-prompt-editor'),{client:()=>harnessClient,refresh:()=>harnessClient?.getHarness()});
 if(harnessMessage)promptApi?.receive(harnessMessage);
 const tabs=[...dialog.querySelectorAll('.settings-tabs button')],panels=[...dialog.querySelectorAll('.settings-panel')];
 tabs.forEach((tab,i)=>{tab.id=`settingsTab${i}`;panels[i].setAttribute('aria-labelledby',tab.id);});
 function revealTab(tab){
  const nav=tab.parentElement,bounds=tab.getBoundingClientRect(),frame=nav.getBoundingClientRect();
  if(bounds.left<frame.left)nav.scrollLeft+=bounds.left-frame.left;
  else if(bounds.right>frame.right)nav.scrollLeft+=bounds.right-frame.right;
 }
 let activePanel=0;
 async function showPanel(index){if(index!==activePanel&&!(await (promptApi?.confirmLeave()??true)))return false;activePanel=index;tabs.forEach((tab,i)=>{const on=i===index;tab.setAttribute('aria-selected',String(on));tab.tabIndex=on?0:-1;panels[i].classList.toggle('on',on);});dialog.querySelector('.settings-current').textContent=tabs[index].querySelector('span').textContent;if(index===2)harnessApi.activate();if(index===3)void promptApi?.activate();if(index===4)componentApi.activate();return true;}
 tabs.forEach((tab,i)=>{tab.tabIndex=i?-1:0;tab.onclick=()=>showPanel(i);tab.onfocus=()=>revealTab(tab);tab.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;void showPanel(next).then(changed=>{if(changed){revealTab(tabs[next]);tabs[next].focus();}});};});
 async function closeSettings(){if(await (promptApi?.confirmLeave()??true))dialog.close();}
 dialog.querySelector('#closeSettings').onclick=closeSettings;dialog.addEventListener('cancel',event=>{event.preventDefault();void closeSettings();});dialog.addEventListener('close',()=>{harnessApi?.render();document.dispatchEvent(new Event('settings-close'));opener?.focus();});
 const current=dialog.querySelector('.mode-'+document.body.dataset.style);current?.setAttribute('aria-current','page');current?.addEventListener('click',e=>{e.preventDefault();void closeSettings();});
 for(const link of dialog.querySelectorAll('.settings-modes a'))if(link!==current)link.addEventListener('click',async event=>{event.preventDefault();if(await (promptApi?.confirmLeave()??true))location.href=link.href;});
 }
 apply(false);document.dispatchEvent(new Event('settings-open'));dialog.showModal();dialog.querySelector('#closeSettings').focus();
}
window.VMSettings={graphLabel(value){const labels={work:'工作',health:'健康',person:'人物',research:'研究',projects:'项目',entertainment:'娱乐',you:'你','经济':'财务'};return prefs.lang==='en'?(value==='经济'?'finance':value):labels[value]||value;},setLanguage(value){prefs.lang=value==='en'?'en':'zh-CN';apply();},get language(){return prefs.lang;},get uiScale(){return prefs.uiLevel*UI_BASE;},get contentScale(){return prefs.contentLevel*CONTENT_BASE;},t,bindHarness,open:openSettings};
const btn=document.getElementById('settingsBtn');if(btn)btn.onclick=openSettings;
apply(false);
let queued=false;
if(window.MutationObserver)new MutationObserver(()=>{if(queued)return;queued=true;queueMicrotask(()=>{queued=false;translate();});}).observe(document.body,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['title','aria-label','placeholder','alt']});
window.addEventListener('storage',e=>{if(e.key===KEY){try{prefs=JSON.parse(e.newValue)||{lang:'zh-CN',uiLevel:1,contentLevel:1,schema:2};apply(false);}catch{}}});
})();
