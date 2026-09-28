(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const label = { 'persona.interaction_style':'陪伴方式', 'speaking_style.speech_rate':'说话速度', 'speaking_style.tone':'语气', 'speaking_style.reply_length':'回答长短', 'reply_modes.reasoning_depth':'思考时间', 'turn_taking.backchannel':'附和频率', 'turn_taking.work_filler':'等待反馈' };
  const phaseLabel = {idle:'',listening:'正在聆听…','short-thinking':'正在思考…','long-thinking':'还在思考…',speaking:'正在回答…'};
  let listening = false, resumeAfterSettings = false, replyMessage = null, memoryPoll = 0, known = new Set(), baselineReady = false;
  const messages = [];
  const memoryKey = (kind, item) => `${kind}:${item.id || item.raw || ''}:${item.text || ''}:${kind === 'right' ? (item.notes || []).length : ''}`;
  const memorySnapshotReady = fetch('/api/memories', {cache:'no-store'}).then(r => r.ok ? r.json() : Promise.reject()).then(data => {
    known = new Set([...(data.left || []).map(item => memoryKey('left', item)), ...(data.right || []).map(item => memoryKey('right', item))]);
    baselineReady = true;
  }).catch(() => {});
  function list(id, values, empty) {
    const host = $(id); host.replaceChildren();
    for (const value of values.slice(0, 6)) { const item = document.createElement('li'); item.textContent = value; host.append(item); }
    if (!values.length) { const item = document.createElement('li'); item.textContent = empty; host.append(item); }
  }
  async function watchStored() {
    const generation = ++memoryPoll;
    await memorySnapshotReady;
    list('storedMemories', [], '正在等待入库…');
    for (let attempt = 0; attempt < 15 && generation === memoryPoll; attempt++) {
      if (attempt) await new Promise(resolve => setTimeout(resolve, 2000));
      try {
        const response = await fetch('/api/memories', {cache:'no-store'});
        if (!response.ok) break;
        const data = await response.json();
        const entries = [...(data.left || []).map(item => ['left', item]), ...(data.right || []).map(item => ['right', item])];
        if (!baselineReady) { entries.forEach(([kind,item]) => known.add(memoryKey(kind,item))); baselineReady = true; continue; }
        const fresh = entries.filter(([kind,item]) => !known.has(memoryKey(kind,item))).map(([kind,item]) => `${kind === 'left' ? '事实' : '感受'} · ${item.text || item.raw || ''}`);
        if (fresh.length) { list('storedMemories', fresh, '本轮没有新记忆'); entries.forEach(([kind,item]) => known.add(memoryKey(kind,item))); return; }
      } catch { break; }
    }
    if (generation === memoryPoll) list('storedMemories', [], '本轮暂无新入库记忆');
  }
  function renderChat() {
    const host = $('chatMessages');
    const atBottom = host.scrollHeight - host.scrollTop - host.clientHeight < 48;
    host.replaceChildren();
    if (!messages.length) {
      const empty = document.createElement('p'); empty.className = 'chat-empty';
      empty.textContent = '开始说话，或输入一条消息。'; host.append(empty);
    }
    for (const message of messages.slice(-100)) {
      const row = document.createElement('div'); row.className = `chat-turn ${message.role}`;
      const body = document.createElement('div'); body.className = 'chat-bubble'; body.textContent = message.text;
      message.element = body; row.append(body); host.append(row);
    }
    if (atBottom || $('chatDialog').open) host.scrollTop = host.scrollHeight;
  }
  function addMessage(message) { messages.push(message); if (messages.length > 100) messages.shift(); renderChat(); }
  const client = VMStudio.create({
    onState(active) { listening = active; $('talkButton').textContent = active ? '结束对话' : '开始对话'; $('talkButton').setAttribute('aria-pressed', String(active)); },
    onPhase(value) { const status = phaseLabel[value] ?? value; $('phase').textContent = status; $('phase').hidden = !status; window.avatar?.setSpeaking(value === 'speaking'); if (value !== 'speaking') window.avatar?.setState(value === 'listening' ? 'listening' : 'idle'); },
    onAudioLevel(rms) { window.avatar?.feedAudioLevel(rms, performance.now()); },
    onEvent(event) {
      if (event.type === 'partial_transcript') { $('partialTranscript').textContent = event.text || ''; $('partialTranscript').hidden = !event.text; }
      if (event.type === 'user_transcript') {
        $('partialTranscript').hidden = true;
        VMStudio.applyUser(messages, event, 'user');
        if (messages.length > 100) messages.shift();
        renderChat();
        if (!$('chatDialog').open) $('chatCount').hidden = false;
        list('retrievedMemories', [], '检索中…'); void watchStored();
      }
      if (event.type === 'answer_start') {
        replyMessage = {role:'assistant', text:'', outputId:event.output_id}; addMessage(replyMessage);
        if (!$('chatDialog').open) $('chatCount').hidden = false;
      }
      if (event.type === 'answer_delta') {
        if (replyMessage) {
          const host = $('chatMessages');
          const atBottom = host.scrollHeight - host.scrollTop - host.clientHeight < 48;
          replyMessage.text += event.text || '';
          if (replyMessage.element) replyMessage.element.textContent = replyMessage.text;
          if (atBottom) host.scrollTop = host.scrollHeight;
        }
      }
      if (event.type === 'answer_interrupt') {
        if (replyMessage) { replyMessage.text = event.heard_text || ''; const index = messages.indexOf(replyMessage); if (!replyMessage.text && index >= 0) messages.splice(index, 1); renderChat(); }
        replyMessage = null;
      }
      if (event.type === 'playback_done' || event.type === 'disconnected' || event.type === 'error') replyMessage = null;
      if (event.type === 'memory_hits') {
        const hits = [...(event.left_brain || []).map(item => item.text), ...(event.right_brain_hits || []).filter(item => !item.internal).map(item => item.content)];
        list('retrievedMemories', hits, '本轮没有召回相关记忆'); $('memoryCount').textContent = hits.length ? `· ${hits.length}` : '';
      }
    },
  });
  window.avatar?.show('sit');
  setTimeout(() => {
    const status = window.avatar?.getStatus();
    if (status && !status.ready && status.modelError)
      VMUI.notify('宠物加载失败，请检查 Live2D Core 和网络连接。');
  }, 8000);
  $('talkButton').onclick = () => { if (listening) client.stop(); else void client.start(); };
  function send() {
    const text = $('message').value.trim(); if (!text) return;
    $('message').value = '';
    const pending = {role:'user', text, pending:true}; addMessage(pending);
    void client.send(text).catch(error => { const index = messages.indexOf(pending); if (index >= 0) messages.splice(index, 1); renderChat(); VMUI.notify(error.message); });
  }
  $('chatForm').onsubmit = event => { event.preventDefault(); send(); };
  function open(id) { const dialog = $(id); if (dialog.open) return; dialog.showModal(); if (id === 'settingsDialog') { resumeAfterSettings = listening; client.pause(); void client.getHarness().catch(error => VMUI.notify(error.message)); } }
  $('memoryButton').onclick = () => open('memoryDialog');
  $('chatButton').onclick = () => { $('chatCount').hidden = true; open('chatDialog'); renderChat(); };
  $('settingsButton').onclick = () => open('settingsDialog');
  for (const button of document.querySelectorAll('[data-close]')) button.onclick = () => $(button.dataset.close).close();
  for (const dialog of document.querySelectorAll('dialog')) dialog.onclick = event => { if (event.target === dialog) dialog.close(); };
  $('settingsDialog').onclose = () => { if (resumeAfterSettings) void client.start(); resumeAfterSettings = false; };
  fetch('/auth/me').then(r => r.ok ? r.json() : null).then(user => { if (user) $('accountName').textContent = user.name; }).catch(() => {});
  $('logoutButton').onclick = async () => { try { await fetch('/auth/logout', {method:'POST'}); location.replace('/ui/login.html'); } catch (error) { VMUI.notify(error.message); } };
  document.addEventListener('self-harness-state', ({detail}) => {
    $('harnessStatus').textContent = detail.error || '已同步到你的对话空间';
    if (detail.error || !detail.schema || !detail.snapshot) return;
    const host = $('harnessControls'); host.replaceChildren();
    const add = (title, control) => { const row = document.createElement('div'); row.className = 'harness-row'; const caption = document.createElement('label'); caption.textContent = title; row.append(caption, control); host.append(row); return row; };
    const prompt = document.createElement('textarea'); prompt.value = detail.snapshot.prompts?.persona || ''; prompt.maxLength = detail.prompt_schema?.persona?.max_length || 2000; prompt.placeholder = '例如：多听我说，别急着给建议。';
    const promptRow = add('Persona Prompt', prompt); const promptSave = document.createElement('button'); promptSave.textContent = '保存'; promptSave.onclick = () => void client.setHarnessPrompt('persona', prompt.value).catch(error => VMUI.notify(error.message)); promptRow.append(promptSave);
    for (const [domain, fields] of Object.entries(detail.schema)) for (const [name, spec] of Object.entries(fields)) {
      const select = document.createElement('select');
      for (const value of spec.values) { const option = document.createElement('option'); option.value = value; option.textContent = spec.labels[value] || value; select.append(option); }
      select.value = detail.snapshot.profile?.[domain]?.[name] || spec.default;
      select.onchange = () => void client.setHarness({[domain]:{[name]:select.value}}).catch(error => VMUI.notify(error.message));
      add(label[`${domain}.${name}`] || name, select);
    }
    const curve = detail.backchannel_curve_schema;
    if (curve) {
      const box = document.createElement('div');
      const mode = detail.snapshot.profile?.turn_taking?.backchannel || 'auto';
      const values = detail.snapshot.backchannel_curve || curve.profiles[mode] || curve.profiles.auto;
      const inputs = curve.phases.map((phase, index) => {
        const item = document.createElement('div'); item.className = 'curve-item';
        const name = document.createElement('span'); name.textContent = phase.label;
        const input = document.createElement('input'); input.type = 'range'; input.min = curve.min; input.max = curve.max; input.step = curve.step; input.value = values[index]; input.setAttribute('aria-label', `${phase.label} 附和次数`);
        const output = document.createElement('span'); output.textContent = Number(input.value).toFixed(1);
        input.oninput = () => { output.textContent = Number(input.value).toFixed(1); };
        input.onchange = () => void client.setBackchannelCurve(inputs.map(node => Number(node.value))).catch(error => VMUI.notify(error.message));
        item.append(name, input, output); box.append(item); return input;
      });
      const row = add('附和节奏', box); const reset = document.createElement('button'); reset.textContent = '恢复默认'; reset.onclick = () => void client.setBackchannelCurve(null).catch(error => VMUI.notify(error.message)); row.append(reset);
    }
  });
})();
