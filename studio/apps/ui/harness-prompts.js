/* Explicit prompt drafts persist locally only after the server accepts them. */
(() => {
  'use strict';
  const KEY = 'voicemem.harness-prompts.v1';
  const GROUPS = [
    ['main', '主模型对话'], ['persona', 'Persona'], ['speaking_style', 'Speaking Style'],
    ['reply_modes', 'Reply Modes'], ['turn_taking', 'Turn Taking'],
    ['self_harness', 'Self Harness'], ['tts', 'TTS'],
  ];
  const escape = value => String(value ?? '').replace(/[&<>"']/g, char =>
    ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const t = text => window.VMSettings?.t(text) || text;
  function saved() {
    try {
      const value = JSON.parse(localStorage.getItem(KEY) || '{}');
      if (!value || Array.isArray(value) || typeof value !== 'object') return {};
      return Object.fromEntries(Object.entries(value).filter(([key, text]) =>
        key.length < 80 && typeof text === 'string' && text.length <= 20000));
    } catch { return {}; }
  }
  function changes(catalog, values) {
    return Object.fromEntries(Object.entries(catalog).filter(([key, spec]) =>
      values[key] !== spec.default).map(([key]) => [key, values[key]]));
  }
  function mount(root, {client, refresh} = {}) {
    let specs = {}, draft = {}, baseline = {}, group = 'main', saving = false;
    let lastMessage = null, leaving = null, promptLanguage = 'zh';
    const dirty = () => Object.keys(specs).some(key => draft[key] !== baseline[key]);
    root.innerHTML = `<div class="prompt-toolbar"><label class="prompt-module"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 7-5 5 5 5m8-10 5 5-5 5M14 4l-4 16"/></svg><select aria-label="Harness"></select></label><span class="prompt-status" role="status"></span></div>
      <div class="prompt-editors"></div><footer class="prompt-actions"><button type="button" class="prompt-reset">恢复默认</button><button type="button" class="prompt-save">保存</button></footer>`;
    const select = root.querySelector('select'), status = root.querySelector('.prompt-status');
    const editors = root.querySelector('.prompt-editors');
    const saveButton = root.querySelector('.prompt-save'), resetButton = root.querySelector('.prompt-reset');
    function numberLines() {
      let line = 1;
      for (const section of editors.querySelectorAll('.prompt-source')) {
        section.querySelector('.prompt-start-number').textContent = line++;
        const rows = section.querySelectorAll('.prompt-measure > span');
        section.querySelector('.prompt-line-numbers').innerHTML = Array.from(rows, row =>
          `<span style="height:${row.offsetHeight || 20}px">${line++}</span>`).join('');
        section.querySelector('.prompt-end-number').textContent = line++;
      }
    }
    // Wrapped display rows share measurements with the editable text, not its stored newlines.
    if (window.ResizeObserver) new window.ResizeObserver(numberLines).observe(editors);
    function paintStatus(text) {
      status.textContent = t(text || (dirty() ? '未保存' : '已保存'));
      saveButton.disabled = saving || !dirty();
      resetButton.disabled = saving || !Object.keys(specs).length;
      select.disabled = saving || !Object.keys(specs).length;
      editors.querySelectorAll('textarea').forEach(input => { input.disabled = saving; });
    }
    function render() {
      select.innerHTML = GROUPS.filter(([id]) => Object.values(specs).some(spec => spec.group === id))
        .map(([id, title]) => `<option value="${id}"${group === id ? ' selected' : ''}>${escape(t(title))}</option>`).join('');
      const entries = Object.entries(specs).filter(([, spec]) => spec.group === group);
      editors.innerHTML = entries
        .map(([key, spec]) => `<section class="prompt-source" data-key="${escape(key)}">
          <div class="prompt-code-line" data-no-i18n><span class="prompt-gutter prompt-start-number" aria-hidden="true"></span><code><span class="prompt-variable">${escape(spec.variable)}</span> = <span class="prompt-string">"""</span></code></div>
          <div class="prompt-code-body"><pre class="prompt-line-numbers" aria-hidden="true" data-no-i18n></pre><div class="prompt-input"><pre class="prompt-measure" aria-hidden="true" data-no-i18n></pre><textarea aria-label="${escape(spec.variable)}" maxlength="${spec.max_length}" spellcheck="false" autocapitalize="off" autocomplete="off" wrap="soft" data-gramm="false" data-gramm_editor="false" data-enable-grammarly="false" data-no-i18n>${escape(draft[key])}</textarea></div></div>
          <div class="prompt-code-line" data-no-i18n><span class="prompt-gutter prompt-end-number" aria-hidden="true"></span><code class="prompt-string">"""</code></div></section>`).join('');
      for (const section of editors.querySelectorAll('.prompt-source')) {
        const input = section.querySelector('textarea'), measure = section.querySelector('.prompt-measure');
        const update = () => {
          measure.innerHTML = input.value.split('\n').map(text =>
            `<span>${escape(text) || '&#8203;'}</span>`).join('');
        };
        input.oninput = () => { draft[section.dataset.key] = input.value; update(); numberLines(); paintStatus(); };
        input.onkeydown = event => {
          if (event.key === 'Tab' && !event.shiftKey) {
            event.preventDefault();
            input.setRangeText('    ', input.selectionStart, input.selectionEnd, 'end');
            input.oninput();
          } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
            event.preventDefault(); void save();
          }
        };
        update();
      }
      numberLines();
      paintStatus(lastMessage?.error || (!Object.keys(specs).length ? '正在连接 Studio…' : ''));
    }
    function receive(message) {
      lastMessage = message;
      if (!message?.prompt_catalog) return;
      const edited = new Set(Object.keys(specs).filter(key => draft[key] !== baseline[key]));
      specs = message.prompt_catalog;
      promptLanguage = message.snapshot?.prompt_language === 'en' ? 'en' : 'zh';
      const overrides = message.snapshot?.prompt_overrides || {};
      baseline = Object.fromEntries(Object.entries(specs).map(([key, spec]) =>
        [key, overrides[key] ?? (key === 'persona' ? message.snapshot?.prompts?.persona : undefined) ?? spec.default]));
      draft = Object.fromEntries(Object.keys(specs).map(key =>
        [key, edited.has(key) ? draft[key] : baseline[key]]));
      render();
    }
    async function save() {
      if (saving || !dirty()) return !saving;
      const target = client();
      if (!target) { paintStatus('连接后即可编辑 Prompt。'); return false; }
      saving = true; paintStatus('正在保存…');
      try {
        const overrides = changes(specs, draft);
        const accepted = await target.setHarnessPrompts(overrides, promptLanguage);
        localStorage.setItem(KEY, JSON.stringify(accepted?.snapshot?.prompt_overrides || overrides));
        baseline = {...draft};
        paintStatus('已保存，下轮生效');
        return true;
      } catch (error) { paintStatus(error.message); return false; }
      finally { saving = false; paintStatus(status.textContent); }
    }
    function confirmLeave() {
      if (saving) return Promise.resolve(false);
      if (!dirty()) return Promise.resolve(true);
      if (leaving) return leaving;
      leaving = new Promise(resolve => {
        const dialog = document.createElement('dialog');
        dialog.className = 'prompt-confirm';
        dialog.innerHTML = `<form method="dialog"><h2>${t('保存 Prompt 修改？')}</h2><div><button value="cancel">${t('继续编辑')}</button><button value="discard">${t('不保存')}</button><button value="save" class="prompt-confirm-save">${t('保存')}</button></div></form>`;
        dialog.addEventListener('close', async () => {
          const action = dialog.returnValue;
          dialog.remove();
          let allowed = false;
          if (action === 'save') allowed = await save();
          else if (action === 'discard') { draft = {...baseline}; render(); allowed = true; }
          leaving = null; resolve(allowed);
        }, {once:true});
        document.body.append(dialog); dialog.showModal();
        dialog.querySelector('button').focus();
      });
      return leaving;
    }
    select.onchange = () => { group = select.value; render(); };
    saveButton.onclick = () => { void save(); };
    resetButton.onclick = () => {
      for (const [key, spec] of Object.entries(specs)) if (spec.group === group) draft[key] = spec.default;
      render();
    };
    window.addEventListener('beforeunload', event => {
      if (dirty()) { event.preventDefault(); event.returnValue = ''; }
    });
    return {receive, confirmLeave, dirty, async activate() {
      if (lastMessage) receive(lastMessage);
      paintStatus(Object.keys(specs).length ? '' : '正在连接 Studio…');
      try { await refresh(); } catch (error) { paintStatus(error.message); }
    }};
  }
  window.VMHarnessPrompts = {mount, saved, changes};
})();
