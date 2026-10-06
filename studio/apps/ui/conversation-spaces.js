/* Conversation groups share the persistent memory of their selected Space. */
(() => {
  'use strict';
  let spaces = [], active = '', fixed = false, demo = false;
  const request = async (url, options) => {
    const response = await fetch(url, options);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Memory Space 请求失败 (${response.status})：${detail.slice(0, 160)}`);
    }
    return response.json();
  };
  async function refresh() {
    const state = await request('/api/spaces');
    demo = !!state.demo;
    spaces = state.spaces || [];
    fixed = spaces.some(space => space.fixed);
    const changed = active !== (state.active || '');
    active = state.active || '';
    if (changed) window.dispatchEvent(new CustomEvent('memory-space-change', { detail: { space: active } }));
    return active;
  }
  const ready = refresh();
  ready.catch(() => {});
  async function use(space) {
    await ready;
    space = space || active;
    if (!space || !spaces.some(item => item.id === space)) throw new Error('Memory Space 不存在，请重新选择。');
    if ((fixed || demo) && space === active) return active;
    const result = await request(`/api/spaces/${encodeURIComponent(space)}/use`, { method: 'POST' });
    const changed = active !== result.active;
    active = result.active;
    if (changed) window.dispatchEvent(new CustomEvent('memory-space-change', { detail: { space: active } }));
    return active;
  }
  async function choose(preferred) {
    await refresh();
    if (demo) return active;
    return new Promise(resolve => {
      const dialog = document.createElement('dialog');
      dialog.className = 'space-picker';
      const heading = document.createElement('h2');
      heading.textContent = '为新对话选择 Memory Space';
      const hint = document.createElement('p');
      hint.textContent = fixed ? '启动时指定了单个记忆库，当前对话固定使用此空间。' : '同一空间的对话共享长期记忆。';
      const list = document.createElement('div');
      list.className = 'space-picker-list';
      const name = document.createElement('input');
      name.placeholder = '新空间名称'; name.maxLength = 20;
      name.setAttribute('aria-label', '新空间名称');
      const add = document.createElement('button'); add.type = 'button'; add.textContent = '创建空间';
      const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = '取消';
      const confirm = document.createElement('button'); confirm.type = 'button';
      confirm.className = 'space-picker__confirm'; confirm.textContent = '确认并创建对话';
      const createRow = document.createElement('div'); createRow.className = 'space-picker-actions';
      createRow.append(name, add);
      createRow.hidden = fixed;
      const footer = document.createElement('div'); footer.className = 'space-picker__footer';
      footer.append(cancel, confirm);
      dialog.append(heading, hint, list, createRow, footer);
      let selected = fixed ? active : preferred || active;
      const close = value => { dialog.close(); dialog.remove(); resolve(value); };
      function render() {
        list.replaceChildren();
        for (const space of spaces) {
          const button = document.createElement('button'); button.type = 'button';
          button.className = space.id === selected ? 'selected' : '';
          button.textContent = `${space.name} · ${space.count} 条记忆`;
          button.onclick = () => { selected = space.id; render(); };
          list.append(button);
        }
        confirm.disabled = !selected;
      }
      add.onclick = async () => {
        const value = name.value.trim();
        if (!value) return name.focus();
        add.disabled = true;
        try {
          const created = await request('/api/spaces', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: value }),
          });
          await refresh(); selected = created.id; name.value = ''; render();
        } catch (error) { window.VMUI?.notify(error.message); }
        finally { add.disabled = false; }
      };
      cancel.onclick = () => close(null);
      confirm.onclick = () => close(selected);
      dialog.addEventListener('cancel', event => { event.preventDefault(); close(null); });
      render(); document.body.append(dialog); dialog.showModal();
    });
  }
  window.VMConversationSpaces = Object.freeze({ ready, refresh, use, choose, get active() { return active; },
    name(id) { return spaces.find(space => space.id === id)?.name || id || 'Memory Space'; } });
})();
