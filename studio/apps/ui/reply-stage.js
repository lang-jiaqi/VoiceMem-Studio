/* Preparation is visible independently of buffered reply text and playback. */
(() => {
  'use strict';
  let output = '', provider = '', mode = '', text = '', timer = null;
  function paint(value) {
    text = value;
    for (const node of document.querySelectorAll('[data-reply-stage]')) {
      node.textContent = window.VMSettings?.t(value) || value;
      node.hidden = !value;
    }
  }
  function clear() { clearTimeout(timer); output = ''; paint(''); }
  window.VMReplyStage = {
    onEvent(message) {
      if (message.type === 'user_transcript') { clear(); paint('检索记忆…'); }
      if (message.type === 'reply_preparing') {
        clearTimeout(timer); output = message.output_id; provider = message.provider; mode = message.reply_mode;
      }
      if (message.output_id && output && message.output_id !== output) return;
      if (message.type === 'memory_hits' && message.stage === 'retrieved') {
        const count = (message.left_brain || []).length + (message.right_brain_hits || []).filter(hit => !hit.internal).length;
        paint(count ? '已找到相关记忆' : '本轮无相关记忆');
        const expected = output;
        timer = setTimeout(() => {
          if (output === expected) paint(provider === 'deepseek'
            ? (mode === 'memory_cot' ? 'DeepSeek 正在思考…' : 'DeepSeek 正在生成…') : '正在生成回复…');
        }, 350);
      }
      if (message.type === 'playback_started') { clearTimeout(timer); paint('正在说话…'); }
      if (['playback_done', 'answer_interrupt', 'disconnected', 'error'].includes(message.type)) clear();
    },
  };
  document.addEventListener('display-settings-change', () => paint(text));
  window.addEventListener('memory-space-change', clear);
})();
