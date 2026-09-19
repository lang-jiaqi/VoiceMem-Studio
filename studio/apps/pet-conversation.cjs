'use strict';

const PAGES = new Set(['/ui/digital.html', '/ui/technical.html']);

function conversationPage(studio, origin) {
  if (!studio || studio.isDestroyed() || !origin) return false;
  try {
    const page = new URL(studio.webContents.getURL());
    return page.origin === origin && PAGES.has(page.pathname);
  } catch { return false; }
}

async function controlConversation(studio, origin, toggle = false) {
  if (!conversationPage(studio, origin)) throw new Error('请先在 Studio App 中进入对话页面。');
  const script = `(() => {
    const button = document.getElementById('startTalk');
    if (!button) throw new Error('对话页面尚未准备好，请更新服务并重试。');
    ${toggle ? 'button.click();' : ''}
    return button.getAttribute('aria-pressed') === 'true';
  })()`;
  return Boolean(await studio.webContents.executeJavaScript(script, toggle));
}

module.exports = { conversationPage, controlConversation };
