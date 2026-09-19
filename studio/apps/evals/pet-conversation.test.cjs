'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { conversationPage, controlConversation } = require('../pet-conversation.cjs');

test('pet toggles only the active Studio voice button with a user gesture', async () => {
  const origin = 'http://127.0.0.1:8787';
  let page = `${origin}/ui/digital.html`, active = false;
  const button = { click() { active = !active; }, getAttribute: name => name === 'aria-pressed' ? String(active) : null };
  const gestures = [];
  const studio = { isDestroyed: () => false, webContents: {
    getURL: () => page,
    executeJavaScript: async (source, userGesture) => {
      gestures.push(userGesture);
      return vm.runInNewContext(source, { document: { getElementById: id => id === 'startTalk' ? button : null } });
    },
  } };
  assert.equal(conversationPage(studio, origin), true);
  assert.equal(await controlConversation(studio, origin), false);
  assert.equal(await controlConversation(studio, origin, true), true);
  assert.equal(await controlConversation(studio, origin, true), false);
  assert.deepEqual(gestures, [false, true, true]);
  page = `${origin}/ui/technical.html`;
  assert.equal(await controlConversation(studio, origin, true), true);
  page = `${origin}/ui/index.html`;
  await assert.rejects(controlConversation(studio, origin, true), /对话页面/);
  page = 'https://other.example/ui/digital.html';
  await assert.rejects(controlConversation(studio, origin, true), /对话页面/);
  assert.equal(gestures.length, 4);
});
