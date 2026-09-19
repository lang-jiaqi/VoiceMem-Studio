'use strict';
const { contextBridge, ipcRenderer } = require('electron');
const send = (name, value) => ipcRenderer.send(`studio-pet:${name}`, value);
const on = (name, callback) => ipcRenderer.on(`studio-pet:${name}`, (_event, value) => callback(value));

// Preserve the existing pet renderer contract without exposing raw IPC or Node.
contextBridge.exposeInMainWorld('pet', {
  initial: () => ipcRenderer.invoke('studio-pet:initial'),
  onMode: callback => on('mode', callback),
  toggleConversation: () => ipcRenderer.invoke('studio-pet:toggle-conversation'),
  conversationState: () => ipcRenderer.invoke('studio-pet:conversation-state'),
  onConversationState: callback => on('conversation-state', callback),
  onConversationError: callback => on('conversation-error', callback),
  initialScale: () => ipcRenderer.invoke('studio-pet:initial-scale'),
  onScale: callback => on('scale', callback),
  resize: step => send('resize', step), resetSize: () => send('reset-size'),
  resizeStart: corner => send('resize-start', corner), resizeMove: () => send('resize-move'), resizeEnd: () => send('resize-end'),
  toggle: () => send('toggle'),
  collapse: () => send('collapse'), activate: pose => send('activate', pose),
  pointer: hit => send('pointer', Boolean(hit)),
  dragStart: () => send('drag-start'), dragMove: () => send('drag-move'), dragEnd: () => send('drag-end'),
});
