const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('pet', {
  initial: () => ipcRenderer.invoke('initial-mode'),
  onMode: callback => ipcRenderer.on('mode', (_event, mode) => callback(mode)),
  toggleConversation: () => Promise.reject(new Error('请在 Studio App 中开启对话。')),
  conversationState: () => Promise.resolve(false),
  onConversationState: () => {},
  onConversationError: () => {},
  initialScale: () => ipcRenderer.invoke('initial-scale'),
  onScale: callback => ipcRenderer.on('scale', (_event, scale) => callback(scale)),
  resize: step => ipcRenderer.send('resize', step), resetSize: () => ipcRenderer.send('reset-size'),
  resizeStart: corner => ipcRenderer.send('resize-start', corner), resizeMove: () => ipcRenderer.send('resize-move'), resizeEnd: () => ipcRenderer.send('resize-end'),
  toggle: () => ipcRenderer.send('toggle'), collapse: () => ipcRenderer.send('collapse'),
  activate: pose => ipcRenderer.send('activate',pose),
  pointer: hit => ipcRenderer.send('pointer', Boolean(hit)),
  dragStart: () => ipcRenderer.send('drag-start'), dragMove: () => ipcRenderer.send('drag-move'), dragEnd: () => ipcRenderer.send('drag-end')
});
