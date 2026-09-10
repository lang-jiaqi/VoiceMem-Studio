const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('pet', {
  initial: () => ipcRenderer.invoke('initial-mode'),
  onMode: callback => ipcRenderer.on('mode', (_event, mode) => callback(mode)),
  actions: () => ipcRenderer.send('actions'),
  onAction: callback => ipcRenderer.on('action', (_event, action) => callback(action)),
  toggle: () => ipcRenderer.send('toggle'), collapse: () => ipcRenderer.send('collapse'),
  activate: pose => ipcRenderer.send('activate',pose),
  quit: () => ipcRenderer.send('quit'), pointer: hit => ipcRenderer.send('pointer', Boolean(hit)),
  dragStart: () => ipcRenderer.send('drag-start'), dragMove: () => ipcRenderer.send('drag-move'), dragEnd: () => ipcRenderer.send('drag-end')
});
