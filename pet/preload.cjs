const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('pet', {
  initial: () => ipcRenderer.invoke('initial-mode'),
  onMode: callback => ipcRenderer.on('mode', (_event, mode) => callback(mode)),
  toggle: () => ipcRenderer.send('toggle'), collapse: () => ipcRenderer.send('collapse'),
  quit: () => ipcRenderer.send('quit'), pointer: hit => ipcRenderer.send('pointer', Boolean(hit)),
  dragStart: () => ipcRenderer.send('drag-start'), dragMove: () => ipcRenderer.send('drag-move'), dragEnd: () => ipcRenderer.send('drag-end')
});
