'use strict';
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('studioModelServices', {
  state: () => ipcRenderer.invoke('studio-desktop:model-services'),
  update: (role, value) => ipcRenderer.invoke('studio-desktop:update-model-service', role, value),
});
