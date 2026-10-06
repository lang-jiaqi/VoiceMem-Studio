/* Shared controls create page-local instances; conversation data never crosses styles. */
(() => {
  'use strict';
  let toastTimer;
  function sizeReading() {
    const width=window.innerWidth||1440;
    const monitor=window.screen?.availWidth||width;
    const allowance=width>=1000?Math.min(1.12,Math.max(1,monitor/1920)):1;
    const font=width<=700?17:Math.min(28,Math.max(18,12+width*.0045)*allowance);
    document.documentElement.style.setProperty('--reading-font',(font*(window.VMSettings?.contentScale||1)).toFixed(2)+'px');
    document.documentElement.style.setProperty('--ui-font',(Math.min(19,Math.max(14,font*.76))*(window.VMSettings?.uiScale||1)).toFixed(2)+'px');
    document.documentElement.style.setProperty('--memory-font',(Math.min(19,Math.max(14,font*.76))*(window.VMSettings?.contentScale||1)).toFixed(2)+'px');
  }
  document.addEventListener('display-settings-change',sizeReading);window.addEventListener('resize',sizeReading);window.addEventListener('pageshow',sizeReading);sizeReading();
  function notify(text) {
    let box = document.getElementById('toast');
    if (!box) { box = document.createElement('div'); box.id = 'toast'; box.className = 'toast'; box.setAttribute('role','status'); document.body.append(box); }
    box.textContent = text; box.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { box.hidden = true; }, 3500);
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); notify('已复制'); }
    catch { notify('复制未成功，请选中文字后复制。'); }
  }
  function voice({onState,onInterim,onFinal}) {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let current = null, active = false;
    const set = value => { active = value; onState(value); };
    function cancel() {
      const old = current; current = null;
      if(old) { old.onresult=old.onend=old.onerror=null; try {old.abort();} catch {} }
      set(false);
    }
    function start() {
      if(active) return;
      if(!Recognition) { notify('此浏览器不支持语音识别，请在输入框中开始对话。'); return; }
      const rec = new Recognition(); current = rec;
      rec.lang=window.VMSettings?.language||'zh-CN'; rec.interimResults=true; rec.continuous=false;
      let finalText='', interimText='';
      rec.onresult = event => {
        if(current!==rec) return;
        interimText='';
        for(let i=event.resultIndex;i<event.results.length;i++) {
          const result=event.results[i];
          if(result.isFinal) finalText+=result[0].transcript; else interimText+=result[0].transcript;
        }
        onInterim(finalText+interimText);
      };
      rec.onerror = event => {
        if(current!==rec) return;
        cancel();
        notify(event.error==='not-allowed' ? '麦克风权限未开启，可继续文字对话。' : '未识别到语音，请重试或输入文字。');
      };
      rec.onend = () => {
        if(current!==rec) return;
        current=null;set(false);
        if(finalText.trim()) onFinal(finalText.trim());
        else if(interimText) notify('语音未确认，文字已保留在输入框。');
      };
      try {set(true);rec.start();} catch {cancel();notify('麦克风暂不可用，请使用文字输入。');}
    }
    function stop() { if(current) {try {current.stop();} catch {cancel();}} }
    document.addEventListener('display-settings-change',()=>{if(active&&current?.lang!==(window.VMSettings?.language||'zh-CN'))cancel();});
    window.addEventListener('pagehide',cancel);
    document.addEventListener('visibilitychange',()=>{if(document.hidden)cancel();});
    return {start,stop,cancel,toggle:()=>active?stop():start()};
  }
  window.addEventListener('pagehide',()=>clearTimeout(toastTimer));
  window.VMUI = {notify,copy,voice,reduced:matchMedia('(prefers-reduced-motion: reduce)').matches};
})();
