/* Observe actual browser playback, never generation-completion events. */
(() => {
  const url=new URLSearchParams(location.search).get('ws');
  if(!url)return;
  const WINDOW_S=1,finished=new Set();
  let socket,retries=0,timer,deadline=0,output=null,heard=false,pending;
  let idleTimer,speaking=false,resting=true;
  const retire=id=>{if(id!==null)finished.add(id);if(finished.size>64)finished.delete(finished.values().next().value);};
  const wake=sit=>window.pet?.activate?.(sit?'sit':undefined);
  function flush(){
    const state=window.petRig?.status();
    if(!state?.ready||!state.active)return;
    if(deadline>Date.now())window.petRig.talk((deadline-Date.now())/1000);
    if(pending&&pending.until>Date.now()&&state.pose==='sit'){
      const name=pending.name;pending=undefined;
      if(name==='tilt')window.petRig.tilt();
      else window.petRig.gesture(name).catch(console.error);
    }
  }
  function armRest(){
    clearTimeout(idleTimer);
    idleTimer=setTimeout(()=>{resting=true;pending=undefined;window.pet?.activate?.('lie');},20000);
  }
  function voice(active){
    if(active){clearTimeout(idleTimer);resting=false;wake(true);}
    else if(speaking)armRest();
    speaking=active;
  }
  function action(name){if(resting)return;pending={name,until:Date.now()+2000};wake(true);flush();}
  function hush(){clearTimeout(timer);deadline=0;window.petRig?.stopTalking();}
  function reset(){hush();retire(output);output=null;heard=false;pending=undefined;}
  function handle(message){
    if(message.type==='conversation_started'){speaking=false;resting=false;wake(true);armRest();return;}
    if(message.type==='user_voice'){voice(message.active===true);return;}
    if(message.type==='conversation_ended'){voice(false);reset();return;}
    if(message.type==='backchannel'){action('Nod');return;}
    if(message.type==='answer_interrupt'){reset();return;}
    if(message.type!=='playback_checkpoint')return;
    const id=message.output_id;
    if(!id||finished.has(id))return;
    if(message.state==='playing'){
      if(output!==id){retire(output);output=id;heard=false;pending=undefined;}
      heard=true;deadline=Date.now()+WINDOW_S*1000;
      clearTimeout(timer);timer=setTimeout(hush,WINDOW_S*1000);
      wake(false);flush();return;
    }
    if(id!==output)return;
    if(['paused','stalled','drained','interrupted'].includes(message.state))hush();
    if(message.state==='drained'){
      const tilt=heard&&Math.random()<.3;
      reset();if(tilt)action('tilt');
    }else if(message.state==='interrupted')reset();
  }
  window.addEventListener('pet-ready',flush);
  function connect(){
    try{socket=new WebSocket(url);}catch{return schedule();}
    socket.onopen=()=>{retries=0;};
    socket.onmessage=event=>{
      let message;try{message=JSON.parse(event.data);}catch{return;}
      if(message&&typeof message==='object')handle(message);
    };
    socket.onclose=()=>{voice(false);reset();schedule();};
    socket.onerror=()=>{try{socket.close();}catch{}};
  }
  function schedule(){setTimeout(connect,Math.min(5000,250*2**retries++));}
  connect();
})();
