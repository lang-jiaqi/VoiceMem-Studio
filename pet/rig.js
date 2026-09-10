/* Real Cubism parameter animation. Mouth is written after motion evaluation. */
window.petRig = (() => {
  const canvas = document.querySelector('#live'), stage = document.querySelector('#stage');
  let app, model, active=false, elapsed=0, mouth=0, talkUntil=0, smileStart=-Infinity, nextSmileAt=Infinity, currentPose='sit',showEpoch=0;
  const loads=new Map();
  const linked=Boolean(new URLSearchParams(location.search).get('ws'));
  const smoothAngles={ParamAngleX:0,ParamAngleY:0,ParamAngleZ:0};
  let lastMotionTime=0;
  let gesture=null,gestureStart=0;
  const ease=x=>{const v=Math.max(0,Math.min(1,x));return v*v*v*(v*(v*6-15)+10);};
  let failure, overridePose, sample;
  const hitCanvas=document.createElement('canvas');hitCanvas.width=100;hitCanvas.height=100;
  const hitContext=hitCanvas.getContext('2d',{willReadFrequently:true});let hitPixels,hitAt=-Infinity;
  function apply(core, t, pose) {
    const blinkPhase=t%4.7;
    const blink=blinkPhase>3.8 && blinkPhase<4.04 ? 1-Math.sin((blinkPhase-3.8)/.24*Math.PI) : 1;
    const actionTime=currentPose==='sit'?t-smileStart:Infinity;
    const action=TiltedSmile.sample(actionTime);
    const desired= t<talkUntil ? .15+.65*Math.abs(Math.sin(t*11)*Math.sin(t*6.3)) : linked?0:action.ParamMouthOpenY;
    mouth += (desired-mouth)*.3;
    const base={ParamAngleX:0,ParamAngleY:.6*Math.sin(t*.7),ParamAngleZ:1.2*Math.sin(t*.9),
      ParamBodyAngleX:0,ParamBodyAngleY:0,ParamBodyAngleZ:0,ParamBreath:.5+.5*Math.sin(t*1.6),
      ParamEyeLOpen:blink,ParamEyeROpen:blink,ParamMouthForm:0,ParamMouthOpenY:mouth};
    const values={...TiltedSmile.compose(base,actionTime,currentPose,mouth),...pose};
    if(gesture && !model.internalModel.motionManager.isFinished()) {
      const age=t-gestureStart,nod=ease(age/.65)*(1-ease((age-.8)/1));
      const ids=gesture==='Nod' ? ['ParamAngleY','ParamBodyAngleY','ParamEyeLOpen','ParamEyeROpen']
        : ['ParamAngleX','ParamBodyAngleX','ParamAngleZ'];
      for(const id of ids) if(!(id in (pose||{}))) {
        const value=gesture==='Nod'?(id==='ParamAngleY'?-18*nod:id.startsWith('ParamEye')?1-.15*nod:0):core.getParameterValueById(id);
        values[id]=id.startsWith('ParamEye')?Math.min(values[id],value):value;
      }
    } else gesture=null;
    if(currentPose==='lie'){values.ParamAngleX=values.ParamAngleY=values.ParamAngleZ=values.ParamBreath=0;}
    const blend=1-Math.exp(-Math.max(0,t-lastMotionTime)/.12);lastMotionTime=t;
    for(const id of Object.keys(smoothAngles)){
      smoothAngles[id]=(pose&&id in pose)?values[id]:smoothAngles[id]+(values[id]-smoothAngles[id])*blend;
      values[id]=smoothAngles[id];
    }
    for(const [id,v] of Object.entries(values)) core.setParameterValueById(id,v);
    sample=values;
  }
  async function load(pose) {
    if(loads.has(pose)) return loads.get(pose);
    const loading=(async()=>{
      if(!app){
      app=new PIXI.Application({view:canvas,width:350,height:440,backgroundAlpha:0,antialias:true,autoStart:false,resolution:window.devicePixelRatio||1,autoDensity:true,preserveDrawingBuffer:true});
      app.ticker.add(()=>{ if(!active||!model) return; elapsed+=Math.min(app.ticker.elapsedMS,50)/1000;
        if(!linked&&currentPose==='sit'&&elapsed>=nextSmileAt){smileStart=elapsed;nextSmileAt=elapsed+TiltedSmile.duration+20+Math.random()*25;}
        model.update(Math.min(app.ticker.elapsedMS,50)); });
      }
      const loaded=await PIXI.live2d.Live2DModel.from(`models/${pose}/noctelle-${pose}.model3.json`,{autoInteract:false,autoUpdate:false});
      loaded.internalModel.motionManager.stopAllMotions();
      await restoreHairTransparency(loaded,pose);
      await calibrateFace(loaded,pose);
      calibrateMotion(loaded,pose);
      const core=loaded.internalModel.coreModel, opacity=core.getDrawableOpacity.bind(core);
      const open=core.getDrawableIndex('ArtMeshMouthOpen'), closed=core.getDrawableIndex('ArtMeshMouthClose');
      const eyes=[core.getDrawableIndex('ArtMeshEyewhiteL'),core.getDrawableIndex('ArtMeshEyewhiteR')];
      let amounts=[0,1,1];
      const update=core.update.bind(core);
      // Capture evaluated values before the framework restores saved parameters.
      core.update=()=>{
        amounts=['ParamMouthOpenY','ParamEyeLOpen','ParamEyeROpen'].map(id=>Math.max(0,Math.min(1,core.getParameterValueById(id))));
        update();
      };
      // The exported model leaves both mouth layers visible even at rest.
      core.getDrawableOpacity=index=>{
        const eye=eyes.indexOf(index),amount=amounts[0];
        return opacity(index)*(index===open?amount:index===closed?1-amount:eye>=0?amounts[eye+1]:1);
      };
      loaded.internalModel.on('beforeModelUpdate',()=>apply(loaded.internalModel.coreModel,elapsed,overridePose));
      loaded.visible=false;app.stage.addChild(loaded);
      return loaded;
    })().catch(e=>{failure=String(e);console.error('Model load failed',e);throw e;});
    loads.set(pose,loading);
    return loading;
  }
  function resize(){
    if(!app||!model) return;
    const w=Math.max(stage.clientWidth,100),h=Math.max(stage.clientHeight,100);
    app.renderer.resize(w,h); model.scale.set(1);
    const scale=Math.min((w-8)/model.width,(h-8)/model.height);
    model.scale.set(scale);model.x=(w-model.width)/2;model.y=h-model.height;
  }
  window.addEventListener('resize',resize);
  return {
    async show(pose='sit'){
      const epoch=++showEpoch;active=true;failure=undefined;
      gesture=null;if(model)model.internalModel.motionManager.stopAllMotions();
      for(const id of Object.keys(smoothAngles))smoothAngles[id]=0;
      const loaded=await load(pose);if(epoch!==showEpoch)return;
      if(model)model.visible=false;model=loaded;currentPose=pose;model.visible=true;
      smileStart=-Infinity;nextSmileAt=elapsed+20+Math.random()*25;hitAt=-Infinity;
      canvas.hidden=false;resize();app.start();
      window.dispatchEvent(new Event('pet-ready'));
    },
    hide(){showEpoch++;active=false;gesture=null;if(model)model.internalModel.motionManager.stopAllMotions();talkUntil=0;mouth=0;smileStart=-Infinity;nextSmileAt=Infinity;if(app)app.stop();canvas.hidden=true;},
    talk(seconds=5){talkUntil=elapsed+seconds;}, stopTalking(){talkUntil=0;mouth=0;},
    async gesture(name){
      if(!active||currentPose!=='sit'||!['Nod','Shake'].includes(name))return false;
      const epoch=showEpoch, current=model;
      smileStart=-Infinity;nextSmileAt=elapsed+25;
      const started=await current.motion(name,0,3);
      if(epoch!==showEpoch){current.internalModel.motionManager.stopAllMotions();return false;}
      if(started){gesture=name;gestureStart=elapsed;}
      return started;
    },
    tilt(){if(currentPose!=='sit'||!active)return false;if(elapsed-smileStart<TiltedSmile.duration)return false;gesture=null;model.internalModel.motionManager.stopAllMotions();smileStart=elapsed;nextSmileAt=elapsed+TiltedSmile.duration+20+Math.random()*25;return true;},
    status(){return {ready:Boolean(model),pose:currentPose,error:failure,active,action:elapsed-smileStart<TiltedSmile.duration?'tilted-smile':null,parameters:sample};},
    hitTest(clientX,clientY){
      if(!active||!model)return false;
      const rect=canvas.getBoundingClientRect(),x=Math.floor((clientX-rect.left)/rect.width*100),y=Math.floor((clientY-rect.top)/rect.height*100);
      if(x<0||y<0||x>=100||y>=100)return false;
      if(performance.now()-hitAt>150){hitContext.clearRect(0,0,100,100);hitContext.drawImage(canvas,0,0,100,100);hitPixels=hitContext.getImageData(0,0,100,100).data;hitAt=performance.now();}
      return Boolean(hitPixels&&hitPixels[(y*100+x)*4+3]>12);
    },
    // Deterministic rendered poses for acceptance checks; normal playback restores afterwards.
    inspectPose(pose={}){
      if(!model)throw new Error('Model is not ready');
      app.stop();
      overridePose=pose;model.update(16);app.renderer.render(app.stage);overridePose=undefined;
      const core=model.internalModel.coreModel;
      const ids=core.getDrawableIds();
      return Object.fromEntries(['ArtMeshFace','ArtMeshMouthOpen','ArtMeshTopwear'].map(prefix=>[prefix,ids.filter(id=>id.startsWith(prefix)).flatMap(id=>Array.from(core.getDrawableVertices(core.getDrawableIndex(id))))]));
    }
  };
})();
