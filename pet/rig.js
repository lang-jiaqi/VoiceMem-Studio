/* Real Cubism parameter animation. Mouth is written after motion evaluation. */
window.petRig = (() => {
  const canvas = document.querySelector('#live'), stage = document.querySelector('#stage');
  let app, model, active=false, elapsed=0, mouth=0, talkUntil=0, smileStart=-Infinity, nextSmileAt=Infinity, nodStart=-Infinity, currentPose='sit',showEpoch=0;
  const loads=new Map();
  let failure, overridePose, sample;
  const hitCanvas=document.createElement('canvas');hitCanvas.width=100;hitCanvas.height=100;
  const hitContext=hitCanvas.getContext('2d',{willReadFrequently:true});let hitPixels,hitAt=-Infinity;
  function enlargeEyes(core) {
    const scale=1.18;
    const eyes=new Set(['ArtMeshEyewhiteR','ArtMeshEyewhiteL','ArtMeshEyeCloseR','ArtMeshEyeCloseL'].map(id=>core.getDrawableIndex(id)));
    const original=core.getDrawableVertices.bind(core), buffers=new Map();
    // Scale a copy of the current deformed mesh; never accumulate changes in Cubism's buffers.
    core.getDrawableVertices=index=>{
      const vertices=original(index);
      if(!eyes.has(index)||!vertices.length)return vertices;
      let result=buffers.get(index);
      if(!result||result.length!==vertices.length){result=new Float32Array(vertices.length);buffers.set(index,result);}
      let left=Infinity,right=-Infinity,bottom=Infinity,top=-Infinity;
      for(let i=0;i<vertices.length;i+=2){left=Math.min(left,vertices[i]);right=Math.max(right,vertices[i]);bottom=Math.min(bottom,vertices[i+1]);top=Math.max(top,vertices[i+1]);}
      const cx=(left+right)/2,cy=(bottom+top)/2;
      for(let i=0;i<vertices.length;i+=2){result[i]=cx+(vertices[i]-cx)*scale;result[i+1]=cy+(vertices[i+1]-cy)*scale;}
      return result;
    };
  }
  /* 待机动作直接用作者写好的 idle.motion3.json（见 idle.js，关键帧逐个照抄）。
     只留一个总倍率：作者给的 ParamAngleZ 是 ±2，量程 ±30，原样播几乎看不出在动。
     scale 只放大角度，呼吸和眨眼是 0~1 的开合量，放大会坏掉——所以在 idle.js 里
     用 SCALABLE 挡住了。

     启动时 --idle=4，或者开着 DevTools 敲 petRig.tune({scale:4})，都能边看边调。
     调大到脖子接缝穿帮就往回收：使用说明里写过坐姿"头颈使用少量重叠，只适合小幅运动"，
     作者把幅度压这么小多半就是为了这个。 */
  const IDLE={scale:1};
  function apply(core, t, pose) {
    const idle=Idle.sample(t,IDLE.scale);
    const actionTime=currentPose==='sit'?t-smileStart:Infinity;
    const action=TiltedSmile.sample(actionTime);
    const desired= t<talkUntil ? .15+.65*Math.abs(Math.sin(t*11)*Math.sin(t*6.3)) : action.ParamMouthOpenY;
    mouth += (desired-mouth)*.3;
    // 作者的 idle 只动这五条；其余参数保持 0，跟原来一样。
    const base={ParamAngleX:0,ParamAngleY:0,ParamAngleZ:idle.ParamAngleZ,
      ParamBodyAngleX:idle.ParamBodyAngleX,ParamBodyAngleY:0,ParamBodyAngleZ:0,
      ParamBreath:idle.ParamBreath,
      ParamEyeLOpen:idle.ParamEyeLOpen,ParamEyeROpen:idle.ParamEyeROpen,
      ParamMouthForm:0,ParamMouthOpenY:mouth};
    const composed=TiltedSmile.compose(base,actionTime,currentPose,mouth);
    // 点头夹在微笑之后、override 之前：inspectPose 传进来的 pose 仍然要能钉死参数。
    // 只在坐姿做——躺姿下面那行本来就把头部三轴清零了（PSD2Live 的横卧形变限制）。
    if(currentPose==='sit') Nod.apply(composed,t-nodStart);
    const values={...composed,...pose};
    // 躺姿只留脸上的动作：身体和头部的形变是 PSD2Live 横卧的短板，一动就穿帮。
    // 眼球、眉毛、眨眼、嘴不涉及身体网格，照常。
    if(currentPose==='lie'){values.ParamAngleX=values.ParamAngleY=values.ParamAngleZ=
      values.ParamBodyAngleX=values.ParamBreath=0;}
    for(const [id,v] of Object.entries(values)) core.setParameterValueById(id,v);
    sample=values;
  }
  async function load(pose) {
    if(loads.has(pose)) return loads.get(pose);
    const loading=(async()=>{
      if(!app){
      app=new PIXI.Application({view:canvas,width:350,height:440,backgroundAlpha:0,antialias:true,autoStart:false,resolution:window.devicePixelRatio||1,autoDensity:true,preserveDrawingBuffer:true});
      app.ticker.add(()=>{ if(!active||!model) return; elapsed+=Math.min(app.ticker.elapsedMS,50)/1000;
        if(currentPose==='sit'&&elapsed>=nextSmileAt){smileStart=elapsed;nextSmileAt=elapsed+TiltedSmile.duration+20+Math.random()*25;}
        model.update(Math.min(app.ticker.elapsedMS,50)); });
      }
      const loaded=await PIXI.live2d.Live2DModel.from(`models/${pose}/noctelle-${pose}.model3.json`,{autoInteract:false,autoUpdate:false});
      loaded.internalModel.motionManager.stopAllMotions();
      enlargeEyes(loaded.internalModel.coreModel);
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
  // --idle=1.6 透传进来的总倍率，省得为了试一个数去改代码。
  const idleArg=Number(new URLSearchParams(location.search).get('idle'));
  if(Number.isFinite(idleArg)&&idleArg>0) IDLE.scale=idleArg;
  return {
    async show(pose='sit'){
      const epoch=++showEpoch;active=true;failure=undefined;
      const loaded=await load(pose);if(epoch!==showEpoch)return;
      if(model)model.visible=false;model=loaded;currentPose=pose;model.visible=true;
      smileStart=-Infinity;nextSmileAt=elapsed+20+Math.random()*25;nodStart=-Infinity;hitAt=-Infinity;
      canvas.hidden=false;resize();app.start();
    },
    hide(){showEpoch++;active=false;talkUntil=0;mouth=0;smileStart=-Infinity;nextSmileAt=Infinity;nodStart=-Infinity;if(app)app.stop();canvas.hidden=true;},
    talk(seconds=5){talkUntil=elapsed+seconds;}, stopTalking(){talkUntil=0;},
    // 接话（"嗯""对啊"）时点一下头。连着来两声不会打断上一个点头，
    // 不然幅度叠在一起会变成抽搐。
    // 边看边调：petRig.tune({scale:1.6}) / tune({angleZ:8})，返回当前全套系数。
    tune(next={}){Object.assign(IDLE,next);return {...IDLE};},
    nod(){if(currentPose!=='sit'||!active)return false;if(elapsed-nodStart<Nod.duration)return false;nodStart=elapsed;return true;},
    tilt(){if(currentPose!=='sit'||!active)return false;if(elapsed-smileStart<TiltedSmile.duration)return false;smileStart=elapsed;nextSmileAt=elapsed+TiltedSmile.duration+20+Math.random()*25;return true;},
    status(){return {ready:Boolean(model),pose:currentPose,error:failure,active,action:elapsed-smileStart<TiltedSmile.duration?'tilted-smile':(elapsed-nodStart<Nod.duration?'nod':null),parameters:sample};},
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
