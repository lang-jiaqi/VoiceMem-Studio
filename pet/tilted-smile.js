// Timing matches mp4/tilted_smile_smooth.mp4 and rebuild_60fps.py (240 frames / 60 fps).
(function(root,factory){const profile=factory();if(typeof module==='object'&&module.exports)module.exports=profile;else root.TiltedSmile=profile;})(globalThis,()=>{
  const duration=4;
  const smoother=x=>{const t=Math.max(0,Math.min(1,x));return Math.max(0,Math.min(1,t*t*t*(t*(t*6-15)+10)));};
  function sample(seconds){
    const weight=Number.isFinite(seconds)&&seconds>.2&&seconds<3.75
      ? smoother((seconds-.2)/1.35)*(1-smoother((seconds-2.2)/1.55)) : 0;
    return {weight,ParamAngleZ:-8*weight,ParamMouthForm:.9*weight,ParamMouthOpenY:.48*weight};
  }
  function compose(base,seconds,pose='sit',speechMouth){
    const action=sample(seconds),w=action.weight;
    return {...base,
      ParamAngleZ:pose==='sit' ? base.ParamAngleZ*(1-w)+action.ParamAngleZ : 0,
      ParamEyeLOpen:base.ParamEyeLOpen*(1-w),ParamEyeROpen:base.ParamEyeROpen*(1-w),
      ParamMouthForm:action.ParamMouthForm,
      ParamMouthOpenY:speechMouth===undefined ? action.ParamMouthOpenY : speechMouth};
  }
  return {duration,sample,compose};
});
