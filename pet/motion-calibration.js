/* Shared upper-body deformation keeps independently exported layers connected. */
function calibrateMotion(model, pose) {
  const core=model.internalModel.coreModel, info=core.getModel().canvasinfo;
  const vertices=core.getDrawableVertices.bind(core), update=core.update.bind(core);
  const ids=['ParamAngleX','ParamAngleY','ParamAngleZ','ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ','ParamBreath'];
  let angles=[0,0,0],breath=0;
  const buffers=new Map(), lids=new Map();
  const centers=pose==='sit'?[[513,216.5],[605.5,204]]:[[301,263.5],[394,237.5]];
  const angle=Math.atan2(centers[1][1]-centers[0][1],centers[1][0]-centers[0][0]);
  const left=core.getDrawableIndex('ArtMeshEyeCloseL'),right=core.getDrawableIndex('ArtMeshEyeCloseR');
  const source=Array.from(vertices(right));
  const xs=source.filter((_,i)=>i%2===0),ys=source.filter((_,i)=>i%2===1);
  const cx=(Math.min(...xs)+Math.max(...xs))/2,cy=(Math.min(...ys)+Math.max(...ys))/2;
  let xx=0,yy=0,xy=0;
  for(let i=0;i<source.length;i+=2){const x=source[i]-cx,y=cy-source[i+1];xx+=x*x;yy+=y*y;xy+=x*y;}
  const sourceAngle=.5*Math.atan2(2*xy,xx-yy);
  // Mirror one eyelid around the face's tilted axis, not the screen's horizontal.
  for(const [side,index] of [right,left].entries()){
    const points=new Float32Array(source.length),[x,y]=centers[side];
    for(let i=0;i<source.length;i+=2){
      const sx=source[i]-cx,sy=cy-source[i+1];
      const dx=(sx*Math.cos(sourceAngle)+sy*Math.sin(sourceAngle))*(side===0?1:-1),dy=-sx*Math.sin(sourceAngle)+sy*Math.cos(sourceAngle);
      points[i]=(x-info.CanvasOriginX-5*Math.sin(angle))/info.PixelsPerUnit+dx*Math.cos(angle)-dy*Math.sin(angle);
      points[i+1]=(info.CanvasOriginY-y-5*Math.cos(angle))/info.PixelsPerUnit-dx*Math.sin(angle)-dy*Math.cos(angle);
    }
    lids.set(index,points);
  }
  for(const method of ['getDrawableVertexCount','getDrawableVertexIndexCount','getDrawableVertexIndices','getDrawableVertexUvs','getDrawableTextureIndices']){
    const original=core[method].bind(core);
    const shared=original(right);
    // Reversed winding preserves culling when the geometry is mirrored.
    const mirrored=method==='getDrawableVertexIndices'?new Uint16Array(Array.from(shared).reduce((out,_,i,a)=>i%3?out:out.concat(a[i],a[i+2],a[i+1]),[])):shared;
    core[method]=index=>index===left?mirrored:original(index);
  }
  core.update=()=>{
    const values=ids.map(id=>core.getParameterValueById(id));
    angles=pose==='sit'?values.slice(0,3):[0,0,0];
    breath=values[6];
    // Disable disconnected native head/body transforms, retaining facial animation.
    for(const id of ids)core.setParameterValueById(id,0);
    update();
    ids.forEach((id,i)=>core.setParameterValueById(id,values[i]));
  };
  core.getDrawableVertices=index=>{
    const raw=lids.get(index)||vertices(index);
    if(pose!=='sit')return raw;
    let out=buffers.get(index);
    if(!out||out.length!==raw.length){out=new Float32Array(raw.length);buffers.set(index,out);}
    const [ax,ay,az]=angles.map((v,i)=>Math.max(-[20,18,8][i],Math.min([20,18,8][i],v)));
    const rotation=-az*.5*Math.PI/180,pivot=(info.CanvasOriginY-420)/info.PixelsPerUnit;
    const pitch=-ay*Math.PI/180,cp=Math.cos(pitch),sp=Math.sin(pitch),ca=Math.cos(angle),sa=Math.sin(angle);
    const neckX=(centers[0][0]+centers[1][0])/2-110*sa,neckY=(centers[0][1]+centers[1][1])/2+110*ca;
    for(let i=0;i<raw.length;i+=2){
      let x=raw[i],y=raw[i+1];
      const pixelX=info.CanvasOriginX+x*info.PixelsPerUnit,pixelY=info.CanvasOriginY-y*info.PixelsPerUnit;
      // Pitch a shallow head surface in perspective, anchored above the shoulders.
      // All layers use the same field so face, hair and neck stay connected.
      const hx=(pixelX-neckX)*ca+(pixelY-neckY)*sa,hy=-(pixelX-neckX)*sa+(pixelY-neckY)*ca;
      const depth=75*Math.max(0,1-(hx/165)**2-((hy+110)/200)**2);
      const perspective=1600/(1600+hy*sp-depth*(cp-1));
      const px=hx*perspective,py=(hy*cp+depth*sp)*perspective;
      const n=Math.max(0,Math.min(1,(360-pixelY)/65)),follow=n*n*(3-2*n);
      x+=((px-hx)*ca-(py-hy)*sa)*follow/info.PixelsPerUnit;
      y-=((px-hx)*sa+(py-hy)*ca)*follow/info.PixelsPerUnit;
      const t=Math.max(0,Math.min(1,(620-pixelY)/300)),w=t*t*(3-2*t),r=rotation*w;
      out[i]=x*Math.cos(r)-(y-pivot)*Math.sin(r)+ax*.00045*w;
      out[i+1]=pivot+x*Math.sin(r)+(y-pivot)*Math.cos(r)+breath*.0005*w;
    }
    return out;
  };
}
