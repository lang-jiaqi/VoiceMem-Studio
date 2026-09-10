/* Restore reference facial artwork while retaining the model's motion. */
async function calibrateFace(model, pose) {
  const boxes = {
    lie: {EyewhiteR:[263,243,339,284],EyewhiteL:[354,218,434,257],EyebrowR:[263,222,332,243],EyebrowL:[360,197,431,218],MouthClose:[340,290,391,318]},
    sit: {EyewhiteR:[474,198,552,235],EyewhiteL:[567,184,644,224],EyebrowR:[479,174,548,198],EyebrowL:[572,158,640,184],MouthClose:[541,257,589,280]}
  }[pose];
  const reference=await PIXI.Texture.fromURL(`assets/${pose}.png`);
  const canvas=document.createElement('canvas');canvas.width=reference.width;canvas.height=reference.height;
  const context=canvas.getContext('2d',{willReadFrequently:true});
  for(const [left,top,right,bottom] of Object.values(boxes)){
    const width=right-left,height=bottom-top;
    context.drawImage(reference.baseTexture.resource.source,left,top,width,height,left,top,width,height);
    const pixels=context.getImageData(left,top,width,height);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)pixels.data[(y*width+x)*4+3]=255*Math.min(1,x/2,y/2,(width-1-x)/2,(height-1-y)/2);
    context.putImageData(pixels,left,top);
  }
  const texture=model.textures.length;model.textures.push(PIXI.Texture.from(canvas));
  const core=model.internalModel.coreModel, info=core.getModel().canvasinfo;
  const vertices=core.getDrawableVertices.bind(core), patches=new Map();
  core.saveParameters();
  for(const id of ['ParamAngleX','ParamAngleY','ParamAngleZ','ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ','ParamBreath','ParamMouthOpenY','ParamMouthForm'])core.setParameterValueById(id,0);
  core.setParameterValueById('ParamEyeLOpen',1);core.setParameterValueById('ParamEyeROpen',1);core.update();
  for(const [name,[left,top,right,bottom]] of Object.entries(boxes)){
    const index=core.getDrawableIndex('ArtMesh'+name), source=Array.from(vertices(index));
    const points=[];
    for(let i=0;i<source.length;i+=2)points.push([source[i]*info.PixelsPerUnit+info.CanvasOriginX,-source[i+1]*info.PixelsPerUnit+info.CanvasOriginY]);
    // Use the widest triangle to carry reference corners with the animated mesh.
    let triangle, area=0;
    for(let a=0;a<points.length;a++)for(let b=a+1;b<points.length;b++)for(let c=b+1;c<points.length;c++){
      const [p,q,r]=[points[a],points[b],points[c]], d=(q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]);
      if(Math.abs(d)>Math.abs(area)){area=d;triangle=[a,b,c];}
    }
    if(!triangle)throw new Error(`Cannot calibrate ${pose}/${name}: missing face geometry`);
    const [p,q,r]=triangle.map(i=>points[i]), corners=[[left,top],[right,top],[right,bottom],[left,bottom]];
    const weights=corners.map(([x,y])=>{
      const b=((x-p[0])*(r[1]-p[1])-(y-p[1])*(r[0]-p[0]))/area;
      const c=((q[0]-p[0])*(y-p[1])-(q[1]-p[1])*(x-p[0]))/area;
      return [1-b-c,b,c];
    });
    patches.set(index,{triangle,weights,buffer:new Float32Array(8),uv:new Float32Array(corners.flatMap(([x,y])=>[x/reference.width,1-y/reference.height]))});
  }
  core.loadParameters();
  core.getDrawableVertices=index=>{
    const raw=vertices(index), patch=patches.get(index);if(!patch)return raw;
    patch.weights.forEach((weights,i)=>{
      for(let axis=0;axis<2;axis++)patch.buffer[i*2+axis]=weights.reduce((sum,w,j)=>sum+w*raw[patch.triangle[j]*2+axis],0);
    });
    return patch.buffer;
  };
  const indices=new Uint16Array([0,1,2,0,2,3]);
  for(const [method,replacement] of Object.entries({getDrawableVertexCount:()=>4,getDrawableVertexIndexCount:()=>6,getDrawableVertexIndices:()=>indices,getDrawableVertexUvs:index=>patches.get(index).uv,getDrawableTextureIndices:()=>texture})){
    const original=core[method].bind(core);core[method]=index=>patches.has(index)?replacement(index):original(index);
  }
}
