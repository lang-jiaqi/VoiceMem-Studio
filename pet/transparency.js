/* Restore background holes without color-keying the character's dark artwork. */
async function restoreHairTransparency(model, pose) {
  const reference=await PIXI.Texture.fromURL(`assets/${pose}.png`);
  const read=texture=>{
    const canvas=document.createElement('canvas');canvas.width=texture.width;canvas.height=texture.height;
    const context=canvas.getContext('2d',{willReadFrequently:true});
    context.drawImage(texture.baseTexture.resource.source,0,0);
    return {canvas,context,pixels:context.getImageData(0,0,canvas.width,canvas.height)};
  };
  const source=read(reference),atlas=read(model.textures[0]),rgba=atlas.pixels.data;
  // Body and head retain their source scale in the v0.3.1 packed atlas.
  const regions=pose==='sit'?[[44,2,1096,1068,-2,313],[373,1096,817,1416,-2,-1091]]:[[17,40,1521,999,-2,-2]];
  const background=(x,y)=>{
    if(x<0||y<0||x>=reference.width||y>=reference.height)return false;
    const i=(y*reference.width+x)*4,p=source.pixels.data;
    return Math.min(p[i],p[i+1],p[i+2])>160&&Math.max(p[i],p[i+1],p[i+2])-Math.min(p[i],p[i+1],p[i+2])<35;
  };
  for(const [left,top,right,bottom,dx,dy] of regions){
    for(let y=top;y<bottom;y++)for(let x=left;x<right;x++){
      const i=(y*atlas.canvas.width+x)*4,value=Math.max(rgba[i],rgba[i+1],rgba[i+2]);
      if(!rgba[i+3]||value>=100)continue;
      const sx=x+dx,sy=y+dy;
      if(background(sx,sy)||(value<24&&[[1,0],[-1,0],[0,1],[0,-1]].some(([a,b])=>background(sx+a,sy+b)))){
        rgba[i]=rgba[i+1]=rgba[i+2]=rgba[i+3]=0;
      }
    }
  }
  atlas.context.putImageData(atlas.pixels,0,0);
  model.textures[0]=PIXI.Texture.from(atlas.canvas);
}
