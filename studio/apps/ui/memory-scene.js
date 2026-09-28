/* The illustrated backdrop is static; graph nodes come from the selected
   Memory Space's live snapshot. */
(() => {
  'use strict';
  const canvas = document.getElementById('scene-canvas');
  const ctx = canvas.getContext('2d');
  const backgroundCanvas = document.getElementById('brain-background-canvas');
  const backgroundCtx = backgroundCanvas.getContext('2d', {alpha: false});
  const digital = document.body.dataset.style === 'digital';
  const stage = document.querySelector(digital ? '.brain' : '#viewSpace .space');
  const memoryIn = document.getElementById('memoryIn');
  const card = document.createElement('div'); card.className = 'memory-node-card';
  card.setAttribute('aria-hidden', 'true');
  const cardHead = document.createElement('div'); cardHead.className = 'memory-node-card__head';
  const cardDot = document.createElement('span'); cardDot.className = 'memory-node-card__dot';
  const cardKind = document.createElement('span'); cardKind.className = 'memory-node-card__kind';
  cardHead.append(cardDot, cardKind);
  const cardTitle = document.createElement('div'); cardTitle.className = 'memory-node-card__title';
  const cardBody = document.createElement('div'); cardBody.className = 'memory-node-card__body';
  const cardFoot = document.createElement('div'); cardFoot.className = 'memory-node-card__foot';
  card.append(cardHead, cardTitle, cardBody, cardFoot); document.body.append(card);
  const isActive = () => digital ? document.body.classList.contains('memory-on') && memoryIn.classList.contains('show') : !document.body.classList.contains('chat-background-only');
  const nodeScale = 1;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let graph = {nodes: [], links: []};
  const image = new Image();
  const P = .953;
  const midX = (.0219 + .9688) / 2;
  const midY = (.0361 + .9475) / 2;
  const crop = {x: 390, y: 85, w: 892, h: 740};
  const colors = {entity:'#4aa8f0',person:'#a583ff',know:'#58e08d',emotion:'#ff5fa2',exper:'#ff9a3c',prefer:'#4ade80',user:'#d02fa8',label:'#fff'};
  const styles = {in:['150,192,235',.19,.75],box:['171,133,255',.46,2.3],field:['255,170,200',.17,.8],cross:['214,228,255',.20,1.45]};
  let neighbors = [];
  let width = 0, height = 0, dpr = 1, frameId = 0, selected = -1, pinned = -1, cardNode = null;
  let cardSize = {width: 236, height: 110};
  let layoutMinGap = Infinity;
  let points = [], pointer = null;
  const newMotion = () => ({
    ph:Math.random()*Math.PI*2, ph2:Math.random()*Math.PI*2,
    sp:.28+Math.random()*.42, ax:.55+Math.random()*.9, ay:.55+Math.random()*.9,
    h:0, p:0, bw:0, bh:0
  });
  let motion = [];
  const pulse = {t:-.6,dur:2.3,gap:1.9,trail:.30,seg:16};
  let crossLinks = [];
  const glowSprites = {};
  let glowScale = -1, previousTime = null;
  let signature = '', snapshot = null, spaceGeneration = 0, watchTimer = 0, watchGeneration = 0;
  const signatureOf = data => JSON.stringify({
    left: (data.left || []).map(item => [item.text, item.slot]),
    right: (data.right || []).map(item => [item.text, item.cluster, (item.notes || []).map(note => note.text)]),
  });
  const slotNames = ['work','health','relationships','finance','knowledge','goals','daily_life','emotion','preference','personality'];
  const domainNames = {
    work:['工作','Work'], health:['健康','Health'], relationships:['关系','Relationships'],
    finance:['财务','Finance'], knowledge:['知识','Knowledge'], goals:['目标','Goals'],
    daily_life:['日常','Daily life'], emotion:['情绪','Emotion'],
    preference:['偏好','Preferences'], personality:['人格','Personality'],
  };
  const domainLabel = domain => domainNames[domain]?.[window.VMSettings?.language === 'en' ? 1 : 0] || domain;
  const anchorPositions = [
    [.3379,.2], [.2616,.3829], [.2507,.6057], [.1199,.4914], [.1226,.7029],
    [.2016,.8171], [.3324,.7657], [.76,.28], [.62,.54], [.79,.73],
  ];
  const hemispheres = {L:{x:.27,y:.49,rx:.17,ry:.37}, R:{x:.73,y:.49,rx:.17,ry:.37}};
  const slotOf = value => slotNames.includes(value) ? value : 'daily_life';
  const hashOf = value => { let hash = 2166136261; for (const char of value) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619); return hash >>> 0; };

  function layoutNodes() {
    layoutMinGap = Infinity;
    const rect = stage.getBoundingClientRect();
    if (!rect.width || !rect.height || graph.nodes.length <= 11) return;
    const plate = fitPlate(rect), imageScale = Math.min(plate.w, plate.h/P);
    const k = imageScale * .98, ky = P*k, unit = imageScale * .82 / 1000;
    const fontSize = Math.max(9.4, 14*unit) * (window.VMSettings?.contentScale || 1);
    ctx.font = `${fontSize.toFixed(1)}px system-ui,sans-serif`;
    const labels = graph.nodes.slice(0, 10).map(node => ({
      x:node.x, y:node.y,
      hw:(ctx.measureText(domainLabel(node.domain)).width + fontSize*1.75)/2,
      hh:fontSize*1.025,
    }));
    const placed = [];
    const clearance = (x,y,radius) => {
      let gap = Infinity;
      for (const label of labels) {
        const dx = Math.max(Math.abs(x-label.x)*k-label.hw, 0);
        const dy = Math.max(Math.abs(y-label.y)*ky-label.hh, 0);
        gap = Math.min(gap, Math.hypot(dx,dy)-radius);
      }
      for (const other of placed) {
        gap = Math.min(gap, Math.hypot((x-other.x)*k,(y-other.y)*ky)-radius-other.r);
      }
      return gap;
    };
    for (const node of graph.nodes.slice(11)) {
      const anchor = graph.nodes[slotNames.indexOf(node.domain)];
      const hemi = hemispheres[node.side];
      const radius = node.r * 2 * unit * nodeScale;
      const phase = (hashOf(node.w) % 360) * Math.PI / 180;
      const start = Math.max(23, radius + 26);
      const step = Math.max(13, radius*2 + 8);
      let found = false;
      for (let ring = 0; ring < 18 && !found; ring++) {
        const distance = start + ring*step;
        const samples = Math.max(16, Math.ceil(2*Math.PI*distance/step));
        for (let sample = 0; sample < samples; sample++) {
          const angle = phase + sample*2*Math.PI/samples;
          const x = anchor.x + Math.cos(angle)*distance/k;
          const y = anchor.y + Math.sin(angle)*distance/ky;
          const edge = Math.hypot((x-hemi.x)/hemi.rx, (y-hemi.y)/hemi.ry);
          if (edge > .93) continue;
          if (clearance(x,y,radius) < 8) continue;
          node.x = x; node.y = y; found = true; break;
        }
      }
      if (!found) {
        // Keep the least crowded valid point when an unusually dense Space fills a hemisphere.
        let best = null;
        for (let i = 0; i < 360; i++) {
          const angle = phase + i*2.399963;
          const radial = Math.sqrt((i+.5)/360)*.88;
          const x = hemi.x + Math.cos(angle)*hemi.rx*radial;
          const y = hemi.y + Math.sin(angle)*hemi.ry*radial;
          const gap = clearance(x,y,radius);
          if (!best || gap > best.gap) best = {x,y,gap};
        }
        node.x = best.x; node.y = best.y;
      }
      layoutMinGap = Math.min(layoutMinGap, clearance(node.x,node.y,radius));
      placed.push({x:node.x,y:node.y,r:radius});
    }
  }

  function buildGraph(data) {
    const anchors = slotNames.map((domain, index) => ({
      x: anchorPositions[index][0], y: anchorPositions[index][1],
      k: 'label', r: 0, w: domain, domain,
    }));
    const nodes = [...anchors, {x: .5, y: .5, k: 'user', r: 8.4, w: 'you', domain: null}];
    const links = anchors.map((_, index) => ({a: anchors.length, b: index, t: 'cross', w: .5}));
    const add = (item, side) => {
      const domain = side === 'L' ? slotOf(item.slot) :
        item.cluster === 'emotion' ? 'emotion' : item.cluster === 'preference' ? 'preference' : 'personality';
      const anchor = slotNames.indexOf(domain), center = anchors[anchor];
      const node = {
        x: center.x, y: center.y,
        k: side === 'L' ? 'entity' : domain === 'emotion' ? 'emotion' : domain === 'preference' ? 'exper' : 'prefer',
        r: side === 'L' ? 2.6 : 5, w: item.text, detail: item.desc || item.text,
        domain, side, memory: item,
      };
      links.push({a: anchor, b: nodes.length, t: 'in', w: .55});
      nodes.push(node);
    };
    (data.left || []).forEach(item => add(item, 'L'));
    (data.right || []).forEach(item => add(item, 'R'));
    graph = {nodes, links};
    layoutNodes();
    motion = nodes.map(newMotion);
    neighbors = nodes.map(() => new Set());
    links.forEach(link => {neighbors[link.a].add(link.b);neighbors[link.b].add(link.a);});
    crossLinks = links.filter(link => link.t === 'cross');
    selected = -1; pinned = -1; hideCard();
    requestDraw();
  }

  async function refresh(generation = spaceGeneration) {
    const response = await fetch('/api/memories', {cache: 'no-store'});
    if (!response.ok) throw new Error(`记忆快照读取失败 (${response.status})`);
    const data = await response.json();
    if (generation !== spaceGeneration) return false;
    const next = signatureOf(data);
    if (next === signature) return false;
    const before = snapshot;
    signature = next; buildGraph(data);
    snapshot = data;
    if (before) {
      const knownLeft = new Set((before.left || []).map(item => item.text));
      const knownRight = new Map((before.right || []).map(item => [item.text, (item.notes || []).length]));
      const until = Date.now() + 3000;
      graph.nodes.forEach((node, index) => {
        if (!node.memory) return;
        if (node.side === 'L' ? !knownLeft.has(node.w) :
          !knownRight.has(node.w) || (node.memory.notes || []).length > knownRight.get(node.w)) {
          motion[index].hitUntil = until;
        }
      });
    }
    return true;
  }
  function watchMemories() {
    clearTimeout(watchTimer);
    const generation = spaceGeneration;
    const watch = ++watchGeneration;
    let remaining = 20;
    const check = async () => {
      if (generation !== spaceGeneration || watch !== watchGeneration) return;
      try { if (await refresh(generation)) return; }
      catch (error) { console.warn('[memory] refresh failed:', error); }
      if (watch !== watchGeneration) return;
      if (--remaining > 0) watchTimer = setTimeout(check, 2000);
    };
    watchTimer = setTimeout(check, 2000);
  }
  function hideCard() {
    card.classList.remove('on'); card.setAttribute('aria-hidden', 'true'); cardNode = null;
  }
  function updateCard(index) {
    if (index < 0 || !isActive()) { hideCard(); return; }
    const node = graph.nodes[index];
    if (cardNode !== node) {
      cardNode = node;
      cardDot.style.background = colors[node.k];
      cardKind.textContent = node.k === 'label' ? '记忆域' : node.k === 'user' ? '记忆中心' :
        node.side === 'L' ? '左脑 · 事实' : '右脑 · 画像';
      cardTitle.textContent = node.memory ? node.w : node.k === 'label' ? domainLabel(node.domain) : '你 · 所有者';
      if (node.memory) {
        const item = node.memory;
        const lines = [];
        if (item.desc && item.desc !== item.text) lines.push(item.desc);
        const notes = (item.notes || []).filter(note => note.text || note.cause);
        notes.slice(0, 2).forEach(note => lines.push(`${note.emotion ? `【${note.emotion}】` : ''}${note.text || note.cause}`));
        if (notes.length > 2) lines.push(`还有 ${notes.length - 2} 条记录`);
        cardBody.textContent = lines.join('\n');
        cardBody.hidden = !lines.length;
        cardFoot.textContent = domainLabel(node.domain);
      } else {
        const count = graph.nodes.filter(item => item.memory && item.domain === node.domain).length;
        cardBody.textContent = node.k === 'label' ? `这个记忆域中有 ${count} 条记忆。` : '所有记忆从这里连接。';
        cardBody.hidden = false;
        cardFoot.textContent = node.k === 'label' ? `${node.domain === 'emotion' || node.domain === 'preference' || node.domain === 'personality' ? '右脑' : '左脑'} · ${count} 条` : 'Memory Space';
      }
      cardSize = {width: card.offsetWidth || 236, height: card.offsetHeight || 110};
    }
    const point = points[index], cardWidth = cardSize.width, cardHeight = cardSize.height;
    const x = point.x < innerWidth / 2 ? point.x + 24 : point.x - cardWidth - 24;
    card.style.left = `${Math.max(8, Math.min(innerWidth - cardWidth - 8, x))}px`;
    card.style.top = `${Math.max(8, Math.min(innerHeight - cardHeight - 8, point.y - cardHeight / 2))}px`;
    card.classList.add('on'); card.setAttribute('aria-hidden', 'false');
  }

  function fitPlate(rect) {
    const w = Math.min(rect.width * .98, Math.max(1, rect.height - 38) * crop.w / crop.h);
    const h = w * crop.h / crop.w;
    return {x:rect.left+(rect.width-w)/2,y:rect.top+(rect.height-38-h)/2,w,h};
  }

  function projectPoint(n, plate, index, time) {
    // Reference coordinates are fractions, NOT a 1000px coordinate system.
    // K is already in pixels; P preserves the reference's y/x proportion.
    const imageScale = Math.min(plate.w, plate.h / P);
    const k = imageScale * .98; // Spread nodes 20% farther across the brain.
    const nodeUnit = imageScale * .82 / 1000; // Size remains independent of spacing.
    const amp = reduced ? 0 : 3.1 * k / 1000;
    const m = motion[index];
    return {
      x:plate.x+midX*plate.w+(n.x-midX)*k+(n.x>=.4906 ? .02*plate.w : 0)+Math.sin(time*m.sp+m.ph)*amp*m.ax,
      y:plate.y+midY*plate.h+(n.y-midY)*P*k+Math.cos(time*m.sp*.83+m.ph2)*amp*m.ay,
      r:n.r*2*nodeUnit*nodeScale, scale:nodeUnit
    };
  }

  function resize() {
    cardNode = null;
    width = innerWidth; height = innerHeight; dpr = Math.min(2, devicePixelRatio || 1);
    const bounds = digital ? stage.getBoundingClientRect() : {width,height};
    for (const [layer, context] of [[canvas,ctx],[backgroundCanvas,backgroundCtx]]) {
      layer.width = Math.round(bounds.width*dpr); layer.height = Math.round(bounds.height*dpr);
      context.setTransform(dpr,0,0,dpr,0,0);
    }
    layoutNodes();
    drawBackground();
  }

  function drawBackground() {
    const origin = digital ? canvas.getBoundingClientRect() : {left:0,top:0};
    backgroundCtx.setTransform(dpr,0,0,dpr,-origin.left*dpr,-origin.top*dpr);
    backgroundCtx.fillStyle='#000';backgroundCtx.fillRect(0,0,width,height);
    if (!isActive() || !image.complete || !image.naturalWidth) return;
    const rect=stage.getBoundingClientRect();
    if(rect.width<=0 || rect.height<=0) return;
    const plate=fitPlate(rect), scale=plate.w/crop.w;
    backgroundCtx.imageSmoothingQuality = 'high';
    backgroundCtx.filter = 'brightness(.82) contrast(1.25)';
    backgroundCtx.drawImage(image,plate.x-crop.x*scale,plate.y-crop.y*scale,
      image.naturalWidth*scale,image.naturalHeight*scale);
    backgroundCtx.filter = 'none';
  }

  // Reference path(): circles on the left, rounded squares and triangles on the right.
  function nodePath(n, p, radius) {
    const {x,y} = p;
    if(n.k === 'exper') {
      const h=radius*1.12;
      ctx.moveTo(x,y-h);ctx.lineTo(x+radius*1.02,y+h*.72);ctx.lineTo(x-radius*1.02,y+h*.72);ctx.closePath();
    } else if(n.k === 'emotion' || n.k === 'prefer') {
      ctx.roundRect(x-radius,y-radius,radius*2,radius*2,Math.min(radius*.42,3.2*p.scale));
    } else {ctx.moveTo(x+radius,y);ctx.arc(x,y,radius,0,Math.PI*2);}
  }

  function hexa(h,a) {
    const n=parseInt(h.slice(1),16);
    return `rgba(${n>>16},${n>>8&255},${n&255},${a})`;
  }

  // Reference radial()/buildGlow(): cache soft, rapidly falling-off glow sprites.
  function buildGlow(scale) {
    if(scale === glowScale) return;
    glowScale=scale;
    for(const kind in colors) {
      if(kind === 'label') continue;
      const radius=Math.max(18,Math.round((kind==='user'?150:56)*scale));
      const sprite=document.createElement('canvas');
      sprite.width=sprite.height=radius*2;
      const g=sprite.getContext('2d');
      const gradient=g.createRadialGradient(radius,radius,0,radius,radius,radius);
      gradient.addColorStop(0,hexa(colors[kind],.55));
      gradient.addColorStop(.28,hexa(colors[kind],.55*.36));
      gradient.addColorStop(.62,hexa(colors[kind],.55*.09));
      gradient.addColorStop(1,hexa(colors[kind],0));
      g.fillStyle=gradient;g.fillRect(0,0,radius*2,radius*2);
      glowSprites[kind]=sprite;
    }
  }

  // Reference ctl(): all cross-brain curves bend through the same central region.
  function controlPoint(a,b,plate) {
    const k=Math.min(plate.w,plate.h/P)*.98;
    const cx=plate.x+midX*plate.w, cy=plate.y+midY*plate.h;
    const mx=(a.x+b.x)/2, my=(a.y+b.y)/2;
    const gx=cx+((.4688+.5094)/2-midX)*k;
    const gy=Math.min(Math.max(my,cy-.12*plate.h),cy+.17*plate.h);
    return [mx*.45+gx*.55,my*.72+gy*.28];
  }

  // Reference trail(): a fine white filament above a soft blue halo.
  const trailBuffer=new Float32Array(128);
  function trail(at,head,len,segments,weight,scale) {
    const step=len/segments;
    let count=0;
    for(let i=segments;i>=0;i--) {
      const u=head-i*step;
      if(u<=0 || u>=1) continue;
      const p=at(u);
      trailBuffer[count*2]=p[0];trailBuffer[count*2+1]=p[1];
      if(++count>=63) break;
    }
    if(count<2) return;
    const lineScale=Math.max(.8,scale), strength=.84;
    ctx.strokeStyle=`rgba(150,200,255,${.085*strength})`;
    ctx.lineWidth=(weight*3+1.2)*lineScale;
    ctx.beginPath();ctx.moveTo(trailBuffer[0],trailBuffer[1]);
    for(let i=1;i<count;i++) ctx.lineTo(trailBuffer[i*2],trailBuffer[i*2+1]);
    ctx.stroke();
    for(let i=1;i<count;i++) {
      const k=i/(count-1), e=k*k*(3-2*k);
      ctx.strokeStyle=`rgba(255,255,255,${.66*e*strength})`;
      ctx.lineWidth=(weight*e+.5)*lineScale;
      ctx.beginPath();ctx.moveTo(trailBuffer[(i-1)*2],trailBuffer[(i-1)*2+1]);
      ctx.lineTo(trailBuffer[i*2],trailBuffer[i*2+1]);ctx.stroke();
    }
  }

  function pickNode() {
    let best=-1, distance=Infinity;
    if(!pointer) return best;
    points.forEach((p,i)=>{
      const n=graph.nodes[i],m=motion[i];
      const d=n.k==='label' && m.bw
        ? Math.hypot(Math.max(Math.abs(p.x-pointer.x)-m.bw/2,0),Math.max(Math.abs(p.y-pointer.y)-m.bh/2,0))
        : Math.max(0,Math.hypot(p.x-pointer.x,p.y-pointer.y)-p.r*.85);
      if(d<distance){distance=d;best=i;}
    });
    return distance>Math.max(24,40*(points[0]?.scale||0)) ? -1 : best;
  }

  function drawGraph(plate,time) {
    const dt=previousTime===null?0:Math.max(0,Math.min(.05,time-previousTime));
    previousTime=time;
    points=graph.nodes.map((n,i)=>projectPoint(n,plate,i,time));
    if(!points.length) return;
    const scale=points[0].scale, lineScale=Math.max(.72,scale), sw=Math.max(.8,scale);
    buildGlow(scale);
    selected=pinned >= 0 ? pinned : pickNode();
    const now = Date.now();
    motion.forEach((m,i)=>{
      const hit = m.hitUntil > now;
      const target = i === selected || hit ? 1 : (selected >= 0 && neighbors[selected].has(i) ? .42 : 0);
      m.h=reduced?target:m.h+(target-m.h)*(target?.22:.10);
      let proximity=0;
      if(pointer) {
        const distance=Math.hypot(points[i].x-pointer.x,points[i].y-pointer.y), field=150*scale;
        proximity=distance<field?Math.pow(1-distance/field,2):0;
      }
      m.p=reduced?proximity:m.p+(proximity-m.p)*.14;
    });
    ctx.lineCap='round';ctx.lineJoin='round';
    graph.links.forEach(l=>{
      const a=points[l.a],b=points[l.b],ma=motion[l.a],mb=motion[l.b],st=styles[l.t]||styles.in;
      const lift=Math.max(ma.h,mb.h)*.38+Math.max(ma.p,mb.p)*.14;
      ctx.strokeStyle=`rgba(${st[0]},${Math.min(.95,st[1]*(l.w||1)+lift)})`;
      ctx.lineWidth=st[2]*(1+lift*1.5)*lineScale;
      ctx.beginPath();ctx.moveTo(a.x,a.y);
      if(l.t==='cross'){const c=controlPoint(a,b,plate);ctx.quadraticCurveTo(c[0],c[1],b.x,b.y);}
      else ctx.lineTo(b.x,b.y);
      ctx.stroke();
    });

    if(!reduced) {
      pulse.t+=dt/pulse.dur;
      if(pulse.t>1+pulse.trail) pulse.t=-pulse.gap/pulse.dur;
      if(pulse.t>-pulse.trail) {
        ctx.globalCompositeOperation='lighter';
        crossLinks.forEach(l=>{
          const a=points[l.a],b=points[l.b],c=controlPoint(a,b,plate);
          trail(u=>{const v=1-u;return [v*v*a.x+2*v*u*c[0]+u*u*b.x,v*v*a.y+2*v*u*c[1]+u*u*b.y];},pulse.t,pulse.trail,pulse.seg,1.5,scale);
        });
        ctx.globalCompositeOperation='source-over';
      }
    }

    // Reference glow pass, composited independently of brain-background opacity.
    ctx.globalCompositeOperation='lighter';
    graph.nodes.forEach((n,i)=>{
      const sprite=glowSprites[n.k];if(!sprite) return;
      const p=points[i],m=motion[i],radius=p.r*2.6*(1+m.h*1.15+m.p*.24);
      ctx.globalAlpha=Math.min(1,.16+m.h*.66+m.p*.22);
      ctx.drawImage(sprite,p.x-radius,p.y-radius,radius*2,radius*2);
    });
    ctx.globalAlpha=1;ctx.globalCompositeOperation='source-over';

    // Reference drawNode(): colored outlines with a faint glass tint, never a solid disk.
    graph.nodes.forEach((n,i)=>{
      if(n.k==='label') return;
      const p=points[i],m=motion[i],r=p.r*(1+m.h*.55+m.p*.13),a=.72+m.h*.28;
      ctx.strokeStyle=hexa(colors[n.k],a);
      ctx.fillStyle=hexa(colors[n.k],a*.16);
      ctx.lineWidth=Math.max(.8,(n.k==='entity'||n.k==='know'?1.05:1.45)*sw*(1+m.h*.35));
      ctx.beginPath();nodePath(n,p,r);
      if(n.k==='user') ctx.fillStyle=hexa(colors.user,.20);
      ctx.fill();ctx.stroke();
      if(n.k==='user') {
        ctx.beginPath();ctx.arc(p.x,p.y,r*.63,0,Math.PI*2);
        ctx.strokeStyle=hexa(colors.user,.55*a);ctx.stroke();
        ctx.beginPath();ctx.arc(p.x,p.y,Math.max(1.6,r*.14),0,Math.PI*2);
        ctx.fillStyle=hexa(colors.user,.95);ctx.fill();
        if(!reduced) {
          ctx.globalCompositeOperation='lighter';
          for(let q=0;q<2;q++) {
            const phase=((time/3.4)+q*.5)%1;
            ctx.strokeStyle=hexa(colors.user,Math.pow(1-phase,2)*.42);
            ctx.lineWidth=Math.max(.7,(1.5-phase)*sw);
            ctx.beginPath();ctx.arc(p.x,p.y,p.r*(1+phase*3.2),0,Math.PI*2);ctx.stroke();
          }
          ctx.globalCompositeOperation='source-over';
        }
      }
    });

    const fontSize=Math.max(9.4,14*scale)*(window.VMSettings?.contentScale||1);
    ctx.font=`${fontSize.toFixed(1)}px system-ui,sans-serif`;
    ctx.textAlign='center';ctx.textBaseline='middle';
    if('letterSpacing' in ctx)ctx.letterSpacing='0px';
    graph.nodes.forEach((n,i)=>{
      if(n.k!=='label' && n.k!=='user')return;
      const text=n.k==='label' ? domainLabel(n.w) : (window.VMSettings?.graphLabel(n.w)||n.w);
      const p=points[i],m=motion[i],bw=ctx.measureText(text).width+fontSize*1.75,bh=fontSize*2.05;
      const y=n.k==='user'?p.y+p.r+bh*.95:p.y;
      m.bw=bw;m.bh=bh;
      const a=.62+m.h*.38;
      ctx.beginPath();ctx.roundRect(p.x-bw/2,y-bh/2,bw,bh,5);
      ctx.fillStyle='rgba(8,8,10,.88)';ctx.fill();
      ctx.strokeStyle=`rgba(255,255,255,${a*.55})`;
      ctx.lineWidth=Math.max(.7,.9*Math.max(.85,scale));ctx.stroke();
      ctx.fillStyle=`rgba(255,255,255,${.82+m.h*.18})`;
      ctx.fillText(text,p.x+fontSize*.055,y+fontSize*.06);
    });
    if('letterSpacing' in ctx)ctx.letterSpacing='0px';

    // Reference rotating dashed focus rings and four radial ticks.
    motion.forEach((m,i)=>{
      if(m.h<.02)return;
      const p=points[i],n=graph.nodes[i];
      const radius=(n.k==='label'?Math.hypot(m.bw,m.bh)/2+8*Math.max(.7,scale):p.r*(1+m.h*.55))+(9+m.h*5)*Math.max(.7,scale);
      ctx.save();ctx.translate(p.x,p.y);ctx.rotate(time*.35);
      ctx.strokeStyle=`rgba(255,255,255,${.42*m.h})`;
      ctx.lineWidth=Math.max(.7,scale*.9);ctx.setLineDash([2.4*sw,5.6*sw]);
      ctx.beginPath();ctx.arc(0,0,radius,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
      ctx.strokeStyle=`rgba(255,255,255,${.6*m.h})`;
      const tick=3.4*sw;
      for(let k=0;k<4;k++) {
        const angle=k*Math.PI/2+Math.PI/4,c=Math.cos(angle),v=Math.sin(angle);
        ctx.beginPath();ctx.moveTo(c*(radius-tick),v*(radius-tick));ctx.lineTo(c*(radius+tick),v*(radius+tick));ctx.stroke();
      }
      ctx.restore();
    });
    updateCard(selected);
  }

  function drawForeground(time) {
    const origin = digital ? canvas.getBoundingClientRect() : {left:0,top:0};
    ctx.setTransform(dpr,0,0,dpr,-origin.left*dpr,-origin.top*dpr);
    ctx.globalAlpha=1;ctx.globalCompositeOperation='source-over';
    ctx.clearRect(0,0,width,height);
    if(!isActive()) {previousTime=null;hideCard();return;}
    const rect=stage.getBoundingClientRect();
    if(rect.width<=0 || rect.height<=0) return;
    const plate=fitPlate(rect);
    drawGraph(plate,time);
  }

  function frame(now) {
    frameId=0;
    if(document.hidden) {hideCard();return;}
    drawForeground(reduced ? 0 : now/1000);
    if(!reduced && isActive()) frameId=requestAnimationFrame(frame);
  }
  function requestDraw() {if(!frameId && !document.hidden) frameId=requestAnimationFrame(frame);}
  backgroundCanvas.style.opacity = '.85';
  canvas.style.opacity = '1';
  window.addEventListener('pointermove',e=>{
    pointer=e.target.closest('button,input,a,.col-side,.col-retrieval,.rail,.stage .scene') ? null : {x:e.clientX,y:e.clientY};
    requestDraw();
  },{passive:true});
  window.addEventListener('pointerdown',e=>{
    pointer=e.target.closest('button,input,a,.col-side,.col-retrieval,.rail,.stage .scene') ? null : {x:e.clientX,y:e.clientY};
    requestDraw();
  },{passive:true});
  document.addEventListener('pointerleave',()=>{pointer=null;requestDraw();});
  window.addEventListener('click',e=>{
    if(e.target.closest('button,input,a,.col-side,.col-retrieval,.rail,.stage .scene') || !isActive()) {pinned=-1;hideCard();return;}
    pointer={x:e.clientX,y:e.clientY};
    selected=pickNode();
    if(selected<0){pinned=-1;hideCard();return;}
    const n=graph.nodes[selected];
    pinned=pinned===selected?-1:selected;
    if(n.k==='label') window.dispatchEvent(new CustomEvent('memory-domain-select',{detail:{domain:n.domain}}));
    requestDraw();
  });
  window.addEventListener('keydown',e=>{if(e.key==='Escape' && pinned>=0){pinned=-1;pointer=null;hideCard();requestDraw();}});
  window.addEventListener('resize',()=>{resize();requestDraw();});
  new ResizeObserver(()=>{resize();requestDraw();}).observe(stage);
  document.querySelector(digital ? '.app' : '.shell').addEventListener('scroll',requestDraw,{passive:true});
  const redrawForView = () => {drawBackground();requestDraw();};
  if(digital) new MutationObserver(redrawForView).observe(memoryIn,{attributes:true,attributeFilter:['class']});
  new MutationObserver(redrawForView).observe(document.body,{attributes:true,attributeFilter:['class']});
  document.addEventListener('visibilitychange',()=>{
    cancelAnimationFrame(frameId);frameId=0;previousTime=null;requestDraw();
    if (!document.hidden) void refresh().catch(error => console.warn('[memory] refresh failed:', error));
  });
  window.addEventListener('pagehide',()=>{cancelAnimationFrame(frameId);frameId=0;});
  window.addEventListener('pageshow',()=>{previousTime=null;resize();requestDraw();
    void refresh().catch(error => console.warn('[memory] refresh failed:', error));});
  document.addEventListener('display-settings-change',()=>{cardNode=null;layoutNodes();requestDraw();});
  image.onload=redrawForView;
  image.src='background.webp';
  window.addEventListener('memory-space-change', () => {
    spaceGeneration += 1; watchGeneration += 1; clearTimeout(watchTimer); signature = ''; snapshot = null;
    void refresh().catch(error => console.warn('[memory] refresh failed:', error));
  });
  window.VMMemoryScene = Object.freeze({
    onStudioEvent(message) {
      if (message.type === 'answer_done' || message.type === 'playback_done') watchMemories();
      if (message.type !== 'memory_hits') return;
      const hits = [...(message.left_brain || []).map(hit => hit.text),
        ...(message.right_brain_hits || []).map(hit => hit.content)];
      const matching = graph.nodes.map((node, index) => hits.some(text => node.memory &&
        (node.w === text || node.memory.notes?.some(note => note.text === text))) ? index : -1).filter(index => index >= 0);
      const until = Date.now() + 3000;
      matching.forEach(index => { motion[index].hitUntil = until; });
      if (matching.length) requestDraw();
    },
    refresh: () => refresh(),
    stats: () => ({left: graph.nodes.filter(node => node.memory && node.side === 'L').length,
      right: graph.nodes.filter(node => node.memory && node.side === 'R').length,
      minGap: Number.isFinite(layoutMinGap) ? layoutMinGap : null}),
  });
  resize();requestDraw();
  void refresh().catch(error => console.warn('[memory] refresh failed:', error));
})();
