// Light-show cue engine. Mirrors swarm/lights.py (parity checked by tests/test_lights_parity.py).
const Lights=(()=>{
  const EFFECTS=['on','off','color','blink','pulse','rainbow','wave','fade_in','fade_out'];
  const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
  function resolve(cues,segments,hold,takeoff=0,landing=0){
    const offs=[];let a=+takeoff;segments.forEach(s=>{offs.push(a);a+=s.total+hold;});
    const out=[],skipped=[];
    (cues||[]).forEach((c,i)=>{
      if(!EFFECTS.includes(c.effect)){skipped.push(i);return;}
      const an=c.anchor||{type:'show'};let t0,t1;
      if(an.type==='show'){t0=0;t1=a+landing;}
      else if(an.type==='takeoff'){t0=0;t1=+takeoff;}
      else if(an.type==='landing'){t0=a;t1=a+landing;}
      else if(an.type==='hold'||an.type==='flight'){
        const k=an.stage|0; if(k<0||k>=segments.length){skipped.push(i);return;}
        t0=offs[k]+(an.type==='hold'?segments[k].total:0);
        t1=offs[k]+segments[k].total+(an.type==='hold'?hold:0);
      } else if(an.type==='abs'){t0=+an.t0;t1=+an.t1;} else {skipped.push(i);return;}
      out.push(Object.assign({},c,{t0,t1}));
    });
    out.skipped=skipped;return out;
  }
  function check(cues){
    const w=[];(cues||[]).forEach((c,i)=>{
      if(c.effect==='blink'||c.effect==='pulse'){const hz=1/Math.max(+c.period||1,1e-3);
        if(hz>3)w.push(`Cue ${i+1} (${c.effect}) flashes ${hz.toFixed(1)} times per second; more than 3/s can trigger photosensitive seizures. Lengthen the period.`);}
    });return w;
  }
  function hsv(h){
    const h6=(((h%1)+1)%1)*6,i=Math.floor(h6)%6,f=h6-Math.floor(h6),q=1-f;
    const t=[[1,f,0],[q,1,0],[0,1,f],[0,q,1],[f,0,1],[1,0,q]][i];
    return [t[0]*255,t[1]*255,t[2]*255];
  }
  function axisPos(axis,i,n,x,y,extent){
    if(axis==='y')return clamp(y/extent+.5);
    if(axis==='id')return i/Math.max(n-1,1);
    if(axis==='radial')return clamp(Math.hypot(x,y)/(extent/2));
    return clamp(x/extent+.5);
  }
  function inScope(s,i,x,y){
    switch(s){case 'left':return x<0;case 'right':return x>=0;case 'top':return y>0;case 'bottom':return y<=0;
      case 'even':return i%2===0;case 'odd':return i%2===1;default:return true;}
  }
  // base: [r,g,b] 0-255; returns [r,g,b]
  function color(base,i,n,x,y,t,rc,extent){
    let out=base;
    for(const c of rc){
      if(t<c.t0||t>=c.t1||!inScope(c.scope||'all',i,x,y))continue;
      const src=c.color||base,per=Math.max(+c.period||1,1e-3),ph=t-c.t0;let col=src,f=1;
      switch(c.effect){
        case 'off':col=[0,0,0];break;
        case 'on':case 'color':col=src;break;
        case 'blink':f=((ph/per)%1)<(c.duty==null?.5:+c.duty)?1:0;break;
        case 'pulse':f=.5-.5*Math.cos(2*Math.PI*ph/per);break;
        case 'rainbow':col=hsv(axisPos(c.axis||'x',i,n,x,y,extent)-ph/per);break;
        case 'wave':f=.5+.5*Math.cos(2*Math.PI*(axisPos(c.axis||'x',i,n,x,y,extent)-ph/per));break;
        case 'fade_in':f=clamp(ph/Math.max(c.t1-c.t0,1e-9));break;
        case 'fade_out':f=1-clamp(ph/Math.max(c.t1-c.t0,1e-9));break;
      }
      out=[col[0]*f,col[1]*f,col[2]*f];
    }
    return out;
  }
  return {EFFECTS,resolve,check,color,hsv};
})();
if(typeof module!=='undefined')module.exports=Lights;
