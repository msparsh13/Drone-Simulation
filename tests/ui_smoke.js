// Runs the real page JavaScript against a live server with a stub DOM: catches runtime errors in the 3D UI.
const fs=require('fs'),path=require('path');
const html=fs.readFileSync(path.resolve(__dirname,'..','swarm','static','index.html'),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const Lights=require(path.resolve(__dirname,'..','swarm','static','lights.js'));
const els={};
function defaults(id){
  const tag=html.match(new RegExp(`<(input|select|textarea)[^>]*id="${id}"[^>]*>`));
  const o={value:'',checked:false,textContent:'',innerHTML:'',files:[],style:{},clientWidth:900,width:0,height:0,dataset:{}};
  if(tag){const tg=tag[0],v=tg.match(/value="([^"]*)"/);if(v)o.value=v[1];o.checked=/\schecked/.test(tg);
    if(tag[1]==='select'){const m=html.match(new RegExp(`id="${id}"[^>]*>\\s*<option(?: value="([^"]*)")?>([^<]*)`));if(m)o.value=m[1]!==undefined?m[1]:m[2];}}
  o.addEventListener=()=>{};o.querySelectorAll=()=>[];o.click=()=>{};o.getBoundingClientRect=()=>({left:0,top:0,width:900,height:562});
  let fills=0;o._fills=()=>fills;
  const ctx=new Proxy({}, {get:(_,k)=>k==='fill'?()=>{fills++;}:()=>{},set:()=>true});
  o.getContext=()=>ctx;return o;
}
global.document={getElementById:id=>els[id]||(els[id]=defaults(id)),createElement:()=>({click(){}})};
global.devicePixelRatio=1;global.Lights=Lights;global.requestAnimationFrame=()=>{};
const realFetch=fetch;global.fetch=(u,o)=>realFetch('http://127.0.0.1:'+process.env.PORT+u,o);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
  new Function('Lights',script+';globalThis.__t={get P(){return P},get SIM(){return SIM},set t(v){t=v},get totalT(){return totalT},draw,stats,updateDash,build,runSim,convertVideo,addFailure,selectDrone,get stages(){return stages},set cues(v){cues=v},refreshCues,get failures(){return failures},cam,get X(){return X},get Y(){return Y},get Z(){return Z},get SX(){return SX},get SY(){return SY}};')(Lights);
  const T=globalThis.__t,$=document.getElementById;let clock=1000;
  const until=async(f,label)=>{for(let i=0;i<150&&!f();i++)await sleep(200);if(!f())throw new Error('timeout waiting for '+label+': '+$('msg').textContent);};
  await until(()=>T.P&&T.SIM&&T.SIM.speed,'plan + telemetry');
  const sweep=label=>{for(let k=0;k<=50;k++){T.t=T.totalT*k/50-(k==50?1e-3:0);T.draw();T.stats();T.updateDash(clock+=300);}
    console.log(label,'ok: segs',T.P.segments.length,'drones',T.P.n,'total',T.totalT.toFixed(0)+'s','triangles drawn',$('cv')._fills());};
  sweep('default 3D show (takeoff+landing, sphere stage)');
  T.selectDrone(7);sweep('selected drone dashboard');
  {const corr=(a,b)=>{const n=a.length,ma=a.reduce((s,v)=>s+v,0)/n,mb=b.reduce((s,v)=>s+v,0)/n;let sab=0,sa=0,sb=0;for(let i=0;i<n;i++){sab+=(a[i]-ma)*(b[i]-mb);sa+=(a[i]-ma)**2;sb+=(b[i]-mb)**2;}return sab/Math.sqrt(sa*sb);};
   T.t=T.totalT*0.45;const chk=(label,yaw,pitch,fn)=>{T.cam.yaw=yaw;T.cam.pitch=pitch;T.draw();const r=fn(Array.from(T.X),Array.from(T.Y),Array.from(T.Z),Array.from(T.SX),Array.from(T.SY));
     console.log(' camera',label,'corr',r.toFixed(3));if(!(r>0.9))throw new Error('camera orientation wrong: '+label);};
   chk('front: +x -> right',0,0.05,(x,y,z,sx,sy)=>corr(x,sx));chk('front: +z -> up',0,0.05,(x,y,z,sx,sy)=>corr(z,sy.map(v=>-v)));
   chk('top: +y -> up',0,1.5,(x,y,z,sx,sy)=>corr(y,sy.map(v=>-v)));chk('top: +x -> right',0,1.5,(x,y,z,sx,sy)=>corr(x,sx));
   T.cam.yaw=.6;T.cam.pitch=.38;}
  T.t=T.totalT*0.3;T.addFailure(7,'motor',T.totalT*0.3);T.addFailure(40,'battery',T.totalT*0.3);T.addFailure(41,'led',10);
  await T.runSim();console.log(' failures',T.SIM.failures.map(f=>f.type+':'+f.state).join(', '),'| verdict',T.SIM.metrics.verdict);sweep('with failures');
  $('placement').value='cylinder';$('takeoff').checked=false;$('landing').checked=false;$('init').value='random';
  T.cues=[{effect:'rainbow',axis:'z',period:3,anchor:{type:'hold',stage:0}},{effect:'blink',anchor:{type:'show'},scope:'top',period:1.2}];
  await T.build();await until(()=>T.SIM&&T.SIM.speed,'rebuild');sweep('cylinder placement + cues');
  await T.convertVideo(null);await until(()=>T.stages.length>3&&T.SIM,'video');await sleep(1500);sweep('video demo');
  console.log('ALL OK');
})().catch(e=>{console.error('FAIL',e);process.exit(1);});
