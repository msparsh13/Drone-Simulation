const fs=require('fs');
const html=fs.readFileSync(require('path').resolve(__dirname,'..','swarm','static','index.html'),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const Lights=require(require('path').resolve(__dirname,'..','swarm','static','lights.js'));
const els={};
function defaults(id){
  const tag=html.match(new RegExp(`<(input|select|textarea)[^>]*id="${id}"[^>]*>`));
  const o={value:'',checked:false,textContent:'',innerHTML:'',files:[],style:{},clientWidth:700,width:0,height:0,dataset:{}};
  if(tag){const t=tag[0]; const v=t.match(/value="([^"]*)"/); if(v)o.value=v[1]; o.checked=/\schecked/.test(t);
    if(tag[1]==='select'){const m=html.match(new RegExp(`id="${id}"[^>]*>\\s*<option(?: value="([^"]*)")?>([^<]*)`)); if(m)o.value=m[1]!==undefined?m[1]:m[2];}}
  o.addEventListener=()=>{}; o.querySelectorAll=()=>[]; o.click=()=>{};
  let fills=0; o._fills=()=>fills;
  o.getContext=()=>({setTransform(){},fillRect(){},strokeRect(){},setLineDash(){},beginPath(){},moveTo(){},lineTo(){},closePath(){},stroke(){},fill(){fills++;},set fillStyle(v){},set strokeStyle(v){}});
  return o;
}
global.document={getElementById:id=>els[id]||(els[id]=defaults(id)),createElement:()=>({click(){}})};
global.devicePixelRatio=1; global.Lights=Lights;
global.requestAnimationFrame=()=>{}; 
const realFetch=fetch; global.fetch=(u,o)=>realFetch('http://127.0.0.1:'+process.env.PORT+u,o);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
  new Function('Lights',script+';globalThis.__t={get P(){return P},set t(v){t=v},get totalT(){return totalT},draw,stats,build,convertVideo,refreshCues,get stages(){return stages},set cues(v){cues=v}};')(Lights);
  const T=globalThis.__t, $=document.getElementById;
  for(let i=0;i<100&&!T.P;i++) await sleep(200);
  if(!T.P) throw new Error('plan never loaded: '+$('msg').textContent);
  const sweep=(label)=>{let n=0; for(let k=0;k<=60;k++){T.t=T.totalT*k/60-(k==60?1e-3:0); T.draw(); T.stats(); n++;} console.log(label,'ok: segs',T.P.segments.length,'total',T.totalT.toFixed(0)+'s fills',$('cv')._fills());};
  sweep('default (image colours, no takeoff)');
  console.log(' first colour',T.P.segments[0].color[0], 'notes',T.P.notes.length);
  $('takeoff').checked=true; $('landing').checked=true; $('n').value='200';
  T.cues=[{effect:'blink',anchor:{type:'takeoff'},period:1},{effect:'rainbow',anchor:{type:'hold',stage:1},axis:'x',period:4},{effect:'off',anchor:{type:'landing'}}];
  await T.build(); sweep('takeoff+landing+cues');
  console.log(' takeoff',T.P.takeoff.duration,'landing',T.P.landing.duration,'segments',T.P.segments.map(s=>s.kind));
  await T.convertVideo(null); for(let i=0;i<50&&T.stages.length===3;i++) await sleep(200); await sleep(2500);
  console.log(' video stages',T.stages.length, $('msg').textContent);
  sweep('video demo');
  console.log('ALL OK');
})().catch(e=>{console.error('FAIL',e);process.exit(1);});
