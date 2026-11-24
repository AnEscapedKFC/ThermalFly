import {RoomMap} from './room-map.js';
const $=id=>document.getElementById(id), map=new RoomMap($('map'));
const statusNames={loading:'加载中',ready:'待命',playing:'执行中',paused:'已暂停',collision:'碰撞终止',stopped:'已停止',finished:'已完成',error:'模拟器异常'};
let state=null,serial=-1,mode='fp',infrared=false,points=[],depth=null,busy=false,networkOK=true;
function message(s,err=false){$('notice').textContent=s;$('notice').className=err?'error':''}
async function request(url,opts={}){const r=await fetch(url,opts);if(!r.ok){let e;try{e=(await r.json()).error}catch{}throw Error(e||('请求失败 '+r.status))}return r}
function updateButtons(){
  const s=state?.status,ready=s&&!['loading','error'].includes(s)&&networkOK;
  $('play').disabled=busy||!ready;$('restart').disabled=busy||!ready;
  $('play').textContent=s==='playing'?'暂停':s==='paused'?'继续':'执行';
  $('sampleAction').disabled=busy||!ready;
}
function sourceImages(){if(!state?.rgb)return;
  $('mainImg').src='data:image/jpeg;base64,'+(infrared?state.thermal:state.rgb);
  $('subImg').src='data:image/jpeg;base64,'+(infrared?state.rgb:state.thermal);
  $('channel').textContent=infrared?'热红外':'RGB 可见光';$('subLabel').textContent=infrared?'RGB 可见光 ↗':'热红外 ↗';
}
function setMode(next){mode=next;$('space').hidden=next!=='space';$('fp').hidden=next!=='fp';
  $('fpTab').setAttribute('aria-pressed',String(next==='fp'));$('mapTab').setAttribute('aria-pressed',String(next==='space'));
  if(next==='space')map.draw();else overlay()}
function overlay(){const el=$('overlay'),w=el.clientWidth,h=el.clientHeight;
  if(!w||!h)return;const px=Math.min(2,devicePixelRatio||1);el.width=Math.round(w*px);el.height=Math.round(h*px);
  if(infrared||!state?.K||!points.length)return;
  const c=el.getContext('2d');c.scale(px,px);const C=state.rgb_pose,K=state.K,iw=state.width,ih=state.height;
  const scale=Math.min(w/iw,h/ih),ox=(w-iw*scale)/2,oy=(h-ih*scale)/2;
  for(const wp of points){const delta=wp.xyz.map((v,k)=>v-C[k][3]);const v=[0,1,2].map(i=>delta.reduce((s,n,k)=>s+n*C[k][i],0));
    if(v[2]<.14)continue;const x=K[0][0]*v[0]/v[2]+K[0][2],y=K[1][1]*v[1]/v[2]+K[1][2];
    if(x<0||x>=iw||y<0||y>=ih)continue;const z=depth?.[(Math.floor(y*60/ih)*80+Math.floor(x*80/iw))]/100;
    if(z>0 && v[2]>z+.14)continue;c.beginPath();c.arc(ox+x*scale,oy+y*scale,2.8,0,7);
    c.fillStyle='#ffad51';c.fill();c.lineWidth=.8;c.strokeStyle='#402b17';c.stroke();}
}
function apply(s){state=s;serial=s.serial;$('playState').textContent=statusNames[s.status]||s.status;
  $('online').textContent=s.error?'模拟器异常':'模拟器已连接';$('lamp').classList.toggle('alert',!!s.error);updateButtons();
  if(s.error){message(s.error,true);return}if(!s.rgb)return;
  sourceImages();const raw=atob(s.depth);const bytes=Uint8Array.from(raw,ch=>ch.charCodeAt(0));depth=new Uint16Array(bytes.buffer);
  $('actionName').textContent=s.name+(s.status==='ready'?'（待执行）':'');
  $('frame').textContent=s.index+' / '+s.total;$('height').textContent=s.p[2].toFixed(2)+' m';
  $('speed').textContent=s.speed.toFixed(2)+' m/s';$('gap').textContent=s.clearance.toFixed(3)+' m';
  const total=Math.round(s.time*100),mm=Math.floor(total/6000),ss=Math.floor(total/100)%60,cs=total%100;
  $('time').textContent=[mm,ss].map(x=>String(x).padStart(2,'0')).join(':')+'.'+String(cs).padStart(2,'0');
  $('progressBar').style.width=(100*s.index/s.total)+'%';map.state=s;if(mode==='space')map.draw();else overlay();
  if(s.contact)message('首次接触 · '+s.contact.point.map(v=>v.toFixed(2)).join(', ')+' m');
}
async function poll(){try{const r=await request('/api/frame?after='+serial,{cache:'no-store'});networkOK=true;
  if(r.status!==204)apply(await r.json());else{$('online').textContent=state?.error?'模拟器异常':'模拟器已连接';updateButtons()}}
  catch(e){networkOK=false;$('online').textContent='连接中断';$('lamp').classList.add('alert');updateButtons()}
  setTimeout(poll,state?.status==='playing'?30:160)}
async function command(url,opts){if(busy)return;busy=true;updateButtons();message('');
  try{await request(url,{method:'POST',...opts})}catch(e){message(e.message,true)}
  finally{busy=false;updateButtons()}}
async function loadActions(file){if(!file)return;if(file.size>64*1024*1024){message('动作文件超过 64 MiB',true);return}
  await command('/api/start?name='+encodeURIComponent(file.name),{body:file,headers:{'Content-Type':'application/octet-stream'}})}
function parseWaypoints(text){const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/);if(lines.length<2||lines.length>30001)throw Error('航点数应为 1 至 30,000');
  const fields=lines.shift().split(',').map(k=>k.trim()),idx=['x_m','y_m','z_m'].map(n=>fields.indexOf(n));
  if(idx.includes(-1))throw Error('CSV 缺少 x_m、y_m 或 z_m 列');
  return lines.map((l,j)=>{const row=l.split(',');const xyz=idx.map(i=>row[i]?.trim()?Number(row[i]):NaN);
    if(!xyz.every(Number.isFinite))throw Error('第 '+(j+2)+' 行坐标无效');return {xyz}})}
function setPoints(w,name){points=w;map.waypoints=w;$('wayName').textContent=w.length?name+' · '+w.length+' 个航点':'尚未加载航点';
  $('clear').hidden=!w.length;setMode(w.length?'space':mode);map.draw();overlay()}
async function loadCsv(file){if(!file)return;try{if(file.size>10*1024*1024)throw Error('航点文件超过 10 MiB');
  setPoints(parseWaypoints(await file.text()),file.name);message('')}catch(e){message(e.message,true)}}
function bindFile(input,drop,fn){$(input).onchange=e=>{fn(e.target.files[0]);e.target.value=''};
  const label=$(drop);label.ondragover=e=>{e.preventDefault();label.classList.add('over')};label.ondragleave=()=>label.classList.remove('over');
  label.ondrop=e=>{e.preventDefault();label.classList.remove('over');fn(e.dataTransfer.files[0])}}
bindFile('actionFile','actionDrop',loadActions);bindFile('wayFile','wayDrop',loadCsv);
$('fpTab').onclick=()=>setMode('fp');$('mapTab').onclick=()=>setMode('space');$('swap').onclick=()=>{infrared=!infrared;sourceImages();overlay()};
$('iso').onclick=()=>{map.yaw=-.49;map.elev=.72;map.zoom=1;map.refresh()};$('top').onclick=()=>{map.yaw=0;map.elev=Math.PI/2;map.refresh()};
$('zoomIn').onclick=()=>map.scale(1.25);$('zoomOut').onclick=()=>map.scale(.8);
$('clear').onclick=()=>setPoints([],'');$('restart').onclick=()=>command('/api/control/restart?episode='+state.episode);
$('play').onclick=()=>command('/api/control/'+(state.status==='playing'?'pause':state.status==='paused'?'play':'restart')+'?episode='+state.episode);
$('sampleAction').onclick=()=>command('/api/start?sample='+encodeURIComponent($('library').value));
$('sampleWay').onclick=async()=>{try{const select=$('library');const r=await request('/api/waypoints/'+select.value);setPoints(parseWaypoints(await r.text()),select.selectedOptions[0].textContent);message('')}catch(e){message(e.message,true)}};
new ResizeObserver(overlay).observe($('stage'));
async function init(){try{const items=await(await request('/api/catalog')).json();for(const item of items){const o=document.createElement('option');o.value=item.id;o.textContent=item.name;$('library').append(o)}
  await map.load()}catch(e){message(e.message,true)}}
// Playback commands are file- or pointer-driven; focused buttons must not steer playback via keyboard.
for (const id of ['play','restart','sampleAction']) {
  $(id).addEventListener('keydown',e=>{if(e.key===' '||e.key==='Enter')e.preventDefault()});
}
init();poll();
