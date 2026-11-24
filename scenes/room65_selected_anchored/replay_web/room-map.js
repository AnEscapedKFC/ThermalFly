export class RoomMap {
  constructor(canvas) {
    this.canvas=canvas;this.points=[];this.waypoints=[];this.state=null;this.yaw=-.49;this.elev=.72;this.zoom=1;this.background=document.createElement('canvas');this.dirty=true;
    let drag=null;
    canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
    canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
    canvas.onpointermove=e=>{if(!drag)return;this.yaw+=(e.clientX-drag[0])*.009;this.elev=Math.max(.12,Math.min(1.55,this.elev+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];this.refresh()};
    canvas.addEventListener('wheel',e=>{e.preventDefault();this.scale(e.deltaY<0?1.12:1/1.12)},{passive:false});
    new ResizeObserver(()=>this.refresh()).observe(canvas);
  }
  async load(){const r=await fetch('/api/surface');if(!r.ok)throw Error('点云加载失败');const b=new DataView(await r.arrayBuffer());
    for(let i=0;i+16<=b.byteLength;i+=16){const x=b.getFloat32(i,true),y=b.getFloat32(i+4,true),z=b.getFloat32(i+8,true);if(z>2.25)continue;
      const g=k=>Math.round(Math.min(255,b.getUint8(i+k)*.76+37));this.points.push([x,y,z,'rgb('+g(12)+','+g(13)+','+g(14)+')']);}
    this.refresh();}
  scale(f){this.zoom=Math.max(.48,Math.min(4,this.zoom*f));this.refresh()}
  refresh(){this.dirty=true;this.draw()}
  draw(){const el=this.canvas,w=el.clientWidth,h=el.clientHeight;if(!w||!h)return;
    const px=Math.min(window.devicePixelRatio||1,2);if(el.width!==Math.round(w*px)||el.height!==Math.round(h*px)){el.width=Math.round(w*px);el.height=Math.round(h*px);this.dirty=true}
    const co=Math.cos(this.yaw),si=Math.sin(this.yaw),ce=Math.cos(this.elev),se=Math.sin(this.elev);
    const scale=Math.min(w/9.2,h/6.7)*this.zoom;
    const project=([x,y,z])=>{y-=.45;z-=.65;const d=-si*x+co*y;return [w/2+(co*x+si*y)*scale,h*.52-(d*se+z*ce)*scale,d*ce-z*se]};
    const line=(ctx,a,b,color,width)=>{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()};
    if(this.dirty){this.background.width=el.width;this.background.height=el.height;const c=this.background.getContext('2d');c.scale(px,px);c.fillStyle='#152522';c.fillRect(0,0,w,h);
      for(let i=-5;i<=5;i++){line(c,project([i,-4,0]),project([i,4,0]),'#30403a',.7);line(c,project([-5,i,0]),project([5,i,0]),'#30403a',.7)}
      const pts=this.points.map(p=>[...project(p),p[3]]).sort((a,b)=>b[2]-a[2]);const sz=Math.max(1.25,Math.min(4,scale*.027));
      for(const [x,y,,color] of pts){c.fillStyle=color;c.fillRect(x,y,sz,sz)}
      this.dirty=false;}
    const c=el.getContext('2d');c.setTransform(1,0,0,1,0,0);c.drawImage(this.background,0,0);c.setTransform(px,0,0,px,0,0);
    const pts=this.waypoints.map(p=>project(p.xyz));c.strokeStyle='#ffae5a99';c.lineWidth=1.3;c.beginPath();pts.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.stroke();
    for(const [x,y] of pts){c.beginPath();c.arc(x,y,2.2,0,2*Math.PI);c.fillStyle='#ffaf5e';c.fill()}
    if(pts.length){c.font='bold 11px system-ui';for(const [k,txt] of [[0,'起点'],[pts.length-1,'终点']]){const [x,y]=pts[k];c.fillStyle='#342a22';c.fillRect(x+4,y-15,32,16);c.fillStyle='#ffd4a4';c.fillText(txt,x+7,y-3)}}
    if(this.state?.p){const p=this.state.p,[x,y]=project(p),ang=this.state.yaw*Math.PI/180;
      const nose=project([p[0]+.5*Math.cos(ang),p[1]+.5*Math.sin(ang),p[2]]);line(c,[x,y],nose,'#aff4be',2.4);
      c.beginPath();c.arc(x,y,5.1,0,7);c.fillStyle='#97efae';c.fill();c.strokeStyle='#183e30';c.lineWidth=2;c.stroke();}
  }
}
