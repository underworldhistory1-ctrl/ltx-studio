const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm'),fs=require('node:fs');
function monitor(initial=[]){
 const elements=new Map(),intervals=[],timeouts=[],listeners={},memory=new Map(),calls=[],alerts=[];let jobs=initial;
 function element(){return {style:{},disabled:false,innerText:'',innerHTML:'',appendChild(){},classList:{add(){},remove(){}}};}
 const document={hidden:false,addEventListener:(type,fn)=>listeners[type]=fn,querySelector:()=>element(),getElementById:id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);},createElement:element};
 const context={document,window:{addEventListener:(type,fn)=>listeners[type]=fn},URL,location:{href:'https://ltx.test/'},history:{replaceState(_a,_b,url){context.location.href=String(url);}},sessionStorage:{getItem:k=>memory.get(k),setItem:(k,v)=>memory.set(k,v),removeItem:k=>memory.delete(k)},
  EventSource:class{},setInterval:fn=>intervals.push(fn),setTimeout(fn,ms){const timer={fn,ms};timeouts.push(timer);return timer;},clearTimeout(timer){if(timer)timer.cleared=true;},console:{error(){},log(){}},alert:message=>alerts.push(message),AbortSignal,AbortController,
  fetch:async(url)=>{calls.push(url);return {ok:true,json:async()=>url==='/api/status'?{gpu:{},comfy:{},active_job:true,queue_length:0}:jobs};}};
 vm.createContext(context);vm.runInContext(fs.readFileSync('studio/static/app.js','utf8')+';globalThis.monitor={initSSEJobs,refreshJobsList,updateHardwareTelemetry,source:renderSourceMedia,submit:submitJob,retry:retryJob,prepare:()=>{currentSource={filename:"Scene.mp4",filepath:"/inputs/scene.mp4"};collectSettings=()=>({mode:"refine"});},select:id=>activeJobId=id,id:()=>activeJobId,stream:()=>sseJobs};',context);
 return {api:context.monitor,context,elements,intervals,timeouts,listeners,memory,calls,alerts,setJobs:value=>jobs=value};
}
const queued={id:'ltx_user',name:'Scene.mp4',status:'queued',queue_position:1,stage:'Waiting for H3',progress:0,settings:{mode:'refine'}};
const running={...queued,status:'running',queue_position:0,stage:'Rendering chunk 1/1',progress:25,step:2,max_steps:8};
test('polling repairs a missed queued-to-running SSE transition',async()=>{
 const m=monitor([queued]);m.api.select(queued.id);m.api.initSSEJobs();await m.api.refreshJobsList();assert.match(m.elements.get('active-progress-pct').innerText,/Queued/);
 m.setJobs([running]);assert.ok(m.intervals.length,'monitor must poll independently of SSE');await m.intervals[0]();assert.equal(m.elements.get('active-step-info').innerText,'Step 2/8');assert.equal(m.elements.get('active-progress-pct').innerText,'25.0%');
});
test('fresh page adopts the running job and leaves a durable job link',async()=>{
 const m=monitor([running,queued]);m.api.initSSEJobs();await m.api.refreshJobsList();assert.equal(m.api.id(),running.id);assert.equal(new URL(m.context.location.href).searchParams.get('job'),running.id);assert.equal(m.memory.get('ltx.activeJobId'),running.id);
});
test('reconnection immediately refreshes the authoritative job snapshot',async()=>{
 const m=monitor([running]);m.api.initSSEJobs();assert.equal(typeof m.api.stream().onopen,'function');await m.api.stream().onopen();assert.equal(m.api.id(),running.id);assert.ok(m.calls.includes('/api/jobs'));
});
test('hardware badge includes the active render instead of overwriting it with zero',async()=>{
 const m=monitor();await m.api.updateHardwareTelemetry();assert.equal(m.elements.get('queue-count-badge').innerText,1);assert.equal(m.elements.get('conn-text').innerText,'LTX Rendering');
});
test('concurrent job refreshes share a single network request',async()=>{
 const m=monitor([running]);let release;m.context.fetch=()=>{m.calls.push('request');return new Promise(resolve=>release=()=>resolve({ok:true,json:async()=>[running]}));};
 const a=m.api.refreshJobsList(),b=m.api.refreshJobsList();assert.equal(m.calls.length,1);release();await Promise.all([a,b]);
});
test('a delayed poll cannot regress a newer running SSE update back to queued',async()=>{
 const m=monitor();m.api.select(queued.id);m.api.initSSEJobs();let release;m.context.fetch=()=>new Promise(resolve=>release=()=>resolve({ok:true,json:async()=>[queued]}));
 const pending=m.api.refreshJobsList();m.api.stream().onmessage({data:JSON.stringify(running)});release();await pending;assert.equal(m.elements.get('active-step-info').innerText,'Step 2/8');
});
test('job link restores its selected task instead of adopting an unrelated running task',async()=>{
 const other={...running,id:'ltx_other'};const m=monitor([other,running]);m.context.location.href='https://ltx.test/?job=ltx_user';m.api.initSSEJobs();await m.api.refreshJobsList();assert.equal(m.api.id(),running.id);
});
test('uploading a new source detaches the old monitor without cancelling its render',async()=>{
 const m=monitor([running]);m.api.initSSEJobs();await m.api.refreshJobsList();m.context.updateGeometryCalculation=()=>{};m.api.source({url:'/inputs/new.mp4',filename:'new.mp4'});await m.api.refreshJobsList();
 assert.equal(m.api.id(),null);assert.equal(m.elements.get('active-job-card').style.display,'none');assert.equal(new URL(m.context.location.href).searchParams.has('job'),false);assert.equal(m.calls.some(url=>url.includes('/cancel')),false);
});
test('snapshot recovery works without AbortSignal.timeout support',async()=>{
 const m=monitor([running]);m.context.AbortSignal={};await m.api.refreshJobsList();assert.equal(m.api.id(),running.id);
});

test('HTML gateway error reports HTTP status instead of a JSON parser error and never re-posts',async()=>{
 const m=monitor();m.api.prepare();let posts=0;
 m.context.fetch=async(_url,options)=>{if(options?.method==='POST')posts++;return new Response('<!DOCTYPE html><title>Bad gateway</title>',{status:502,headers:{'Content-Type':'text/html'}});};
 await m.api.submit();assert.equal(posts,1);assert.match(m.alerts[0],/HTTP 502/);assert.match(m.alerts[0],/gateway/i);assert.match(m.alerts[0],/Queue.*History/i);assert.doesNotMatch(m.alerts[0],/Unexpected token|DOCTYPE/);assert.equal(m.elements.get('enhance-btn').disabled,false);
});

test('a JSON API rejection preserves its real error and HTTP status',async()=>{
 const m=monitor();m.api.prepare();m.context.fetch=async()=>new Response(JSON.stringify({error:'Source file is missing'}),{status:400,headers:{'Content-Type':'application/json'}});
 await m.api.submit();assert.match(m.alerts[0],/Source file is missing/);assert.match(m.alerts[0],/HTTP 400/);
});

test('malformed successful API response is treated as unconfirmed submission',async()=>{
 const m=monitor();m.api.prepare();m.context.fetch=async()=>new Response('{broken',{status:200,headers:{'Content-Type':'application/json'}});
 await m.api.submit();assert.match(m.alerts[0],/could not be confirmed/i);assert.doesNotMatch(m.alerts[0],/Unexpected token/);
});

test('full render and quick test cannot submit simultaneously',async()=>{
 const m=monitor();m.api.prepare();let release,posts=0;
 m.context.fetch=(_url,options)=>{if(options?.method==='POST'){posts++;return new Promise(resolve=>release=()=>resolve(new Response(JSON.stringify({success:true,job:queued}),{status:200,headers:{'Content-Type':'application/json'}})));}return Promise.resolve({ok:true,json:async()=>[queued]});};
 const first=m.api.submit(false);assert.equal(m.elements.get('test-render-btn').disabled,true);await m.api.submit(true);assert.equal(posts,1);release();await first;assert.equal(m.api.id(),queued.id);assert.equal(m.elements.get('test-render-btn').disabled,false);assert.equal(m.alerts.length,0);
});

test('re-queue also handles an HTML gateway error without exposing parser text',async()=>{
 const m=monitor();m.context.fetch=async()=>new Response('<!DOCTYPE html><title>Gateway timeout</title>',{status:504,headers:{'Content-Type':'text/html'}});
 await m.api.retry('ltx_user');assert.match(m.alerts[0],/HTTP 504/);assert.doesNotMatch(m.alerts[0],/Unexpected token|DOCTYPE/);
});

test('a stalled submission times out, releases both buttons, and is not automatically repeated',async()=>{
 const m=monitor();m.api.prepare();let posts=0;
 m.context.fetch=(_url,options)=>{posts++;return new Promise((_resolve,reject)=>options.signal?.addEventListener('abort',()=>reject(new Error('Request timed out'))));};
 const pending=m.api.submit();const deadline=m.timeouts.find(timer=>timer.ms===45000);assert.ok(deadline,'submission needs a finite deadline');deadline.fn();await pending;
 assert.equal(posts,1);assert.equal(m.elements.get('enhance-btn').disabled,false);assert.equal(m.elements.get('test-render-btn').disabled,false);assert.match(m.alerts[0],/could not be confirmed/i);assert.equal(deadline.cleared,true);
});

test('a successful response without a job receipt cannot masquerade as an accepted render',async()=>{
 const m=monitor();m.api.prepare();m.context.fetch=async()=>new Response(JSON.stringify({success:true}),{status:200,headers:{'Content-Type':'application/json'}});
 await m.api.submit();assert.equal(m.api.id(),null);assert.match(m.alerts[0],/valid job receipt/);
});
