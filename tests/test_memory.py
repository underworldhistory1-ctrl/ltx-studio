import ast,asyncio,json,types,unittest
from pathlib import Path
from unittest.mock import AsyncMock
p=Path(__file__).resolve().parents[1]/'studio';tree=ast.parse((p/'engine.py').read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine');keep={'__init__','reusable_vram_mb','execute','_delayed_vram_free','release_temporary_memory'};cl.body=[n for n in cl.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in keep];ns={'asyncio':asyncio,'ComfyClient':object,'sysmon':types.SimpleNamespace(),'read_json':lambda *a:{'pid':123},'DIRS':{'state':Path('.')},'aiohttp':types.SimpleNamespace(ClientError=ConnectionError),'load_config':lambda:{}};exec(compile(ast.fix_missing_locations(ast.Module(body=[cl],type_ignores=[])),'engine.py','exec'),ns);Engine=ns['Engine']
class Tests(unittest.IsolatedAsyncioTestCase):
 def make(self):
  c=types.SimpleNamespace(queue=AsyncMock(return_value={'queue_running':[],'queue_pending':[]}),free=AsyncMock());e=Engine(c);e._execute=AsyncMock();return e
 def test_own_memory_reusable(self):
  ns['sysmon'].gpu_info=lambda:{'available':True,'vram_total_mb':32000,'vram_used_mb':24000,'procs':[{'pid':123,'mb':23000}]};self.assertEqual(self.make().reusable_vram_mb(),31000)
 def test_external_memory_not_credited(self):
  ns['sysmon'].gpu_info=lambda:{'available':True,'vram_total_mb':32000,'vram_used_mb':24000,'procs':[{'pid':999,'mb':23000}]};self.assertEqual(self.make().reusable_vram_mb(),8000)
 async def test_stale_timer(self):
  e=self.make();e._idle_generation=2;await e._delayed_vram_free(0,1);e.comfy.free.assert_not_awaited()
 async def test_busy_queue(self):
  for q in [{'queue_running':[1],'queue_pending':[]},{'queue_running':[],'queue_pending':[1]}]:
   e=self.make();e.comfy.queue.return_value=q;await e._delayed_vram_free(0,0);e.comfy.free.assert_not_awaited()
 async def test_next_execute_cancels_timer(self):
  e=self.make();e._idle_free_task=asyncio.create_task(e._delayed_vram_free(10,0));await e.execute({},None);e.comfy.free.assert_not_awaited();e._execute.assert_awaited_once()
 async def test_inflight_free_finishes_before_execute(self):
  e=self.make();entered=asyncio.Event();release=asyncio.Event();events=[]
  async def free(**kw):events.append('free_start');entered.set();await release.wait();events.append('free_end')
  async def run(*a):events.append('execute')
  e.comfy.free.side_effect=free;e._execute.side_effect=run;e._idle_free_task=asyncio.create_task(e._delayed_vram_free(0,0));await entered.wait();t=asyncio.create_task(e.execute({},None));await asyncio.sleep(0);self.assertEqual(events,['free_start']);release.set();await t;self.assertEqual(events,['free_start','free_end','execute'])
 async def test_unknown_queue_is_not_idle(self):
  for q in ({}, {'queue_running':[]}, {'queue_running':None,'queue_pending':[]}, []):
   e=self.make();e.comfy.queue.return_value=q;await e._delayed_vram_free(0,0);e.comfy.free.assert_not_awaited()
 async def test_failed_queue_is_not_idle(self):
  e=self.make();e.comfy.queue.side_effect=ConnectionError();await e._delayed_vram_free(0,0);e.comfy.free.assert_not_awaited()
 async def test_adaptive_cache_not_reset(self):
  for cfg in ({'cache_mode':'ram'},{'cache_mode':'auto','comfy_extra_args':['--cache-ram','8','16']}):
   ns['load_config']=lambda:cfg;e=self.make();await e.release_temporary_memory();e.comfy.free.assert_not_awaited()
 async def test_no_cache_keeps_cleanup(self):
  ns['load_config']=lambda:{'cache_mode':'none'};e=self.make();await e.release_temporary_memory();e.comfy.free.assert_awaited_once_with(unload=False,free_memory=True)
if __name__=='__main__':unittest.main()
