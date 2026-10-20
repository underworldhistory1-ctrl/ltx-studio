import ast,asyncio,types,unittest
from pathlib import Path
from unittest.mock import AsyncMock
root=Path(__file__).resolve().parents[1]/'studio'
tree=ast.parse((root/'engine.py').read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine');cl.body=[n for n in cl.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='free_h3_cache']
class Response:
 def __init__(self,q,status=200):self.q=q;self.status=status
 async def __aenter__(self):return self
 async def __aexit__(self,*a):pass
 async def json(self):return self.q
class Session:
 def __init__(self,q,status):self.q=q;self.status=status;self.post=AsyncMock()
 async def __aenter__(self):return self
 async def __aexit__(self,*a):pass
 def get(self,*a,**kw):return Response(self.q,self.status)
class Tests(unittest.IsolatedAsyncioTestCase):
 async def test_release_only_verified_idle(self):
  for q,status,expected in [({'queue_running':[],'queue_pending':[]},200,1),({'queue_running':[1],'queue_pending':[]},200,0),({'queue_running':[],'queue_pending':[1]},200,0),({},200,0),({'queue_running':[],'queue_pending':[]},503,0),([],200,0)]:
   s=Session(q,status);ns={'load_config':lambda:{},'aiohttp':types.SimpleNamespace(ClientSession=lambda:s,ClientTimeout=lambda **kw:None)};exec(compile(ast.fix_missing_locations(ast.Module(body=[cl],type_ignores=[])),'engine.py','exec'),ns);await ns['Engine']().free_h3_cache();self.assertEqual(s.post.await_count,expected)
if __name__=='__main__':unittest.main()
