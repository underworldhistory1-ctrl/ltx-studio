import os,pathlib,subprocess,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class BaseEnvironmentTests(unittest.TestCase):
    @unittest.skipIf(os.name=='nt','Tests Linux venv symlink identity')
    def test_resume_selects_host_packages_even_when_python_binaries_resolve_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);host=root/'host';target=root/'studio'
            for env in (host,target/'venv'):
                subprocess.run([sys.executable,'-m','venv','--without-pip',str(env)],check=True)
            hostpy=host/'bin/python';targetpy=target/'venv/bin/python'
            self.assertEqual(hostpy.resolve(),targetpy.resolve())
            sp=pathlib.Path(subprocess.check_output([str(hostpy),'-c','import site;print(site.getsitepackages()[0])'],text=True).strip())
            for name,body in [('torch',"class cuda:\n @staticmethod\n def is_available(): return True\n__version__='fixture'"),('comfy_kitchen',"__version__='fixture'")]:
                (sp/name).mkdir();(sp/name/'__init__.py').write_text(body)
            code="import sys,pathlib,subprocess,ast;from pathlib import Path;tree=ast.parse(Path(sys.argv[1],'installer/install.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('phase_venv','venv_python')];exec(compile(ast.Module(body=nodes,type_ignores=[]),'installer-fixture','exec'));S=type('S',(),{'phase':lambda *a:None,'log':lambda *a:None});phase_venv(S(),Path(sys.argv[2]),None,[Path(sys.executable),Path(sys.argv[3])])"
            subprocess.run([str(targetpy),'-c',code,str(ROOT),str(target),str(hostpy)],check=True,capture_output=True)
            targetsp=pathlib.Path(subprocess.check_output([str(targetpy),'-c','import site;print(site.getsitepackages()[0])'],text=True).strip())
            self.assertEqual((targetsp/'zz_ltx_base_env.pth').read_text().strip(),str(sp))
