import ast
import json
import pathlib
import tempfile
import types
import unittest
from unittest.mock import Mock

SOURCE = pathlib.Path(__file__).resolve().parents[1] / 'installer' / 'install.py'


def functions(*names, **context):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'Path': pathlib.Path, 'json': json, **context}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope


class RecoveryTests(unittest.TestCase):
    def test_first_service_request_creates_private_token_instead_of_deadlock(self):
        import os
        request = Mock(return_value=types.SimpleNamespace(status_code=200, text='{"success":true}'))
        scope = functions('studio_api', os=os, requests=types.SimpleNamespace(request=request))
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'config').mkdir()
            (root / 'config/studio.json').write_text('{"port":7860}')
            self.assertEqual(scope['studio_api'](root, '/api/internal/comfy/start')[0], 200)
            token = (root / 'state/internal.token').read_text()
            self.assertEqual(len(token), 48)
            self.assertEqual(request.call_args.kwargs['headers']['X-Internal-Token'], token)
            scope['studio_api'](root, '/api/internal/comfy/start')
            self.assertEqual((root / 'state/internal.token').read_text(), token)

    def test_explicit_gguf_opt_in_reaches_installer(self):
        tree = ast.parse((SOURCE.parents[1] / 'launcher.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_installer')
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            call = Mock(return_value=0)
            scope = {'ROOT': root, 'INSTALLER_DIR': root / 'installer', 'STATE_DIR': root, 'get_python_exe': lambda: 'python', 'log': Mock(), 'subprocess': types.SimpleNamespace(call=call)}
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'launcher-fixture', 'exec'), scope)
            for optout, expected in [(True, '--no-gguf'), (False, '--with-gguf')]:
                scope['run_installer'](types.SimpleNamespace(no_gguf=optout, skip_models=False, verify_only=True, hf_token=None))
                self.assertIn(expected, call.call_args.args[0])
                self.assertIn('--verify-only', call.call_args.args[0])

    def test_cli_bootstraps_missing_studio_without_recursive_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            dit = root / 'models/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors'
            dit.parent.mkdir(parents=True)
            with dit.open('wb') as f:
                f.truncate(10_000_001)
            run = Mock()
            api = Mock(side_effect=[None, (200, '{"success":true}')])
            scope = functions('phase_services', studio_api=api, run=run, venv_python=lambda r: r / 'venv/bin/python')
            self.assertTrue(scope['phase_services'](Mock(), root, None, True))
            command = run.call_args.args[0]
            self.assertIn('--run-only', command)
            self.assertIn('--daemon', command)
            self.assertNotIn('--install', command)

    def test_http_200_with_failed_start_is_not_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            dit = root / 'models/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors'
            dit.parent.mkdir(parents=True)
            with dit.open('wb') as f:
                f.truncate(10_000_001)
            scope = functions('phase_services', studio_api=lambda *a: (200, '{"success":false}'))
            self.assertFalse(scope['phase_services'](Mock(), root, None, True))

    def test_missing_required_weight_fails_integrity_phase(self):
        scope = functions('phase_verify', MANIFEST={'models': [{'name': 'missing', 'dir': 'vae', 'group': 'core', 'required': True}]})
        with self.assertRaisesRegex(RuntimeError, 'required model is missing'):
            scope['phase_verify'](Mock(), pathlib.Path('absent-fixture'), types.SimpleNamespace(with_gguf=False), [])

    def test_missing_file_is_reported_without_stat_exception(self):
        verify = functions('verify_file')['verify_file']
        self.assertEqual(verify(pathlib.Path('absent-fixture'), 'safetensors', 100)[0], False)

    def test_resume_uses_actual_model_store_and_only_missing_bytes(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'models').mkdir()
            usage = Mock(return_value=types.SimpleNamespace(free=3_000_000_000))
            scope = functions('phase_preflight', os=os, shutil=types.SimpleNamespace(disk_usage=usage, which=lambda x: x), subprocess=types.SimpleNamespace(check_output=lambda *a, **kw: 'GPU'), MANIFEST={'models': [{'dir': 'vae', 'name': 'existing', 'kind': 'safetensors', 'size': 42_000_000_000, 'group': 'core'}]}, verify_file=lambda *a: (True, 'ok'))
            scope['phase_preflight'](Mock(), root, types.SimpleNamespace(with_gguf=False, skip_models=False))
            usage.assert_called_once_with(root / 'models')
            self.assertEqual(list(root.glob('.ltx-write-probe-*')), [])


if __name__ == '__main__':
    unittest.main()
